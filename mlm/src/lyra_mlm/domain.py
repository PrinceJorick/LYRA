"""Domain-shift checks: LYRA pod recordings vs the SeizeIT2 training distribution.

Reference = SeizeIT2 Train/CV windows only (never held-out). Descriptive statistics only: nothing
here is fitted into, or changes, a model. Two pod feature paths are compared:
- "env": MyoWare hardware ENV (what the deployed pod computes); its emg_env_* features are compared
  with the trained emg_simenv_* features by name suffix, which is a cross-signal-type comparison;
- "simenv": SIMULATED_ENV_FROM_RAW computed from the pod's RAW channel (data-collection mode), which
  is exactly the trained schema, so this gap is placement/device only.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy import stats

from .annotations import AnnotationKind
from .features import FeatureConfig, extract_features, feature_names
from .prep import QUALITY_CODE, PrepConfig, prepare_recording
from .preprocess import SimulatedEnvConfig, raw_to_simulated_env
from .quality import USABLE, QualityConfig
from .seizeit2 import SATURATION_LIMITS
from .signals import EmgSignalType, Modality, Recording, SignalStream
from .training import PRIMARY_NAMES
from .windows import segment

ENV_CFG = FeatureConfig("multimodal", EmgSignalType.ENV)
assert [n.replace("emg_env_", "emg_simenv_") for n in feature_names(ENV_CFG)] == list(PRIMARY_NAMES)
USABLE_CODES = [QUALITY_CODE[s] for s in USABLE]
PCTS = (1, 5, 50, 95, 99)


def pod_quality(meta: dict) -> QualityConfig:
    """IMU full scale as trained; MyoWare ENV/RAW saturate at the ADC full scale, if declared."""
    limits = dict(SATURATION_LIMITS)
    for name in ("env", "raw"):
        fs = meta["streams"].get(name, {}).get("full_scale_mv")
        if fs:
            limits[name] = (-1e9, float(fs))
    return QualityConfig(saturation_limits=limits)


def window_events(rec: Recording, start: np.ndarray, end: np.ndarray) -> list[str]:
    """Verbatim label of the annotation with the largest overlap per window ('unannotated' if none)."""
    out = []
    for s, e in zip(start, end):
        best, label = 0.0, "unannotated"
        for a in rec.annotations:
            ov = min(e, a.end_s) - max(s, a.onset_s)
            if ov > best:
                best, label = ov, a.raw_label
        out.append(label)
    return out


def session_features(rec: Recording, raw: SignalStream | None, qcfg: QualityConfig) -> dict[str, dict]:
    """Per path: window times, overall quality code, event label, features on the trained schema."""
    ws = list(segment(rec, PrepConfig().window, qcfg, (Modality.EMG, Modality.IMU)))
    start, end = np.array([w.start_s for w in ws]), np.array([w.end_s for w in ws])
    out = {"env": {"start_s": start, "overall": np.array([QUALITY_CODE[w.quality.overall] for w in ws], np.int8),
                   "events": window_events(rec, start, end),
                   "features": np.array([extract_features(w, ENV_CFG).values for w in ws]).reshape(-1, len(PRIMARY_NAMES))}}
    if raw is not None:
        a = prepare_recording(replace(rec, emg=raw), PrepConfig(quality=qcfg))
        out["simenv"] = {"start_s": a["start_s"], "overall": a["overall"], "label": a["label"],
                         "events": window_events(rec, a["start_s"], a["end_s"]), "features": a["primary"].astype(np.float64)}
    return out


def compare(features: np.ndarray, usable: np.ndarray, reference: np.ndarray) -> dict[str, dict]:
    """Per trained feature: pod vs reference percentiles, share of pod windows outside the reference
    1st-99th percentile range, and the two-sample KS statistic."""
    out = {}
    for j, name in enumerate(PRIMARY_NAMES):
        pod, ref = features[usable, j], reference[:, j]
        pod, ref = pod[np.isfinite(pod)], ref[np.isfinite(ref)]
        if pod.size == 0 or ref.size == 0:
            out[name] = {"pod_windows": int(pod.size)}
            continue
        rp = np.percentile(ref, PCTS)
        out[name] = {"pod_windows": int(pod.size), "ref_pct": rp.tolist(), "pod_pct": np.percentile(pod, PCTS).tolist(),
                     "outside_ref_1_99": float(np.mean((pod < rp[0]) | (pod > rp[-1]))),
                     "ks": float(stats.ks_2samp(pod, ref).statistic)}
    return out


def env_vs_simenv(env: SignalStream, raw: SignalStream) -> dict:
    """MyoWare hardware ENV against SIMULATED_ENV_FROM_RAW from the same electrodes (same pod clock).
    Descriptive: correlation, median ratio and a least-squares line. Nothing is applied."""
    sim = raw_to_simulated_env(raw, SimulatedEnvConfig())
    idx = np.clip(np.searchsorted(env.timestamps, sim.timestamps), 0, env.n - 1)
    prev = np.clip(idx - 1, 0, env.n - 1)
    idx = np.where(np.abs(env.timestamps[prev] - sim.timestamps) < np.abs(env.timestamps[idx] - sim.timestamps), prev, idx)
    ok = (np.abs(env.timestamps[idx] - sim.timestamps) < 0.5 / env.nominal_rate_hz) & sim.usable & env.usable[idx]
    e, s = env.values[idx[ok], 0], sim.values[ok, 0]
    if ok.sum() < 3 or np.std(e) == 0 or np.std(s) == 0:
        return {"paired_samples": int(ok.sum())}
    slope, intercept = np.polyfit(s, e, 1)
    pos = s > 0
    return {"paired_samples": int(ok.sum()), "pearson_r": float(np.corrcoef(e, s)[0, 1]),
            "median_ratio_env_over_simenv": float(np.median(e[pos] / s[pos])) if pos.any() else None,
            "line_env_mv_from_simenv": {"slope": float(slope), "intercept": float(intercept)}}


def report(rec: Recording, raw: SignalStream | None, meta: dict, ref_negative: np.ndarray,
           ref_positive: np.ndarray) -> dict:
    """Full per-session domain report (JSON-serializable)."""
    paths = session_features(rec, raw, pod_quality(meta))
    out = {"recording_id": rec.recording_id, "origin": str(rec.origin), "placement": rec.placement,
           "reference": {"negative_windows": int(len(ref_negative)), "positive_windows": int(len(ref_positive)),
                         "source": "SeizeIT2 Train/CV only (sub-001..sub-096)"}}
    for name, p in paths.items():
        usable = np.isin(p["overall"], USABLE_CODES)
        sim = np.array([e == "sim_convulsion" for e in p["events"]])
        out[name] = {
            "windows": int(len(p["overall"])), "usable_windows": int(usable.sum()),
            "quality": {int(k): int(v) for k, v in zip(*np.unique(p["overall"], return_counts=True))},
            "events": {e: int(sum(1 for x in p["events"] if x == e)) for e in sorted(set(p["events"]))},
            "vs_seizeit2_negatives": compare(p["features"], usable & ~sim, ref_negative),
            "sim_convulsion_vs_seizeit2_fbtcs": compare(p["features"], usable & sim, ref_positive) if (usable & sim).any() else {},
        }
        if "label" in p:
            out[name]["sim_convulsion_windows_never_negative"] = bool(
                not np.any((p["label"] == 0) & sim))
    if raw is not None and rec.emg is not None:
        out["env_vs_simenv_signal"] = env_vs_simenv(rec.emg, raw)
    out["annotation_kinds"] = sorted({str(a.kind) for a in rec.annotations})
    out["note"] = ("Descriptive domain-shift report. Simulated convulsions are scripted movements by healthy "
                   "volunteers: never seizures and never sensitivity evidence.")
    if any(a.kind is AnnotationKind.UNMAPPED for a in rec.annotations):
        out["warning"] = "UNMAPPED event labels present; check events.tsv against the protocol vocabulary"
    return out
