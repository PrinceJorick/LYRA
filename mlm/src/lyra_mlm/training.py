"""Stage 4 (authorized 2026-11-22): training-set construction, models, event confirmation over
stored windows, and event-level metrics.

Rules implemented here (locked decisions, not tunable from results):
- Train/CV is sub-001..sub-096 only; training_rows() raises LeakageError for anything else.
- Three modality tracks, never zero-imputed: a window is scored only when its track's quality
  gate passes and every feature of the track is available; otherwise it is SUPPRESSED.
- Training loss uses POSITIVE/NEGATIVE windows only (no IGNORE), usable for the track, outside
  impedance-check ('impd') annotations. Negatives are subsampled deterministically and
  proportionally per participant, inside training folds only; validation and held-out data are
  never subsampled.
- Event matching: a POSSIBLE_EVENT entry within [onset - 30 s, offset + 30 s] detects that FBTCS
  (one per event). Other alerts are non-FBTCS-seizure alerts (within 30 s of another annotated
  seizure) or interictal false alarms. FA/24h uses valid (non-suppressed) monitored hours.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .annotations import Annotation, AnnotationKind
from .confirmation import ConfirmationConfig
from .evaluation import SEIZEIT2_PLANNED_SPLIT, LeakageError, Partition
from .features import feature_names
from .prep import QUALITY_CODE, PrepConfig
from .quality import USABLE, QualityState

PRIMARY_NAMES = feature_names(PrepConfig().primary)
USABLE_CODES = np.array([QUALITY_CODE[s] for s in USABLE])
DEGRADED, UNSYNC = QUALITY_CODE[QualityState.DEGRADED], QUALITY_CODE[QualityState.UNSYNCHRONIZED]
SUPPRESSED, VALID_STATUS, DEGRADED_STATUS = 0, 1, 2
STEP_S = PrepConfig().window.step_s
MATCH_S = 30.0  # event-matching margin around an annotated seizure
SEIZURES = (AnnotationKind.SEIZURE_FBTCS, AnnotationKind.SEIZURE_GTCS, AnnotationKind.SEIZURE_OUT_OF_SCOPE)


@dataclass(frozen=True)
class Track:
    name: str
    columns: tuple[int, ...]  # indices into the stored primary feature array
    gate: str  # stored quality code that gates the track ("overall" = both modalities)


def _cols(prefix: str) -> tuple[int, ...]:
    return tuple(i for i, n in enumerate(PRIMARY_NAMES) if n.startswith(prefix))


TRACKS = {
    "multimodal": Track("multimodal", tuple(range(len(PRIMARY_NAMES))), "overall"),
    "simenv": Track("simenv", _cols("emg_simenv_"), "emg_state"),
    "imu": Track("imu", _cols("imu_"), "imu_state"),
}


def track_status(z, track: Track) -> np.ndarray:
    """SUPPRESSED / VALID / DEGRADED per window (degraded windows allowed with a flag)."""
    state = z[track.gate]
    ok = np.isin(state, USABLE_CODES) & z["primary_available"][:, track.columns].all(axis=1)
    return np.where(ok, np.where(state == DEGRADED, DEGRADED_STATUS, VALID_STATUS), SUPPRESSED).astype(np.int8)


def fallback_status(z) -> np.ndarray:
    """Windows where missing_modality_policy='fallback_unimodal' routes to the SIMULATED_ENV-only
    model: IMU unusable, EMG usable, and no cross-modal clock problem (mirrors QualityGatedInference)."""
    sim = track_status(z, TRACKS["simenv"])
    imu_down = ~np.isin(z["imu_state"], USABLE_CODES)
    clock_ok = (z["overall"] != UNSYNC) | (z["imu_state"] == UNSYNC)  # UNSYNC overall from a clock mismatch: suppress
    return np.where(imu_down & clock_ok, sim, SUPPRESSED).astype(np.int8)


def train_mask(z, track: Track) -> np.ndarray:
    return np.isin(z["label"], (0, 1)) & (track_status(z, track) != SUPPRESSED) & ~z["artifact_overlap"]


def window_keys(recording_id: str, n: int) -> np.ndarray:
    """Deterministic uniform [0, 1) key per window, used to subsample negatives reproducibly."""
    return np.random.default_rng(zlib.crc32(recording_id.encode())).random(n)


def require_train_cv(participant_ids: Iterable[str]) -> None:
    bad = sorted({p for p in participant_ids if SEIZEIT2_PLANNED_SPLIT.partition_of(p) is not Partition.TRAIN})
    if bad:
        raise LeakageError(f"not Train/CV participants: {bad[:5]}{'...' if len(bad) > 5 else ''}")


def training_rows(participant: np.ndarray, label: np.ndarray, key: np.ndarray, neg_total: dict[str, int],
                  ratio: int) -> np.ndarray:
    """Indices of a training set: every positive, plus round(f * negatives_p) lowest-key negatives
    of each participant p, with f = ratio * positives / negatives (proportional stratification).
    participant/label/key describe the candidate rows; neg_total counts each participant's usable
    negatives (all of them, not just candidates)."""
    require_train_cv(np.unique(participant))
    pos = np.flatnonzero(label == 1)
    f = ratio * pos.size / max(1, sum(neg_total.values()))
    keep = [pos]
    for p in np.unique(participant):
        cand = np.flatnonzero((participant == p) & (label == 0))
        n = min(cand.size, int(round(f * neg_total.get(str(p), 0))))
        keep.append(cand[np.argsort(key[cand], kind="stable")[:n]])
    return np.sort(np.concatenate(keep))


def make_model(family: str, params: dict, n_pos: int, n_neg: int):
    """Fresh, unfitted estimator. Scaling (LR only) lives inside the pipeline, so it is fitted on
    the training fold and nowhere else."""
    if family == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000, **params))
    if family == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(class_weight="balanced", random_state=0, n_jobs=-1, **params)
    if family == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(scale_pos_weight=n_neg / max(1, n_pos), random_state=0, n_jobs=-1,
                             tree_method="hist", eval_metric="logloss", **params)
    raise ValueError(family)


def confirm(ts: np.ndarray, score: np.ndarray, status: np.ndarray,
            cfg: ConfirmationConfig = ConfirmationConfig()) -> np.ndarray:
    """POSSIBLE_EVENT entry times. Same state machine as EventConfirmation (tested equal) on plain
    arrays: VALID and DEGRADED windows are evidence (allow_with_flag), SUPPRESSED windows hold."""
    if cfg.degraded_window_policy != "allow_with_flag" or cfg.invalid_window_policy != "hold":
        raise NotImplementedError("offline runner implements the locked policies only")
    idle, acc, active, refr = 0, 1, 2, 3
    state, count, last, until = idle, 0, None, -np.inf
    entries = []
    gap, cool, k, hi, lo = cfg.max_timestamp_gap_s, cfg.refractory_cooldown_s, cfg.consecutive_positive_required, \
        cfg.entry_threshold, cfg.exit_threshold
    for t, s, st in zip(ts.tolist(), score.tolist(), status.tolist()):
        if state == refr and t >= until:
            state = idle
        if last is not None and t - last > gap:
            count = 0
            if state == active:
                state, until = refr, t + cool
            elif state == acc:
                state = idle
        if st == SUPPRESSED:
            continue
        last = t
        if state == refr:
            continue
        if state == active:
            if s < lo:
                count, state, until = 0, refr, t + cool
            continue
        if s >= hi:
            count += 1
            if count >= k:
                state = active
                entries.append(t)
            else:
                state = acc
        else:
            count, state = 0, idle
    return np.array(entries)


def seizure_spans(annotations: Sequence[Annotation]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """(FBTCS, other annotated seizures) as (onset, offset) pairs."""
    fb = [(a.onset_s, a.onset_s + a.duration_s) for a in annotations if a.kind is AnnotationKind.SEIZURE_FBTCS]
    other = [(a.onset_s, a.onset_s + a.duration_s) for a in annotations
             if a.kind in SEIZURES and a.kind is not AnnotationKind.SEIZURE_FBTCS]
    return fb, other


BINS = 1 << 14  # window-score histogram bins for AUPRC (int32 counts keep CV tallies small)


@dataclass
class Tally:
    """Event- and window-level counts for one participant (sums over recordings)."""
    fbtcs: int = 0
    detected: int = 0
    latencies: list = field(default_factory=list)
    other_seizures: int = 0
    other_detected: int = 0
    alerts: int = 0
    fbtcs_alerts: int = 0
    other_alerts: int = 0
    interictal_fa: int = 0
    interictal_valid_h: float = 0.0
    suppressed_h: float = 0.0
    monitored_h: float = 0.0
    neg_windows: int = 0
    neg_below_entry: int = 0
    pos_hist: np.ndarray = field(default_factory=lambda: np.zeros(BINS, np.int32))
    neg_hist: np.ndarray = field(default_factory=lambda: np.zeros(BINS, np.int32))

    def add(self, other: Tally) -> Tally:
        for k, v in vars(other).items():
            setattr(self, k, getattr(self, k) + v)
        return self


def _inside(t: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(t.shape, bool)
    for on, off in spans:
        m |= (t >= on - MATCH_S) & (t <= off + MATCH_S)
    return m


def score_recording(ts: np.ndarray, label: np.ndarray, status: np.ndarray, score: np.ndarray,
                    annotations: Sequence[Annotation], cfg: ConfirmationConfig = ConfirmationConfig()) -> Tally:
    """All windows in recording order (IGNORE / DEGRADED / INVALID / MISSING included)."""
    alerts = confirm(ts, score, status, cfg)
    fb, other = seizure_spans(annotations)
    t = Tally(fbtcs=len(fb), other_seizures=len(other), alerts=int(alerts.size))
    for on, off in fb:
        hit = alerts[(alerts >= on - MATCH_S) & (alerts <= off + MATCH_S)]
        if hit.size:
            t.detected += 1
            t.latencies.append(float(hit[0] - on))
    t.other_detected = sum(bool(((alerts >= on - MATCH_S) & (alerts <= off + MATCH_S)).any()) for on, off in other)
    in_fb, in_other = _inside(alerts, fb), _inside(alerts, other)
    t.fbtcs_alerts = int(in_fb.sum())
    t.other_alerts = int((~in_fb & in_other).sum())
    t.interictal_fa = int((~in_fb & ~in_other).sum())
    scored = status != SUPPRESSED
    interictal = ~_inside(ts, fb + other)
    t.monitored_h = float(scored.sum()) * STEP_S / 3600
    t.suppressed_h = float((~scored).sum()) * STEP_S / 3600
    t.interictal_valid_h = float((scored & interictal).sum()) * STEP_S / 3600
    neg, pos = scored & (label == 0), scored & (label == 1)
    t.neg_windows = int(neg.sum())
    t.neg_below_entry = int((score[neg] < cfg.entry_threshold).sum())
    b = np.minimum((np.nan_to_num(score) * BINS).astype(np.int64), BINS - 1)
    t.pos_hist = np.bincount(b[pos], minlength=BINS).astype(np.int32)
    t.neg_hist = np.bincount(b[neg], minlength=BINS).astype(np.int32)
    return t


def binned_average_precision(pos_hist: np.ndarray, neg_hist: np.ndarray) -> float | None:
    """Average precision (step interpolation, as sklearn) from score histograms, thresholds high to low."""
    tp, fp = np.cumsum(pos_hist[::-1], dtype=np.int64), np.cumsum(neg_hist[::-1], dtype=np.int64)
    if tp[-1] == 0:
        return None
    keep = (pos_hist[::-1] + neg_hist[::-1]) > 0
    tp, fp = tp[keep], fp[keep]
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.concatenate(([0.0], recall))) * tp / (tp + fp)))


def summarise(t: Tally, per_participant: dict[str, Tally] | None = None) -> dict:
    """The seven evaluation metrics (evaluation.METRICS), FA split into interictal and non-FBTCS."""
    out = {
        "fbtcs_events": t.fbtcs, "detected": t.detected,
        "event_sensitivity": t.detected / t.fbtcs if t.fbtcs else None,
        "interictal_fa_per_24h": t.interictal_fa / t.interictal_valid_h * 24 if t.interictal_valid_h else None,
        "interictal_false_alarms": t.interictal_fa, "interictal_valid_monitored_h": round(t.interictal_valid_h, 2),
        "non_fbtcs_seizures": t.other_seizures,
        "non_fbtcs_seizure_alert_rate": t.other_detected / t.other_seizures if t.other_seizures else None,
        "non_fbtcs_seizure_alerts": t.other_alerts,
        "detection_latency_s_median": float(np.median(t.latencies)) if t.latencies else None,
        "detection_latency_s_mean": float(np.mean(t.latencies)) if t.latencies else None,
        "precision": t.fbtcs_alerts / t.alerts if t.alerts else None, "alerts": t.alerts,
        "auprc": binned_average_precision(t.pos_hist, t.neg_hist),
        "specificity": t.neg_below_entry / t.neg_windows if t.neg_windows else None,
        "valid_monitored_h": round(t.monitored_h, 2), "suppressed_h": round(t.suppressed_h, 2),
        "suppressed_fraction": t.suppressed_h / (t.monitored_h + t.suppressed_h) if t.monitored_h + t.suppressed_h else None,
    }
    if per_participant is not None:
        sens = [p.detected / p.fbtcs for p in per_participant.values() if p.fbtcs]
        out["event_sensitivity_macro"] = float(np.mean(sens)) if sens else None
        out["participants_with_fbtcs"] = len(sens)
    return out
