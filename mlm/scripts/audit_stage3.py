"""Stage 3 audit of mlm/data/processed/ against the raw SeizeIT2 files and the locked decisions.

Re-derives every window label from events.tsv, checks the 4 s / 2 s grid and the expected window
count per recording (from EDF header durations), availability masks, that unusable modalities are
NaN (never 0), and the SIMULATED_ENV_FROM_RAW provenance. Files written before provenance was
stored are backfilled by re-running the conditioning on their raw EMG. Only labels, quality codes
and availability are read for held-out participants; the raw-scale check uses Train/CV only.
Usage: python scripts/audit_stage3.py [--data mlm/data]
"""

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from lyra_mlm.annotations import LabelConfig, label_window
from lyra_mlm.edf import read_edf_header
from lyra_mlm.evaluation import SEIZEIT2_PLANNED_SPLIT, Partition
from lyra_mlm.features import feature_names
from lyra_mlm.prep import QUALITY_CODE, PrepConfig, primary_emg_metadata
from lyra_mlm.preprocess import raw_to_simulated_env
<<<<<<< HEAD
from lyra_mlm.quality import QUALITY_RANK, USABLE, QualityState
=======
from lyra_mlm.quality import USABLE
>>>>>>> 414a2140836e55fed760a18fff6c49fe2bc0082d
from lyra_mlm.seizeit2 import SeizeIT2Adapter

DATA = Path(__file__).resolve().parents[1] / "data"
STATE = {v: k.value for k, v in QUALITY_CODE.items()}
USABLE_CODES = [QUALITY_CODE[s] for s in USABLE]
<<<<<<< HEAD
RANK_OF_CODE = np.array([QUALITY_RANK[s] for s in QualityState])  # indexed by stored code
=======
>>>>>>> 414a2140836e55fed760a18fff6c49fe2bc0082d


def expected_windows(adapter: SeizeIT2Adapter, rid: str, cfg: PrepConfig) -> int:
    p, info = adapter._paths(rid), adapter.info(rid)
    spans = [read_edf_header(p[k]).duration_s for k, ok in (("emg", info.emg_rate_hz), ("mov", info.imu_rate_hz)) if ok]
    d = max(spans, default=0.0)
    return int((d - cfg.window.length_s) // cfg.window.step_s) + 1 if d >= cfg.window.length_s else 0


def backfill(path: Path, adapter: SeizeIT2Adapter, cfg: PrepConfig) -> None:
    rec = adapter.load_recording(str(np.load(path)["recording_id"]))
    sim = raw_to_simulated_env(rec.emg, cfg.sim_env) if rec.emg is not None else None
    meta = primary_emg_metadata(replace(rec, emg=sim if sim is not None and sim.n else None))
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays, **meta)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    args = ap.parse_args()
    raw, out = args.data / "seizeit2", args.data / "processed"
    adapter, cfg, labels_cfg = SeizeIT2Adapter(raw), PrepConfig(), LabelConfig()
    manifest = {m["recording_id"]: m for m in json.loads((out / "windows_manifest.json").read_text())}
    rids = adapter.recording_ids()
    problems, backfilled = [], 0
    totals = Counter()
    by_partition = {Partition.TRAIN: Counter(), Partition.TEST: Counter()}
    quality = {k: Counter() for k in ("overall", "emg_state", "imu_state", "raw_emg_state")}
    train_scale = []
    n_primary, i_rms = len(feature_names(cfg.primary)), feature_names(cfg.primary).index("emg_simenv_rms")
    emg_block = [i for i, n in enumerate(feature_names(cfg.primary)) if n.startswith("emg_")]
    imu_block = [i for i, n in enumerate(feature_names(cfg.primary)) if n.startswith("imu_")]

    missing_from_manifest = sorted(set(rids) - set(manifest))
    excluded = {r: m["excluded"] for r, m in manifest.items() if "excluded" in m}
    for rid in rids:
        if rid in excluded or rid not in manifest:
            continue
        path = out / "windows" / rid.split("_")[0] / f"{rid}.npz"
        with np.load(path) as z:
            has_prov = "primary_emg_provenance" in z.files
        if not has_prov:
            backfill(path, adapter, cfg)
            backfilled += 1
        with np.load(path) as z:
            start, end, lab = z["start_s"], z["end_s"], z["label"]
            n = lab.size
            if n != expected_windows(adapter, rid, cfg):
                problems.append(f"{rid}: {n} windows, expected {expected_windows(adapter, rid, cfg)}")
            if not (np.allclose(end - start, 4.0) and np.allclose(np.diff(start), 2.0) and np.isclose(start[0], 0.0)):
                problems.append(f"{rid}: window grid is not 4 s / 2 s from t=0")
            anns = adapter.annotations(rid)
            relabel = np.array([int(label_window(s, e, anns, labels_cfg)) for s, e in zip(start, end)], np.int8)
            if not np.array_equal(relabel, lab):
                problems.append(f"{rid}: {int((relabel != lab).sum())} stored labels differ from re-derived labels")
            if not (np.array_equal(z["primary_available"], ~np.isnan(z["primary"]))
                    and np.array_equal(z["secondary_available"], ~np.isnan(z["secondary"]))):
                problems.append(f"{rid}: availability mask does not match NaN pattern")
            emg_bad = ~np.isin(z["emg_state"], USABLE_CODES)
            imu_bad = ~np.isin(z["imu_state"], USABLE_CODES)
            if z["primary_available"][np.ix_(emg_bad, emg_block)].any() or \
                    z["primary_available"][np.ix_(imu_bad, imu_block)].any():
                problems.append(f"{rid}: features present for an unusable modality")
<<<<<<< HEAD
            if (RANK_OF_CODE[z["emg_state"]] < RANK_OF_CODE[z["raw_emg_state"]]).any():
                problems.append(f"{rid}: simulated ENV rated better than its RAW source")
=======
>>>>>>> 414a2140836e55fed760a18fff6c49fe2bc0082d
            if z["primary"].shape[1] != n_primary:
                problems.append(f"{rid}: primary feature width {z['primary'].shape[1]} != {n_primary}")
            st, prov = str(z["primary_emg_signal_type"]), str(z["primary_emg_provenance"])
            if st and (st != "SIMULATED_ENV_FROM_RAW" or "SIMULATED_ENV_FROM_RAW:" not in prov
                       or "HARDWARE" in prov.upper()):
                problems.append(f"{rid}: primary EMG provenance not SIMULATED_ENV_FROM_RAW")
            usable = np.isin(z["overall"], USABLE_CODES)
            part = SEIZEIT2_PLANNED_SPLIT.partition_of(rid.split("_")[0])
            for name, code in (("positive", 1), ("negative", 0), ("ignore", -1)):
                k = int((lab == code).sum())
                totals[f"all_{name}"] += k
                totals[f"usable_{name}"] += int(((lab == code) & usable).sum())
                by_partition[part][f"usable_{name}"] += int(((lab == code) & usable).sum())
                by_partition[part][f"all_{name}"] += k
            totals["total"] += n
            totals["signal_excluded"] += int((~usable).sum())
            by_partition[part]["total"] += n
            for k in quality:
                quality[k].update(STATE[int(c)] for c in z[k])
            if part is Partition.TRAIN:
                rms = z["primary"][:, i_rms]
                train_scale.append(rms[~np.isnan(rms)])
    t = totals
    conservation = t["total"] == t["usable_positive"] + t["usable_negative"] + t["usable_ignore"] + t["signal_excluded"]
    rms = np.concatenate(train_scale) if train_scale else np.empty(0)
    report = {
        "recordings_in_dataset": len(rids),
        "recordings_processed": len(rids) - len(excluded) - len(missing_from_manifest),
        "recordings_excluded": len(excluded),
        "exclusion_reasons": dict(Counter(v.split(": ", 1)[1] for v in excluded.values())),
        "recordings_missing_from_manifest": missing_from_manifest, "provenance_backfilled": backfilled,
        "windows": dict(t), "conservation_holds": bool(conservation),
        "by_partition": {str(p): dict(c) for p, c in by_partition.items()},
        "quality": {k: dict(v) for k, v in quality.items()},
        "train_only_raw_scale_check": {
            "emg_simenv_rms_min": float(rms.min()) if rms.size else None,
            "emg_simenv_rms_median": float(np.median(rms)) if rms.size else None,
            "note": "positive, uV-scale values: not standardized (a z-scored feature would centre on 0)"},
        "problems": problems,
    }
    (out / "audit.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "problems"}, indent=1))
    print(f"{len(problems)} problems", *problems[:20], sep="\n")


if __name__ == "__main__":
    main()
