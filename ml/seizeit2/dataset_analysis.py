"""Aggregate dataset summary from the annotations and the feature-cache index.

    python -m ml.seizeit2.dataset_analysis   # -> models/seizeit2/dataset_summary.json

Only counts and distributions are written; no raw samples and no per-recording signal values.
"""

from __future__ import annotations

import collections
import csv
import json

import numpy as np

from . import config as C
from . import data_loader as D


def _stats(x) -> dict:
    x = np.asarray(x, float)
    if not x.size:
        return {}
    return {"n": int(x.size), "min": float(x.min()), "median": float(np.median(x)),
            "mean": round(float(x.mean()), 2), "max": float(x.max())}


def main() -> None:
    runs = D.list_runs()
    index = json.loads((C.CACHE / "features" / "index.json").read_text())
    missing_index = [r.run for r in runs if r.run not in index]
    by_label = collections.Counter(a.raw_label for r in runs for a in r.annotations)
    by_kind = collections.Counter(a.kind.value for r in runs for a in r.annotations)
    tgt = [(r, a) for r in runs for a in r.annotations if a.kind in D.TARGET_KINDS]
    with (C.DATASET / "participants.tsv").open() as fh:
        sex = collections.Counter(row["sex"] for row in csv.DictReader(fh, delimiter="\t"))

    def both(r: D.Run) -> bool:
        m = index.get(r.run, {})
        return bool(m.get("has_emg") and m.get("has_imu"))

    vig = collections.Counter()
    with_both = sum(both(r) for r, _ in tgt)
    for ev in C.DATASET.glob("sub-*/ses-*/eeg/*_events.tsv"):
        with ev.open() as fh:
            vig.update(row["vigilance"] for row in csv.DictReader(fh, delimiter="\t")
                       if row["eventType"] in C.TARGET_LABELS)

    windows = {}
    for W in C.WINDOWS_S:
        tot = collections.Counter()
        for m in index.values():
            w = m.get("windows", {}).get(str(W))
            if w:
                tot.update(w)
        windows[f"W{W}"] = dict(tot)

    summary = {
        "dataset": "SeizeIT2, BIDS (OpenNeuro ds005873 v1.1.0, CC0)",
        "subjects": len({r.subject for r in runs}), "sex": dict(sex),
        "recordings_runs": len(runs),
        "recorded_hours": round(sum(r.duration_s for r in runs) / 3600, 1),
        "modality_files": {
            "runs_without_emg_file": sum(r.emg_path is None for r in runs),
            "runs_without_mov_file": sum(r.mov_path is None for r in runs),
            "runs_with_mov_file_but_no_chest_imu": sum(
                1 for r in runs if r.mov_path is not None and r.emg_path is not None
                and r.run in index and not index[r.run]["has_imu"]),
            "runs_with_emg_and_chest_imu": sum(both(r) for r in runs),
            "runs_with_invalid_acc_scale": sum(1 for m in index.values() if m.get("acc_scale_invalid")),
            "hours_usable_after_quality_checks": round(sum(m.get("usable_duration_s", 0) for m in index.values()) / 3600, 1),
            "runs_not_processed": missing_index,
        },
        "annotations_by_raw_label": dict(by_label.most_common()),
        "annotations_by_lyra_kind": dict(by_kind),
        "dropped_corrupt_rows": [i for r in runs for i in r.issues],
        "target_fbtcs": {
            "events": len(tgt), "subjects": len({r.subject for r, _ in tgt}),
            "events_with_emg_and_chest_imu": with_both,
            "events_usable_after_quality_checks": sum(
                len(r.intervals(D.TARGET_KINDS)) for r in runs if index.get(r.run, {}).get("windows")),
            "duration_s": _stats([a.duration_s for _, a in tgt]),
            "vigilance": dict(vig),
        },
        "windows": windows,
        "acc_scale_regimes_runs": dict(collections.Counter(
            "~1/2" if 0.3 <= a <= 0.7 else "~1/6" if 0.12 <= a < 0.3 else "pinned_32.768" if a > 32 else "other_invalid"
            for a in (m["acc_norm_median"] for m in index.values() if "acc_norm_median" in m))),
        "gyr_norm_median_per_run_file_units": _stats([m["gyr_norm_median"] for m in index.values() if "gyr_norm_median" in m]),
        "quality_notes": [
            "EDF content is git-annexed and absent from the local clone; it is fetched from the public bucket "
            "and verified against the SHA256 in each annex key.",
            "Annotation onsets are relative to the run start; all modality files of a run share that origin "
            "(EDF+C, 1 s records, record counts equal recordingDuration).",
            "Accelerometer files are labelled g but use per-file scales: median |acc| (gravity) is ~1/2 or ~1/6 "
            "in valid runs; in others it is pinned at 32.768 or zero. Valid runs are rescaled to g by their own "
            "gravity estimate; the rest are excluded (ACC_SCALE_VALID). Gyroscope channels are labelled 'g', their "
            "unit is undocumented and their scale differs between runs, so only scale-invariant gyro features are used.",
            "EMG is a single left-deltoid channel at 256 Hz with strong low-frequency (ECG/motion) content.",
            "No age, centre or medication metadata is distributed; only sex.",
            "Seizure annotations come from vEEG review; FBTCS onset is the focal onset, so the convulsive "
            "phase starts some time after the annotated onset.",
        ],
    }
    out = C.MODEL_DIR / "dataset_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("subjects", "recordings_runs", "recorded_hours",
                                              "modality_files", "target_fbtcs", "windows",
                                              "acc_scale_regimes_runs")}, indent=1))


if __name__ == "__main__":
    main()
