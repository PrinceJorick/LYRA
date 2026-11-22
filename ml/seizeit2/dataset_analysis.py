"""Aggregate summary of SeizeIT2 and of the regenerated lyra_mlm Stage 3 windows.

    python -m ml.seizeit2.dataset_analysis   # -> models/seizeit2/dataset_summary.json

Only counts and distributions are written; no raw samples and no per-window values.
"""

from __future__ import annotations

import collections
import csv
import json

import numpy as np
from lyra_mlm.annotations import AnnotationKind
from lyra_mlm.seizeit2 import ACC_SCALE_VALID, GYRO_DPS_PER_UNIT, SeizeIT2Adapter

from . import config as C
from .data_loader import load_windows, window_file

QUALITY = ("VALID", "DEGRADED", "INVALID", "MISSING", "UNSYNCHRONIZED")


def main() -> None:
    ad = SeizeIT2Adapter(C.DATASET)
    rids = ad.recording_ids()
    labels, skipped, durations, hours = collections.Counter(), [], [], 0.0
    fbtcs_by_part = collections.Counter()
    for rid in rids:
        anns, sk = ad.parse_annotations(rid)
        skipped += sk
        hours += float(ad._rows(rid)[0]["recordingDuration"]) / 3600
        for a in anns:
            labels[a.raw_label] += 1
            if a.kind is AnnotationKind.SEIZURE_FBTCS:
                fbtcs_by_part[rid.split("_")[0]] += 1
                durations.append(a.duration_s)
    with (C.DATASET / "participants.tsv").open() as fh:
        sex = collections.Counter(r["sex"] for r in csv.DictReader(fh, delimiter="\t"))

    win, states, imu_states = collections.Counter(), collections.Counter(), collections.Counter()
    no_windows, imu_never_usable, ev = [], 0, {"in_recordings_with_windows": 0, "with_evidence": 0, "without": 0}
    evidence_windows = monitored = 0
    for rid in rids:
        if not window_file(rid).exists():
            no_windows.append(rid)
            continue
        with np.load(window_file(rid)) as z:
            win.update({{1: "positive", 0: "negative", -1: "ignore"}[int(k)]: int(v)
                        for k, v in zip(*np.unique(z["label"], return_counts=True))})
            states.update(QUALITY[int(k)] for k in z["overall"])
            imu_states.update(QUALITY[int(k)] for k in z["imu_state"])
            imu_never_usable += int(not np.isin(z["imu_state"], (0, 1)).any())
        r = load_windows(rid, ad)
        evidence_windows += int(r.evidence.sum())
        for on, off in r.fbtcs:
            ev["in_recordings_with_windows"] += 1
            covered = r.evidence[(r.t_end > on) & (r.t_end - C.WINDOW_S < off)].any()
            ev["with_evidence" if covered else "without"] += 1
    ev["in_recordings_without_windows"] = sum(
        1 for rid in no_windows for a in ad.annotations(rid) if a.kind is AnnotationKind.SEIZURE_FBTCS)
    monitored = evidence_windows * C.HOP_S / 3600
    d = np.asarray(durations)
    summary = {
        "dataset": "SeizeIT2, BIDS (OpenNeuro ds005873 v1.1.0, CC0)",
        "participants": len({r.split("_")[0] for r in rids}), "sex": dict(sex), "recordings": len(rids),
        "recorded_hours": round(hours, 1),
        "annotations_by_raw_label": dict(labels.most_common()),
        "dropped_corrupt_annotation_rows": skipped,
        "target_fbtcs": {"events": int(d.size), "participants": len(fbtcs_by_part),
                         "duration_s": {"min": float(d.min()), "median": float(np.median(d)), "max": float(d.max())},
                         **ev},
        "stage3": {
            "pipeline": "lyra_mlm prepare_recording via ml/seizeit2/prepare.py (D12-D16; 4 s windows, 2 s step)",
            "recordings_with_windows": len(rids) - len(no_windows),
            "recordings_without_windows (no EMG and no chest IMU)": len(no_windows),
            "recordings_with_imu_never_usable": imu_never_usable,
            "windows_by_label": dict(win), "windows_by_overall_quality": dict(states),
            "windows_by_imu_quality": dict(imu_states),
            "evidence_windows (both modalities usable, all features available)": evidence_windows,
            "valid_monitored_hours": round(monitored, 1),
        },
        "imu_calibration": {
            "rule": "lyra_mlm.seizeit2.calibrate_imu: ACC and GYR / median |acc|; GYR x deg/s per unit; "
                    "median |acc| outside ACC_SCALE_VALID -> all IMU samples INVALID",
            "acc_scale_valid": list(ACC_SCALE_VALID), "gyro_deg_s_per_unit": GYRO_DPS_PER_UNIT,
            "acc_scale_regimes_from_full_scan": {"~1/2": 245, "~1/6": 1837, "pinned_32.768": 347, "other_invalid": 58,
                                                 "scanned_recordings": 2487,
                                                 "source": "session 4 scan of every EMG+MOV recording (same rule)"},
            "gyro_unit_evidence": "evaluation/gyro_calibration.json",
        },
    }
    out = C.MODEL_DIR / "dataset_summary.json"
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: summary[k] for k in ("participants", "recordings", "recorded_hours", "target_fbtcs",
                                              "stage3")}, indent=1))


if __name__ == "__main__":
    main()
