"""Stage 4: the single held-out evaluation (sub-097..sub-125) of the frozen pipeline.

Runs once: refuses if mlm/models/heldout_report.json exists. Refuses changed artifacts or data
(sha256 against frozen_pipeline.json). Nothing is tuned here; every threshold comes frozen.
Configurations: multimodal RF with missing_modality_policy "suppress" and "fallback_unimodal"
(SIMULATED_ENV-only RF when IMU is unusable), SIMULATED_ENV-only RF, IMU-only RF (ablation), each at
its two pre-declared operating points (frozen_pipeline.json): the CV-selected entry threshold
(primary) and the locked Decision 4 entry threshold 0.65.
Usage: python scripts/evaluate_heldout.py
"""

import json
import sys
import time
from collections import defaultdict

import joblib

from lyra_mlm.evaluation import SEIZEIT2_PLANNED_SPLIT, Partition
from lyra_mlm.seizeit2 import SeizeIT2Adapter
from lyra_mlm.training import PRIMARY_NAMES, TRACKS, summarise
from train_stage4 import DATA, MODELS, PROC, d_groups, operating_points, report, sha256, total


def main() -> None:
    out = MODELS / "heldout_report.json"
    if out.exists():
        sys.exit(f"{out} exists: the held-out set has been evaluated once already. Not running again.")
    pipe = json.loads((MODELS / "frozen_pipeline.json").read_text())
    for name, digest in pipe["data"].items():
        if sha256(PROC / name) != digest:
            sys.exit(f"{name} changed since the pipeline was frozen")
    models = {}
    for track, m in pipe["models"].items():
        if sha256(MODELS / m["file"]) != m["sha256"]:
            sys.exit(f"{m['file']} does not match its frozen sha256")
        if m["feature_names"] != [PRIMARY_NAMES[c] for c in TRACKS[track].columns]:
            sys.exit(f"{track}: feature order differs from the frozen model")
        models[track] = joblib.load(MODELS / m["file"])
    if set(pipe["training"]["participants"]) & SEIZEIT2_PLANNED_SPLIT.held_out:
        sys.exit("frozen models were trained on held-out participants")

    recs = defaultdict(list)
    for m in json.loads((PROC / "windows_manifest.json").read_text()):
        if "excluded" not in m and SEIZEIT2_PLANNED_SPLIT.partition_of(m["participant_id"]) is Partition.TEST:
            recs[m["participant_id"]].append(m["recording_id"])
    t0, adapter, groups = time.time(), SeizeIT2Adapter(DATA / "seizeit2"), d_groups()
    pts = pipe["operating_points"]["by_configuration"]
    entries = sorted({pt[k] for pt in pts.values() for k in ("cv_selected", "locked_decision4")})
    tallies = defaultdict(dict)
    for p, rids in sorted(recs.items()):
        for k, t in operating_points(rids, models, adapter, entries).items():
            tallies[k][p] = t
    results = defaultdict(dict)
    for config, pt in pts.items():
        for point in ("cv_selected", "locked_decision4"):
            per_part = tallies[(config, pt[point])]
            r = {"entry_threshold": pt[point]} | report(per_part, groups)
            top2 = sorted(per_part, key=lambda q: per_part[q].fbtcs, reverse=True)[:2]
            rest = {q: t for q, t in per_part.items() if q not in top2}
            r["excluding_top2_fbtcs_participants"] = {"participants_removed": top2} | {
                k: v for k, v in summarise(total(rest.values()), rest).items()
                if k in ("fbtcs_events", "detected", "event_sensitivity", "event_sensitivity_macro")}
            r["per_participant"] = {q: {k: v for k, v in summarise(t).items() if k in (
                "fbtcs_events", "detected", "event_sensitivity", "interictal_false_alarms", "interictal_valid_monitored_h",
                "interictal_fa_per_24h", "non_fbtcs_seizures", "non_fbtcs_seizure_alert_rate", "suppressed_fraction")}
                for q, t in per_part.items()}
            results[config][point] = r
    out.write_text(json.dumps({"evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pipeline_commit": pipe["git_commit"],
                               "participants": sorted(recs), "recordings": sum(map(len, recs.values())),
                               "results": results}, indent=1, default=float))
    for name, by_point in results.items():
        for point, r in by_point.items():
            print(f"{name:30s} {point:16s} entry {r['entry_threshold']:.2f} detected {r['detected']}/{r['fbtcs_events']} "
                  f"(macro {r['event_sensitivity_macro']}) FA/24h {r['interictal_fa_per_24h']} auprc {r['auprc']}", flush=True)
    print(f"written {out}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
