"""Assemble reports/lyra/ from a completed run (benchmark -> deploy -> final_eval [-> parity, adl]).

    python -m ml.lyra.report --run ml/lyra/runs/bench-<id>
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from . import config as C
from . import data as Dt

PKGS = ("numpy", "scipy", "scikit-learn", "xgboost", "emlearn", "m2cgen", "pyedflib", "pandas", "pyarrow")


def _j(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(fmt(v, nd) for v in x) + "]"
    return str(x)


def dataset_manifest() -> dict:
    ms = Dt.manifests()
    recs = []
    for m in ms:
        recs.append({"recording_id": m["recording_id"], "subject": m["subject"], "partition": m["partition"],
                     "emg": bool(m.get("emg")), "mov": bool(m.get("mov")),
                     "emg_rate_hz": (m.get("emg") or {}).get("channels", [{}])[0].get("rate_hz"),
                     "imu_rate_hz": (m.get("mov") or {}).get("channels", [{}])[0].get("rate_hz"),
                     "emg_duration_s": (m.get("emg") or {}).get("duration_s"),
                     "mov_duration_s": (m.get("mov") or {}).get("duration_s"),
                     "events_recording_duration_s": (m.get("events") or {}).get("recording_duration_s"),
                     # EDF start dates are dropped: they can be real recording dates and this manifest is published
                     "sync": {k: v for k, v in m["sync"].items() if not k.endswith("_start")} if m.get("sync") else None,
                     "events": m.get("events"), "windows": m.get("windows"),
                     "excluded_reason": m.get("excluded_reason"), "problems": m.get("problems"), "notes": m.get("notes"),
                     "source_sha256": m.get("source_keys")})
    return {"dataset": f"SeizeIT2 OpenNeuro {C.DATASET_ID} v{C.DATASET_VERSION}", "feature_config_hash": C.FEATURE_HASH,
            "recordings": recs}


def summarise_dataset(dm: dict) -> dict:
    out = {}
    for part in ("dev", "test"):
        rs = [r for r in dm["recordings"] if r["partition"] == part]
        inc = [r for r in rs if not r["excluded_reason"]]
        w4 = [r["windows"]["w4"] for r in inc if r["windows"]]
        ev = [r["events"] for r in rs if r["events"]]
        out[part] = {
            "subjects": len({r["subject"] for r in rs}), "subjects_with_windows": len({r["subject"] for r in inc}),
            "recordings": len(rs), "recordings_included": len(inc),
            "recordings_without_emg": sum(not r["emg"] for r in rs), "recordings_without_mov": sum(not r["mov"] for r in rs),
            "recordings_excluded": {r["recording_id"]: r["excluded_reason"] for r in rs if r["excluded_reason"]},
            "imu_uncalibratable_recordings": sum(any("IMU rejected" in n for n in (r["notes"] or [])) for r in rs),
            "recorded_hours": round(sum((r["events_recording_duration_s"] or 0) for r in rs) / 3600, 1),
            "w4_fusion_valid_hours": round(sum(w["fusion_valid"] for w in w4) * 2 / 3600, 1),
            "w4_windows": {k: sum(w[k] for w in w4) for k in ("n", "emg_ok", "imu_ok", "artifact", "warmup", "fusion_valid",
                                                             "fusion_valid_positive", "fusion_valid_negative",
                                                             "fusion_valid_partial_excluded",
                                                             "fusion_valid_nontarget_seizure")},
            "target_events": sum(e["target"] for e in ev), "nontarget_seizures": sum(e["nontarget"] for e in ev),
            "impedance_checks": sum(e["artifact"] for e in ev),
            "target_events_in_excluded_recordings": sum(r["events"]["target"] for r in rs if r["excluded_reason"] and r["events"]),
            "unmapped_codes": sorted({c for e in ev for c in e["unmapped_codes"]}),
            "malformed_rows_skipped": sum(len(e["skipped_rows"]) for e in ev),
            "duration_mismatches_gt_1s": sum(1 for r in rs if r["sync"] and abs(r["sync"]["duration_mismatch_s"]) > 1),
        }
    codes = {c for r in dm["recordings"] if r["events"] for c in r["events"]["codes"]}
    out["target_codes"] = {c: (c in codes) for c in C.TARGET_CODES}
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    run = ap.parse_args(argv).run
    out = C.REPORTS
    out.mkdir(parents=True, exist_ok=True)
    results = _j(run / "results.json")
    sel = _j(run / "selection.json")
    fin = _j(run / "final_eval.json")
    parity = _j(run / "parity_real.json")
    adl = _j(run / "adl_eval.json")
    folds = _j(run / "folds.json")
    scaling = _j(run / "tree_scaling_check.json")

    dm = dataset_manifest()
    (out / "dataset_manifest.json").write_text(json.dumps(dm, indent=1, default=str))
    ds = summarise_dataset(dm)

    # ablation table rows: best dev configuration per (window, modality, model) at FAR <= target
    abl = []
    for e in results:
        ok = [r for r in e["sweep"] if r["far_per_24h"] is not None and r["far_per_24h"] <= C.MAX_FAR_PER_24H]
        b = max(ok, key=lambda r: (r["detected"], -r["far_per_24h"]), default=None)
        abl.append({"window": e["window"], "modality": e["modality"], "model": e["model"],
                    "window_auprc_oof": round(e["window_auprc"], 4),
                    **({k: b[k] for k in ("threshold", "n_open", "m_close", "detected", "evaluable_events",
                                         "event_sensitivity", "far_per_24h", "latency_s_median", "event_precision")}
                       if b else {"threshold": None}),
                    "fit_s_mean": round(sum(f["fit_s"] for f in e["folds"]) / len(e["folds"]), 2),
                    "infer_us_per_window": round(sum(f["infer_us_per_window"] for f in e["folds"]) / len(e["folds"]), 3),
                    "complexity_fold0": json.dumps(e["folds"][0]["complexity"])})
    with (out / "ablation_results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in abl for k in r}, key=lambda k: list(abl[0]).index(k)
                                                  if k in abl[0] else 99))
        w.writeheader()
        w.writerows(abl)
    (out / "oof_metrics.json").write_text(json.dumps(
        {"scope": "development pool sub-001..sub-096, patient-wise out-of-fold, common fusion-valid exposure",
         "experiments": [{k: e[k] for k in ("window", "modality", "model", "experiment_hash", "window_auprc", "sweep")}
                         for e in results]}, indent=1))
    dep = sel["deployed"]
    resource = {"targets": {"flash_bytes": C.FLASH_TARGET, "sram_bytes": C.SRAM_TARGET}, "budget": dep["resources"],
                "xtensa_objects": dep["xtensa_objects"], "esp_idf_application": dep["esp_idf_app"],
                "measured_candidates": sel["measured_candidates"]}
    (out / "resource_report.json").write_text(json.dumps(resource, indent=1, default=str))
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO, capture_output=True, text=True).stdout.strip()
    gcc = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    exp = {"python": sys.version.split()[0], "packages": {p: version(p) for p in PKGS}, "os": platform.platform(),
           "git_commit": git, "host_compiler": gcc, "esp32s3_compiler": dep["xtensa_objects"].get("compiler"),
           "esp_idf": dep["esp_idf_app"].get("esp_idf"), "seed": C.SEED, "feature_config": C.feature_config(),
           "feature_config_hash": C.FEATURE_HASH, "experiment_hashes": {f"{e['window']}|{e['modality']}|{e['model']}":
                                                                        e["experiment_hash"] for e in results},
           "subject_folds": folds, "tree_scaling_check": scaling, "selection_rule": sel["rule"],
           "selected": {k: sel[k] for k in ("best_statistical", "best_deployable", "per_modality")},
           "export": dep["export"], "final_fit": dep["final_fit"], "lock": fin["lock"] if fin else None,
           "validation": {"golden_host": dep["golden_host_result"], "golden": dep["golden"],
                          "exported_vs_original_on_dev": dep["exported_vs_original_on_dev"], "real_data_parity": parity}}
    (out / "experiment_manifest.json").write_text(json.dumps(exp, indent=1, default=str))
    rep = {"dataset": ds, "development_oof": {"best_statistical": sel["best_statistical"],
                                              "best_deployable": sel["best_deployable"],
                                              "per_modality": sel["per_modality"],
                                              "far_target_met": sel["far_target_met_on_dev"],
                                              "latency_target_met": sel["latency_target_met_on_dev"]},
           "final_evaluation": fin["results"] if fin else None, "adl": adl, "resources": dep["resources"],
           "parity": {"golden": dep["golden_host_result"], "real": parity}}
    (out / "lyra_report.json").write_text(json.dumps(rep, indent=1, default=str))
    (out / "lyra_report.md").write_text(markdown(ds, sel, fin, parity, adl, abl, dep, scaling, exp))
    print(f"wrote {out}")


def markdown(ds, sel, fin, parity, adl, abl, dep, scaling, exp) -> str:
    bs, bd = sel["best_statistical"], sel["best_deployable"]
    pm = sel["per_modality"]
    R = fin["results"] if fin else {}
    L = []
    a = L.append
    a("# LYRA convulsive-seizure detector: ESP32-S3 pipeline report\n")
    a("> Experimental research prototype. Not clinically validated, not a diagnostic device, and not a "
      "substitute for prescribed monitoring or medical care. A detection is a *possible* event, never a confirmed "
      "seizure. All evidence is from in-hospital SeizeIT2 recordings; nothing was run on LYRA hardware.\n")
    a("## 1. Dataset and split audit\n")
    d, t = ds["dev"], ds["test"]
    a("| | Development (sub-001..096) | Final evaluation (sub-097..125) |\n|---|---:|---:|")
    for k, lab in (("subjects", "Subjects"), ("subjects_with_windows", "Subjects with usable windows"),
                   ("recordings", "Recordings"), ("recordings_included", "Recordings with windows"),
                   ("recordings_without_emg", "Recordings without EMG"), ("recordings_without_mov", "Recordings without MOV"),
                   ("imu_uncalibratable_recordings", "Recordings with uncalibratable IMU"),
                   ("recorded_hours", "Recorded hours"), ("w4_fusion_valid_hours", "Valid EMG+IMU hours (4 s windows)"),
                   ("target_events", "Target seizures"), ("target_events_in_excluded_recordings", "  of which in recordings without data"),
                   ("nontarget_seizures", "Non-target seizures (hard negatives)"), ("impedance_checks", "Impedance checks (excluded)")):
        a(f"| {lab} | {fmt(d[k])} | {fmt(t[k])} |")
    for k in ("fusion_valid_positive", "fusion_valid_negative", "fusion_valid_partial_excluded",
              "fusion_valid_nontarget_seizure", "artifact", "warmup"):
        a(f"| 4 s windows: {k.replace('_', ' ')} | {d['w4_windows'][k]:,} | {t['w4_windows'][k]:,} |")
    a("")
    a(f"Target codes present in the data: {', '.join(f'`{c}`: {v}' for c, v in ds['target_codes'].items())}. "
      "SeizeIT2 v1.1.0 contains no generalised or unknown-onset tonic-clonic codes, so every target event is a "
      "focal-to-bilateral tonic-clonic seizure.\n")
    a("- **EMG sampling rate:** EDF headers and JSON sidecars agree on 256 Hz for every EMG file. The 250 Hz "
      "stated in the task brief is not what the files contain; 256 Hz was used.")
    a("- **Synchronisation:** EMG and MOV EDF start times are de-identified differently (EMG 2000-01-01, MOV a "
      "real-looking date), so the absolute offset cannot be verified. Both come from the same ECGEMG Sensor Dot "
      f"module and have identical durations ({d['duration_mismatches_gt_1s']} development and "
      f"{t['duration_mismatches_gt_1s']} final-evaluation recordings differ by more than 1 s). Offset 0 is assumed.")
    a("- **Split:** patient-wise. Development folds use GroupKFold(5) by subject; nothing in development code can load "
      "the final cohort (`ml/lyra/data.py` refuses until `deploy.py` writes the lock).")
    a("- **Caveat on the final cohort:** sub-097..125 were used to train earlier LYRA models (v1/v2, PR #1). They "
      "are held out for this pipeline only, not a pristine cohort.\n")
    a("## 2. Development results (out-of-fold, sub-001..096)\n")
    a(f"Selection rule: {sel['rule']}\n")
    a(f"- FAR <= {C.MAX_FAR_PER_24H}/24 h reachable on development data: **{sel['far_target_met_on_dev']}**.")
    a(f"- Median latency < {C.MAX_LATENCY_S:g} s reachable together with it: **{sel['latency_target_met_on_dev']}**.")
    sg = sel.get("spec_grid_only", {})
    a(f"- Within the brief's threshold grid (0.20-0.80) alone: any detection **{sg.get('any_detection')}**, FAR target met "
      f"**{sg.get('far_target_met')}**. Calibrated probabilities target the true window prevalence, so they rarely exceed "
      "~0.1; the grid was extended downwards (0.005-0.15) without changing the FAR or latency criteria.\n")
    a("| Role | Window | Modality | Classifier | Threshold | N | M | Detected | FAR/24 h | Median latency (s) | Precision |")
    a("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    rows = [("Best statistical", bs), ("Best deployable", bd)] + [(f"Best {m}", r) for m, r in pm.items() if r]
    for role, r in rows:
        a(f"| {role} | {r['window']} | {r['modality']} | {r['model']} | {r['threshold']:g} | {r['n_open']} | "
          f"{r['m_close']} | {r['detected']}/{r['evaluable_events']} | {fmt(r['far_per_24h'], 3)} | "
          f"{fmt(r['latency_s_median'], 1)} | {fmt(r['event_precision'])} |")
    a("\nFull grid: `ablation_results.csv` (best configuration per window x modality x classifier) and "
      "`oof_metrics.json` (every threshold/N/M).")
    if scaling:
        a(f"\nTree models and feature scaling: standardising before a random forest changed predictions by at most "
          f"{scaling['max_abs_diff']:.2g} on {scaling['windows_compared']:,} windows, so tree models are trained "
          "without normalisation; LR and SVC standardise inside each training fold.\n")
    a("## 3. Final evaluation (sub-097..125, evaluated once)\n")
    if R:
        a("| | sEMG-only | IMU-only | sEMG + IMU |\n|---|---:|---:|---:|")
        roles = ["modality_emg", "modality_imu", "modality_fusion"]
        g = lambda k, nd=2: " | ".join(fmt(R[x][k], nd) if x in R else "n/a" for x in roles)  # noqa: E731
        a(f"| Event-based sensitivity | {g('event_sensitivity')} |")
        a("| Detected / evaluable | " + " | ".join(f"{R[x]['detected']}/{R[x]['evaluable_events']}" for x in roles) + " |")
        a(f"| FAR/24 h | {g('far_per_24h', 3)} |")
        a(f"| Median detection latency (s) | {g('latency_s_median', 1)} |")
        a(f"| Event-level precision | {g('event_precision')} |")
        a(f"| Window-level AUPRC | {g('window_auprc', 3)} |")
        a("| Selected window / classifier | " + " | ".join(f"{R[x]['selection']['window']} / {R[x]['selection']['model']}"
                                                       for x in roles) + " |")
        a("")
        for role, label in (("deployable", "Deployed model (exported C replica)"), ("best_statistical", "Best statistical model")):
            r = R[role]
            b = r["bootstrap"]
            a(f"**{label}** ({r['selection']['window']}, {r['selection']['modality']}, {r['selection']['model']}, "
              f"threshold {r['selection']['threshold']:g}, N={r['selection']['n_open']}, M={r['selection']['m_close']}):")
            a(f"- Event sensitivity {r['detected']}/{r['evaluable_events']} = {fmt(r['event_sensitivity'])} "
              f"(patient-clustered bootstrap 95% CI {fmt(b['event_sensitivity_95ci'])}); "
              f"{r['target_events'] - r['evaluable_events']} target events had no valid window.")
            a(f"- {r['false_alerts']} false alerts over {fmt(r['valid_negative_hours'], 1)} valid target-negative hours = "
              f"{fmt(r['far_per_24h'], 3)}/24 h (CI {fmt(b['far_per_24h_95ci'], 3)}); "
              f"{r['false_alerts_during_nontarget_seizures']} of them during non-target seizures.")
            a(f"- Median latency {fmt(r['latency_s_median'], 1)} s from annotated onset; event precision "
              f"{fmt(r['event_precision'])}; {r['emitted_alerts']} alerts emitted, {r['suppressed_alerts']} suppressed by "
              f"the 60 s refractory, {r['candidate_events']} candidate events; window AUPRC {fmt(r['window_auprc'], 3)}.\n")
        r = R["deployable"]
        a("Per-patient breakdown (deployed model; subjects with target events or false alerts):\n")
        a("| Subject | Valid h | Events | Evaluable | Detected | Missed | Sensitivity | False alerts | FAR/24 h | Latency (s) |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for p in r["per_subject"]:
            if p["target_events"] or p["false_alerts"]:
                a(f"| {p['subject']} | {fmt(p['valid_negative_hours'], 1)} | {p['target_events']} | {p['evaluable_events']} | "
                  f"{p['detected']} | {p['missed']} | {fmt(p['event_sensitivity'])} | {p['false_alerts']} | "
                  f"{fmt(p['far_per_24h'], 2)} | {fmt(p['latency_s_median'], 1)} |")
        a("")
    else:
        a("Not run.\n")
    a("## 4. Volunteer ADL negative cohort\n")
    a("Not provided: no pod recordings exist yet. `python -m ml.lyra.adl --lyra-negatives-dir PATH` implements the "
      "schema checks, the volunteer-wise calibration/evaluation split and the false-alarm report.\n" if not adl else
      "```json\n" + json.dumps(adl, indent=1)[:4000] + "\n```\n")
    a("## 5. C99 export, parity and ESP32-S3 resources\n")
    ex = dep["export"]
    rb = dep["resources"]
    a(f"- Exported: `{bd['model']}` via {ex['backend']}. Files: `firmware/model/` (`lyra_model.h`, "
      "`lyra_detector_config.h`, `lyra_feature_spec.h`, `lyra_features.[ch]`, `lyra_detector.[ch]`, `lyra_model_test.c`).")
    ag = dep["exported_vs_original_on_dev"]
    a(f"- Exported vs original estimator on {ag['windows']:,} development windows: max |Δp| {ag['max_abs_prob_diff']:.3g}, "
      f"decision agreement {ag['decision_agreement']:.5f}. The final evaluation scores the exported replica.")
    a(f"- Golden vectors (synthetic, `firmware/model/test/golden_vectors.bin`), host gcc C99: {dep['golden_host_result']}")
    if parity:
        fp, mp = parity["features_real_dev_windows"], parity["model_real_dev_feature_vectors"]
        a(f"- Real development data (local only): {fp['windows']:,} windows from {len(fp['recordings'])} recordings, "
          f"{fp['exact']:,}/{fp['values']:,} feature values bit-identical, max relative difference {fp['max_rel_diff']:.2g} "
          f"(passed: {fp['passed']}); model on {mp['rows']:,} feature vectors: max |Δp| {mp['max_abs_prob_diff']:.2g}, "
          f"decision agreement {mp['decision_agreement']:.6f} (passed: {mp['passed']}).")
        a(f"- Float32 pod inputs vs float64 training inputs: features differ by at most "
          f"{fp['max_rel_diff_vs_float64_training_inputs']:.2g} (relative).")
    a(f"\n| Measured with {exp.get('esp32s3_compiler')} (-Os) | Bytes | Target |\n|---|---:|---:|")
    a(f"| Flash: model | {rb['flash_model_bytes']:,} | |")
    a(f"| Flash: feature extraction | {rb['flash_features_bytes']:,} | |")
    a(f"| Flash: detector | {rb['flash_detector_bytes']:,} | |")
    a(f"| **Flash: LYRA package** | **{rb['flash_bytes']:,}** | < {C.FLASH_TARGET:,} |")
    a(f"| SRAM: static (.data/.bss scratch) | {rb['sram_static_bytes']:,} | |")
    a(f"| SRAM: peak stack (features + model) | {rb['sram_peak_stack_bytes']:,} | |")
    a(f"| SRAM: one input window (ENV + 6-axis IMU, float32; requirement, computed) | {rb['sram_input_window_bytes']:,} | |")
    a(f"| **SRAM: total** | **{rb['sram_total_bytes']:,}** | < {C.SRAM_TARGET:,} |")
    app = dep["esp_idf_app"]
    if app.get("status") == "built":
        a(f"\nESP-IDF {app['esp_idf']} application build for esp32s3 (`firmware/esp32s3_app`, size probe linking the "
          "package): per-object linked sizes " + "; ".join(f"`{k}` flash {v['flash_bytes']:,} B, RAM {v['ram_bytes']:,} B"
                                                          for k, v in app["objects"].items()) + ".")
    a(f"\nScope: {rb['scope']}. Stack figures come from `-fstack-usage` (static, per function); no run on a "
      "board, so no runtime peak was measured.\n")
    a("## 6. Deviations from the task brief\n")
    for line in (
        "EMG is 256 Hz in every file (brief: 250 Hz); the files are authoritative.",
        "Signal path follows locked decision D12 (MyoWare **ENV** pin at 50 Hz), as the owner directed: the pod applies no "
        "digital 20-120 Hz band-pass or mains notch (the ENV output has neither). Training reproduces the MyoWare analog "
        "chain causally (1st-order HPF 20.8 Hz, rectification, 1st-order LPF 3.6 Hz, 20 ms bin averaging); no filtfilt anywhere.",
        "The 5 s configuration uses a 2.48 s hop: 2.5 s is not an integer number of 25 Hz samples.",
        "`sz_gen_m_tonicClonic` and `sz_uo_m_tonicClonic` do not occur in SeizeIT2 v1.1.0; all targets are `sz_foc_f2b`.",
        "Threshold grid extended below 0.20 (see section 2).",
        "m2cgen 0.10.0 cannot convert XGBoost 3.4 models (TypeError on base_score); XGBoost is exported by an own "
        "converter from XGBoost's JSON model. emlearn 0.23 (inline, 8-bit leaf probabilities) exports RF/ET; its "
        "quantisation is part of the deployed model and its deviation from scikit-learn is reported.",
        f"RBF SVC is trained on a patient-stratified subsample of at most {C.SVC_MAX_TRAIN:,} windows per fit (O(n^2) kernel); "
        "its probabilities come from patient-wise inner-fold Platt calibration, not `probability=True`.",
        "No startup/recording-local normalisation is used: tree models need none and LR/SVC scalers are fitted per training fold.",
        "Final cohort sub-097..125 was used to train LYRA v1/v2 before this pipeline existed.",
    ):
        a(f"- {line}")
    a("")
    a("## 7. Limitations and hardware validation still required\n")
    a("- **Sensor domain:** training uses a software model of the MyoWare 2.0 ENV chain applied to SeizeIT2 "
      "Sensor Dot RAW EMG, not a MyoWare. Gain, offset, rectifier behaviour and the ADC must be calibrated on the "
      "pod so `env` arrives in the same uV-equivalent scale; amplitude features depend on it.")
    a("- **IMU:** SeizeIT2's chest module (25 Hz) stands in for the BMI270; its gyro unit is an estimate "
      "(125 deg/s per unit). Placement (D10) differs.")
    a("- **Labels and data:** in-hospital recordings only; latency is measured from the annotated focal onset, not "
      "from the convulsive phase; few target events; no daily-life negatives.")
    a("- **Hardware:** nothing has run on an ESP32-S3 board; acquisition, ring buffers, BLE and power are not part "
      "of the measured package.")
    a("- **Final cohort:** previously used to train LYRA v1/v2, so these numbers are not a pristine external test.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
