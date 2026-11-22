"""Select on development out-of-fold results, fit the final models on the whole development pool,
export C99, verify parity, measure ESP32-S3 resources, and freeze everything in the evaluation lock.

    python -m ml.lyra.deploy --run ml/lyra/runs/bench-<id>

Selection rule (predefined): among (window, modality, classifier, threshold, N, M) with dev
out-of-fold FAR <= MAX_FAR_PER_24H: 1) most detected events, 2) median latency < 20 s if any
candidate achieves it, 3) smaller model, 4) lower FAR, 5) narrower between-subject sensitivity IQR.
Best statistical = first candidate. Best deployable = first candidate whose exported package
measures under the flash and SRAM targets on the ESP32-S3 toolchain.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import subprocess
from pathlib import Path

import numpy as np

from . import benchmark as Bm
from . import config as C
from . import data as Dt
from . import export_c as X
from . import golden as G
from . import resources as R

MODEL_ID = "lyra-esp32s3-convulsive"
MODEL_VERSION = "1.0.0"


def size_proxy(exp: dict) -> float:
    c = exp["folds"][0]["complexity"]
    return c.get("nodes", 0) * 3 + c.get("parameters", 0)


def rank(results: list[dict]) -> tuple[list[dict], bool, bool]:
    rows = []
    for e in results:
        for r in e["sweep"]:
            rows.append({**{k: e[k] for k in ("window", "modality", "model", "experiment_hash", "window_auprc")},
                         **r, "size_proxy": size_proxy(e)})
    ok = [r for r in rows if r["far_per_24h"] is not None and r["far_per_24h"] <= C.MAX_FAR_PER_24H and r["detected"]]
    far_met = bool(ok)
    pool = ok or rows
    lat_met = any(r["latency_s_median"] is not None and r["latency_s_median"] < C.MAX_LATENCY_S for r in pool)
    iqr = lambda r: (r["subject_sensitivity_iqr"][1] - r["subject_sensitivity_iqr"][0]) if r["subject_sensitivity_iqr"] else 1  # noqa: E731
    key = lambda r: (-r["detected"], -(r["latency_s_median"] is not None and r["latency_s_median"] < C.MAX_LATENCY_S),  # noqa: E731
                     r["size_proxy"], r["far_per_24h"] if r["far_per_24h"] is not None else 1e9, iqr(r))
    ranked = sorted(pool, key=key)
    if not ranked or ranked[0]["detected"] == 0:
        raise SystemExit("no configuration detects any target event on the development pool")
    return ranked, far_met, lat_met


def final_fit(sel: dict, D: Dt.Windows):
    cols = list(C.MODALITIES[sel["modality"]])
    model, A, B, best_c, idx, t_fit = Bm.train(sel["model"], D, cols, D.valid(sel["modality"]), C.SEED)
    return {"model": model, "A": A, "B": B, "lr_c": best_c, "fit_windows": int(idx.size), "cols": cols,
            "fit_s": round(t_fit, 2), "complexity": Bm.complexity(sel["model"], model)}


def package(sel: dict, fm: dict, out: Path) -> tuple[dict, callable]:
    out.mkdir(parents=True, exist_ok=True)
    hdr, replica, meta = X.model_header(sel["model"], fm["model"], fm["cols"], fm["A"], fm["B"], MODEL_ID)
    (out / "lyra_model.h").write_text(hdr)
    (out / "lyra_feature_spec.h").write_text(X.spec_header(int(C.WINDOWS[sel["window"]][0] * C.IMU_RATE)))
    (out / "lyra_detector_config.h").write_text(X.detector_config(sel, MODEL_ID))
    for f in ("lyra_features.c", "lyra_features.h", "lyra_detector.c", "lyra_detector.h", "lyra_model_test.c"):
        if out.resolve() != C.FIRMWARE.resolve():
            shutil.copy(C.FIRMWARE / f, out / f)
    return meta, replica


def resource_budget(sel: dict, objs: dict) -> dict:
    w = C.WINDOWS[sel["window"]][0]
    o = objs["objects"]
    stack = (o["features"]["stack_bytes_per_function"].get("lyra_features_compute", 0)
             + o["model"]["stack_bytes_per_function"].get("lyra_model_entry", 0))
    inputs = int(4 * (w * C.ENV_RATE + 6 * w * C.IMU_RATE))
    flash = sum(v["flash_bytes"] for v in o.values())
    static = sum(v["static_ram_bytes"] for v in o.values())
    return {"flash_bytes": flash, "flash_model_bytes": o["model"]["flash_bytes"],
            "flash_features_bytes": o["features"]["flash_bytes"], "flash_detector_bytes": o["detector"]["flash_bytes"],
            "sram_static_bytes": static, "sram_peak_stack_bytes": stack, "sram_input_window_bytes": inputs,
            "sram_total_bytes": static + stack + inputs,
            "meets_flash_target": flash < C.FLASH_TARGET, "meets_sram_target": static + stack + inputs < C.SRAM_TARGET,
            "scope": "LYRA package only (model + features + detector + one input window); excludes RTOS, BLE/Wi-Fi, "
                     "drivers and acquisition ring buffers beyond one window"}


def host_test(d: Path) -> str:
    exe = d / "lyra_model_test"
    subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I", str(d),
                    str(d / "lyra_model_test.c"), str(d / "lyra_features.c"), str(d / "lyra_detector.c"), "-lm",
                    "-o", str(exe)], check=True, capture_output=True)
    r = subprocess.run([str(exe), str(d / "test" / "golden_vectors.bin")], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"golden parity failed:\n{r.stdout[-3000:]}")
    return r.stdout.strip().splitlines()[-1]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    run = ap.parse_args(argv).run
    if Dt.LOCK.exists():
        raise SystemExit(f"{Dt.LOCK} exists: the configuration is already frozen")
    results = json.loads((run / "results.json").read_text())
    ranked, far_met, lat_met = rank(results)
    best_stat = ranked[0]
    spec_only, spec_far_met, _ = rank([{**e, "sweep": [r for r in e["sweep"] if r["grid"] == "spec"]} for e in results]) \
        if any(r["detected"] for e in results for r in e["sweep"] if r["grid"] == "spec") else ([None], False, False)
    per_modality = {m: next((r for r in ranked if r["modality"] == m), None) for m in C.MODALITIES}

    fitted: dict[tuple, dict] = {}
    loaded: dict[str, Dt.Windows] = {}

    def get_fit(r):
        k = (r["window"], r["modality"], r["model"])
        if k not in fitted:
            if r["window"] not in loaded:
                loaded.clear()
                loaded[r["window"]] = Dt.load(r["window"], "dev")
            fitted[k] = final_fit(r, loaded[r["window"]])
        return fitted[k]

    measured: dict[tuple, dict] = {}
    deployable = None
    work = run / "deploy_candidates"
    for r in ranked:
        k = (r["window"], r["modality"], r["model"])
        if k not in measured:
            fm = get_fit(r)
            d = work / "_".join(k)
            package(r, fm, d)
            measured[k] = resource_budget(r, R.object_sizes(d, d / "xtensa"))
            print(f"measured {k}: flash {measured[k]['flash_bytes']} B, SRAM {measured[k]['sram_total_bytes']} B", flush=True)
        if measured[k]["meets_flash_target"] and measured[k]["meets_sram_target"]:
            deployable = r
            break
        if len(measured) >= 12:
            break
    if deployable is None:
        raise SystemExit("no ranked candidate met the flash/SRAM targets among the first 12 measured models")

    # export the deployable package into firmware/model
    sel = deployable
    fm = get_fit(sel)
    meta, replica = package(sel, fm, C.FIRMWARE)
    (C.FIRMWARE / "test").mkdir(exist_ok=True)
    gold = G.write(C.FIRMWARE / "test" / "golden_vectors.bin", replica, sel["threshold"], sel["n_open"], sel["m_close"])
    host = host_test(C.FIRMWARE)
    objs = R.object_sizes(C.FIRMWARE, run / "xtensa_final")
    budget = resource_budget(sel, objs)
    app = R.idf_app_size(C.FIRMWARE)
    R.clean_app_build()

    # deployed (exported) model vs the original estimator on development windows (agreement only)
    D = loaded.get(sel["window"]) or Dt.load(sel["window"], "dev")
    v = np.flatnonzero(D.valid(sel["modality"]))
    rng = np.random.default_rng(C.SEED)
    s = np.sort(rng.choice(v, min(300_000, v.size), replace=False))
    from scipy.special import expit
    p_orig = expit(fm["A"] * Bm.raw_score(sel["model"], fm["model"], D.X[s][:, fm["cols"]].astype(np.float64)) + fm["B"])
    p_dep = replica(D.X[s]).astype(np.float64)
    thr = float(np.float32(sel["threshold"]))
    agree = {"windows": int(s.size), "max_abs_prob_diff": float(np.abs(p_orig - p_dep).max()),
             "decision_agreement": float(((p_orig >= thr) == (p_dep >= thr)).mean())}

    models = {"deployable": sel, "best_statistical": best_stat, **{f"modality_{m}": r for m, r in per_modality.items()}}
    for role, r in models.items():
        if r is None:
            continue
        fm_r = get_fit(r)
        with (run / f"final_{role}.pkl").open("wb") as fh:
            pickle.dump({k: fm_r[k] for k in ("model", "A", "B", "cols", "lr_c")} | {"selection": r}, fh)

    selection = {
        "rule": __doc__.split("Selection rule (predefined): ")[1].split("\n\n")[0].replace("\n", " "),
        "far_target_met_on_dev": far_met, "latency_target_met_on_dev": lat_met,
        "spec_grid_only": {"any_detection": spec_only[0] is not None, "far_target_met": spec_far_met,
                           "best": spec_only[0]},
        "best_statistical": best_stat, "best_deployable": sel, "per_modality": per_modality,
        "measured_candidates": {"|".join(k): v for k, v in measured.items()},
        "deployed": {"model_id": MODEL_ID, "version": MODEL_VERSION, "export": meta, "golden": gold,
                     "golden_host_result": host, "resources": budget, "xtensa_objects": objs, "esp_idf_app": app,
                     "exported_vs_original_on_dev": agree, "final_fit": {k: fm[k] for k in ("A", "B", "lr_c", "fit_windows",
                                                                                         "fit_s", "complexity")}},
    }
    (run / "selection.json").write_text(json.dumps(selection, indent=1, default=str))
    lock = {"frozen_by": "ml/lyra/deploy.py", "run": run.name, "feature_hash": C.FEATURE_HASH,
            "selection": {k: {kk: r[kk] for kk in ("window", "modality", "model", "threshold", "n_open", "m_close")}
                          for k, r in models.items() if r},
            "final_models_sha256": {p.name: sha(p) for p in sorted(run.glob("final_*.pkl"))},
            "firmware_sha256": {p.name: sha(p) for p in sorted(C.FIRMWARE.glob("lyra_*.[ch]"))}}
    Dt.LOCK.write_text(json.dumps(lock, indent=1))
    print(json.dumps({"best_statistical": {k: best_stat[k] for k in ("window", "modality", "model", "threshold", "n_open",
                                                                      "m_close", "detected", "evaluable_events",
                                                                      "far_per_24h", "latency_s_median")},
                      "deployable": {k: sel[k] for k in ("window", "modality", "model", "threshold", "n_open", "m_close",
                                                          "detected", "far_per_24h")},
                      "resources": budget, "host": host, "agree": agree}, indent=1, default=str))


if __name__ == "__main__":
    from ml.lyra.deploy import main as _main

    _main()
