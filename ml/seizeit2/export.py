"""Export the frozen, test-evaluated model into models/seizeit2/ for the app's TypeScript runtime.

    python -m ml.seizeit2.export --run ml/seizeit2/runs/<id>

Why JSON and not .tflite: LYRA has no native inference runtime (Expo Go compatible, no native ML
module), and the selected model is a small feature-based classifier. A JSON tree ensemble / linear
model evaluated in TypeScript runs unchanged on iOS, Android and web without a development build.

Writes model.json, config.json, labels.json, metadata.json, evaluation/*, and the synthetic
parity vectors used by ml/seizeit2/parity_check.mjs. Refuses to export a run without test metrics.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from . import config as C
from . import data_loader as D
from . import evaluate as E
from .preprocessing import FEATURE_NAMES, feature_path, window_features
from .train import SkModel, training_pool

FORMAT = "lyra-seizeit2-model/1"
MODEL_ID = "lyra-seizeit2-fbtcs"
MODEL_VERSION = "1.0.0"


def to_json(m) -> dict:
    if isinstance(m, SkModel) and isinstance(m.est, HistGradientBoostingClassifier):
        if m.est.n_trees_per_iteration_ != 1:
            raise ValueError("binary classifier expected")
        trees = []
        for (pred,) in m.est._predictors:  # sklearn internals; parity is verified below
            n = pred.nodes
            if n["is_categorical"].any():
                raise ValueError("categorical splits are not supported by the TS runtime")
            leaf = n["is_leaf"].astype(bool)
            trees.append({"feature": np.where(leaf, -1, n["feature_idx"]).tolist(),
                          "threshold": np.where(leaf, 0.0, n["num_threshold"]).tolist(),
                          "left": n["left"].tolist(), "right": n["right"].tolist(),
                          "value": np.where(leaf, n["value"], 0.0).tolist()})
        return {"format": FORMAT, "kind": "gbt", "feature_names": list(FEATURE_NAMES),
                "columns": m.cols, "baseline": float(np.ravel(m.est._baseline_prediction)[0]),
                "trees": trees}
    if isinstance(m, SkModel) and isinstance(m.est, Pipeline):
        sc, lr = m.est.named_steps["standardscaler"], m.est.named_steps["logisticregression"]
        return {"format": FORMAT, "kind": "logreg", "feature_names": list(FEATURE_NAMES),
                "columns": m.cols, "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
                "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    # shortcut: the rule baseline has no exporter; add one only if it is ever selected
    raise NotImplementedError(f"no exporter for {type(m).__name__}")


def score_json(mj: dict, X: np.ndarray) -> np.ndarray:
    """Reference evaluator of model.json (the TS runtime mirrors this)."""
    Z = np.asarray(X, np.float64)[:, mj["columns"]]
    if mj["kind"] == "logreg":
        raw = mj["intercept"] + ((Z - mj["mean"]) / mj["scale"]) @ np.asarray(mj["coef"])
    else:
        raw = np.full(Z.shape[0], mj["baseline"])
        for t in mj["trees"]:
            f, thr, left, right, val = (np.asarray(t[k]) for k in ("feature", "threshold", "left", "right", "value"))
            node = np.zeros(Z.shape[0], int)
            while True:
                inner = f[node] >= 0
                if not inner.any():
                    break
                i = np.flatnonzero(inner)
                go_left = Z[i, f[node[i]]] <= thr[node[i]]
                node[i] = np.where(go_left, left[node[i]], right[node[i]])
            raw += val[node]
    return 1 / (1 + np.exp(-raw))


def synthetic_windows(W: int, rng: np.random.Generator) -> list[dict]:
    """Deterministic synthetic signals only (never dataset samples), rounded so TS sees the same input."""
    ne, ni = C.EMG_FS * W, C.IMU_FS * W
    te, ti = np.arange(ne) / C.EMG_FS, np.arange(ni) / C.IMU_FS
    cases = []
    for name, emg_amp, burst_hz, mov_amp, mov_hz in [
        ("rest", 5, 0, 0.002, 0.3), ("rhythmic_3hz", 200, 3, 0.15, 3),
        ("walking_like", 30, 0, 0.08, 1.8), ("strong_tonic", 400, 0, 0.02, 5)]:
        burst = (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * burst_hz * te))) if burst_hz else 1.0
        emg = emg_amp * rng.normal(size=ne) * burst + 300 * np.sin(2 * np.pi * 1.2 * te)
        osc = mov_amp * np.sin(2 * np.pi * mov_hz * ti)
        acc = np.stack([0.80 + osc, 0.30 + 0.5 * osc, -0.52 + 0.3 * osc], 1) + 0.003 * rng.normal(size=(ni, 3))
        gyr = np.stack([osc, -0.7 * osc, 0.4 * osc], 1) + 0.002 * rng.normal(size=(ni, 3))
        cases.append({"name": name, "emg": np.round(emg, 3), "acc": np.round(acc, 5), "gyr": np.round(gyr, 5)})
    return cases


def stream_alarms(t_end, scores, k: int, thr: float) -> list[int]:
    """Window-by-window detector exactly as the app runs it (createDetector in src/services/ml/seizeit2/model.ts)."""
    buf, out, prev, last = [], [], None, None
    for t, sc in zip(t_end, scores):
        if prev is not None and t - prev > 1.5 * C.HOP_S:
            buf = []
        prev = t
        buf = (buf + [float(sc)])[-k:]
        if sum(buf) / len(buf) >= thr and (last is None or t - last >= C.REFRACTORY_S):
            out.append(int(t))
            last = t
    return out


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Export the selected model to models/seizeit2/")
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args(argv)
    run = args.run
    if not (run / "test_metrics.json").exists():
        raise SystemExit("evaluate the test set first (python -m ml.seizeit2.evaluate --run ...)")
    sel = json.loads((run / "selection.json").read_text())
    test = json.loads((run / "test_metrics.json").read_text())
    split = json.loads((run / "split.json").read_text())
    cv = json.loads((run / "cv_results.json").read_text())
    with (run / "final_model.pkl").open("rb") as fh:
        model = pickle.load(fh)
    W, k, thr = sel["window_s"], sel["k"], sel["threshold"]
    mj = to_json(model)

    # 1. exported model == sklearn model, on real dev windows (checked locally, not exported)
    dev = set().union(*map(set, split["folds"]))
    X, _, _ = training_pool(D.list_runs(), dev, W)
    X = X[np.random.default_rng(C.SEED).permutation(X.shape[0])[:200_000]]
    ref, got = model.score(X), score_json(mj, X)
    max_diff = float(np.abs(ref - got).max())
    agree = float(((ref >= thr) == (got >= thr)).mean())
    if max_diff > 1e-9 or agree < 1:
        raise SystemExit(f"exported model disagrees with sklearn: max diff {max_diff}, decisions {agree}")

    # 2. parity vectors for the TS runtime: synthetic signals + random in-range feature vectors
    rng = np.random.default_rng(C.SEED)
    wins = synthetic_windows(W, rng)
    feats = window_features(np.stack([c["emg"] for c in wins]), np.stack([c["acc"] for c in wins]),
                            np.stack([c["gyr"] for c in wins]), W)
    lo, hi = np.percentile(X, 0.5, axis=0), np.percentile(X, 99.5, axis=0)
    rand = rng.uniform(lo, hi, size=(500, X.shape[1]))
    seq = np.clip(rng.normal(thr, 2 * thr + 0.05, size=400), 0, 1)
    seq_t = np.arange(seq.size) * C.HOP_S + W
    seq_t[200:] += 60  # one gap in the stream: the detector restarts its smoothing buffer
    contiguous = stream_alarms(seq_t[:200], seq[:200], k, thr)
    if contiguous != E.alarms(seq_t[:200], E.smooth(seq[:200], k), thr).tolist():
        raise SystemExit("streaming detector disagrees with the evaluation post-processing")
    kept = stream_alarms(seq_t, seq, k, thr)
    parity = {
        "window_s": W,
        "windows": [{"name": c["name"], "emg": c["emg"].tolist(),
                     "acc": {a: c["acc"][:, i].tolist() for i, a in enumerate("xyz")},
                     "gyr": {a: c["gyr"][:, i].tolist() for i, a in enumerate("xyz")},
                     "features": f.tolist(), "score": float(s)}
                    for c, f, s in zip(wins, feats, score_json(mj, feats))],
        "feature_vectors": rand.tolist(), "feature_vector_scores": score_json(mj, rand).tolist(),
        "detector": {"k": k, "threshold": thr, "refractory_s": C.REFRACTORY_S, "hop_s": C.HOP_S,
                     "t_end_s": seq_t.tolist(), "scores": seq.tolist(), "alarm_t_end_s": [int(t) for t in kept]},
    }
    (C.REPO / "ml" / "seizeit2" / "parity_vectors.json").write_text(json.dumps(parity))

    # 3. deployable assets
    md = C.MODEL_DIR
    (md / "evaluation").mkdir(parents=True, exist_ok=True)
    (md / "model.json").write_text(json.dumps(mj, separators=(",", ":")))
    (md / "labels.json").write_text(json.dumps({
        "window_classes": {"0": "background", "1": "convulsive_pattern"},
        "app_event_types": {
            "no_event": "No alarm from the post-processed score stream.",
            "possible_event": "Smoothed score reached the threshold: a pattern resembling the annotated "
                              "focal-to-bilateral tonic-clonic seizures in SeizeIT2. Not a diagnosis; the app's "
                              "cancel countdown and caregiver workflow still apply."},
        "class_1_definition": "window overlapping an annotated sz_foc_f2b event by >= 50%",
    }, indent=1))
    (md / "config.json").write_text(json.dumps({
        "format": FORMAT, "model_file": "model.json",
        "input": {
            "window_s": W, "hop_s": C.HOP_S,
            "emg": {"samples": C.EMG_FS * W, "rate_hz": C.EMG_FS, "unit": "uV",
                    "source": "single sEMG channel, SeizeIT2 left deltoid placement"},
            "accelerometer": {"samples_per_axis": C.IMU_FS * W, "rate_hz": C.IMU_FS, "unit": "g",
                              "note": "training data rescaled per run so that median |acc| = 1 g"},
            "gyroscope": {"samples_per_axis": C.IMU_FS * W, "rate_hz": C.IMU_FS,
                          "unit": "any fixed unit (only scale-invariant gyroscope features are used)"},
            "axes": "any orientation: all features are rotation invariant",
            "timestamp": "ms of the last sample in the window; windows arrive every hop_s",
        },
        "features": list(FEATURE_NAMES),
        "quality": {"flat_std": C.FLAT_STD, "rule": "a flat EMG or IMU window scores 0"},
        "postprocessing": {"smoothing_k": k, "threshold": thr, "refractory_s": C.REFRACTORY_S,
                           "gap_reset_s": 1.5 * C.HOP_S,
                           "rule": "alarm when the mean of the last k scores >= threshold and no alarm "
                                   "in the previous refractory_s seconds; a gap in the stream restarts the mean"},
        "output": {"score": "internal only, never shown to users", "event_types": ["no_event", "possible_event"]},
    }, indent=1))
    (md / "metadata.json").write_text(json.dumps({
        "model_id": MODEL_ID, "version": MODEL_VERSION, "format": FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_run": run.name, "git_commit_at_export": git_commit(),
        "model": sel["candidate"], "kind": mj["kind"],
        "trees": len(mj.get("trees", [])), "sklearn": sklearn.__version__, "numpy": np.__version__,
        "dataset": "SeizeIT2, OpenNeuro ds005873 v1.1.0 (CC0); trained only on subjects in split.json folds",
        "task": "window-level detection of focal-to-bilateral tonic-clonic (convulsive) seizure patterns "
                "from one sEMG channel + one 6-axis IMU",
        "selection": {k_: sel[k_] for k_ in ("candidate", "window_s", "k", "threshold", "sensitivity",
                                             "far_per_24h", "latency_s_median", "met_far_cap")},
        "export_check": {"dev_windows_compared": int(X.shape[0]), "max_abs_score_diff": max_diff,
                         "decision_agreement": agree},
        "status": "experimental research component; not clinically validated; not a diagnostic device",
    }, indent=1))
    (md / "split.json").write_text(json.dumps(split, indent=1))
    metrics = {"test": test, "cv_out_of_fold": cv, "selected": sel["candidate"] + f"@W{W}"}
    (md / "evaluation" / "metrics.json").write_text(json.dumps(metrics, indent=1))
    for f in ("confusion_matrix.png", "training_history.png", "cv_operating_curves.png"):
        if (run / f).exists():
            shutil.copy(run / f, md / "evaluation" / f)
    print(f"exported {MODEL_ID} {MODEL_VERSION}: {mj['kind']}, {(md / 'model.json').stat().st_size / 1e3:.0f} kB, "
          f"max diff vs sklearn {max_diff:.2e}")


if __name__ == "__main__":
    main()
