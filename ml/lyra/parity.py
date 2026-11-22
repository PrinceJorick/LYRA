"""Python/C parity on REAL development-pool data (kept local: nothing from it is written to the repo).

    python -m ml.lyra.parity --run ml/lyra/runs/bench-<id>

1. Features: two development recordings are streamed and verified as in extract.py, every window
   of the deployed configuration is cast to float32 (what the pod delivers) and scored by
   ml/lyra/features.py and by firmware/model/lyra_features.c.
2. Model: up to 200,000 cached development feature vectors are scored by the Python replica of the
   exported model and by firmware/model/lyra_model.h.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import bids
from . import config as C
from . import data as Dt
from . import extract as E
from . import features as F
from . import signals as S
from . import windows as Wn

HARNESS = r"""
#include <stdio.h>
#include <stdint.h>
#include "lyra_features.h"
#include "lyra_model.h"
int main(int argc, char **argv) {
    (void)argv;
    if (argc > 1) { /* model mode: rows of LYRA_N_FEATURES floats -> probability */
        float x[LYRA_N_FEATURES];
        while (fread(x, sizeof x, 1, stdin) == 1) { float p = lyra_model_predict(x); fwrite(&p, 4, 1, stdout); }
        return 0;
    }
    int32_t n; static float env[2048], acc[3072], gyr[3072]; float out[LYRA_N_FEATURES];
    while (fread(&n, 4, 1, stdin) == 1) {
        if (fread(env, 4, 2 * n, stdin) != (size_t)(2 * n) || fread(acc, 4, 3 * n, stdin) != (size_t)(3 * n) ||
            fread(gyr, 4, 3 * n, stdin) != (size_t)(3 * n)) return 2;
        int32_t c = lyra_features_compute(env, 2 * n, acc, gyr, n, out);
        fwrite(&c, 4, 1, stdout); fwrite(out, 4, LYRA_N_FEATURES, stdout);
    }
    return 0;
}
"""


def build(work: Path) -> Path:
    for f in ("lyra_features.c", "lyra_features.h", "lyra_feature_spec.h", "lyra_model.h", "lyra_detector_config.h"):
        shutil.copy(C.FIRMWARE / f, work / f)
    (work / "h.c").write_text(HARNESS)
    exe = work / "h"
    subprocess.run(["gcc", "-std=c99", "-O2", "-I", str(work), str(work / "h.c"), str(work / "lyra_features.c"), "-lm",
                    "-o", str(exe)], check=True)
    return exe


def feature_parity(exe: Path, window: str, n_recordings: int = 2) -> dict:
    recs = [r for r in bids.match_recordings()[0] if C.is_dev(r.recording_id) and r.emg and r.mov]
    rng = np.random.default_rng(C.SEED)
    pick = [recs[i] for i in rng.choice(len(recs), n_recordings, replace=False)]
    w, hop = C.WINDOWS[window]
    wi = int(w * C.IMU_RATE)
    stats = {"recordings": [r.recording_id for r in pick], "windows": 0, "exact": 0, "values": 0,
             "max_rel_diff": 0.0, "code_mismatch": 0, "max_rel_diff_vs_float64_training_inputs": 0.0}
    for rec in pick:
        work = E.TMP / f"parity_{rec.recording_id}"
        work.mkdir(parents=True, exist_ok=True)
        try:
            _, dig, phys = S.read_edf(E._local(rec.emg, work), (C.EMG_CHANNEL,))
            env = S.simulate_env(phys[:, 0], 256)
            _, _, mov = S.read_edf(E._local(rec.mov, work), C.IMU_CHANNELS)
            acc, gyr, ok, _ = S.imu_physical(mov)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if not ok:
            continue
        end = Wn.grid(min(len(acc), len(env) // 2), w, hop)
        e32 = np.stack([env[2 * (k - wi):2 * k] for k in end]).astype(np.float32)
        a32 = np.stack([acc[k - wi:k] for k in end]).astype(np.float32)
        g32 = np.stack([gyr[k - wi:k] for k in end]).astype(np.float32)
        py = F.compute(e32.astype(np.float64), a32.astype(np.float64), g32.astype(np.float64), len(end)).astype(np.float32)
        py64 = F.compute(np.stack([env[2 * (k - wi):2 * k] for k in end]), np.stack([acc[k - wi:k] for k in end]),
                         np.stack([gyr[k - wi:k] for k in end]), len(end)).astype(np.float32)
        payload = b"".join(np.int32(wi).tobytes() + e32[i].tobytes() + a32[i].tobytes() + g32[i].tobytes()
                           for i in range(len(end)))
        raw = subprocess.run([str(exe)], input=payload, capture_output=True, check=True).stdout
        rows = np.frombuffer(raw, np.float32).reshape(len(end), 1 + len(C.FEATURES))
        codes, cx = rows[:, 0].view(np.int32), rows[:, 1:]
        ok_rows = codes == 0
        stats["code_mismatch"] += int((~ok_rows).sum())
        rel = np.abs(cx[ok_rows] - py[ok_rows]) / np.maximum(1, np.abs(py[ok_rows]))
        stats["windows"] += int(ok_rows.sum())
        stats["values"] += int(rel.size)
        stats["exact"] += int((cx[ok_rows] == py[ok_rows]).sum())
        stats["max_rel_diff"] = max(stats["max_rel_diff"], float(rel.max()))
        rel64 = np.abs(py - py64) / np.maximum(1, np.abs(py64))
        stats["max_rel_diff_vs_float64_training_inputs"] = max(stats["max_rel_diff_vs_float64_training_inputs"],
                                                               float(np.nanmax(rel64)))
    stats["tolerance"] = "relative 1e-5 (max(1,|ref|) scale)"
    stats["passed"] = stats["max_rel_diff"] <= 1e-5 and stats["code_mismatch"] == 0
    return stats


def model_parity(exe: Path, replica, window: str, threshold: float, modality: str, n: int = 200_000) -> dict:
    D = Dt.load(window, "dev")
    v = np.flatnonzero(D.valid(modality))
    rng = np.random.default_rng(C.SEED + 1)
    s = np.sort(rng.choice(v, min(n, v.size), replace=False))
    X = np.ascontiguousarray(D.X[s], np.float32)
    raw = subprocess.run([str(exe), "model"], input=X.tobytes(), capture_output=True, check=True).stdout
    pc = np.frombuffer(raw, np.float32)
    pp = replica(X)
    thr = np.float32(threshold)
    near = np.abs(pp - thr) <= 1e-6
    return {"rows": int(s.size), "max_abs_prob_diff": float(np.abs(pc - pp).max()),
            "decision_agreement": float(((pc >= thr) == (pp >= thr))[~near].mean()), "rows_within_1e-6_of_threshold":
            int(near.sum()), "passed": float(np.abs(pc - pp).max()) <= 1e-6}


def main(argv: list[str] | None = None) -> None:
    import pickle

    from . import export_c as X

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    run = ap.parse_args(argv).run
    with (run / "final_deployable.pkl").open("rb") as fh:
        fm = pickle.load(fh)
    sel = fm["selection"]
    _, replica, _ = X.model_header(sel["model"], fm["model"], fm["cols"], fm["A"], fm["B"], "parity")
    with tempfile.TemporaryDirectory() as td:
        exe = build(Path(td))
        out = {"features_real_dev_windows": feature_parity(exe, sel["window"]),
               "model_real_dev_feature_vectors": model_parity(exe, replica, sel["window"], sel["threshold"],
                                                              sel["modality"])}
    (run / "parity_real.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
