"""Estimate the SeizeIT2 gyroscope unit from rigid-body kinematics (no external reference needed).

    python -m ml.seizeit2.gyro_calibration      # -> models/seizeit2/evaluation/gyro_calibration.json

SeizeIT2 stores ACC and GYR of the chest module in a per-recording scale (median |acc| ~1/2 or
~1/6 instead of 1 g), and labels GYR as "g". After dividing both by the recording's gravity
estimate (lyra_mlm.seizeit2.calibrate_imu), the gyro is consistent across recordings but its unit
is still unknown. For a rigid sensor, the gravity direction u (from the accelerometer, in g) turns
as du/dt = -w x u, with w in rad/s. Fitting du/dt against -(w_file x u) by least squares gives
rad/s per calibrated gyro unit, and trying every axis permutation/sign checks the axis mapping.
Samples are selected on the gyro side (|w x u| large, |acc| close to 1 g), so the slope is not
biased by selecting on the noisier accelerometer derivative.
"""

from __future__ import annotations

import itertools
import json
import shutil

import numpy as np
from lyra_mlm.seizeit2 import ACC_SCALE_VALID, IMU_LABELS
from lyra_mlm.edf import read_edf

from . import config as C
from . import data_loader as D

N_PER_REGIME = 12
FS = 25


def fit_recording(acc: np.ndarray, gyr: np.ndarray) -> dict:
    """acc in g, gyr in calibrated file units; returns slope (rad/s per unit) and the best axis map."""
    k = np.ones(5) / 5  # 0.2 s smoothing on both, so they see the same bandwidth
    a = np.stack([np.convolve(acc[:, i], k, "same") for i in range(3)], 1)
    w = np.stack([np.convolve(gyr[:, i], k, "same") for i in range(3)], 1)
    n = np.linalg.norm(a, axis=1)
    u = a / n[:, None]
    du = (u[2:] - u[:-2]) * FS / 2
    uc, wc, nc = u[1:-1], w[1:-1], n[1:-1]
    best = None
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            pred = -np.cross(wc[:, perm] * np.array(sg), uc)
            m = (np.abs(nc - 1) < 0.05) & (np.linalg.norm(pred, axis=1) > 0.08)
            if m.sum() < 500:
                continue
            c = float((pred[m] * du[m]).sum() / (pred[m] ** 2).sum())
            res = float(np.linalg.norm(du[m] - c * pred[m], axis=1).mean() / np.linalg.norm(du[m], axis=1).mean())
            if best is None or res < best["residual"]:
                best = {"rad_s_per_unit": c, "axes": list(perm), "signs": list(sg), "residual": res, "n": int(m.sum())}
    return best or {}


def main() -> None:
    from lyra_mlm.seizeit2 import SeizeIT2Adapter, calibrate_imu

    adapter = SeizeIT2Adapter(C.DATASET)
    rng = np.random.default_rng(C.SEED)
    rids = [r for r in adapter.recording_ids()]
    rng.shuffle(rids)
    picked = {"~1/2": [], "~1/6": []}
    results = []
    tmp = C.CACHE / "tmp" / "gyro_calibration"
    for rid in rids:
        if all(len(v) >= N_PER_REGIME for v in picked.values()):
            break
        mov = adapter._paths(rid)["mov"]
        if not mov.is_symlink() or D.annex_key(mov)[0] < 20e6:  # skip short (< ~6 h) recordings
            continue
        p = D.fetch(mov, tmp / mov.name)
        try:
            _, sig = read_edf(p, IMU_LABELS)
        except Exception:
            continue
        finally:
            p.unlink(missing_ok=True)
        if not all(lb in sig for lb in IMU_LABELS):
            continue
        vals = np.column_stack([sig[lb][1] for lb in IMU_LABELS])
        cal, flags, note = calibrate_imu(vals, gyro_to_dps=1.0)  # keep calibrated file units here
        if flags is not None:
            continue
        scale = float(np.median(np.linalg.norm(vals[::FS, :3], axis=1)))
        regime = "~1/2" if scale >= 0.3 else "~1/6"
        if len(picked[regime]) >= N_PER_REGIME:
            continue
        fit = fit_recording(cal[:, :3], cal[:, 3:])
        if not fit:
            continue
        picked[regime].append(rid)
        results.append({"recording_id": rid, "regime": regime, "acc_scale": scale, **fit})
        print(f"{rid} {regime} {fit['rad_s_per_unit']:.3f} rad/s/unit axes {fit['axes']} {fit['signs']} "
              f"res {fit['residual']:.2f}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    c = np.array([r["rad_s_per_unit"] for r in results])
    identity = sum(r["axes"] == [0, 1, 2] and r["signs"] == [1, 1, 1] for r in results)
    summary = {
        "method": __doc__.strip().splitlines()[0],
        "recordings": len(results), "identity_axis_map": identity,
        "rad_s_per_unit_median": float(np.median(c)), "rad_s_per_unit_iqr": [float(x) for x in np.percentile(c, [25, 75])],
        "deg_s_per_unit_median": float(np.degrees(np.median(c))),
        "deg_s_per_unit_iqr": [float(np.degrees(x)) for x in np.percentile(c, [25, 75])],
        "by_regime_median_deg_s": {g: float(np.degrees(np.median([r["rad_s_per_unit"] for r in results if r["regime"] == g])))
                                   for g in picked},
        "acc_scale_valid": list(ACC_SCALE_VALID),
        "per_recording": results,
    }
    out = C.MODEL_DIR / "evaluation" / "gyro_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_recording"}, indent=1))


if __name__ == "__main__":
    main()
