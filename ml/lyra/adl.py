"""Optional independent no-seizure ADL cohort recorded with the pod (ESP32-S3 + MyoWare ENV + BMI270).

    python -m ml.lyra.adl --lyra-negatives-dir PATH [--column-map map.json] [--calibrate-far 1.0]

Schema (CSV or Parquet, one file per recording; column names remappable with --column-map):
  subject_id, recording_id, timestamp (s), emg (MyoWare ENV, uV-equivalent after gain calibration),
  ax, ay, az, gx, gy, gz.
Rows are a common timeline; a modality's columns are empty on rows where it has no sample.
Units and rates: PATH/metadata.json (or a per-file <name>.json) with
  {"emg_rate_hz": 50, "imu_rate_hz": 25, "acc_unit": "m/s^2" | "g", "gyr_unit": "rad/s" | "deg/s",
   "activities": {"<recording_id>": "walking"}}  (activities optional)
Rates must equal the pod rates (50 / 25 Hz); nothing is resampled or interpolated.

These recordings are NEGATIVES ONLY. They never train, select features, fit scalers or change
tree splits, and they cannot show seizure sensitivity. With >= 4 volunteers they are split by
volunteer into a calibration subset (may move the threshold to a predeclared FAR target, locked
before anything else is looked at) and an untouched evaluation subset.
"""

from __future__ import annotations

import argparse
import json
import pickle
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import data as Dt
from . import events as Ev
from . import export_c as X
from . import features as F
from . import golden as G
from . import windows as Wn

REQUIRED = ("subject_id", "recording_id", "timestamp", "emg", "ax", "ay", "az", "gx", "gy", "gz")


class AdlSchemaError(ValueError):
    pass


def read_file(path: Path, colmap: dict[str, str], meta: dict) -> dict:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    df = df.rename(columns={v: k for k, v in colmap.items()})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise AdlSchemaError(f"{path.name}: missing columns {missing}")
    for k in ("subject_id", "recording_id"):
        if df[k].nunique() != 1:
            raise AdlSchemaError(f"{path.name}: one {k} per file expected")
    t = df["timestamp"].to_numpy(float)
    if not np.all(np.diff(t) > 0):
        raise AdlSchemaError(f"{path.name}: timestamps must be strictly increasing")
    if meta.get("emg_rate_hz") != C.ENV_RATE or meta.get("imu_rate_hz") != C.IMU_RATE:
        raise AdlSchemaError(f"{path.name}: rates {meta.get('emg_rate_hz')}/{meta.get('imu_rate_hz')} Hz, "
                             f"need {C.ENV_RATE}/{C.IMU_RATE} Hz (no resampling is done)")
    out = {}
    for mod, cols, rate in (("env", ["emg"], C.ENV_RATE), ("imu", ["ax", "ay", "az", "gx", "gy", "gz"], C.IMU_RATE)):
        rows = df[cols].notna().all(axis=1).to_numpy()
        ts = t[rows]
        if ts.size < 2:
            raise AdlSchemaError(f"{path.name}: no {mod} samples")
        dt = np.diff(ts)
        if np.abs(dt - 1 / rate).max() > 0.25 / rate:
            raise AdlSchemaError(f"{path.name}: {mod} sampling is not uniform at {rate} Hz (gaps or jitter)")
        out[mod] = (ts, df.loc[rows, cols].to_numpy(np.float64))
    acc, gyr = out["imu"][1][:, :3], out["imu"][1][:, 3:]
    acc = acc * C.GRAVITY if meta.get("acc_unit") == "g" else acc
    gyr = np.radians(gyr) if meta.get("gyr_unit") == "deg/s" else gyr
    if meta.get("acc_unit") not in ("g", "m/s^2") or meta.get("gyr_unit") not in ("rad/s", "deg/s"):
        raise AdlSchemaError(f"{path.name}: acc_unit/gyr_unit missing or unknown")
    if abs(out["env"][0][0] - out["imu"][0][0]) > 1 / C.IMU_RATE:
        raise AdlSchemaError(f"{path.name}: EMG and IMU streams start more than one IMU sample apart")
    hours = min(out["env"][0][-1] - out["env"][0][0], out["imu"][0][-1] - out["imu"][0][0]) / 3600
    return {"subject": str(df["subject_id"].iloc[0]), "recording": str(df["recording_id"].iloc[0]),
            "env": out["env"][1][:, 0], "acc": acc, "gyr": gyr, "hours": hours}


def load_dir(root: Path, colmap: dict[str, str]) -> list[dict]:
    base_meta = json.loads((root / "metadata.json").read_text()) if (root / "metadata.json").exists() else {}
    recs = []
    for p in sorted([*root.glob("*.csv"), *root.glob("*.parquet")]):
        side = p.with_suffix(".json")
        meta = base_meta | (json.loads(side.read_text()) if side.exists() else {})
        r = read_file(p, colmap, meta)
        r["activity"] = meta.get("activities", {}).get(r["recording"], "unlabelled")
        recs.append(r)
    if not recs:
        raise AdlSchemaError(f"no CSV/Parquet recordings in {root}")
    return recs


def score(rec: dict, model_prob, window: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(window end times, probabilities with NaN for invalid windows, valid mask)."""
    w, hop = C.WINDOWS[window]
    n_imu = min(len(rec["acc"]), len(rec["env"]) // 2)
    end = Wn.grid(n_imu, w, hop)
    wi = int(w * C.IMU_RATE)
    p = np.full(end.size, np.nan)
    valid = np.zeros(end.size, bool)
    for k, e in enumerate(end):
        env = rec["env"][2 * (e - wi):2 * e].astype(np.float32)
        acc = rec["acc"][e - wi:e].astype(np.float32)
        gyr = rec["gyr"][e - wi:e].astype(np.float32)
        if G.py_valid(env, acc, gyr) != 0 or (e - wi) / C.IMU_RATE < C.WARMUP_S:
            continue
        x = F.compute(env[None].astype(np.float64), acc[None].astype(np.float64), gyr[None].astype(np.float64), 1)
        p[k] = model_prob(x.astype(np.float32))[0]
        valid[k] = True
    return end / C.IMU_RATE, p, valid


def false_alarms(recs: list[dict], scored: list, threshold: float, n_open: int, m_close: int, hop: float) -> dict:
    per_subject: dict[str, dict] = {}
    for r, (t, p, v) in zip(recs, scored):
        opens = Ev.openings(t[v], p[v] >= threshold, n_open, m_close)
        emitted, suppressed = Ev.alerts(opens)
        s = per_subject.setdefault(r["subject"], {"false_alerts": 0, "valid_hours": 0.0, "recordings": 0})
        s["false_alerts"] += len(emitted)
        s["valid_hours"] += float(v.sum()) * hop / 3600
        s["recordings"] += 1
    fa = sum(s["false_alerts"] for s in per_subject.values())
    h = sum(s["valid_hours"] for s in per_subject.values())
    ci = None
    if h > 0:  # exact Poisson CI on the count
        from scipy.stats import chi2
        ci = [chi2.ppf(0.025, 2 * fa) / 2 / h * 24 if fa else 0.0, chi2.ppf(0.975, 2 * fa + 2) / 2 / h * 24]
    return {"volunteers": len(per_subject), "recordings": len(recs), "valid_hours": round(h, 3), "false_alerts": fa,
            "far_per_24h": fa / h * 24 if h else None, "far_per_24h_95ci_poisson": ci, "per_volunteer": per_subject}


def split_volunteers(subjects: list[str]) -> tuple[set[str], set[str]]:
    order = sorted(set(subjects), key=lambda s: zlib.crc32(s.encode()) ^ C.SEED)
    half = len(order) // 2
    return set(order[:half]), set(order[half:])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Independent ADL negative cohort (pod recordings)")
    ap.add_argument("--lyra-negatives-dir", type=Path, required=True)
    ap.add_argument("--column-map", type=Path, help="JSON {schema_name: file_column_name}")
    ap.add_argument("--run", type=Path, required=True, help="benchmark run holding final_deployable.pkl")
    ap.add_argument("--calibrate-far", type=float, default=None,
                    help="predeclared FAR target (/24 h) for the calibration subset; omit to keep the locked threshold")
    args = ap.parse_args(argv)
    if not Dt.LOCK.exists():
        raise SystemExit("deploy.py must freeze the configuration first")
    colmap = json.loads(args.column_map.read_text()) if args.column_map else {}
    recs = load_dir(args.lyra_negatives_dir, colmap)
    with (args.run / "final_deployable.pkl").open("rb") as fh:
        fm = pickle.load(fh)
    sel = fm["selection"]
    _, replica, _ = X.model_header(sel["model"], fm["model"], fm["cols"], fm["A"], fm["B"], "adl")
    hop = C.WINDOWS[sel["window"]][1]
    scored = [score(r, replica, sel["window"]) for r in recs]
    thr = float(np.float32(sel["threshold"]))
    out = {"cohort": "independent no-seizure ADL volunteers (pod hardware)", "locked_threshold": thr,
           "activities": sorted({r["activity"] for r in recs}),
           "note": "negatives only: no seizure sensitivity and no clinical generalisation can be inferred"}
    cal, ev = split_volunteers([r["subject"] for r in recs])
    if args.calibrate_far is not None and len(cal) >= 2 and len(ev) >= 2:
        ci = [i for i, r in enumerate(recs) if r["subject"] in cal]
        grid = [t for t in C.THRESHOLDS if t >= thr] + [0.85, 0.9, 0.95]
        new = next((t for t in grid if (false_alarms([recs[i] for i in ci], [scored[i] for i in ci], t, sel["n_open"],
                                                      sel["m_close"], hop)["far_per_24h"] or 0) <= args.calibrate_far),
                   None)
        out["calibration"] = {"volunteers": sorted(cal), "far_target": args.calibrate_far, "adjusted_threshold": new,
                              "locked_before_evaluation": True}
        ei = [i for i, r in enumerate(recs) if r["subject"] in ev]
        out["evaluation_subset"] = false_alarms([recs[i] for i in ei], [scored[i] for i in ei], new if new else thr,
                                                sel["n_open"], sel["m_close"], hop)
    else:
        out["all_volunteers_at_locked_threshold"] = false_alarms(recs, scored, thr, sel["n_open"], sel["m_close"], hop)
    path = args.run / "adl_eval.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
