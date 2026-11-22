"""One-time evaluation of the frozen configuration on the final cohort (sub-097..sub-125).

    python -m ml.lyra.final_eval --run ml/lyra/runs/bench-<id>

Requires the lock written by deploy.py and refuses to run twice. Nothing is tuned here: models,
calibration, thresholds, N/M and refractory come from the lock, and the deployed model is scored
with the replica of the exported C code (bit-exact with firmware/model). The cohort was used to
train earlier LYRA models (v1/v2, PR #1), so it is held out for this pipeline only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score

from . import benchmark as Bm
from . import config as C
from . import data as Dt
from . import events as Ev
from . import export_c as X

BOOT = 2000


def per_subject(D: Dt.Windows, p: np.ndarray, valid: np.ndarray, sel: dict) -> dict[int, Ev.EventCounts]:
    starts = np.searchsorted(D.rec, np.arange(len(D.recordings) + 1))
    out: dict[int, Ev.EventCounts] = {}
    thr = float(np.float32(sel["threshold"]))
    for r in range(len(D.recordings)):
        a, b = starts[r], starts[r + 1]
        v = valid[a:b]
        ec = Ev.score_recording(D.t[a:b], p[a:b], v, v & (D.label[a:b] == 0), D.window_s, D.hop_s, D.target_iv[r],
                                D.nontarget_iv[r], thr, sel["n_open"], sel["m_close"])
        out.setdefault(int(D.subjects[r]), Ev.EventCounts()).add(ec)
    return out


def bootstrap(subj: dict[int, Ev.EventCounts], seed: int = C.SEED) -> dict:
    """Patient-level clustered bootstrap (subjects resampled with replacement)."""
    keys = sorted(subj)
    det = np.array([subj[k].detected for k in keys])
    ev = np.array([subj[k].evaluable for k in keys])
    fa = np.array([subj[k].false_alerts for k in keys])
    h = np.array([subj[k].negative_hours for k in keys])
    rng = np.random.default_rng(seed)
    sens, far = [], []
    for _ in range(BOOT):
        i = rng.integers(0, len(keys), len(keys))
        if ev[i].sum():
            sens.append(det[i].sum() / ev[i].sum())
        if h[i].sum():
            far.append(fa[i].sum() / h[i].sum() * 24)
    ci = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if x else None  # noqa: E731
    return {"event_sensitivity_95ci": ci(sens), "far_per_24h_95ci": ci(far), "resamples": BOOT,
            "method": "subjects resampled with replacement"}


def breakdown(subj: dict[int, Ev.EventCounts]) -> list[dict]:
    rows = []
    for s, c in sorted(subj.items()):
        lat = np.asarray(c.latencies)
        rows.append({"subject": f"sub-{s:03d}", "valid_negative_hours": round(c.negative_hours, 2),
                     "target_events": c.events, "evaluable_events": c.evaluable, "detected": c.detected,
                     "missed": c.evaluable - c.detected,
                     "event_sensitivity": c.detected / c.evaluable if c.evaluable else None,
                     "false_alerts": c.false_alerts,
                     "far_per_24h": c.false_alerts / c.negative_hours * 24 if c.negative_hours >= 1 else None,
                     "latency_s_median": float(np.median(lat)) if lat.size else None})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    run = ap.parse_args(argv).run
    if not Dt.LOCK.exists():
        raise SystemExit("no lock: run deploy.py first")
    out_path = run / "final_eval.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists: the final cohort is evaluated once")
    lock = json.loads(Dt.LOCK.read_text())
    for name, digest in lock["final_models_sha256"].items():
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{name} changed since the lock")
    for name, digest in lock["firmware_sha256"].items():
        if hashlib.sha256((C.FIRMWARE / name).read_bytes()).hexdigest() != digest:
            raise SystemExit(f"firmware/model/{name} changed since the lock")
    results, cache = {}, {}
    for role in lock["selection"]:
        with (run / f"final_{role}.pkl").open("rb") as fh:
            fm = pickle.load(fh)
        sel = fm["selection"]
        w = sel["window"]
        if w not in cache:
            cache.clear()
            cache[w] = Dt.load(w, "test")
        D = cache[w]
        valid = D.valid("fusion")  # common exposure for every role, as on the development pool
        p = np.full(D.t.size, np.nan)
        if role == "deployable":
            hdr, replica, _ = X.model_header(sel["model"], fm["model"], fm["cols"], fm["A"], fm["B"], "check")
            p[valid] = replica(D.X[valid])
            scored_with = "replica of the exported C model (bit-exact with firmware/model)"
        else:
            z = Bm.raw_score(sel["model"], fm["model"], D.X[valid][:, fm["cols"]].astype(np.float64))
            p[valid] = expit(fm["A"] * z + fm["B"])
            scored_with = f"{sel['model']} (scikit-learn/XGBoost) + Platt"
        subj = per_subject(D, p, valid, sel)
        tot = Ev.EventCounts()
        for c in subj.values():
            tot.add(c)
        ok = valid & (D.label >= 0)
        results[role] = {"selection": {k: sel[k] for k in ("window", "modality", "model", "threshold", "n_open", "m_close")},
                         "scored_with": scored_with, **tot.summary(),
                         "window_auprc": float(average_precision_score(D.label[ok], p[ok])) if (D.label[ok] == 1).any() else None,
                         "bootstrap": bootstrap(subj), "per_subject": breakdown(subj)}
        print(role, {k: results[role][k] for k in ("detected", "evaluable_events", "far_per_24h", "latency_s_median")},
              flush=True)
    audit = []
    for m in Dt.manifests("test"):
        audit.append({k: m.get(k) for k in ("recording_id", "events", "windows", "excluded_reason", "problems")})
    out_path.write_text(json.dumps({"cohort": "sub-097..sub-125", "caveat": __doc__.split("The cohort")[1].strip(),
                                    "lock": lock, "results": results, "test_manifest": audit}, indent=1, default=str))


if __name__ == "__main__":
    from ml.lyra.final_eval import main as _main

    _main()
