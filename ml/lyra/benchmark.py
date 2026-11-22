"""Patient-wise benchmark on the development pool (sub-001..sub-096) only.

    python -m ml.lyra.benchmark            # -> ml/lyra/runs/<timestamp>/ (+ OOF caches in ml/lyra/cache/oof)

For every window configuration x modality (emg / imu / fusion) x classifier (LR, RF 40x7, ET 40x7,
XGBoost 40x4, RBF SVC):
1. GroupKFold(5) by subject, identical folds for every experiment.
2. In each training part: negatives subsampled per recording for fitting (all positives kept), class
   weighting, scaling inside the pipeline (LR/SVC), LR's C chosen by an inner GroupKFold(3), and a
   Platt calibration fitted on inner patient-wise out-of-fold scores (weights restore the true
   prevalence). Validation subjects are scored on every valid window, never subsampled.
3. The calibrated out-of-fold probabilities feed the event sweep (thresholds 0.20..0.80, N, M,
   60 s refractory) on the common fusion-valid exposure, so the three modalities are compared on
   the same windows and events.
Nothing here can read sub-097..sub-125 (data.load refuses).
"""

from __future__ import annotations

import argparse
import json
import pickle
import resource
import time
import zlib
from datetime import datetime

import numpy as np
from scipy.special import expit, logit
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config as C
from . import data as Dt
from . import events as Ev

LR_C_GRID = (0.01, 0.1, 1.0, 10.0)


def make_model(name: str, n_pos: int = 1, n_neg: int = 1, lr_c: float = 1.0):
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(C=lr_c, class_weight="balanced", max_iter=3000))
    if name == "RF":
        return RandomForestClassifier(n_estimators=40, max_depth=7, class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=C.SEED)
    if name == "ET":
        return ExtraTreesClassifier(n_estimators=40, max_depth=7, class_weight="balanced_subsample", n_jobs=-1,
                                    random_state=C.SEED)
    if name == "XGB":
        return XGBClassifier(n_estimators=40, max_depth=4, tree_method="hist", n_jobs=12, random_state=C.SEED,
                             scale_pos_weight=n_neg / max(n_pos, 1), eval_metric="aucpr")
    if name == "SVC":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                                                   cache_size=2000))
    raise KeyError(name)


MODELS = ("LR", "RF", "ET", "XGB", "SVC")


def raw_score(name: str, model, X: np.ndarray) -> np.ndarray:
    """Uncalibrated log-odds-like score z; calibrated probability = expit(A z + B)."""
    if name in ("LR", "SVC"):
        if name == "SVC" and len(X) > 200_000:
            from joblib import Parallel, delayed
            parts = np.array_split(X, 24)
            return np.concatenate(Parallel(n_jobs=12)(delayed(model.decision_function)(p) for p in parts))
        return model.decision_function(X)
    if name == "XGB":
        return model.predict(X, output_margin=True)
    return logit(np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6))


def complexity(name: str, model) -> dict:
    if name in ("RF", "ET"):
        nodes = int(sum(e.tree_.node_count for e in model.estimators_))
        return {"trees": len(model.estimators_), "nodes": nodes, "leaves": (nodes + len(model.estimators_)) // 2}
    if name == "XGB":
        df = model.get_booster().trees_to_dataframe()
        return {"trees": int(df["Tree"].nunique()), "nodes": int(len(df)), "leaves": int((df["Feature"] == "Leaf").sum())}
    if name == "LR":
        return {"parameters": int(model[-1].coef_.size + 1)}
    return {"support_vectors": int(model[-1].support_vectors_.shape[0]),
            "parameters": int(model[-1].support_vectors_.size + model[-1].dual_coef_.size + 1)}


# ---- folds and pools -------------------------------------------------------------------------------


def subject_folds() -> dict[int, int]:
    """GroupKFold(5) at subject level from each dev subject's w4 window count; reused everywhere."""
    counts: dict[int, int] = {}
    for m in Dt.manifests("dev"):
        if m.get("windows"):
            s = C.subject_number(m["recording_id"])
            counts[s] = counts.get(s, 0) + m["windows"]["w4"]["n"]
    subs = np.array(sorted(counts))
    groups = np.repeat(subs, [max(1, counts[s] // 1000) for s in subs])
    out = {}
    for f, (_, va) in enumerate(GroupKFold(n_splits=C.OUTER_FOLDS).split(groups, groups=groups)):
        for s in np.unique(groups[va]):
            out[int(s)] = f
    assert all(s in C.DEV for s in out)
    return out


def pool(D: Dt.Windows, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(indices, weights) for fitting: every positive, NEG_FRACTION of each recording's negatives
    (deterministic), weights = inverse sampling rate so calibration sees the true prevalence."""
    idx, wts = [], []
    starts = np.searchsorted(D.rec, np.arange(len(D.recordings) + 1))
    for r in range(len(D.recordings)):
        a, b = starts[r], starts[r + 1]
        if a == b:
            continue
        m = mask[a:b]
        lab = D.label[a:b]
        pos = np.flatnonzero(m & (lab == 1)) + a
        neg = np.flatnonzero(m & (lab == 0)) + a
        if neg.size:
            k = min(neg.size, max(C.NEG_MIN, round(C.NEG_FRACTION * neg.size)))
            rng = np.random.default_rng(zlib.crc32(D.recordings[r].encode()) ^ C.SEED)
            neg_s = np.sort(rng.choice(neg, k, replace=False))
            idx += [pos, neg_s]
            wts += [np.ones(pos.size), np.full(k, neg.size / k)]
        elif pos.size:
            idx.append(pos)
            wts.append(np.ones(pos.size))
    return (np.concatenate(idx), np.concatenate(wts)) if idx else (np.empty(0, int), np.empty(0))


def svc_subsample(idx: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos, neg = idx[y == 1], idx[y == 0]
    kp = min(pos.size, C.SVC_MAX_TRAIN // 2)
    kn = min(neg.size, C.SVC_MAX_TRAIN - kp)
    return np.sort(np.concatenate([rng.choice(pos, kp, replace=False), rng.choice(neg, kn, replace=False)]))


def fit(name: str, D: Dt.Windows, cols: list[int], idx: np.ndarray, lr_c: float = 1.0, seed: int = C.SEED):
    y = D.label[idx]
    if name == "SVC":
        idx = svc_subsample(idx, y, seed)
        y = D.label[idx]
    model = make_model(name, int((y == 1).sum()), int((y == 0).sum()), lr_c)
    model.fit(D.X[idx][:, cols].astype(np.float64), y)
    return model


def platt(z: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    cal = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y, sample_weight=w)
    return float(cal.coef_[0, 0]), float(cal.intercept_[0])


# ---- one experiment ----------------------------------------------------------------------------------


def train(name: str, D: Dt.Windows, cols: list[int], mask: np.ndarray, seed: int):
    """The full training procedure on the subjects in `mask`: pool, inner patient-wise CV (LR's C and
    the Platt calibration), final fit. Used for every outer fold and for the final model."""
    idx, wts = pool(D, mask)
    inner = list(GroupKFold(n_splits=C.INNER_FOLDS).split(idx, groups=D.subject[idx]))
    grid = LR_C_GRID if name == "LR" else (1.0,)
    zin = {c: np.empty(idx.size) for c in grid}
    for i_tr, i_va in inner:
        assert not set(D.subject[idx[i_tr]]) & set(D.subject[idx[i_va]]), "inner subject leak"
        for c in grid:
            mdl = fit(name, D, cols, idx[i_tr], c, seed)
            zin[c][i_va] = raw_score(name, mdl, D.X[idx[i_va]][:, cols].astype(np.float64))
    y_in = D.label[idx]
    best_c = max(grid, key=lambda c: average_precision_score(y_in, zin[c], sample_weight=wts))
    A, B = platt(zin[best_c], y_in, wts)
    t0 = time.perf_counter()
    model = fit(name, D, cols, idx, best_c, seed)
    return model, A, B, best_c, idx, time.perf_counter() - t0


def oof(name: str, D: Dt.Windows, modality: str, folds: dict[int, int]) -> dict:
    cols = list(C.MODALITIES[modality])
    valid = D.valid(modality)
    fold_of = np.array([folds.get(s, -1) for s in D.subjects])[D.rec]
    z = np.full(D.t.size, np.nan)
    p = np.full(D.t.size, np.nan)
    meta = []
    for f in range(C.OUTER_FOLDS):
        tr = valid & (fold_of != f) & (fold_of >= 0)
        va = valid & (fold_of == f)
        tr_subj = np.unique(D.subject[tr])
        assert not set(tr_subj) & set(np.unique(D.subject[va])), "subject leak"
        model, A, B, best_c, idx, t_fit = train(name, D, cols, tr, C.SEED + f)
        Xv = D.X[va][:, cols].astype(np.float64)
        t0 = time.perf_counter()
        z[va] = raw_score(name, model, Xv)
        t_inf = time.perf_counter() - t0
        p[va] = expit(A * z[va] + B)
        meta.append({"fold": f, "train_subjects": int(tr_subj.size), "val_subjects": int(np.unique(D.subject[va]).size),
                     "fit_windows": int(idx.size if name != "SVC" else min(idx.size, C.SVC_MAX_TRAIN)),
                     "fit_positive": int((D.label[idx] == 1).sum()), "lr_c": best_c if name == "LR" else None,
                     "platt": [A, B], "fit_s": round(t_fit, 3), "infer_us_per_window": round(t_inf / max(va.sum(), 1) * 1e6, 3),
                     "complexity": complexity(name, model), "pickle_bytes": len(pickle.dumps(model))})
    return {"z": z, "p": p, "folds": meta}


def sweep(D: Dt.Windows, p: np.ndarray, exposure_mask: np.ndarray) -> list[dict]:
    """Event metrics for every (threshold, N, M) on the given exposure (valid windows)."""
    starts = np.searchsorted(D.rec, np.arange(len(D.recordings) + 1))
    recs = []
    for r in range(len(D.recordings)):
        a, b = starts[r], starts[r + 1]
        v = exposure_mask[a:b]
        if not v.any() and not len(D.target_iv[r]):
            continue
        pr = np.where(v, p[a:b], np.nan)
        recs.append((r, a, b, v, np.nanmax(pr) if v.any() else -1.0))
    out = []
    for thr in C.THRESHOLDS:
        for n in C.N_OPEN:
            for m in C.M_CLOSE:
                tot, per_subj = Ev.EventCounts(), {}
                for r, a, b, v, pmax in recs:
                    if pmax < thr and not len(D.target_iv[r]):
                        ec = Ev.EventCounts(negative_hours=float((v & (D.label[a:b] == 0)).sum()) * D.hop_s / 3600)
                    else:
                        ec = Ev.score_recording(D.t[a:b], p[a:b], v, v & (D.label[a:b] == 0), D.window_s, D.hop_s,
                                                D.target_iv[r], D.nontarget_iv[r], thr, n, m)
                    tot.add(ec)
                    s = int(D.subjects[r])
                    per_subj.setdefault(s, Ev.EventCounts()).add(ec)
                row = {"threshold": thr, "grid": "spec" if thr in C.THRESHOLDS_SPEC else "extended", "n_open": n,
                       "m_close": m, **tot.summary()}
                det = [c for c in per_subj.values() if c.evaluable]
                row["subjects_with_events"] = len(det)
                row["subject_sensitivity_iqr"] = (np.percentile([c.detected / c.evaluable for c in det], [25, 75]).tolist()
                                                  if det else None)
                out.append(row)
    return out


def exp_hash(name: str, w: str, modality: str, folds: dict) -> str:
    return C.config_hash({"features": C.FEATURE_HASH, "window": w, "modality": modality, "model": name,
                          "params": repr(make_model(name).get_params()) if name != "XGB" else "xgb40x4",
                          "folds": sorted(folds.items()), "seed": C.SEED, "inner": C.INNER_FOLDS,
                          "neg": [C.NEG_FRACTION, C.NEG_MIN], "svc_max": C.SVC_MAX_TRAIN, "lr_grid": LR_C_GRID,
                          "calibration": "platt-inner-oof-v1"})


def tree_scaling_check(D: Dt.Windows, folds: dict) -> dict:
    """Trees split on per-feature order, so standardising should change nothing: verify on fold 0."""
    cols = list(C.MODALITIES["fusion"])
    fold_of = np.array([folds.get(s, -1) for s in D.subjects])[D.rec]
    valid = D.valid("fusion")
    idx, _ = pool(D, valid & (fold_of != 0) & (fold_of >= 0))
    va = np.flatnonzero(valid & (fold_of == 0))[:200_000]
    a = fit("RF", D, cols, idx).predict_proba(D.X[va][:, cols].astype(np.float64))[:, 1]
    sc = StandardScaler().fit(D.X[idx][:, cols].astype(np.float64))
    m2 = clone(make_model("RF")).fit(sc.transform(D.X[idx][:, cols].astype(np.float64)), D.label[idx])
    b = m2.predict_proba(sc.transform(D.X[va][:, cols].astype(np.float64)))[:, 1]
    return {"windows_compared": int(va.size), "max_abs_diff": float(np.abs(a - b).max()),
            "decision_agreement_at_0.5": float(((a >= 0.5) == (b >= 0.5)).mean())}


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description="Subject-wise OOF benchmark on the development cohort").parse_args(argv)
    run = C.RUNS / datetime.now().strftime("bench-%Y%m%d-%H%M%S")
    run.mkdir(parents=True)
    oof_dir = C.CACHE / "oof"
    oof_dir.mkdir(parents=True, exist_ok=True)
    folds = subject_folds()
    (run / "folds.json").write_text(json.dumps({"method": "GroupKFold(5) by subject (dev pool only)",
                                                "subject_fold": {f"sub-{s:03d}": f for s, f in folds.items()}}, indent=1))
    results, t_start = [], time.time()
    for w in C.WINDOWS:
        D = Dt.load(w, "dev")
        common = D.valid("fusion")
        print(f"[{datetime.now():%H:%M:%S}] {w}: {D.t.size:,} windows, {len(D.recordings)} recordings", flush=True)
        if w == "w4":
            (run / "tree_scaling_check.json").write_text(json.dumps(tree_scaling_check(D, folds), indent=1))
        for modality in C.MODALITIES:
            for name in MODELS:
                h = exp_hash(name, w, modality, folds)
                cache = oof_dir / f"{h}.npz"
                if cache.exists():
                    with np.load(cache, allow_pickle=False) as z:
                        p, meta = z["p"], json.loads(str(z["meta"]))
                else:
                    t0 = time.time()
                    res = oof(name, D, modality, folds)
                    p, meta = res["p"], {"folds": res["folds"], "seconds": round(time.time() - t0, 1)}
                    np.savez_compressed(cache, p=p, z=res["z"], meta=json.dumps(meta))
                ok = common & (D.label >= 0)
                auprc = float(average_precision_score(D.label[ok], p[ok]))
                rows = sweep(D, p, common)
                results.append({"window": w, "modality": modality, "model": name, "experiment_hash": h,
                                "window_auprc": auprc, "folds": meta["folds"], "seconds": meta.get("seconds"),
                                "sweep": rows})
                best = max((r for r in rows if r["far_per_24h"] is not None and r["far_per_24h"] <= C.MAX_FAR_PER_24H),
                           key=lambda r: r["detected"], default=None)
                print(f"[{datetime.now():%H:%M:%S}] {w} {modality:6s} {name:4s} AUPRC {auprc:.3f} "
                      f"best@FAR<=1: {best and (best['detected'], best['evaluable_events'], round(best['far_per_24h'], 3))}",
                      flush=True)
                (run / "results.json").write_text(json.dumps(results))
        del D
    print(f"done in {(time.time() - t_start) / 60:.1f} min; peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB", flush=True)


if __name__ == "__main__":
    from ml.lyra.benchmark import main as _main  # pickled objects must resolve to ml.lyra.benchmark

    _main()
