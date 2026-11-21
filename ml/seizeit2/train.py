"""Subject-level split, grouped cross-validation of candidate models, selection, final fit.

    python -m ml.seizeit2.train            # writes ml/seizeit2/runs/<timestamp>/

Protocol (test subjects are never loaded here):
1. Subjects are split once, before any windowing decision is evaluated: TEST_FRACTION of the
   subjects with and without a target seizure go to the test set. Every window of a subject
   stays on one side, so overlapping windows cannot leak.
2. Dev subjects are split into CV_FOLDS subject folds (target-seizure subjects spread evenly).
3. Each candidate (model x window length) is fitted on 4 folds and scores every window of the
   5th. Scaling/normalisation lives inside the model, so it is fitted on training folds only.
4. The operating point (smoothing k, threshold) and the candidate are chosen on the pooled
   out-of-fold events with the rule in evaluate.select_operating_point.
5. The chosen candidate is refitted on all dev subjects; evaluate.py then scores the test set once.
"""

from __future__ import annotations

import json
import pickle
import time
import zlib
from datetime import datetime

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import data_loader as D
from . import evaluate as E
from .preprocessing import FEATURE_NAMES, feature_path

# ---- Candidates --------------------------------------------------------------------------------


def _idx(prefixes: tuple[str, ...]) -> list[int]:
    return [i for i, n in enumerate(FEATURE_NAMES) if n.startswith(prefixes)]


ALL = list(range(len(FEATURE_NAMES)))
EMG_ONLY = _idx(("emg_", "env_"))
IMU_ONLY = _idx(("acc_", "gyr_", "orient_"))


class RuleBaseline:
    """M0: no learned weights. AND-rule on sEMG high-frequency power and accelerometer variability,
    each z-scored with training-fold background statistics; score = the smaller z."""

    name = "M0_rule_emg_and_acc"
    cols = [FEATURE_NAMES.index("emg_hf_logpow"), FEATURE_NAMES.index("acc_mag_std")]

    def fit(self, X, y):
        neg = X[y == 0][:, self.cols]
        self.mu, self.sd = neg.mean(0), neg.std(0) + 1e-9
        return self

    def score(self, X):
        return ((X[:, self.cols] - self.mu) / self.sd).min(1)


class SkModel:
    def __init__(self, name: str, est, cols=ALL):
        self.name, self.est, self.cols = name, est, list(cols)

    def fit(self, X, y):
        self.est.fit(X[:, self.cols], y)
        return self

    def score(self, X):
        return self.est.predict_proba(X[:, self.cols])[:, 1]


def hgb(max_iter: int, leaves: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=max_iter, max_leaf_nodes=leaves, min_samples_leaf=200,
        l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=C.SEED)


def candidates() -> list:
    return [
        RuleBaseline(),
        SkModel("M1_logreg", make_pipeline(StandardScaler(), LogisticRegression(
            C=0.1, class_weight="balanced", max_iter=2000))),
        SkModel("M2_hgb_100x15", hgb(100, 15)),
        SkModel("M2_hgb_300x15", hgb(300, 15)),
    ]


def ablations() -> list:
    """Reported for insight only; never eligible for selection."""
    return [SkModel("A_hgb_emg_only", hgb(300, 15), EMG_ONLY),
            SkModel("A_hgb_imu_only", hgb(300, 15), IMU_ONLY)]


# ---- Split -------------------------------------------------------------------------------------


def make_split(runs: list[D.Run]) -> dict:
    subjects = sorted({r.subject for r in runs})
    events = {s: 0 for s in subjects}
    for r in runs:
        events[r.subject] += len(r.intervals(D.TARGET_KINDS))
    rng = np.random.default_rng(C.SEED)
    test, folds = [], [[] for _ in range(C.CV_FOLDS)]
    for stratum in ([s for s in subjects if events[s]], [s for s in subjects if not events[s]]):
        order = list(rng.permutation(stratum))
        n_test = round(C.TEST_FRACTION * len(order))
        test += order[:n_test]
        dev = sorted(order[n_test:], key=lambda s: -events[s])  # spread multi-event subjects
        for i, s in enumerate(dev):
            folds[i % C.CV_FOLDS].append(s)
    return {"seed": C.SEED, "test": sorted(test), "folds": [sorted(f) for f in folds],
            "target_events": {"test": sum(events[s] for s in test),
                              "folds": [sum(events[s] for s in f) for f in folds]}}


# ---- Data assembly -----------------------------------------------------------------------------


def training_pool(runs: list[D.Run], subjects: set[str], W: int):
    """All usable positives + a deterministic NEG_PER_RUN_FRACTION sample of usable negatives."""
    Xs, ys, gs = [], [], []
    for r in runs:
        p = feature_path(W, r.run)
        if r.subject not in subjects or not p.exists():
            continue
        z = np.load(p)
        ok, y = z["ok"], z["y"]
        pos = np.flatnonzero(ok & (y == 1))
        neg = np.flatnonzero(ok & (y == 0))
        rng = np.random.default_rng(zlib.crc32(r.run.encode()) ^ C.SEED)
        n = min(neg.size, max(C.NEG_MIN_PER_RUN, round(C.NEG_PER_RUN_FRACTION * neg.size)))
        sel = np.concatenate([pos, np.sort(rng.choice(neg, n, replace=False))])
        Xs.append(z["X"][sel])
        ys.append(y[sel])
        gs += [r.subject] * sel.size
    return np.concatenate(Xs).astype(np.float64), np.concatenate(ys).astype(int), np.asarray(gs)


def thresholds_for(scores: np.ndarray) -> np.ndarray:
    """Candidate thresholds in the extreme upper tail, where FAR/24h is in a usable range."""
    return np.unique(np.quantile(scores, 1 - np.logspace(-1.5, -5.5, 60)))


# ---- Main --------------------------------------------------------------------------------------


def log(run_dir, msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with (run_dir / "log.txt").open("a") as fh:
        fh.write(line + "\n")


def hgb_history(model: SkModel, X, y, Xv, yv) -> dict:
    """Balanced log-loss per boosting iteration on the training pool and the held-out fold."""
    from sklearn.metrics import log_loss

    def curve(Xa, ya):
        w = np.where(ya == 1, 0.5 / max(1, ya.sum()), 0.5 / max(1, (ya == 0).sum()))
        return [float(log_loss(ya, p[:, 1], sample_weight=w, labels=[0, 1]))
                for i, p in enumerate(model.est.staged_predict_proba(Xa[:, model.cols])) if i % 10 == 9]

    return {"iteration": list(range(10, model.est.n_iter_ + 1, 10)), "train": curve(X, y), "val": curve(Xv, yv)}


def main() -> None:
    run_dir = C.RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    (run_dir / "folds").mkdir()
    runs = D.list_runs()
    split = make_split(runs)
    (run_dir / "split.json").write_text(json.dumps(split, indent=1))
    log(run_dir, f"split: {len(split['test'])} test subjects ({split['target_events']['test']} target events), "
                 f"folds events {split['target_events']['folds']}")
    dev = set().union(*map(set, split["folds"]))
    results, histories = {}, {}

    for W in C.WINDOWS_S:
        X, y, g = training_pool(runs, dev, W)
        log(run_dir, f"W={W}: dev pool {X.shape[0]} windows, {int(y.sum())} positive")
        oof = {m.name: [] for m in candidates() + ablations()}
        for f, fold in enumerate(split["folds"]):
            tr = ~np.isin(g, fold)
            held = set(fold)
            fitted = []
            for m in candidates() + ablations():
                t0 = time.time()
                m.fit(X[tr], y[tr])
                fitted.append(m)
                with (run_dir / "folds" / f"W{W}_fold{f}_{m.name}.pkl").open("wb") as fh:
                    pickle.dump(m, fh)
                log(run_dir, f"W={W} fold {f} {m.name}: fit {time.time() - t0:.0f}s")
                if m.name == "M2_hgb_300x15":
                    histories.setdefault(str(W), []).append(hgb_history(m, X[tr], y[tr], X[~tr], y[~tr]))
            # every window of the held-out subjects, scored by every fitted candidate
            for r in runs:
                p = feature_path(W, r.run)
                if r.subject not in held or not p.exists():
                    continue
                z = np.load(p)
                ok = z["ok"]
                Xr = z["X"][ok].astype(np.float64)
                for m in fitted:
                    sc = np.zeros(ok.size, np.float32)
                    if ok.any():
                        sc[ok] = m.score(Xr)
                    oof[m.name].append(E.RunScores(r, z["t_end"], sc, z["y"], ok))
        for name, rs_list in oof.items():
            allsc = np.concatenate([r.score[r.ok] for r in rs_list])
            op = E.select_operating_point(rs_list, thresholds_for(allsc))
            sel = op["selected"]
            res = {"candidate": name, "window_s": W, **sel, "met_far_cap": op["met_far_cap"],
                   "selectable": not name.startswith("A_"),
                   "event": E.event_eval(rs_list, sel["k"], sel["threshold"], per_subject=True).summary(),
                   "window": E.window_metrics(rs_list, sel["k"], sel["threshold"])}
            results[f"{name}@W{W}"] = res
            (run_dir / f"curve_{name}_W{W}.json").write_text(json.dumps(op["curve"]))
            log(run_dir, f"OOF {name} W={W}: sens {sel['sensitivity']:.3f} FAR/24h {sel['far_per_24h']:.3f} "
                         f"k={sel['k']} thr={sel['threshold']:.4g} lat {sel['latency_s_median']}")
        (run_dir / "cv_results.json").write_text(json.dumps(results, indent=1))
        del X, y, g, oof

    # ---- selection: same rule as the operating point, across selectable candidates ----
    def key(r):
        ok = r["met_far_cap"]
        return (ok, r["sensitivity"] if ok else -r["far_per_24h"], -r["far_per_24h"], -r["latency_s_median"])

    best = max((r for r in results.values() if r["selectable"]), key=key)
    log(run_dir, f"selected {best['candidate']} W={best['window_s']} k={best['k']} thr={best['threshold']:.4g}")
    (run_dir / "selection.json").write_text(json.dumps(best, indent=1))
    (run_dir / "training_history.json").write_text(json.dumps(histories))

    W = best["window_s"]
    X, y, _ = training_pool(runs, dev, W)
    final = next(m for m in candidates() if m.name == best["candidate"])
    final.fit(X, y)
    with (run_dir / "final_model.pkl").open("wb") as fh:
        pickle.dump(final, fh)
    log(run_dir, f"final model fitted on {X.shape[0]} dev windows ({int(y.sum())} positive)")
    plot_history(histories, run_dir / "training_history.png")
    plot_curves(run_dir, results)


def plot_curves(run_dir, results: dict) -> None:
    """Out-of-fold event sensitivity vs FAR/24h for every candidate, at its selected smoothing k."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(C.WINDOWS_S), figsize=(5.5 * len(C.WINDOWS_S), 4), squeeze=False)
    for ax, W in zip(axes[0], C.WINDOWS_S):
        for key, r in sorted(results.items()):
            if r["window_s"] != W:
                continue
            pts = [p for p in json.loads((run_dir / f"curve_{r['candidate']}_W{W}.json").read_text())
                   if p["k"] == r["k"] and p["far_per_24h"] > 0]
            pts.sort(key=lambda p: p["far_per_24h"])
            ax.plot([p["far_per_24h"] for p in pts], [p["sensitivity"] for p in pts],
                    ls="--" if not r["selectable"] else "-", label=f"{r['candidate']} (k={r['k']})")
        ax.axvline(C.MAX_FAR_PER_24H, color="grey", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("false alarms per 24 h (dev, out-of-fold)")
        ax.set_ylabel("event sensitivity")
        ax.set_title(f"W={W}s", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(run_dir / "cv_operating_curves.png", dpi=120)
    plt.close(fig)


def plot_history(histories: dict, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 3.6), squeeze=False)
    for ax, (W, folds) in zip(axes[0], sorted(histories.items())):
        for i, h in enumerate(folds):
            ax.plot(h["iteration"], h["train"], color="#6E56CF", alpha=0.5, label="train" if i == 0 else None)
            ax.plot(h["iteration"], h["val"], color="#D97706", alpha=0.7, label="held-out fold" if i == 0 else None)
        ax.set_title(f"M2_hgb_300x15, W={W}s, 5 subject folds", fontsize=9)
        ax.set_xlabel("boosting iteration")
        ax.set_ylabel("class-balanced log-loss")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    from ml.seizeit2.train import main as _main  # pickled classes must live in ml.seizeit2.train, not __main__

    _main()
