"""Nested, participant-grouped cross-validation over ALL SeizeIT2 participants, then the final fit.

    python -m ml.seizeit2.train            # writes ml/seizeit2/runs/<timestamp>/

Owner decision (Session 5): train on all 125 participants and every usable FBTCS; there is no
held-out test set. Performance is therefore estimated by nested CV:
1. Participants go to CV_FOLDS outer folds (FBTCS participants spread evenly). A participant's
   windows never cross folds, so overlapping windows cannot leak.
2. For each outer fold, the candidate is fitted on the other folds. Its D17 entry threshold
   (exit = entry - 0.20) is chosen ONLY from an inner INNER_FOLDS CV on those training
   participants, with the predefined rule in evaluate.select_operating_point.
3. The outer fold is then scored once with that threshold. Pooled outer folds = the estimate.
Scaling lives inside the pipelines (fitted on training folds only). Negatives are subsampled for
fitting only; evaluation scores every window.

Candidates follow locked decision D18: logistic regression (baseline), random forest 50 trees /
depth 8 (primary embedded model, chosen in advance, not by results), random forest 25 / 6 (smaller
embedded budget), gradient boosting (offline comparison, standing in for XGBoost), and two RF
ablations (simulated-ENV-only, IMU-only). The final model is the primary RF refitted on all
participants, with the threshold chosen by the same rule on the pooled out-of-fold scores.
"""

from __future__ import annotations

import json
import pickle
import time
import zlib
from datetime import datetime

import numpy as np
from lyra_mlm.features import feature_names
from lyra_mlm.prep import PrepConfig
from lyra_mlm.seizeit2 import SeizeIT2Adapter
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import evaluate as E
from .data_loader import RecordingWindows, load_windows

FEATURES = feature_names(PrepConfig().primary)  # lyra_mlm Strategy D primary schema (22 features)
ALL = list(range(len(FEATURES)))
EMG_ONLY = [i for i, n in enumerate(FEATURES) if n.startswith("emg_")]
IMU_ONLY = [i for i, n in enumerate(FEATURES) if n.startswith("imu_")]
PRIMARY = "RF_50x8"


class Model:
    def __init__(self, name: str, est, cols=ALL):
        self.name, self.est, self.cols = name, est, list(cols)

    def fit(self, X, y):
        self.est.fit(X[:, self.cols], y)
        return self

    def score(self, X):
        return self.est.predict_proba(X[:, self.cols])[:, 1]


def rf(n: int, depth: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n, max_depth=depth, min_samples_leaf=20, max_features="sqrt",
                                  class_weight="balanced_subsample", n_jobs=-1, random_state=C.SEED)


def candidates() -> list[Model]:
    return [
        Model("LR", make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000))),
        Model(PRIMARY, rf(50, 8)),
        Model("RF_25x6", rf(25, 6)),
        Model("HGB_300x15", HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=C.SEED)),
        Model("A_RF_simenv_only", rf(50, 8), EMG_ONLY),
        Model("A_RF_imu_only", rf(50, 8), IMU_ONLY),
    ]


# ---- Data ----------------------------------------------------------------------------------------


def load_all() -> list[RecordingWindows]:
    adapter = SeizeIT2Adapter(C.DATASET)
    out = []
    for rid in adapter.recording_ids():
        r = load_windows(rid, adapter)
        if r is not None:
            out.append(r)
    return out


def make_folds(recs: list[RecordingWindows], participants: list[str], n: int, seed: int) -> list[list[str]]:
    events = {p: 0 for p in participants}
    for r in recs:
        if r.participant_id in events:
            events[r.participant_id] += len(r.fbtcs)
    rng = np.random.default_rng(seed)
    folds: list[list[str]] = [[] for _ in range(n)]
    i = 0
    for stratum in ([p for p in participants if events[p]], [p for p in participants if not events[p]]):
        order = sorted(rng.permutation(stratum), key=lambda p: -events[p])  # spread multi-event participants
        for p in order:
            folds[i % n].append(str(p))
            i += 1
    return [sorted(f) for f in folds]


def pool(recs: list[RecordingWindows], parts: set[str]):
    """All usable positives + a deterministic subsample of usable negatives (fitting only)."""
    Xs, ys = [], []
    for r in recs:
        if r.participant_id not in parts:
            continue
        pos = np.flatnonzero(r.evidence & (r.label == 1))
        neg = np.flatnonzero(r.evidence & (r.label == 0))
        rng = np.random.default_rng(zlib.crc32(r.recording_id.encode()) ^ C.SEED)
        k = min(neg.size, max(C.NEG_MIN_PER_RECORDING, round(C.NEG_PER_RECORDING_FRACTION * neg.size)))
        sel = np.concatenate([pos, np.sort(rng.choice(neg, k, replace=False))])
        Xs.append(r.x[sel])
        ys.append(r.label[sel])
    return np.concatenate(Xs).astype(np.float64), np.concatenate(ys).astype(int)


def score_recs(model: Model, recs: list[RecordingWindows], parts: set[str]) -> list[E.Scored]:
    sel = [r for r in recs if r.participant_id in parts]
    X = np.concatenate([r.x[r.evidence] for r in sel]).astype(np.float64)
    s = model.score(X) if len(X) else np.empty(0)
    out, i = [], 0
    for r in sel:
        sc = np.full(r.t_end.size, np.nan)
        m = int(r.evidence.sum())
        sc[r.evidence] = s[i:i + m]
        i += m
        out.append(E.Scored(r, sc))
    return out


def merge(results: list[E.EventResult]) -> E.EventResult:
    m = E.EventResult()
    for r in results:
        for f in ("n_events", "detected", "unevaluable", "false_alarms", "oos_alarms", "artefact_alarms",
                  "monitored_h", "suppressed_h"):
            setattr(m, f, getattr(m, f) + getattr(r, f))
        m.latencies += r.latencies
        m.per_participant.update(r.per_participant)
    return m


# ---- Main ----------------------------------------------------------------------------------------


def log(run_dir, msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with (run_dir / "log.txt").open("a") as fh:
        fh.write(line + "\n")


def nested_cv(m_factory, recs, folds, run_dir) -> dict:
    """Outer folds scored with thresholds chosen in inner CV only. Returns pooled results + OOF scores."""
    name = m_factory().name
    outer_results, outer_scored, per_fold, history = [], [], [], []
    for f, test in enumerate(folds):
        train_parts = sorted(set().union(*map(set, folds)) - set(test))
        inner = make_folds(recs, train_parts, C.INNER_FOLDS, C.SEED + f + 1)
        inner_scored = []
        for g in inner:
            fit_parts = set(train_parts) - set(g)
            X, y = pool(recs, fit_parts)
            inner_scored += score_recs(m_factory().fit(X, y), recs, set(g))
        op = E.select_operating_point(inner_scored, E.candidate_entries(inner_scored))["selected"]
        del inner_scored
        X, y = pool(recs, set(train_parts))
        model = m_factory().fit(X, y)
        scored = score_recs(model, recs, set(test))
        res = E.event_eval(scored, op["entry_threshold"], op["exit_threshold"], per_participant=True)
        Xt, yt = pool(recs, set(test))
        history.append({"fold": f, "train_pr_auc": float(average_precision_score(y, model.score(X))),
                        "heldout_pr_auc": float(average_precision_score(yt, model.score(Xt)))})
        per_fold.append({"fold": f, "inner_selected": op, "outer": res.summary()})
        outer_results.append(res)
        outer_scored += scored
        log(run_dir, f"{name} fold {f}: entry {op['entry_threshold']:.4f} -> outer sens "
                     f"{res.detected}/{res.n_events - res.unevaluable} FA {res.false_alarms} "
                     f"({res.monitored_h:.0f} h)")
    pooled = merge(outer_results)
    d17 = E.event_eval(outer_scored, C.D17_DEFAULTS["entry_threshold"], C.D17_DEFAULTS["exit_threshold"])
    thr_w = float(np.median([p["inner_selected"]["entry_threshold"] for p in per_fold]))
    return {"candidate": name, "nested_outer": pooled.summary(), "per_participant": pooled.per_participant,
            "per_fold": per_fold, "d17_default_thresholds": d17.summary(),
            "window_at_median_fold_entry": E.window_metrics(outer_scored, thr_w),
            "training_history": history, "_oof": outer_scored}


def main() -> None:
    run_dir = C.RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    t0 = time.time()
    recs = load_all()
    participants = sorted({r.participant_id for r in recs})
    folds = make_folds(recs, participants, C.CV_FOLDS, C.SEED)
    ev = {p: sum(len(r.fbtcs) for r in recs if r.participant_id == p) for p in participants}
    (run_dir / "folds.json").write_text(json.dumps({
        "seed": C.SEED, "participants": len(participants), "folds": folds,
        "fbtcs_events_per_fold": [sum(ev[p] for p in f) for f in folds]}, indent=1))
    log(run_dir, f"{len(recs)} recordings, {len(participants)} participants, "
                 f"{sum(r.t_end.size for r in recs):,} windows, folds events "
                 f"{[sum(ev[p] for p in f) for f in folds]} ({time.time() - t0:.0f} s)")

    results, oof_primary = {}, None
    for factory in candidates():
        name = factory.name
        res = nested_cv(lambda n=name: next(m for m in candidates() if m.name == n), recs, folds, run_dir)
        oof = res.pop("_oof")
        if name == PRIMARY:
            oof_primary = oof
        results[name] = res
        o = res["nested_outer"]
        log(run_dir, f"NESTED {name}: sens {o['detected']}/{o['evaluable_events']} = {o['event_sensitivity']:.3f} "
                     f"FAR/24h {o['far_per_24h']:.3f} latency {o['latency_s_median']}")
        (run_dir / "cv_results.json").write_text(json.dumps(results, indent=1))

    # final model: primary RF on all participants; threshold from pooled out-of-fold scores
    op = E.select_operating_point(oof_primary, E.candidate_entries(oof_primary))
    (run_dir / "operating_curve_primary.json").write_text(json.dumps(op["curve"]))
    final_cm = E.window_metrics(oof_primary, op["selected"]["entry_threshold"])
    X, y = pool(recs, set(participants))
    final = next(m for m in candidates() if m.name == PRIMARY).fit(X, y)
    with (run_dir / "final_model.pkl").open("wb") as fh:
        pickle.dump(final, fh)
    selection = {"candidate": PRIMARY, **op["selected"], "met_far_cap": op["met_far_cap"],
                 "chosen_on": "pooled out-of-fold scores of the 5 outer folds (all participants)",
                 "fit_windows": int(X.shape[0]), "fit_positive_windows": int(y.sum()),
                 "window_metrics_oof": final_cm}
    (run_dir / "selection.json").write_text(json.dumps(selection, indent=1))
    log(run_dir, f"final {PRIMARY}: fitted on {X.shape[0]:,} windows ({int(y.sum())} positive); entry "
                 f"{op['selected']['entry_threshold']:.4f} exit {op['selected']['exit_threshold']:.4f} "
                 f"(pooled OOF sens {op['selected']['sensitivity']:.3f}, FAR {op['selected']['far_per_24h']:.3f})")
    plots(run_dir, results, op["curve"], final_cm)
    log(run_dir, f"done in {(time.time() - t0) / 60:.1f} min")


def plots(run_dir, results: dict, curve: list, cm: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    names = list(results)
    for i, n in enumerate(names):
        h = results[n]["training_history"]
        ax.scatter([i - 0.1] * len(h), [x["train_pr_auc"] for x in h], color="#6E56CF", s=14,
                   label="training folds" if i == 0 else None)
        ax.scatter([i + 0.1] * len(h), [x["heldout_pr_auc"] for x in h], color="#D97706", s=14,
                   label="held-out fold" if i == 0 else None)
    ax.set_xticks(range(len(names)), names, rotation=20, fontsize=8)
    ax.set_ylabel("window PR-AUC (subsampled negatives)")
    ax.set_title("Per-fold fit vs held-out participants (5 outer folds)", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "training_history.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    pts = sorted((p for p in curve if p["far_per_24h"] > 0), key=lambda p: p["far_per_24h"])
    ax.plot([p["far_per_24h"] for p in pts], [p["sensitivity"] for p in pts], color="#6E56CF")
    ax.axvline(C.MAX_FAR_PER_24H, color="grey", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("false alarms per 24 h (pooled out-of-fold, D17 confirmation)")
    ax.set_ylabel("event sensitivity")
    ax.set_title(f"{PRIMARY}: operating curve over entry thresholds", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "cv_operating_curve.png", dpi=120)
    plt.close(fig)
    E.confusion_png(cm["confusion_matrix"], run_dir / "confusion_matrix.png",
                    f"Out-of-fold windows, {PRIMARY}, entry {cm['threshold']:.3f}")


if __name__ == "__main__":
    from ml.seizeit2.train import main as _main  # pickled classes must live in ml.seizeit2.train, not __main__

    _main()
