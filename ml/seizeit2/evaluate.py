"""Event-level and window-level evaluation, shared by CV (train.py) and the one-time test run.

Post-processing (mirrored by createDetector in src/services/ml/seizeit2/model.ts):
  window scores every HOP_S -> unusable windows score 0 -> moving average of the last k scores
  -> alarm when the average >= threshold and no alarm in the previous REFRACTORY_S seconds.

Alarm classification per run:
  detection    fires in [onset - MATCH_PRE_S, offset + MATCH_POST_S] of a target seizure
  out-of-scope fires near a non-target seizure (OOS_GUARD_S) -> reported, not a false alarm
  artefact     fires inside an impedance-check annotation -> reported, not a false alarm
  false alarm  everything else; FAR/24h = false alarms / evaluated hours * 24

Run as a module to evaluate the frozen final model on the held-out test subjects (once):
    python -m ml.seizeit2.evaluate --run ml/seizeit2/runs/<id>
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lyra_mlm.annotations import AnnotationKind
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from . import data_loader as D


@dataclass
class RunScores:
    run: D.Run
    t_end: np.ndarray  # int32 seconds
    score: np.ndarray  # float, unusable windows already 0
    y: np.ndarray  # window labels (1/0/-1)
    ok: np.ndarray  # window usable


def smooth(score: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return score
    c = np.cumsum(np.concatenate([[0.0], score]))
    i = np.arange(1, score.size + 1)
    return (c[i] - c[np.maximum(i - k, 0)]) / np.minimum(i, k)


def alarms(t_end: np.ndarray, smoothed: np.ndarray, thr: float) -> np.ndarray:
    """Alarm times with the refractory rule (a new alarm needs >= REFRACTORY_S since the last)."""
    cand = t_end[smoothed >= thr]
    out, i = [], 0
    while i < cand.size:
        out.append(cand[i])
        i = np.searchsorted(cand, cand[i] + C.REFRACTORY_S, side="left")
    return np.asarray(out, dtype=np.int64)


def _inside(times: np.ndarray, iv: list[tuple[float, float]]) -> np.ndarray:
    m = np.zeros(times.shape, bool)
    for a, b in iv:
        m |= (times >= a) & (times <= b)
    return m


@dataclass
class EventResult:
    n_events: int = 0
    detected: int = 0
    false_alarms: int = 0
    oos_alarms: int = 0
    artefact_alarms: int = 0
    hours: float = 0.0
    latencies: list[float] = field(default_factory=list)
    per_subject: dict = field(default_factory=dict)

    @property
    def sensitivity(self) -> float:
        return self.detected / self.n_events if self.n_events else float("nan")

    @property
    def far24(self) -> float:
        return self.false_alarms / self.hours * 24 if self.hours else float("nan")

    def summary(self) -> dict:
        from scipy.stats import beta, chi2

        lat = np.asarray(self.latencies)
        n, x, fa = self.n_events, self.detected, self.false_alarms
        sens_ci = [float(beta.ppf(0.025, x, n - x + 1)) if x else 0.0,
                   float(beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0] if n else None  # Clopper-Pearson
        far_ci = [float(chi2.ppf(0.025, 2 * fa) / 2) if fa else 0.0, float(chi2.ppf(0.975, 2 * fa + 2) / 2)]
        return {
            "events": n, "detected": x,
            "event_sensitivity": self.sensitivity, "event_sensitivity_95ci": sens_ci,
            "false_alarms": fa, "evaluated_hours": round(self.hours, 2),
            "far_per_24h": self.far24,
            "far_per_24h_95ci": [c / self.hours * 24 for c in far_ci] if self.hours else None,
            "alarms_near_out_of_scope_seizures": self.oos_alarms,
            "alarms_during_impedance_checks": self.artefact_alarms,
            "latency_s_median": float(np.median(lat)) if lat.size else None,
            "latency_s_iqr": [float(np.percentile(lat, 25)), float(np.percentile(lat, 75))] if lat.size else None,
            "latency_s_all": [float(x) for x in lat],
        }


def event_eval(runs: list[RunScores], k: int, thr: float, per_subject: bool = False) -> EventResult:
    res = EventResult()
    opre, opost = C.OOS_GUARD_S
    for rs in runs:
        t = alarms(rs.t_end, smooth(rs.score, k), thr)
        tgt = rs.run.intervals(D.TARGET_KINDS)
        oos = [(a - opre, b + opost) for a, b in rs.run.intervals((AnnotationKind.SEIZURE_OUT_OF_SCOPE,))]
        art = rs.run.intervals((AnnotationKind.ARTIFACT,))
        match = [(a - C.MATCH_PRE_S, b + C.MATCH_POST_S) for a, b in tgt]
        hours = rs.t_end.size * C.HOP_S / 3600
        det = 0
        for (on, _), (lo, hi) in zip(tgt, match):
            hit = t[(t >= lo) & (t <= hi)]
            if hit.size:
                det += 1
                res.latencies.append(float(hit[0] - on))
        is_tp = _inside(t, match)
        is_oos = ~is_tp & _inside(t, oos)
        is_art = ~is_tp & ~is_oos & _inside(t, art)
        fa = int((~is_tp & ~is_oos & ~is_art).sum())
        res.n_events += len(tgt)
        res.detected += det
        res.false_alarms += fa
        res.oos_alarms += int(is_oos.sum())
        res.artefact_alarms += int(is_art.sum())
        res.hours += hours
        if per_subject:
            s = res.per_subject.setdefault(rs.run.subject, {"events": 0, "detected": 0, "false_alarms": 0, "hours": 0.0})
            s["events"] += len(tgt)
            s["detected"] += det
            s["false_alarms"] += fa
            s["hours"] += hours
    for s in res.per_subject.values():
        s["hours"] = round(s["hours"], 2)
        s["far_per_24h"] = s["false_alarms"] / s["hours"] * 24 if s["hours"] else None
    return res


def select_operating_point(runs: list[RunScores], thresholds: np.ndarray) -> dict:
    """Predefined rule: max event sensitivity with FAR/24h <= MAX_FAR_PER_24H; ties -> lower FAR,
    then lower median latency. Falls back to the lowest-FAR point if nothing meets the cap."""
    best, curve = None, []
    for k in C.SMOOTH_K:
        for thr in thresholds:
            r = event_eval(runs, k, float(thr))
            lat = float(np.median(r.latencies)) if r.latencies else float("inf")
            pt = {"k": k, "threshold": float(thr), "sensitivity": r.sensitivity, "far_per_24h": r.far24,
                  "latency_s_median": lat}
            curve.append(pt)
            key = (r.far24 <= C.MAX_FAR_PER_24H, r.sensitivity if r.far24 <= C.MAX_FAR_PER_24H else -r.far24,
                   -r.far24, -lat)
            if best is None or key > best[0]:
                best = (key, pt)
    return {"selected": best[1], "met_far_cap": bool(best[0][0]), "curve": curve}


def window_metrics(runs: list[RunScores], k: int, thr: float) -> dict:
    """Window-level metrics on usable windows with a definite label (y in {0,1})."""
    s = np.concatenate([smooth(r.score, k) for r in runs])
    y = np.concatenate([r.y for r in runs])
    m = np.concatenate([r.ok for r in runs]) & (y >= 0)
    s, y = s[m], y[m]
    p = s >= thr
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "windows": int(m.sum()), "positive_windows": int(y.sum()),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy_not_meaningful_under_imbalance": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": prec, "recall_sensitivity": rec,
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else None,
        "pr_auc": float(average_precision_score(y, s)) if 0 < y.sum() < y.size else None,
    }


def load_scored_runs(subjects: set[str], W: int, predict) -> list[RunScores]:
    """Score every cached window of the given subjects with `predict(X) -> scores`."""
    from .preprocessing import feature_path

    out = []
    for run in D.list_runs():
        if run.subject not in subjects:
            continue
        p = feature_path(W, run.run)
        if not p.exists():
            continue  # run without both modalities: counted by `unevaluable_events`
        z = np.load(p)
        score = np.zeros(z["t_end"].size)
        if z["ok"].any():
            score[z["ok"]] = predict(z["X"][z["ok"]])
        out.append(RunScores(run, z["t_end"], score, z["y"], z["ok"]))
    return out


def unevaluable_events(subjects: set[str], W: int) -> int:
    """Target seizures in runs with no feature file (a modality missing, or accelerometer scale invalid)."""
    from .preprocessing import feature_path

    return sum(len(r.intervals(D.TARGET_KINDS)) for r in D.list_runs()
               if r.subject in subjects and not feature_path(W, r.run).exists())


def confusion_png(cm: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    shade = np.log10(m + 1)
    ax.imshow(shade, cmap="Purples")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if shade[i, j] > shade.max() / 2 else "black")
    ax.set_xticks([0, 1], ["background", "convulsive"])
    ax.set_yticks([0, 1], ["background", "convulsive"])
    ax.set_xlabel("predicted (smoothed score >= threshold)")
    ax.set_ylabel("annotated")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    import pickle


    ap = argparse.ArgumentParser(description="Evaluate the frozen final model on the test subjects")
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args(argv)
    sel = json.loads((args.run / "selection.json").read_text())
    split = json.loads((args.run / "split.json").read_text())
    if (args.run / "test_metrics.json").exists():
        raise SystemExit("test set already evaluated for this run; re-running invites test-set tuning")
    with (args.run / "final_model.pkl").open("rb") as fh:
        model = pickle.load(fh)
    W, k, thr = sel["window_s"], sel["k"], sel["threshold"]
    test = set(split["test"])
    runs = load_scored_runs(test, W, model.score)
    ev = event_eval(runs, k, thr, per_subject=True)
    unev = unevaluable_events(test, W)
    out = {
        "dataset": "SeizeIT2 (OpenNeuro ds005873 v1.1.0)", "split": "held-out test subjects",
        "test_subjects": len(test), "model": sel["candidate"], "window_s": W, "k": k, "threshold": thr,
        "event": ev.summary(),
        "events_in_runs_missing_or_invalid_modality": unev,
        "event_sensitivity_counting_unevaluable_as_missed": ev.detected / (ev.n_events + unev) if ev.n_events + unev else None,
        "window": window_metrics(runs, k, thr),
        "per_subject": ev.per_subject,
    }
    (args.run / "test_metrics.json").write_text(json.dumps(out, indent=1))
    confusion_png(out["window"]["confusion_matrix"], args.run / "confusion_matrix.png",
                  f"Test windows ({sel['candidate']}, W={W}s, k={k})")
    print(json.dumps({k_: out[k_] for k_ in ("event", "events_in_runs_missing_or_invalid_modality")}, indent=1))


if __name__ == "__main__":
    main()
