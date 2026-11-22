"""Event-level and window-level evaluation with lyra_mlm's D17 event confirmation.

Alarms come from `confirm_alarms`, an exact, fast replica of lyra_mlm.confirmation.EventConfirmation
for a monotonic stream (test_evaluate.py checks equivalence against the real class on random
streams). Suppressed windows (lyra_mlm inference routing: a modality unusable or a feature
unavailable) carry no evidence and follow D17's "hold" policy.

Alarm classification per recording:
  detection    fires in [onset - MATCH_PRE_S, offset + MATCH_POST_S] of an FBTCS
  out-of-scope fires near another seizure (OOS_GUARD_S) -> reported, not a false alarm
  artefact     fires inside an impedance check -> reported, not a false alarm
  false alarm  everything else; FAR/24h = false alarms / valid monitored hours * 24
Monitored hours count windows with evidence; suppressed hours are reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import beta, chi2
from sklearn.metrics import average_precision_score, roc_auc_score

from . import config as C
from .data_loader import RecordingWindows


def confirm_alarms(t: np.ndarray, evidence: np.ndarray, score: np.ndarray, entry: float, exit_: float,
                   k: int = C.CONSECUTIVE, max_gap: float = C.MAX_GAP_S,
                   refractory: float = C.REFRACTORY_S) -> list[float]:
    """Times at which EventConfirmation enters POSSIBLE_EVENT (D17, hold policy, no stale check).

    From a quiescent state (IDLE with no count, or REFRACTORY) nothing can change until an evidence
    window scores >= entry at or after the refractory end, so the loop jumps to that candidate and
    then steps window by window, exactly as EventConfirmation.update, until quiescent again.
    """
    cand = np.flatnonzero(evidence & (score >= entry))
    ev_idx = np.flatnonzero(evidence)
    alarms: list[float] = []
    until, pos, n = -np.inf, 0, t.size
    while True:
        ci = int(np.searchsorted(cand, pos))
        if ci < cand.size and t[cand[ci]] < until:
            ci += int(np.searchsorted(t[cand[ci:]], until, "left"))
        if ci >= cand.size:
            return alarms
        j = int(cand[ci])
        p = int(np.searchsorted(ev_idx, j)) - 1
        last_ev = t[ev_idx[p]] if p >= 0 else None
        state, count = "idle", 0
        while j < n:
            ts = t[j]
            if last_ev is not None and ts - last_ev > max_gap:  # evidence not continuous: the run ends
                count = 0
                if state == "possible":
                    until, state = ts + refractory, "refractory"
                elif state == "accumulating":
                    state = "idle"
            j += 1
            if not evidence[j - 1]:  # suppressed: held, never negative evidence
                if state == "refractory" or (state == "idle" and count == 0):
                    break
                continue
            last_ev = ts
            if state == "refractory":
                break
            s = score[j - 1]
            if state == "possible":
                if s < exit_:
                    until, state = ts + refractory, "refractory"
                    break
            elif s >= entry:
                count += 1
                state = "accumulating"
                if count >= k:
                    state = "possible"
                    alarms.append(float(ts))
            else:
                break  # valid negative window: count resets, IDLE
        pos = j


def _inside(times: np.ndarray, iv) -> np.ndarray:
    m = np.zeros(times.shape, bool)
    for a, b in iv:
        m |= (times >= a) & (times <= b)
    return m


@dataclass
class Scored:
    rec: RecordingWindows
    score: np.ndarray  # NaN where the window carries no evidence


@dataclass
class EventResult:
    n_events: int = 0
    detected: int = 0
    unevaluable: int = 0  # FBTCS with no evidence window overlapping it
    false_alarms: int = 0
    oos_alarms: int = 0
    artefact_alarms: int = 0
    monitored_h: float = 0.0
    suppressed_h: float = 0.0
    latencies: list[float] = field(default_factory=list)
    per_participant: dict = field(default_factory=dict)

    @property
    def sensitivity(self) -> float:
        n = self.n_events - self.unevaluable
        return self.detected / n if n else float("nan")

    @property
    def far24(self) -> float:
        return self.false_alarms / self.monitored_h * 24 if self.monitored_h else float("nan")

    def summary(self) -> dict:
        lat = np.asarray(self.latencies)
        n, x, fa = self.n_events - self.unevaluable, self.detected, self.false_alarms
        sens_ci = [float(beta.ppf(0.025, x, n - x + 1)) if x else 0.0,
                   float(beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0] if n else None  # Clopper-Pearson
        far_ci = [float(chi2.ppf(0.025, 2 * fa) / 2) if fa else 0.0, float(chi2.ppf(0.975, 2 * fa + 2) / 2)]
        return {
            "fbtcs_events": self.n_events, "evaluable_events": n, "unevaluable_events": self.unevaluable,
            "detected": x, "event_sensitivity": self.sensitivity, "event_sensitivity_95ci": sens_ci,
            "event_sensitivity_counting_unevaluable_as_missed": x / self.n_events if self.n_events else None,
            "false_alarms": fa, "valid_monitored_hours": round(self.monitored_h, 2),
            "suppressed_hours": round(self.suppressed_h, 2),
            "far_per_24h": self.far24,
            "far_per_24h_95ci": [c / self.monitored_h * 24 for c in far_ci] if self.monitored_h else None,
            "alarms_near_out_of_scope_seizures": self.oos_alarms,
            "alarms_during_impedance_checks": self.artefact_alarms,
            "latency_s_median": float(np.median(lat)) if lat.size else None,
            "latency_s_iqr": [float(np.percentile(lat, 25)), float(np.percentile(lat, 75))] if lat.size else None,
        }


def event_eval(scored: list[Scored], entry: float, exit_: float, per_participant: bool = False) -> EventResult:
    res = EventResult()
    opre, opost = C.OOS_GUARD_S
    for s in scored:
        r = s.rec
        ev = r.evidence
        al = np.asarray(confirm_alarms(r.t_end, ev, np.nan_to_num(s.score, nan=-1.0), entry, exit_))
        match = [(a - C.MATCH_PRE_S, b + C.MATCH_POST_S) for a, b in r.fbtcs]
        det = unev = 0
        for (on, off), (lo, hi) in zip(r.fbtcs, match):
            covered = ev[(r.t_end > on) & (r.t_end - C.WINDOW_S < off)]  # windows overlapping the event
            hit = al[(al >= lo) & (al <= hi)] if al.size else al
            if hit.size:
                det += 1
                res.latencies.append(float(hit[0] - on))
            elif not covered.any():
                unev += 1
        is_tp = _inside(al, match)
        is_oos = ~is_tp & _inside(al, [(a - opre, b + opost) for a, b in r.out_of_scope])
        is_art = ~is_tp & ~is_oos & _inside(al, r.artifacts)
        fa = int((~is_tp & ~is_oos & ~is_art).sum())
        mon, sup = ev.sum() * C.HOP_S / 3600, (~ev).sum() * C.HOP_S / 3600
        res.n_events += len(r.fbtcs)
        res.detected += det
        res.unevaluable += unev
        res.false_alarms += fa
        res.oos_alarms += int(is_oos.sum())
        res.artefact_alarms += int(is_art.sum())
        res.monitored_h += mon
        res.suppressed_h += sup
        if per_participant:
            p = res.per_participant.setdefault(r.participant_id, {"events": 0, "detected": 0, "unevaluable": 0,
                                                                  "false_alarms": 0, "monitored_h": 0.0})
            p["events"] += len(r.fbtcs)
            p["detected"] += det
            p["unevaluable"] += unev
            p["false_alarms"] += fa
            p["monitored_h"] += mon
    for p in res.per_participant.values():
        p["monitored_h"] = round(p["monitored_h"], 2)
        p["far_per_24h"] = p["false_alarms"] / p["monitored_h"] * 24 if p["monitored_h"] else None
    return res


def select_operating_point(scored: list[Scored], entries: np.ndarray) -> dict:
    """Predefined rule: max event sensitivity with FAR/24h <= MAX_FAR_PER_24H; ties -> lower FAR,
    then lower median latency. Falls back to the lowest-FAR point if nothing meets the cap."""
    best, curve = None, []
    for entry in entries:
        exit_ = max(0.0, float(entry) - C.EXIT_BELOW_ENTRY)
        r = event_eval(scored, float(entry), exit_)
        lat = float(np.median(r.latencies)) if r.latencies else float("inf")
        pt = {"entry_threshold": float(entry), "exit_threshold": exit_, "sensitivity": r.sensitivity,
              "far_per_24h": r.far24, "latency_s_median": lat}
        curve.append(pt)
        ok = r.far24 <= C.MAX_FAR_PER_24H
        key = (ok, r.sensitivity if ok else -r.far24, -r.far24, -lat)
        if best is None or key > best[0]:
            best = (key, pt)
    return {"selected": best[1], "met_far_cap": bool(best[0][0]), "curve": curve}


def candidate_entries(scored: list[Scored]) -> np.ndarray:
    """Entry thresholds in the upper tail of the evidence-window scores (where FAR/24h is usable)."""
    s = np.concatenate([x.score[x.rec.evidence] for x in scored])
    return np.unique(np.round(np.quantile(s, 1 - np.logspace(-1.3, -5.3, 50)), 6))


def window_metrics(scored: list[Scored], thr: float) -> dict:
    """Window level, on evidence windows with a definite D14 label (IGNORE excluded)."""
    s = np.concatenate([x.score for x in scored])
    y = np.concatenate([x.rec.label for x in scored])
    m = np.concatenate([x.rec.evidence for x in scored]) & (y >= 0)
    s, y = s[m], y[m]
    p = s >= thr
    tp, fp = int((p & (y == 1)).sum()), int((p & (y == 0)).sum())
    fn, tn = int((~p & (y == 1)).sum()), int((~p & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "windows": int(m.sum()), "positive_windows": int(y.sum()),
        "threshold": thr, "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy_not_meaningful_under_imbalance": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": prec, "recall_sensitivity": rec,
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "roc_auc": float(roc_auc_score(y, s)) if 0 < y.sum() < y.size else None,
        "pr_auc": float(average_precision_score(y, s)) if 0 < y.sum() < y.size else None,
    }


def confusion_png(cm: dict, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    shade = np.log10(m + 1)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(shade, cmap="Purples")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if shade[i, j] > shade.max() / 2 else "black")
    ax.set_xticks([0, 1], ["background", "convulsive"])
    ax.set_yticks([0, 1], ["background", "convulsive"])
    ax.set_xlabel("predicted (window score >= entry threshold)")
    ax.set_ylabel("annotated (D14 labels)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
