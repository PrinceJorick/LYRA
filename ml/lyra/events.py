"""Deterministic event state machine (reference for firmware/model/lyra_detector.c) and event metrics.

Per window, at its completion time T (ms):
- invalid window (no score, NaN or outside [0, 1]): hold, nothing changes;
- valid window after a gap > MAX_GAP_S since the previous valid one: the positive run resets and
  an open candidate event closes (consecutiveness cannot be assumed across missing data);
- p >= threshold: positive run += 1; a closed detector opens a candidate event when the run
  reaches N. The opening emits a caregiver alert unless an alert was emitted less than
  REFRACTORY_S earlier; then the alert is suppressed (the event is still tracked and counted);
- p < threshold: positive run = 0; an open event closes after M consecutive such windows.
State resets at every recording boundary. One candidate event never emits more than one alert.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import config as C

OPENED, ALERT, SUPPRESSED, CLOSED, INVALID = 1, 2, 4, 8, 16
U32 = 1 << 32


@dataclass
class Detector:
    """Sequential reference, identical to lyra_detector.c (uint32 millisecond arithmetic)."""

    threshold: float
    n_open: int
    m_close: int
    refractory_ms: int = int(C.REFRACTORY_S * 1000)
    max_gap_ms: int = int(C.MAX_GAP_S * 1000)
    run_pos: int = 0
    run_neg: int = 0
    open: bool = False
    last_valid_ms: int | None = None
    last_alert_ms: int | None = None

    def reset(self) -> None:
        self.run_pos = self.run_neg = 0
        self.open = False
        self.last_valid_ms = self.last_alert_ms = None

    def update(self, prob: float, t_ms: int) -> int:
        t_ms %= U32
        if not (prob == prob and 0.0 <= prob <= 1.0):  # NaN or out of range: hold
            return INVALID
        flags = 0
        if self.last_valid_ms is not None and (t_ms - self.last_valid_ms) % U32 > self.max_gap_ms:
            self.run_pos = self.run_neg = 0
            if self.open:
                self.open = False
                flags |= CLOSED
        self.last_valid_ms = t_ms
        if prob >= self.threshold:
            self.run_pos += 1
            self.run_neg = 0
            if not self.open and self.run_pos >= self.n_open:
                self.open = True
                flags |= OPENED
                if self.last_alert_ms is None or (t_ms - self.last_alert_ms) % U32 >= self.refractory_ms:
                    self.last_alert_ms = t_ms
                    flags |= ALERT
                else:
                    flags |= SUPPRESSED
        else:
            self.run_pos = 0
            if self.open:
                self.run_neg += 1
                if self.run_neg >= self.m_close:
                    self.open = False
                    self.run_neg = 0
                    flags |= CLOSED
        return flags


def openings(t: np.ndarray, positive: np.ndarray, n_open: int, m_close: int,
             max_gap: float = C.MAX_GAP_S) -> np.ndarray:
    """Fast equivalent of Detector for one recording: opening times. t/positive cover VALID
    windows only (invalid windows hold, so they are simply absent), in time order."""
    if t.size == 0:
        return np.empty(0)
    pos = positive.astype(np.int8)
    seg_break = np.flatnonzero(np.diff(t) > max_gap + 1e-9) + 1
    out = []
    for a, b in zip(np.r_[0, seg_break], np.r_[seg_break, t.size]):
        p = pos[a:b]
        if not p.any():
            continue
        d = np.diff(np.r_[0, p, 0])
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        is_open, prev_end = False, None
        for s, e in zip(starts, ends):
            if is_open and s - prev_end >= m_close:
                is_open = False
            if not is_open and e - s >= n_open:
                is_open = True
                out.append(t[a + s + n_open - 1])
            prev_end = e
    return np.asarray(out)


def alerts(opens: np.ndarray, refractory: float = C.REFRACTORY_S) -> tuple[np.ndarray, int]:
    """(emitted alert times, suppressed count) for one recording's openings."""
    emitted, last = [], None
    for t in opens:
        if last is None or t - last >= refractory - 1e-9:
            emitted.append(t)
            last = t
    return np.asarray(emitted), len(opens) - len(emitted)


@dataclass
class EventCounts:
    events: int = 0
    evaluable: int = 0
    detected: int = 0
    alerts: int = 0
    true_alerts: int = 0
    false_alerts: int = 0
    false_alerts_during_nontarget_seizures: int = 0
    suppressed: int = 0
    candidate_events: int = 0
    negative_hours: float = 0.0
    latencies: list = field(default_factory=list)

    def add(self, o: "EventCounts") -> None:
        for k in ("events", "evaluable", "detected", "alerts", "true_alerts", "false_alerts",
                  "false_alerts_during_nontarget_seizures", "suppressed", "candidate_events", "negative_hours"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        self.latencies += o.latencies

    def summary(self) -> dict:
        lat = np.asarray(self.latencies)
        return {"target_events": self.events, "evaluable_events": self.evaluable, "detected": self.detected,
                "event_sensitivity": self.detected / self.evaluable if self.evaluable else None,
                "false_alerts": self.false_alerts, "valid_negative_hours": round(self.negative_hours, 3),
                "far_per_24h": self.false_alerts / self.negative_hours * 24 if self.negative_hours else None,
                "latency_s_median": float(np.median(lat)) if lat.size else None,
                "event_precision": self.true_alerts / self.alerts if self.alerts else None,
                "emitted_alerts": self.alerts, "suppressed_alerts": self.suppressed,
                "candidate_events": self.candidate_events,
                "false_alerts_during_nontarget_seizures": self.false_alerts_during_nontarget_seizures}


def score_recording(t: np.ndarray, p: np.ndarray, valid: np.ndarray, negative: np.ndarray, window_s: float,
                    hop_s: float, target: np.ndarray, nontarget: np.ndarray, threshold: float, n_open: int,
                    m_close: int) -> EventCounts:
    """Event metrics for one recording. t: window end times; valid: windows the model may score;
    negative: valid windows with no target overlap (exposure); target/nontarget: (k, 2) intervals."""
    tv = t[valid]
    opens = openings(tv, p[valid] >= threshold, n_open, m_close)
    emitted, suppressed = alerts(opens)
    ec = EventCounts(suppressed=suppressed, candidate_events=len(opens), alerts=len(emitted),
                     negative_hours=float(negative.sum()) * hop_s / 3600)
    matched = np.zeros(emitted.size, bool)
    for on, off in target:
        ec.events += 1
        if not (valid & (t > on) & (t - window_s < off)).any():
            continue
        ec.evaluable += 1
        hit = (emitted >= on - C.MATCH_PRE_S) & (emitted <= off + C.MATCH_POST_S)
        matched |= hit
        if hit.any():
            ec.detected += 1
            ec.latencies.append(float(emitted[hit].min() - on))
    ec.true_alerts = int(matched.sum())
    ec.false_alerts = int((~matched).sum())
    if nontarget.size and ec.false_alerts:
        fa = emitted[~matched]
        near = np.zeros(fa.size, bool)
        for on, off in nontarget:
            near |= (fa >= on - C.MATCH_PRE_S) & (fa <= off + C.MATCH_POST_S)
        ec.false_alerts_during_nontarget_seizures = int(near.sum())
    return ec
