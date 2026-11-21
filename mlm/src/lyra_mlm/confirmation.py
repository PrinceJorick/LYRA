"""Event confirmation, separate from the classifier.

Turns per-window predictions into possible-event state. It never claims that a seizure
occurred: POSSIBLE_EVENT only means the configured evidence rule was met. The phone then shows
"Possible convulsive seizure detected. This is not a diagnosis." Defaults follow locked
Decision 4: unvalidated engineering parameters, not clinical thresholds.

Evidence rules:
- VALID predictions are evidence. DEGRADED ones are evidence only with
  degraded_window_policy == "allow_with_flag" (the default).
- SUPPRESSED (and ignored DEGRADED) predictions are never negative evidence. They follow
  invalid_window_policy: "hold" (default) keeps the positive count, "reset" clears it.
- If the time since the last evidence-bearing prediction exceeds max_timestamp_gap_s, evidence is
  not continuous: an accumulating run resets to IDLE, and an active possible event ends in
  REFRACTORY (not IDLE), so a data gap cannot re-notify the same episode.
- Out-of-order/duplicate timestamps and stale predictions (older than max_latency_s at now_s)
  are rejected without changing state.
- Hysteresis: a score >= entry_threshold starts or extends the positive run; an active possible
  event continues while score >= exit_threshold.
- After a possible event ends, REFRACTORY lasts refractory_cooldown_s from the end, so the same episode
  cannot notify twice.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from .model import Prediction, PredictionStatus
from .signals import Modality


class ConfirmationState(StrEnum):
    IDLE = "idle"
    ACCUMULATING = "accumulating"
    POSSIBLE_EVENT = "possible_event"
    REFRACTORY = "refractory"


@dataclass(frozen=True)
class ConfirmationConfig:
    entry_threshold: float = 0.65  # classification threshold for a positive window
    exit_threshold: float = 0.45
    consecutive_positive_required: int = 3  # >= 8 s confirmation latency at 4 s / 2 s
    max_timestamp_gap_s: float = 6.0  # tolerates up to two consecutive missing windows at the 2 s step
    refractory_cooldown_s: float = 60.0
    max_latency_s: float | None = 10.0
    degraded_window_policy: Literal["allow_with_flag", "ignore"] = "allow_with_flag"
    invalid_window_policy: Literal["hold", "reset"] = "hold"

    def __post_init__(self) -> None:
        if not 0 <= self.exit_threshold <= self.entry_threshold <= 1:
            raise ValueError("need 0 <= exit_threshold <= entry_threshold <= 1")
        if self.consecutive_positive_required < 1 or self.max_timestamp_gap_s <= 0 or self.refractory_cooldown_s < 0:
            raise ValueError("invalid count, gap or refractory parameters")


@dataclass(frozen=True)
class TransitionRecord:
    timestamp_s: float
    previous: ConfirmationState
    state: ConfirmationState
    reason: str
    accepted: bool  # False when the prediction was rejected
    event_started: bool  # True only on entry into POSSIBLE_EVENT: notify once
    positive_count: int
    quality: tuple[tuple[Modality, str], ...]  # per-modality states of the triggering window


class EventConfirmation:
    def __init__(self, cfg: ConfirmationConfig = ConfirmationConfig()):
        self.cfg = cfg
        self.history: list[TransitionRecord] = []
        self.state = ConfirmationState.IDLE
        self.positive_count = 0
        self._last_ts: float | None = None
        self._last_evidence_ts: float | None = None
        self._refractory_until = float("-inf")
        self._last_quality: tuple[tuple[Modality, str], ...] | None = None

    def _log(self, ts: float, state: ConfirmationState, reason: str, quality, accepted: bool = True,
             started: bool = False) -> TransitionRecord:
        rec = TransitionRecord(ts, self.state, state, reason, accepted, started, self.positive_count, quality)
        self.state = state
        self.history.append(rec)
        return rec

    def _end_run(self, ts: float, reason: str, quality) -> None:
        self.positive_count = 0
        if self.state is ConfirmationState.POSSIBLE_EVENT:
            self._refractory_until = ts + self.cfg.refractory_cooldown_s
            self._log(ts, ConfirmationState.REFRACTORY, "possible event ended: " + reason, quality)
        elif self.state is ConfirmationState.ACCUMULATING:
            self._log(ts, ConfirmationState.IDLE, reason, quality)

    def update(self, p: Prediction, now_s: float | None = None) -> TransitionRecord:
        ts, cfg = p.timestamp_s, self.cfg
        quality = tuple((m, str(q.state)) for m, q in p.quality.per_modality.items())
        if self._last_ts is not None and ts <= self._last_ts:
            return self._log(ts, self.state, "rejected: out-of-order or duplicate timestamp", quality, accepted=False)
        if cfg.max_latency_s is not None and now_s is not None and now_s - ts > cfg.max_latency_s:
            return self._log(ts, self.state, f"rejected: stale prediction ({now_s - ts:.1f} s old)", quality,
                             accepted=False)
        self._last_ts = ts
        if self._last_quality is not None and quality != self._last_quality:
            change = ", ".join(f"{m} {a}->{b}" for (m, a), (_, b) in zip(self._last_quality, quality) if a != b)
            self._log(ts, self.state, f"quality changed: {change}", quality)
        self._last_quality = quality

        if self.state is ConfirmationState.REFRACTORY and ts >= self._refractory_until:
            self._log(ts, ConfirmationState.IDLE, "refractory period over", quality)
        if self._last_evidence_ts is not None and ts - self._last_evidence_ts > cfg.max_timestamp_gap_s:
            gap = ts - self._last_evidence_ts
            self._end_run(ts, f"{gap:.1f} s since last valid prediction exceeds max_timestamp_gap_s",
                          quality)

        evidence = p.status is PredictionStatus.VALID or (
            p.status is PredictionStatus.DEGRADED and cfg.degraded_window_policy == "allow_with_flag")
        if not evidence:
            note = f"{p.status} window carries no evidence"
            if cfg.invalid_window_policy == "reset" and self.state in (
                    ConfirmationState.ACCUMULATING, ConfirmationState.POSSIBLE_EVENT):
                self._end_run(ts, note + " (invalid_window_policy=reset)", quality)
                return self.history[-1]
            return self._log(ts, self.state, note + " (held)", quality)

        self._last_evidence_ts = ts
        score = p.score
        if self.state is ConfirmationState.REFRACTORY:
            return self._log(ts, self.state, "refractory: evidence not accumulated", quality)
        if self.state is ConfirmationState.POSSIBLE_EVENT:
            if score >= cfg.exit_threshold:
                return self._log(ts, self.state, "possible event continuing (score >= exit threshold)", quality)
            self._end_run(ts, "score below exit threshold", quality)
            return self.history[-1]
        if score >= cfg.entry_threshold:
            self.positive_count += 1
            if self.positive_count >= cfg.consecutive_positive_required:
                return self._log(ts, ConfirmationState.POSSIBLE_EVENT,
                                 f"{self.positive_count} consecutive valid positive windows", quality, started=True)
            return self._log(ts, ConfirmationState.ACCUMULATING, "valid positive window", quality)
        self.positive_count = 0
        return self._log(ts, ConfirmationState.IDLE, "valid negative window", quality)

    def reset(self, timestamp_s: float, reason: str) -> TransitionRecord:
        """External reset/recovery, e.g. the user cancelled on the pod button or the phone."""
        self.positive_count = 0
        self._last_evidence_ts = None
        self._refractory_until = float("-inf")
        return self._log(timestamp_s, ConfirmationState.IDLE, "reset: " + reason, self._last_quality or ())
