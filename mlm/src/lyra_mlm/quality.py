"""Signal-quality assessment, independent of any classifier.

These checks are engineering safeguards with placeholder thresholds. They are not
clinically validated seizure-detection rules. A poor-quality or missing signal is never
interpreted as a negative seizure classification.

State semantics, per modality and per window:
- VALID: enough usable samples, no detected problems.
- DEGRADED: usable, with recorded problems (some missing samples, irregular timing, partial
  saturation, some flat channels). Downstream use requires an explicit policy.
- INVALID: samples exist but cannot be trusted (flatline, heavy saturation).
- MISSING: modality not recorded, no samples in the window, or too few usable samples.
- UNSYNCHRONIZED: duplicate/out-of-order timestamps, or the clocks of the modalities disagree.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from .signals import Modality, SampleFlag, SignalStream, check_timestamps
from .sync import SyncReport


class QualityState(StrEnum):
    VALID = "VALID"
    DEGRADED = "DEGRADED"
    INVALID = "INVALID"
    MISSING = "MISSING"
    UNSYNCHRONIZED = "UNSYNCHRONIZED"


USABLE = frozenset({QualityState.VALID, QualityState.DEGRADED})
_WORST_FIRST = (QualityState.UNSYNCHRONIZED, QualityState.INVALID, QualityState.MISSING,
                QualityState.DEGRADED, QualityState.VALID)


@dataclass(frozen=True)
class QualityConfig:
    min_coverage_valid: float = 0.95  # usable / expected samples needed for VALID
    min_coverage_usable: float = 0.5  # below this the modality is MISSING for the window
    flatline_ptp: float = 0.0  # channel peak-to-peak at or below this counts as flat
    saturation_limits: Mapping[str, tuple[float, float]] = field(default_factory=dict)  # channel -> full scale
    max_saturated_fraction: float = 0.05  # above: INVALID; any saturation: DEGRADED
    max_irregular_fraction: float = 0.1  # irregular sampling intervals above this: DEGRADED
    interval_tolerance: float = 0.25
    gap_factor: float = 3.0

    def __post_init__(self) -> None:
        if not 0 <= self.min_coverage_usable <= self.min_coverage_valid <= 1:
            raise ValueError("need 0 <= min_coverage_usable <= min_coverage_valid <= 1")
        for ch, (lo, hi) in self.saturation_limits.items():
            if not lo < hi:
                raise ValueError(f"saturation limits for {ch} must be increasing")


@dataclass(frozen=True)
class ModalityQuality:
    modality: Modality
    state: QualityState
    reasons: tuple[str, ...]
    coverage: float  # usable / expected samples, capped at 1


def assess_modality(stream: SignalStream | None, modality: Modality, start_s: float, end_s: float,
                    cfg: QualityConfig) -> ModalityQuality:
    def result(state: QualityState, reasons: list[str], coverage: float = 0.0) -> ModalityQuality:
        return ModalityQuality(modality, state, tuple(reasons), coverage)

    if stream is None:
        return result(QualityState.MISSING, ["modality not recorded"])
    win = stream.slice_time(start_s, end_s)
    if win.n == 0:
        return result(QualityState.MISSING, ["no samples in window: sensor disconnected or not acquired"])
    tr = check_timestamps(win.timestamps, win.nominal_rate_hz, cfg.interval_tolerance, cfg.gap_factor)
    if tr.duplicates or tr.out_of_order:
        return result(QualityState.UNSYNCHRONIZED,
                      [f"{tr.duplicates} duplicate and {tr.out_of_order} out-of-order timestamps"])
    ok = win.flags == SampleFlag.OK
    coverage = min(1.0, ok.sum() / ((end_s - start_s) * win.nominal_rate_hz))
    n_invalid = int((win.flags == SampleFlag.INVALID).sum())
    n_missing = int((win.flags == SampleFlag.MISSING).sum())
    sample_notes = []
    if n_missing:
        sample_notes.append(f"{n_missing} missing samples")
    if n_invalid:
        sample_notes.append(f"{n_invalid} invalid samples")
    if coverage < cfg.min_coverage_usable:
        return result(QualityState.MISSING, [f"coverage {coverage:.0%} below usable minimum"] + sample_notes, coverage)

    vals = win.values[ok]
    flat = [ch for ch, p in zip(win.channels, np.ptp(vals, axis=0)) if p <= cfg.flatline_ptp]
    if len(flat) == len(win.channels):
        return result(QualityState.INVALID, ["flatline signal"], coverage)
    degraded = sample_notes + ([f"flat channels: {', '.join(flat)}"] if flat else [])
    for c, ch in enumerate(win.channels):
        if ch in cfg.saturation_limits:
            lo, hi = cfg.saturation_limits[ch]
            frac = float(((vals[:, c] <= lo) | (vals[:, c] >= hi)).mean())
            if frac > cfg.max_saturated_fraction:
                return result(QualityState.INVALID, [f"{ch} saturated in {frac:.0%} of samples"], coverage)
            if frac > 0:
                degraded.append(f"{ch} saturated in {frac:.0%} of samples")
    if coverage < cfg.min_coverage_valid:
        degraded.append(f"coverage {coverage:.0%}")
    if tr.gaps:
        degraded.append(f"{len(tr.gaps)} timestamp gap(s)")
    if tr.irregular > cfg.max_irregular_fraction * max(1, win.n - 1):
        degraded.append(f"{tr.irregular} irregular sampling intervals")
    return result(QualityState.DEGRADED if degraded else QualityState.VALID, degraded, coverage)


<<<<<<< HEAD
QUALITY_RANK = {s: i for i, s in enumerate(reversed(_WORST_FIRST))}  # VALID=0 .. UNSYNCHRONIZED=4


def inherit_source_quality(derived: ModalityQuality, source: ModalityQuality) -> ModalityQuality:
    """A derived stream (e.g. a simulated envelope) is never rated better than the source it was
    computed from: filtering a flatlined raw signal leaves a decaying transient that would
    otherwise pass the flatline check."""
    if QUALITY_RANK[source.state] <= QUALITY_RANK[derived.state]:
        return derived
    return ModalityQuality(derived.modality, source.state,
                           tuple(f"source: {r}" for r in source.reasons) + derived.reasons,
                           min(derived.coverage, source.coverage))


=======
>>>>>>> 414a2140836e55fed760a18fff6c49fe2bc0082d
@dataclass(frozen=True)
class WindowQuality:
    per_modality: Mapping[Modality, ModalityQuality]
    required: tuple[Modality, ...]
    overall: QualityState
    sync_reasons: tuple[str, ...] = ()  # cross-modality clock problems, if any

    @property
    def available_modalities(self) -> tuple[Modality, ...]:
        """Modalities usable in this window, whether or not the current mode requires them."""
        return tuple(m for m in Modality if self.per_modality[m].state in USABLE)


def combine(per_modality: Mapping[Modality, ModalityQuality], required: tuple[Modality, ...],
            sync: SyncReport | None) -> WindowQuality:
    if not required:
        raise ValueError("at least one modality must be required")
    if len(required) > 1 and sync is not None and not sync.synchronized:
        return WindowQuality(per_modality, required, QualityState.UNSYNCHRONIZED, sync.reasons)
    states = {per_modality[m].state for m in required}
    return WindowQuality(per_modality, required, next(s for s in _WORST_FIRST if s in states))
