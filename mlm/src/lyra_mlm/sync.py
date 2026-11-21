"""Multimodal timestamp synchronization checks and explicit resampling.

Modalities are aligned by shared window boundaries on a common time base (seconds), not by
forcing a common sampling grid. Each stream keeps its native rate. resample() is never called
implicitly by the pipeline. Passing these checks does not prove the sensors are physically
synchronized; it only rules out gross clock discrepancies.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Literal

import numpy as np
from scipy import signal as sps

from .signals import SignalStream, check_timestamps


@dataclass(frozen=True)
class SyncConfig:
    max_rate_error_ppm: float = 5000.0  # placeholder engineering tolerance, not validated


@dataclass(frozen=True)
class SyncReport:
    synchronized: bool
    reasons: tuple[str, ...]


def check_sync(streams: list[SignalStream], cfg: SyncConfig) -> SyncReport:
    """Compare each stream's measured rate with its nominal rate (detects clock drift or a
    misdeclared rate, which would misalign modalities within a window)."""
    reasons = []
    for s in streams:
        measured = check_timestamps(s.timestamps, s.nominal_rate_hz).measured_rate_hz
        if measured is None:
            reasons.append(f"{s.modality}: cannot measure rate")
        elif abs(measured / s.nominal_rate_hz - 1) * 1e6 > cfg.max_rate_error_ppm:
            reasons.append(f"{s.modality}: measured {measured:.4g} Hz vs nominal {s.nominal_rate_hz:g} Hz")
    return SyncReport(not reasons, tuple(reasons))


ResampleMethod = Literal["linear", "polyphase"]


def resample(stream: SignalStream, target_rate_hz: float, method: ResampleMethod,
             gap_factor: float = 3.0) -> SignalStream:
    """Explicit resampling. Returns a new stream and records both rates in its provenance.

    linear: interpolation on timestamps; upsampling only (downsampling by interpolation aliases).
        Output samples that fall in a gap or next to a missing/invalid sample are NaN.
    polyphase: anti-aliased scipy resample_poly; needs a clean, gap-free, fully usable stream.
    """
    src = stream.nominal_rate_hz
    if not target_rate_hz > 0:
        raise ValueError("target_rate_hz must be positive")
    ts = stream.timestamps
    if not (np.isfinite(ts).all() and (np.diff(ts) > 0).all()):
        raise ValueError("resampling needs finite, strictly increasing timestamps")
    if method == "linear":
        if target_rate_hz < src:
            raise ValueError("linear interpolation cannot downsample without aliasing; use 'polyphase'")
        ok = stream.usable
        t_out = np.arange(ts[0], ts[-1] + 1e-12, 1.0 / target_rate_hz)
        left = np.clip(np.searchsorted(ts, t_out, side="right") - 1, 0, ts.size - 1)
        right = np.minimum(left + 1, ts.size - 1)
        exact = np.isclose(t_out, ts[left], rtol=0, atol=1e-9)  # lands on a real sample: neighbour irrelevant
        bad = ~ok[left] | (~exact & (~ok[right] | (ts[right] - ts[left] > gap_factor / src)))
        filled = np.where(ok[:, None], stream.values, 0.0)
        out = np.column_stack([np.interp(t_out, ts, filled[:, c]) for c in range(filled.shape[1])])
        out[bad] = np.nan
    elif method == "polyphase":
        if not (check_timestamps(ts, src, gap_factor=gap_factor).is_clean and stream.usable.all()):
            raise ValueError("polyphase resampling needs a regular, gap-free stream with no missing samples")
        ratio = Fraction(target_rate_hz / src).limit_denominator(1000)
        if not np.isclose(float(ratio) * src, target_rate_hz):
            raise ValueError(f"{src:g} -> {target_rate_hz:g} Hz is not a usable rational ratio")
        out = sps.resample_poly(stream.values, ratio.numerator, ratio.denominator, axis=0)
        t_out = ts[0] + np.arange(out.shape[0]) / target_rate_hz
    else:
        raise ValueError(f"unknown resampling method {method!r}")
    bw = stream.effective_bandwidth_hz
    return replace(
        stream, timestamps=t_out, values=out, nominal_rate_hz=target_rate_hz, flags=None,
        bandwidth_hz=None if bw is None else min(bw, target_rate_hz / 2),
        provenance=stream.provenance + (f"resampled {src:g} Hz -> {target_rate_hz:g} Hz ({method})",),
    )
