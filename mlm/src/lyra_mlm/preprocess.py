"""Leakage-safe preprocessing: validated filters, fixed transforms and fitted normalization.

Sensor-side vs software-side filtering: the MyoWare 2.0 RECT/ENV outputs are already
rectified/smoothed in analog hardware, and the BMI270 applies its own digital filter for the
configured ODR. FilterConfig describes only additional software processing. Never re-apply a
stage the sensor already performed.

Fixed transforms (apply_fixed_scale) estimate nothing from data. Data-fitted transforms
(fit_normalization) may only be fitted on the training partition. Applying them never changes
the fitted parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy import signal as sps

from .evaluation import LeakageError, Partition
from .signals import EmgSignalType, Modality, SampleFlag, SignalStream

FilterKind = Literal["lowpass", "highpass", "bandpass", "bandstop"]


class FilterConfigError(ValueError):
    pass


@dataclass(frozen=True)
class FilterConfig:
    modality: Modality
    signal_type: EmgSignalType | None  # EMG only
    sample_rate_hz: float
    kind: FilterKind
    order: int
    cutoff_hz: tuple[float, ...]
    causal: bool  # True: sosfilt (streaming/embedded-compatible). False: zero-phase, offline only.
    rationale: str  # assumptions and limitations, required

    def __post_init__(self) -> None:
        nyq = self.sample_rate_hz / 2
        errs = []
        if not (math.isfinite(self.sample_rate_hz) and self.sample_rate_hz > 0):
            errs.append("sample_rate_hz must be positive")
        if self.kind not in ("lowpass", "highpass", "bandpass", "bandstop"):
            errs.append(f"unknown filter kind {self.kind!r}")
        if not (isinstance(self.order, int) and 1 <= self.order <= 8):
            errs.append("order must be an integer in 1..8")
        n_cut = 2 if self.kind in ("bandpass", "bandstop") else 1
        if len(self.cutoff_hz) != n_cut:
            errs.append(f"{self.kind} needs {n_cut} cutoff frequency(ies)")
        elif not all(0 < f < nyq for f in self.cutoff_hz):
            errs.append(f"cutoffs {self.cutoff_hz} Hz must lie strictly between 0 and Nyquist ({nyq:g} Hz)")
        elif n_cut == 2 and not self.cutoff_hz[0] < self.cutoff_hz[1]:
            errs.append("band edges must be increasing")
        if (self.modality is Modality.EMG) != (self.signal_type is not None):
            errs.append("signal_type is required for EMG and must be None for IMU")
        if self.signal_type in (EmgSignalType.RECT, EmgSignalType.ENV) and self.kind != "lowpass":
            errs.append(f"{self.signal_type} is already rectified on the sensor; only lowpass smoothing is "
                        "allowed (high-pass or band-pass would remove the envelope level being measured)")
        if not self.rationale.strip():
            errs.append("rationale (assumptions and limitations) is required")
        if errs:
            raise FilterConfigError("; ".join(errs))  # rejected, never adjusted

    def describe(self) -> str:
        mode = "causal" if self.causal else "zero-phase"
        return f"butterworth {self.kind} order {self.order} at {self.cutoff_hz} Hz, {mode}"


def apply_filter(stream: SignalStream, cfg: FilterConfig, gap_factor: float = 3.0) -> SignalStream:
    """Filter each contiguous run of usable samples on its own.

    Missing/invalid samples and timestamp gaps are never filled. Runs too short for zero-phase
    padding stay unfiltered and are flagged INVALID. Causal runs start with a zero initial state
    (a start-up transient that the embedded port will share).
    """
    if stream.modality is not cfg.modality or stream.signal_type != cfg.signal_type:
        raise FilterConfigError(f"filter is for {cfg.modality}/{cfg.signal_type}, "
                                f"stream is {stream.modality}/{stream.signal_type}")
    if not math.isclose(stream.nominal_rate_hz, cfg.sample_rate_hz):
        raise FilterConfigError(f"filter designed for {cfg.sample_rate_hz} Hz, stream is {stream.nominal_rate_hz} Hz")
    btype = {"lowpass": "low", "highpass": "high", "bandpass": "bandpass", "bandstop": "bandstop"}[cfg.kind]
    wn = cfg.cutoff_hz[0] if len(cfg.cutoff_hz) == 1 else cfg.cutoff_hz
    sos = sps.butter(cfg.order, wn, btype=btype, fs=cfg.sample_rate_hz, output="sos")
    padlen = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    out = np.full_like(stream.values, np.nan)
    flags = stream.flags.copy()
    short = 0
    for a, b in stream.contiguous_runs(gap_factor):
        seg = stream.values[a:b]
        if cfg.causal:
            out[a:b] = sps.sosfilt(sos, seg, axis=0)
        elif b - a > padlen:
            out[a:b] = sps.sosfiltfilt(sos, seg, axis=0)
        else:
            flags[a:b] = SampleFlag.INVALID
            short += b - a
    note = cfg.describe() + (f"; {short} samples in runs too short to filter flagged INVALID" if short else "")
    return replace(stream, values=out, flags=flags, provenance=stream.provenance + (note,))


def apply_fixed_scale(stream: SignalStream, scale: float, offset: float, units: tuple[str, ...],
                      reason: str) -> SignalStream:
    """Fixed signal-level transform y = x * scale + offset (e.g. ADC counts to volts)."""
    if not (math.isfinite(scale) and scale != 0 and math.isfinite(offset)):
        raise ValueError("scale must be finite and non-zero, offset finite")
    return replace(stream, values=stream.values * scale + offset, units=units,
                   provenance=stream.provenance + (f"fixed scale x{scale:g} {offset:+g}: {reason}",))


def imu_magnitudes(imu: SignalStream) -> tuple[np.ndarray, np.ndarray]:
    """Acceleration magnitude (g, gravity included) and angular-velocity magnitude (deg/s)."""
    if imu.modality is not Modality.IMU:
        raise ValueError("imu_magnitudes needs an IMU stream")
    return np.linalg.norm(imu.values[:, :3], axis=1), np.linalg.norm(imu.values[:, 3:], axis=1)


NormMethod = Literal["zscore", "robust"]


@dataclass(frozen=True)
class NormalizationParams:
    """Fitted on the training partition only. Stored with the model config for embedded parity."""

    method: NormMethod
    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    n_fit_rows: int
    fitted_on: Partition = Partition.TRAIN

    def __post_init__(self) -> None:
        if self.fitted_on is not Partition.TRAIN:
            raise LeakageError("normalization parameters must come from the training partition")
        if not (len(self.feature_names) == len(self.center) == len(self.scale)):
            raise ValueError("feature_names, center and scale must align")

    def to_dict(self) -> dict:
        return {"method": self.method, "feature_names": list(self.feature_names), "center": list(self.center),
                "scale": list(self.scale), "n_fit_rows": self.n_fit_rows, "fitted_on": str(self.fitted_on)}

    @classmethod
    def from_dict(cls, d: dict) -> NormalizationParams:
        return cls(d["method"], tuple(d["feature_names"]), tuple(d["center"]), tuple(d["scale"]),
                   d["n_fit_rows"], Partition(d["fitted_on"]))


def fit_normalization(x: np.ndarray, feature_names: tuple[str, ...], partition: Partition,
                      method: NormMethod) -> NormalizationParams:
    """Estimate per-feature center/scale. NaN (unavailable) entries are ignored, never imputed."""
    if partition is not Partition.TRAIN:
        raise LeakageError(f"refusing to fit normalization on the {partition} partition")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(feature_names):
        raise ValueError("x must be (rows, len(feature_names))")
    with np.errstate(all="ignore"):
        if method == "zscore":
            center, scale = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
        elif method == "robust":
            q1, center, q3 = np.nanpercentile(x, [25, 50, 75], axis=0)
            scale = q3 - q1
        else:
            raise ValueError(f"unknown method {method!r}")
    bad = [n for n, c, s in zip(feature_names, center, scale) if not (np.isfinite(c) and np.isfinite(s) and s > 0)]
    if bad:
        raise ValueError(f"cannot normalize constant or all-unavailable features: {bad}")
    return NormalizationParams(method, tuple(feature_names), tuple(map(float, center)),
                               tuple(map(float, scale)), int(x.shape[0]))


def apply_normalization(x: np.ndarray, feature_names: tuple[str, ...], params: NormalizationParams) -> np.ndarray:
    """Apply previously fitted parameters unchanged. NaN stays NaN."""
    if tuple(feature_names) != params.feature_names:
        raise ValueError("feature order differs from the fitted normalization")
    return (np.asarray(x, dtype=np.float64) - np.array(params.center)) / np.array(params.scale)
