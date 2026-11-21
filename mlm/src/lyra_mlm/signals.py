"""Core data contracts: signal streams, recordings and timestamp checks.

Conventions
- timestamps: float64 seconds on the stream's own clock, shape (n,). NaN = missing timestamp.
- values: float64, shape (n, channels). Each stream keeps its native sampling rate.
- flags (per sample): OK; MISSING = slot exists but has no value (NaN); INVALID = a value
  exists but was rejected (non-finite, or flagged corrupt by the source).
- Samples that were never acquired have no slot at all. They show up as timestamp gaps.
Streams are immutable. Every transformation returns a new stream and appends to `provenance`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import IntEnum, StrEnum

import numpy as np

from .annotations import Annotation


class Modality(StrEnum):
    EMG = "emg"
    IMU = "imu"


class EmgSignalType(StrEnum):
    """MyoWare 2.0 outputs. Not interchangeable: each has different sensor-side processing."""

    RAW = "RAW"
    RECT = "RECT"
    ENV = "ENV"


class DataOrigin(StrEnum):
    HARDWARE = "hardware"  # recorded by a LYRA pod
    DATASET = "dataset"  # loaded from an external dataset
    SYNTHETIC = "synthetic"  # generated in software; never physiological data


class SampleFlag(IntEnum):
    OK = 0
    MISSING = 1
    INVALID = 2


IMU_CHANNELS = ("acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z")
IMU_UNITS = ("g", "g", "g", "deg/s", "deg/s", "deg/s")


class SignalValidationError(ValueError):
    pass


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
class SignalStream:
    modality: Modality
    timestamps: np.ndarray
    values: np.ndarray
    nominal_rate_hz: float
    channels: tuple[str, ...]
    units: tuple[str, ...]
    origin: DataOrigin
    signal_type: EmgSignalType | None = None  # required for EMG, None for IMU
    bandwidth_hz: float | None = None  # sensor-side analog bandwidth as delivered, if known
    flags: np.ndarray | None = None  # optional source flags; merged with flags derived from values
    provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ts = np.array(self.timestamps, dtype=np.float64)
        vals = np.array(self.values, dtype=np.float64)
        if vals.ndim == 1:
            vals = vals[:, None]
        errors = []
        if ts.ndim != 1 or vals.ndim != 2 or vals.shape[0] != ts.shape[0]:
            errors.append(f"timestamps {ts.shape} and values {vals.shape} do not describe the same samples")
        if not (math.isfinite(self.nominal_rate_hz) and self.nominal_rate_hz > 0):
            errors.append("nominal_rate_hz must be a positive finite number")
        if vals.ndim == 2 and not (len(self.channels) == len(self.units) == vals.shape[1]):
            errors.append("channels and units must name every value column")
        if self.modality is Modality.EMG:
            if self.signal_type is None:
                errors.append("EMG streams must declare signal_type (RAW, RECT or ENV)")
            if vals.ndim == 2 and vals.shape[1] != 1:
                errors.append("EMG streams carry exactly one channel; dataset adapters select it")
        else:
            if self.signal_type is not None:
                errors.append("signal_type applies to EMG only")
            if tuple(self.channels) != IMU_CHANNELS:
                errors.append(f"IMU channels must be {IMU_CHANNELS}")
        if self.bandwidth_hz is not None and not self.bandwidth_hz > 0:
            errors.append("bandwidth_hz must be positive when given")
        if errors:
            raise SignalValidationError("; ".join(errors))

        flags = np.where(np.isnan(vals).any(axis=1), SampleFlag.MISSING, SampleFlag.OK).astype(np.int8)
        flags[np.isinf(vals).any(axis=1)] = SampleFlag.INVALID
        if self.flags is not None:
            src = np.asarray(self.flags, dtype=np.int8)
            if src.shape != flags.shape:
                raise SignalValidationError("flags must have one entry per sample")
            flags = np.maximum(flags, src)
        object.__setattr__(self, "timestamps", _readonly(ts))
        object.__setattr__(self, "values", _readonly(vals))
        object.__setattr__(self, "flags", _readonly(flags))

    @property
    def n(self) -> int:
        return self.timestamps.shape[0]

    @property
    def nyquist_hz(self) -> float:
        return self.nominal_rate_hz / 2

    @property
    def effective_bandwidth_hz(self) -> float | None:
        """Highest frequency the delivered signal can represent, or None if unknown.

        RECT/ENV outputs are smoothed on the sensor, so without a declared bandwidth their
        frequency content is unknown and frequency-dependent features must not be computed.
        """
        if self.bandwidth_hz is not None:
            return min(self.nyquist_hz, self.bandwidth_hz)
        if self.signal_type in (EmgSignalType.RECT, EmgSignalType.ENV):
            return None
        return self.nyquist_hz

    @property
    def usable(self) -> np.ndarray:
        """Samples with a finite timestamp and an OK value."""
        return (self.flags == SampleFlag.OK) & np.isfinite(self.timestamps)

    def slice_time(self, start_s: float, end_s: float) -> SignalStream:
        """Samples with start_s <= t < end_s, original order kept (so disorder stays detectable)."""
        m = (self.timestamps >= start_s) & (self.timestamps < end_s)
        return replace(self, timestamps=self.timestamps[m], values=self.values[m], flags=self.flags[m])

    def contiguous_runs(self, gap_factor: float = 3.0) -> list[tuple[int, int]]:
        """[start, stop) index ranges of usable samples with forward steps <= gap_factor periods."""
        ok = self.usable
        d = np.diff(self.timestamps)
        joins = ok[1:] & ok[:-1] & (d > 0) & (d <= gap_factor / self.nominal_rate_hz)  # i joins i+1
        starts = np.flatnonzero(ok & ~np.concatenate(([False], joins)))
        stops = np.flatnonzero(ok & ~np.concatenate((joins, [False]))) + 1
        return list(zip(starts.tolist(), stops.tolist()))


@dataclass(frozen=True)
class TimestampReport:
    n_samples: int
    missing: int  # non-finite timestamps
    duplicates: int
    out_of_order: int  # backward steps
    irregular: int  # forward steps off the nominal period by more than the tolerance (gaps excluded)
    gaps: tuple[tuple[float, float], ...]  # forward steps longer than gap_factor periods
    measured_rate_hz: float | None  # from the median forward step

    @property
    def is_clean(self) -> bool:
        return not (self.missing or self.duplicates or self.out_of_order or self.irregular or self.gaps)


def check_timestamps(
    timestamps: np.ndarray, nominal_rate_hz: float, interval_tolerance: float = 0.25, gap_factor: float = 3.0
) -> TimestampReport:
    ts = np.asarray(timestamps, dtype=np.float64)
    finite = ts[np.isfinite(ts)]
    d = np.diff(finite)
    period = 1.0 / nominal_rate_hz
    fwd = d > 0
    gap = d > gap_factor * period
    irregular = fwd & ~gap & (np.abs(d - period) > interval_tolerance * period)
    return TimestampReport(
        n_samples=ts.size,
        missing=int(ts.size - finite.size),
        duplicates=int((d == 0).sum()),
        out_of_order=int((d < 0).sum()),
        irregular=int(irregular.sum()),
        gaps=tuple((float(finite[i]), float(finite[i + 1])) for i in np.flatnonzero(gap)),
        measured_rate_hz=float(1.0 / np.median(d[fwd])) if fwd.any() else None,
    )


@dataclass(frozen=True)
class Recording:
    participant_id: str
    recording_id: str
    origin: DataOrigin
    emg: SignalStream | None = None
    imu: SignalStream | None = None
    placement: str = "NOT SPECIFIED"  # e.g. "left deltoid"; never changed silently between sources
    annotations: tuple[Annotation, ...] = ()

    def __post_init__(self) -> None:
        if not self.participant_id or not self.recording_id:
            raise SignalValidationError("participant_id and recording_id are required")
        if self.emg is None and self.imu is None:
            raise SignalValidationError("a recording needs at least one stream")
        for slot, modality in (("emg", Modality.EMG), ("imu", Modality.IMU)):
            s = getattr(self, slot)
            if s is not None and (s.modality is not modality or s.origin is not self.origin):
                raise SignalValidationError(f"{slot} stream modality/origin does not match the recording")

    def stream(self, modality: Modality) -> SignalStream | None:
        return self.emg if modality is Modality.EMG else self.imu
