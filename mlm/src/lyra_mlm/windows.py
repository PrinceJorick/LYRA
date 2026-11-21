"""Window segmentation on a common analysis timeline.

Windows share start/end times across modalities. Each modality's samples keep their native
rate (e.g. 400 sEMG samples and 100 IMU samples in a 4 s window at 100 Hz / 25 Hz).
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .quality import QualityConfig, WindowQuality, assess_modality, combine
from .signals import DataOrigin, Modality, Recording, SignalStream
from .sync import SyncConfig, check_sync


@dataclass(frozen=True)
class WindowConfig:
    """Initial defaults (4 s, 50% overlap). Not tuned for seizure detection (context.md U8)."""

    length_s: float = 4.0
    step_s: float = 2.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.length_s) and self.length_s > 0 and 0 < self.step_s <= self.length_s):
            raise ValueError("need length_s > 0 and 0 < step_s <= length_s")

    @property
    def overlap(self) -> float:
        return 1 - self.step_s / self.length_s


@dataclass(frozen=True)
class Window:
    participant_id: str
    recording_id: str
    origin: DataOrigin
    start_s: float
    end_s: float
    emg: SignalStream | None
    imu: SignalStream | None
    quality: WindowQuality
    annotation_refs: tuple[str, ...] = ()  # overlapping annotation ids; labelling convention unresolved

    @property
    def available_modalities(self) -> tuple[Modality, ...]:
        return self.quality.available_modalities


def segment(recording: Recording, window_cfg: WindowConfig, quality_cfg: QualityConfig,
            required: tuple[Modality, ...], sync_cfg: SyncConfig | None = SyncConfig()) -> Iterator[Window]:
    """Yield fixed windows over the span covered by any stream; quality is assessed per modality."""
    streams = [s for s in (recording.emg, recording.imu) if s is not None]
    finite = [s.timestamps[np.isfinite(s.timestamps)] for s in streams]
    if not any(f.size for f in finite):
        return
    t0 = min(f.min() for f in finite if f.size)
    t_stop = max(f.max() + 1.0 / s.nominal_rate_hz for f, s in zip(finite, streams) if f.size)
    sync = check_sync(streams, sync_cfg) if len(streams) == 2 and sync_cfg else None
    k = 0
    while (start := t0 + k * window_cfg.step_s) + window_cfg.length_s <= t_stop + 1e-9:
        end = start + window_cfg.length_s
        per = {m: assess_modality(recording.stream(m), m, start, end, quality_cfg) for m in Modality}
        yield Window(
            recording.participant_id, recording.recording_id, recording.origin, start, end,
            recording.emg.slice_time(start, end) if recording.emg else None,
            recording.imu.slice_time(start, end) if recording.imu else None,
            combine(per, required, sync),
            tuple(a.annotation_id for a in recording.annotations if a.overlaps(start, end)),
        )
        k += 1
