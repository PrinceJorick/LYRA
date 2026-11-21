import ast
from pathlib import Path

import numpy as np
from synthetic import emg, imu, recording

from lyra_mlm.quality import QualityConfig, QualityState, assess_modality
from lyra_mlm.signals import Modality
from lyra_mlm.windows import WindowConfig, segment

CFG = QualityConfig(saturation_limits={"env": (0.0, 3.3)})
BOTH = (Modality.EMG, Modality.IMU)


def assess(stream, start=0.0, end=4.0, modality=Modality.EMG):
    return assess_modality(stream, modality, start, end, CFG)


def env_values():
    return emg().values[:, 0].copy()


def test_clean_signal_is_valid():
    q = assess(emg())
    assert q.state is QualityState.VALID and q.coverage == 1.0 and q.reasons == ()


def test_flatline_is_invalid():
    assert assess(emg(values=np.full(1200, 1.0))).state is QualityState.INVALID


def test_saturation_invalid_when_heavy_degraded_when_slight():
    v = env_values()
    v[:200] = 3.3
    assert assess(emg(values=v)).state is QualityState.INVALID
    v = env_values()
    v[:2] = 3.3
    q = assess(emg(values=v))
    assert q.state is QualityState.DEGRADED and "saturated" in q.reasons[0]


def test_missing_samples_degrade_then_make_modality_missing():
    v = env_values()
    v[0:40] = np.nan  # 10% of the window
    assert assess(emg(values=v)).state is QualityState.DEGRADED
    v[0:240] = np.nan  # 60%
    assert assess(emg(values=v)).state is QualityState.MISSING


def test_corrupted_samples_are_recorded():
    v = env_values()
    v[10:15] = np.inf
    q = assess(emg(values=v))
    assert q.state is QualityState.DEGRADED and any("invalid samples" in r for r in q.reasons)


def test_sensor_disconnection():
    dropped = emg(seconds=6.0)  # stream stops at 6 s
    assert assess(dropped, 8.0, 12.0).state is QualityState.MISSING
    assert assess(dropped, 4.0, 8.0).state is QualityState.DEGRADED  # half the window
    v = env_values()
    v[:] = 3.3  # output pinned at the rail
    assert assess(emg(values=v)).state is QualityState.INVALID


def test_out_of_order_timestamps_are_unsynchronized():
    t = emg().timestamps.copy()
    t[50], t[51] = t[51], t[50]
    assert assess(emg(timestamps=t)).state is QualityState.UNSYNCHRONIZED


def test_timestamp_gap_degrades():
    keep = np.r_[0:100, 120:1200]
    s = emg()
    q = assess(emg(values=s.values[keep, 0], timestamps=s.timestamps[keep]))
    assert q.state is QualityState.DEGRADED and any("gap" in r for r in q.reasons)


def test_one_missing_modality_keeps_per_modality_breakdown():
    w = next(segment(recording(emg_stream=None), WindowConfig(), CFG, BOTH))
    assert w.quality.overall is QualityState.MISSING
    assert w.quality.per_modality[Modality.EMG].state is QualityState.MISSING
    assert w.quality.per_modality[Modality.IMU].state is QualityState.VALID
    assert w.available_modalities == (Modality.IMU,)


def test_unimodal_requirement_ignores_the_other_modality():
    w = next(segment(recording(emg_stream=None), WindowConfig(), CFG, (Modality.IMU,)))
    assert w.quality.overall is QualityState.VALID


def test_clock_discrepancy_marks_multimodal_windows_unsynchronized():
    rec = recording(imu_stream=imu(timestamps=np.arange(300) / 26.0))
    w = next(segment(rec, WindowConfig(), CFG, BOTH))
    assert w.quality.overall is QualityState.UNSYNCHRONIZED and w.quality.sync_reasons
    assert w.quality.per_modality[Modality.IMU].state is QualityState.VALID  # per-modality kept


def test_quality_module_does_not_depend_on_the_classifier():
    src = Path(__file__).parents[1] / "src/lyra_mlm/quality.py"
    imported = {n.module for n in ast.walk(ast.parse(src.read_text())) if isinstance(n, ast.ImportFrom)}
    assert not imported & {"model", "inference", "features", "confirmation"}
