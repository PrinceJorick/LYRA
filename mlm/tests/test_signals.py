import numpy as np
import pytest
from synthetic import emg, imu

from lyra_mlm.annotations import Annotation, AnnotationKind, activity_group
from lyra_mlm.signals import (
    IMU_UNITS, DataOrigin, EmgSignalType, Modality, Recording, SampleFlag, SignalStream, SignalValidationError,
    check_timestamps,
)

T = np.arange(10) / 100.0


def test_emg_requires_signal_type():
    with pytest.raises(SignalValidationError, match="signal_type"):
        SignalStream(Modality.EMG, T, np.zeros(10), 100.0, ("env",), ("V",), DataOrigin.SYNTHETIC)


def test_imu_requires_six_named_channels():
    with pytest.raises(SignalValidationError, match="IMU channels"):
        SignalStream(Modality.IMU, T, np.zeros((10, 6)), 25.0, ("a", "b", "c", "d", "e", "f"), IMU_UNITS,
                     DataOrigin.SYNTHETIC)


def test_shape_mismatch_and_multichannel_emg_rejected():
    with pytest.raises(SignalValidationError, match="same samples"):
        SignalStream(Modality.EMG, T, np.zeros(9), 100.0, ("env",), ("V",), DataOrigin.SYNTHETIC, EmgSignalType.ENV)
    with pytest.raises(SignalValidationError, match="exactly one channel"):
        SignalStream(Modality.EMG, T, np.zeros((10, 2)), 100.0, ("a", "b"), ("V", "V"), DataOrigin.SYNTHETIC,
                     EmgSignalType.RAW)


def test_flags_distinguish_missing_and_invalid():
    v = np.ones(10)
    v[2], v[3] = np.nan, np.inf
    src = np.zeros(10, dtype=np.int8)
    src[5] = SampleFlag.INVALID  # e.g. corrupt packet reported by the source
    s = emg(values=v, timestamps=T, flags=src)
    assert s.flags[2] == SampleFlag.MISSING
    assert s.flags[3] == SampleFlag.INVALID and s.flags[5] == SampleFlag.INVALID
    assert s.usable.sum() == 7


def test_stream_arrays_are_read_only():
    s = emg()
    with pytest.raises(ValueError):
        s.values[0, 0] = 1.0


def test_check_timestamps_detects_each_irregularity():
    ts = np.array([0, .01, .02, np.nan, .03, .03, .025, .05, .2])
    r = check_timestamps(ts, 100.0)
    assert (r.missing, r.duplicates, r.out_of_order, r.irregular) == (1, 1, 1, 1)
    assert r.gaps == ((0.05, 0.2),)
    assert not r.is_clean


def test_clean_timestamps():
    r = check_timestamps(np.arange(100) / 25.0, 25.0)
    assert r.is_clean and r.measured_rate_hz == pytest.approx(25.0)


def test_effective_bandwidth_respects_signal_type_and_nyquist():
    assert emg(bandwidth_hz=None).effective_bandwidth_hz is None  # ENV smoothing unknown
    assert emg(bandwidth_hz=20.0).effective_bandwidth_hz == 20.0
    assert emg(bandwidth_hz=80.0).effective_bandwidth_hz == 50.0  # capped at Nyquist
    assert emg(signal_type=EmgSignalType.RAW, bandwidth_hz=None).effective_bandwidth_hz == 50.0


def test_contiguous_runs_break_at_missing_samples_and_gaps():
    t = np.concatenate([np.arange(11) / 100.0, 0.5 + np.arange(5) / 100.0])
    v = np.ones(16)
    v[5] = np.nan
    assert emg(values=v, timestamps=t).contiguous_runs() == [(0, 5), (6, 11), (11, 16)]


def test_recording_rejects_mismatched_origin_and_empty():
    hw = SignalStream(Modality.IMU, T, np.zeros((10, 6)), 25.0, imu().channels, IMU_UNITS, DataOrigin.HARDWARE)
    with pytest.raises(SignalValidationError, match="origin"):
        Recording("p", "r", DataOrigin.SYNTHETIC, imu=hw)
    with pytest.raises(SignalValidationError, match="at least one"):
        Recording("p", "r", DataOrigin.SYNTHETIC)


def test_annotation_validation_and_grouping():
    with pytest.raises(ValueError, match="unknown activity"):
        Annotation("a", 0, 1, AnnotationKind.HARD_NEGATIVE, "x", activity="juggling")
    with pytest.raises(ValueError, match="only to HARD_NEGATIVE"):
        Annotation("a", 0, 1, AnnotationKind.SEIZURE_FBTCS, "x", activity="walking")
    a = Annotation("a", 5, 2, AnnotationKind.HARD_NEGATIVE, "brushing", activity="brushing_teeth")
    assert activity_group(a.activity) == "daily_activity"
    assert a.overlaps(6, 10) and not a.overlaps(7, 10)
