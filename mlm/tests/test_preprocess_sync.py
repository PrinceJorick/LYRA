import dataclasses

import numpy as np
import pytest
from synthetic import emg, imu, recording

from lyra_mlm.evaluation import LeakageError, Partition
from lyra_mlm.preprocess import (
    FilterConfig, FilterConfigError, NormalizationParams, apply_filter, apply_fixed_scale, apply_normalization,
    fit_normalization,
)
from lyra_mlm.quality import QualityConfig
from lyra_mlm.signals import EmgSignalType, Modality, SampleFlag
from lyra_mlm.sync import SyncConfig, check_sync, resample
from lyra_mlm.windows import WindowConfig, segment

RAW = EmgSignalType.RAW


def lowpass(rate=100.0, cutoff=(20.0,), kind="lowpass", signal_type=RAW, causal=False, **kw):
    return FilterConfig(Modality.EMG, signal_type, rate, kind, kw.pop("order", 4), cutoff, causal,
                        kw.pop("rationale", "synthetic test filter"))


# --- resampling and synchronization ----------------------------------------------------------

def test_resample_method_must_be_explicit():
    with pytest.raises(TypeError):
        resample(emg(), 50.0)


def test_linear_interpolation_refuses_to_downsample():
    with pytest.raises(ValueError, match="aliasing"):
        resample(emg(signal_type=RAW), 25.0, "linear")


def test_polyphase_records_rates_and_leaves_input_untouched():
    s = emg(signal_type=RAW, bandwidth_hz=None)
    out = resample(s, 50.0, "polyphase")
    assert out.nominal_rate_hz == 50.0 and out.n == s.n // 2
    assert out.provenance[-1] == "resampled 100 Hz -> 50 Hz (polyphase)"
    assert s.nominal_rate_hz == 100.0 and s.provenance == ()
    assert out.effective_bandwidth_hz == 25.0


def test_polyphase_rejects_missing_samples():
    v = emg().values[:, 0].copy()
    v[10] = np.nan
    with pytest.raises(ValueError, match="no missing samples"):
        resample(emg(values=v), 50.0, "polyphase")


def test_linear_upsampling_does_not_bridge_gaps():
    t = np.concatenate([np.arange(50) / 25.0, 3.0 + np.arange(50) / 25.0])  # 1 s gap
    out = resample(imu(timestamps=t), 50.0, "linear")
    in_gap = (out.timestamps > 1.96) & (out.timestamps < 3.0)  # last sample before the gap is at 1.96 s
    assert np.isnan(out.values[in_gap]).all()
    assert np.isfinite(out.values[~in_gap]).all()


def test_segmentation_preserves_native_rates_and_timestamps():
    rec = recording()
    w = next(segment(rec, WindowConfig(), QualityConfig(), (Modality.EMG, Modality.IMU)))
    assert (w.emg.nominal_rate_hz, w.imu.nominal_rate_hz) == (100.0, 25.0)
    assert (w.emg.n, w.imu.n) == (400, 100)  # sEMG not downsampled to the IMU rate
    np.testing.assert_array_equal(w.emg.timestamps, rec.emg.timestamps[:400])
    assert w.emg.provenance == () and w.imu.provenance == ()


def test_check_sync_flags_a_misdeclared_rate():
    t = np.arange(300) / 26.0  # declared 25 Hz, actually 26 Hz
    report = check_sync([emg(), imu(timestamps=t)], SyncConfig())
    assert not report.synchronized and "imu" in report.reasons[0]
    assert check_sync([emg(), imu()], SyncConfig()).synchronized


# --- filters ---------------------------------------------------------------------------------

@pytest.mark.parametrize("cutoff", [(50.0,), (60.0,), (0.0,)])
def test_filter_rejects_cutoffs_outside_nyquist(cutoff):
    with pytest.raises(FilterConfigError, match="Nyquist"):
        lowpass(cutoff=cutoff)


@pytest.mark.parametrize("kwargs, match", [
    ({"kind": "bandpass", "cutoff": (10.0,)}, "needs 2"),
    ({"kind": "bandpass", "cutoff": (20.0, 10.0)}, "increasing"),
    ({"order": 0}, "order"),
    ({"kind": "highpass", "signal_type": EmgSignalType.ENV}, "only lowpass"),
    ({"rationale": "  "}, "rationale"),
])
def test_filter_rejects_invalid_configs(kwargs, match):
    with pytest.raises(FilterConfigError, match=match):
        lowpass(**kwargs)


def test_apply_filter_rejects_mismatched_stream():
    with pytest.raises(FilterConfigError, match="designed for"):
        apply_filter(emg(signal_type=RAW), lowpass(rate=200.0))
    with pytest.raises(FilterConfigError, match="filter is for"):
        apply_filter(emg(), lowpass())  # ENV stream, RAW filter


def test_filter_never_fills_missing_samples():
    v = emg(signal_type=RAW).values[:, 0].copy()
    v[300:310] = np.nan
    out = apply_filter(emg(signal_type=RAW, values=v), lowpass())
    assert np.isnan(out.values[300:310]).all()
    assert np.isfinite(out.values[:300]).all() and np.isfinite(out.values[310:]).all()
    assert out.provenance[-1].startswith("butterworth lowpass order 4")


def test_lowpass_attenuates_out_of_band_tone():
    t = np.arange(4000) / 1000.0
    v = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 200 * t)
    s = emg(rate=1000.0, signal_type=RAW, values=v, timestamps=t)
    out = apply_filter(s, lowpass(rate=1000.0)).values[500:-500, 0]
    residual = out - np.sin(2 * np.pi * 5 * t[500:-500])
    assert np.abs(residual).max() < 0.05


def test_zero_phase_runs_too_short_are_flagged_not_guessed():
    v = emg(signal_type=RAW).values[:, 0].copy()
    v[100], v[105] = np.nan, np.nan  # 4-sample run in between
    out = apply_filter(emg(signal_type=RAW, values=v), lowpass())
    assert (out.flags[101:105] == SampleFlag.INVALID).all() and np.isnan(out.values[101:105]).all()


def test_fixed_scale_is_recorded():
    out = apply_fixed_scale(emg(), 3.3 / 4095, 0.0, ("V",), "ADC counts to volts (synthetic)")
    assert "ADC counts" in out.provenance[-1]


# --- normalization and leakage ---------------------------------------------------------------

NAMES = ("f1", "f2")
TRAIN_X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, np.nan], [4.0, 40.0]])


@pytest.mark.parametrize("partition", [Partition.VALIDATION, Partition.TEST])
def test_normalization_cannot_be_fitted_outside_training(partition):
    with pytest.raises(LeakageError):
        fit_normalization(TRAIN_X, NAMES, partition, "zscore")


def test_validation_data_cannot_modify_fitted_parameters():
    params = fit_normalization(TRAIN_X, NAMES, Partition.TRAIN, "zscore")
    before = params.to_dict()
    apply_normalization(np.array([[1e6, -1e6]]), NAMES, params)  # wildly different validation rows
    assert params.to_dict() == before
    with pytest.raises(dataclasses.FrozenInstanceError):
        params.center = (0.0, 0.0)
    with pytest.raises(LeakageError):
        NormalizationParams("zscore", NAMES, (0, 0), (1, 1), 4, fitted_on=Partition.TEST)


def test_normalization_ignores_and_preserves_unavailable_values():
    params = fit_normalization(TRAIN_X, NAMES, Partition.TRAIN, "robust")
    out = apply_normalization(np.array([[2.5, np.nan]]), NAMES, params)
    assert np.isnan(out[0, 1]) and np.isfinite(out[0, 0])
    assert NormalizationParams.from_dict(params.to_dict()) == params


def test_normalization_rejects_constant_features_and_reordering():
    with pytest.raises(ValueError, match="constant"):
        fit_normalization(np.ones((3, 2)), NAMES, Partition.TRAIN, "zscore")
    params = fit_normalization(TRAIN_X, NAMES, Partition.TRAIN, "zscore")
    with pytest.raises(ValueError, match="order"):
        apply_normalization(TRAIN_X, ("f2", "f1"), params)
