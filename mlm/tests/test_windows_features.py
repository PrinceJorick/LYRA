import math

import numpy as np
import pytest
from synthetic import emg, imu, recording

from lyra_mlm.annotations import Annotation, AnnotationKind
from lyra_mlm.features import FeatureConfig, extract_features, feature_names
from lyra_mlm.quality import QualityConfig
from lyra_mlm.signals import EmgSignalType, Modality
from lyra_mlm.windows import WindowConfig, segment

ENV, RAW = EmgSignalType.ENV, EmgSignalType.RAW
BOTH = (Modality.EMG, Modality.IMU)
MM = FeatureConfig("multimodal", ENV)


def windows(rec=None, required=BOTH):
    return list(segment(rec or recording(), WindowConfig(), QualityConfig(), required))


def test_window_boundaries_and_metadata():
    ws = windows()
    assert [w.start_s for w in ws] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert all(w.end_s - w.start_s == 4.0 for w in ws)
    assert {(w.participant_id, w.recording_id) for w in ws} == {("SYN-P01", "SYN-R01")}


def test_window_config_validation():
    assert WindowConfig().overlap == 0.5
    with pytest.raises(ValueError):
        WindowConfig(length_s=4.0, step_s=5.0)


def test_annotation_references_are_attached_not_labels():
    a = Annotation("syn-1", 5.0, 2.0, AnnotationKind.UNMAPPED, "synthetic")
    refs = [w.annotation_refs for w in windows(recording(annotations=(a,)))]
    assert refs == [(), ("syn-1",), ("syn-1",), ("syn-1",), ()]


def test_feature_schema_is_deterministic_and_signal_type_specific():
    env_names = feature_names(FeatureConfig("emg", ENV))
    raw_names = feature_names(FeatureConfig("emg", RAW))
    assert env_names == feature_names(FeatureConfig("emg", ENV))
    assert env_names[0] == "emg_env_rms" and raw_names[0] == "emg_raw_rms"
    assert "emg_env_autocorr_peak" in env_names and len(raw_names) == 5
    with pytest.raises(ValueError, match="ENV path"):
        FeatureConfig("multimodal", RAW)
    with pytest.raises(ValueError):
        FeatureConfig("imu", ENV)


def test_feature_vector_matches_schema():
    fv = extract_features(windows()[0], MM)
    assert fv.names == feature_names(MM) and fv.values.shape == (len(fv.names),)
    assert fv.available.all(), fv.unavailable_reasons


def test_unavailable_is_distinct_from_valid_zero():
    t_e, t_i = np.arange(1200) / 100.0, np.arange(300) / 25.0
    env_v = 0.5 + 0.2 * np.sin(2 * np.pi * 2 * t_e)
    imu_v = np.column_stack([np.zeros(300), np.zeros(300), 1 + 0.1 * np.sin(2 * np.pi * 2 * t_i),
                             np.sin(2 * np.pi * 2 * t_i), np.zeros(300), np.zeros(300)])
    fv = extract_features(windows(recording(emg(values=env_v), imu(values=imu_v)))[0], MM)
    i = fv.names.index("xmod_lag_s")
    assert fv.values[i] == 0.0 and fv.available[i]  # a real, available zero

    fv = extract_features(windows(recording(emg_stream=None))[0], MM)
    emg_idx = [k for k, n in enumerate(fv.names) if n.startswith(("emg_", "xmod_"))]
    assert np.isnan(fv.values[emg_idx]).all() and not fv.available[emg_idx].any()
    assert fv.unavailable_reasons["emg_env_rms"] == "quality: emg MISSING"
    assert fv.available[fv.names.index("imu_acc_mag_mean")]


def test_env_autocorrelation_requires_declared_bandwidth():
    fv = extract_features(windows(recording(emg(bandwidth_hz=None)))[0], FeatureConfig("emg", ENV))
    assert "bandwidth not declared" in fv.unavailable_reasons["emg_env_autocorr_peak"]
    fv = extract_features(windows()[0], FeatureConfig("emg", ENV))
    assert fv.values[fv.names.index("emg_env_autocorr_peak")] > 0.8  # synthetic 2 Hz envelope rhythm


def test_dominant_frequency_matches_synthetic_oscillation():
    fv = extract_features(windows(recording(imu_stream=imu(freq_hz=3.0)), (Modality.IMU,))[0],
                          FeatureConfig("imu"))
    assert fv.values[fv.names.index("imu_acc_dominant_freq_hz")] == pytest.approx(3.0, abs=0.25)


def test_fusion_preserves_quality_and_identity_metadata():
    w = windows(recording(emg_stream=None))[0]
    fv = extract_features(w, MM)
    assert fv.quality is w.quality and fv.available_modalities == (Modality.IMU,)
    assert (fv.participant_id, fv.recording_id, fv.start_s, fv.end_s) == ("SYN-P01", "SYN-R01", 0.0, 4.0)


def test_emg_signal_type_mismatch_is_a_contract_error():
    with pytest.raises(ValueError, match="expects EMG"):
        extract_features(windows(recording(emg(signal_type=RAW)))[0], MM)


def test_imu_only_mode_ignores_emg():
    fv = extract_features(windows(recording(emg_stream=None), (Modality.IMU,))[0], FeatureConfig("imu"))
    assert fv.available.all() and all(math.isfinite(v) for v in fv.values)
