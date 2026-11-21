import numpy as np
import pytest
from synthetic import StubScorePredictor, emg, imu, model_config, recording

from lyra_mlm.features import extract_features
from lyra_mlm.inference import InferenceConfig, QualityGatedInference
from lyra_mlm.model import Prediction, PredictionStatus
from lyra_mlm.preprocess import NormalizationParams
from lyra_mlm.quality import QualityConfig, QualityState
from lyra_mlm.signals import Modality
from lyra_mlm.windows import WindowConfig, segment

BOTH = (Modality.EMG, Modality.IMU)
SUPPRESS = InferenceConfig(degraded_policy="suppress", missing_modality_policy="suppress")
FALLBACK = InferenceConfig(degraded_policy="proceed", missing_modality_policy="fallback_unimodal")


def first_window(rec=None, required=BOTH):
    return next(segment(rec or recording(), WindowConfig(), QualityConfig(), required))


def test_valid_window_gives_valid_prediction():
    stub = StubScorePredictor(model_config(), 0.9)
    p = QualityGatedInference(SUPPRESS, stub).process(first_window())
    assert (p.status, p.score, p.predicted_class, p.model_id) == (PredictionStatus.VALID, 0.9, 1, "stub")
    assert p.available_modalities == BOTH and p.timestamp_s == 4.0


def test_missing_modality_with_suppress_policy():
    stub = StubScorePredictor(model_config())
    p = QualityGatedInference(SUPPRESS, stub).process(first_window(recording(emg_stream=None)))
    assert p.status is PredictionStatus.SUPPRESSED and p.score is None and p.predicted_class is None
    assert "emg: MISSING" in p.reason and stub.calls == 0
    assert p.available_modalities == (Modality.IMU,)


def test_missing_modality_falls_back_to_configured_unimodal_model():
    imu_stub = StubScorePredictor(model_config("imu", model_id="imu-stub"), 0.2)
    inf = QualityGatedInference(FALLBACK, StubScorePredictor(model_config()), {Modality.IMU: imu_stub})
    p = inf.process(first_window(recording(emg_stream=None)))
    assert (p.status, p.model_id, p.predicted_class) == (PredictionStatus.VALID, "imu-stub", 0)
    assert "fallback to imu-only model" in p.reason


def test_fallback_policy_without_a_fallback_model_suppresses():
    p = QualityGatedInference(FALLBACK, StubScorePredictor(model_config())).process(
        first_window(recording(imu_stream=None)))
    assert p.status is PredictionStatus.SUPPRESSED


def test_degraded_policies():
    v = imu().values.copy()
    v[:10] = np.nan  # 10% of the first window
    w = first_window(recording(imu_stream=imu(values=v)), (Modality.IMU,))
    assert w.quality.per_modality[Modality.IMU].state is QualityState.DEGRADED
    stub = StubScorePredictor(model_config("imu"), 0.7)
    assert QualityGatedInference(FALLBACK, stub).process(w).status is PredictionStatus.DEGRADED
    assert QualityGatedInference(SUPPRESS, stub).process(w).status is PredictionStatus.SUPPRESSED


def test_unavailable_features_suppress_instead_of_guessing():
    p = QualityGatedInference(SUPPRESS, StubScorePredictor(model_config())).process(
        first_window(recording(emg(bandwidth_hz=None))))
    assert p.status is PredictionStatus.SUPPRESSED
    assert "emg_env_autocorr_peak" in p.reason and "xmod_coherence_band" in p.reason


def test_unsynchronized_modalities_suppress_multimodal():
    rec = recording(imu_stream=imu(timestamps=np.arange(300) / 26.0))
    p = QualityGatedInference(FALLBACK, StubScorePredictor(model_config())).process(first_window(rec))
    assert p.status is PredictionStatus.SUPPRESSED and "unsynchronized" in p.reason


def test_fallback_configuration_is_validated():
    with pytest.raises(ValueError, match="multimodal primary"):
        QualityGatedInference(FALLBACK, StubScorePredictor(model_config("imu")),
                              {Modality.IMU: StubScorePredictor(model_config("imu"))})
    with pytest.raises(ValueError, match="imu-only"):
        QualityGatedInference(FALLBACK, StubScorePredictor(model_config()),
                              {Modality.IMU: StubScorePredictor(model_config("emg"))})


def test_fixed_normalization_is_applied_before_scoring():
    w = first_window()
    names = model_config().feature_names
    params = NormalizationParams("zscore", names, tuple([1.0] * len(names)), tuple([2.0] * len(names)), 10)
    stub = StubScorePredictor(model_config(normalization=params))
    QualityGatedInference(SUPPRESS, stub).process(w)
    raw = extract_features(w, stub.config.feature_config).values
    np.testing.assert_allclose(stub.last_x, (raw - 1.0) / 2.0)


def test_prediction_schema_rules():
    q = first_window().quality
    with pytest.raises(ValueError, match="suppressed"):
        Prediction(1.0, PredictionStatus.SUPPRESSED, 0.0, None, (), q, None)
    with pytest.raises(ValueError, match="score"):
        Prediction(1.0, PredictionStatus.VALID, None, None, (), q, "m")
    with pytest.raises(ValueError, match="score"):
        Prediction(1.0, PredictionStatus.VALID, 1.5, 1, (), q, "m")
