"""SYNTHETIC software fixtures for tests. Generated signals, not physiological data.

StubScorePredictor is a TEST-ONLY stand-in that returns fixed scores so pipeline wiring can be
exercised. It is not a model and not a seizure detector; it must never be exported.
"""

from __future__ import annotations

import numpy as np

from lyra_mlm.features import FeatureConfig, feature_names
from lyra_mlm.model import ModelConfig, ModelFamily, Prediction, PredictionStatus
from lyra_mlm.quality import ModalityQuality, QualityState, WindowQuality
from lyra_mlm.signals import IMU_CHANNELS, IMU_UNITS, DataOrigin, EmgSignalType, Modality, Recording, SignalStream
from lyra_mlm.windows import WindowConfig

EMG_RATE = 100.0  # synthetic choice; LYRA's MyoWare ENV acquisition rate is not yet specified
IMU_RATE = 25.0  # BMI270 initial ODR
_DEFAULT = object()


def emg(seconds=12.0, rate=EMG_RATE, signal_type=EmgSignalType.ENV, bandwidth_hz=20.0, values=None,
        timestamps=None, flags=None) -> SignalStream:
    n = int(round(seconds * rate))
    t = np.arange(n) / rate if timestamps is None else timestamps
    if values is None:
        values = 0.5 + 0.2 * np.sin(2 * np.pi * 2.0 * t) + 0.01 * np.random.default_rng(0).standard_normal(len(t))
    return SignalStream(Modality.EMG, t, values, rate, (signal_type.lower(),), ("V",), DataOrigin.SYNTHETIC,
                        signal_type, bandwidth_hz, flags)


def imu(seconds=12.0, rate=IMU_RATE, values=None, timestamps=None, freq_hz=3.0) -> SignalStream:
    n = int(round(seconds * rate))
    t = np.arange(n) / rate if timestamps is None else timestamps
    if values is None:
        rng = np.random.default_rng(1)
        osc = np.sin(2 * np.pi * freq_hz * t)
        values = np.column_stack([
            0.02 * osc, 0.01 * rng.standard_normal(len(t)), 1.0 + 0.05 * osc,
            10 * osc, rng.standard_normal(len(t)), rng.standard_normal(len(t)),
        ])
    return SignalStream(Modality.IMU, t, values, rate, IMU_CHANNELS, IMU_UNITS, DataOrigin.SYNTHETIC)


def recording(emg_stream=_DEFAULT, imu_stream=_DEFAULT, annotations=()) -> Recording:
    return Recording("SYN-P01", "SYN-R01", DataOrigin.SYNTHETIC,
                     emg() if emg_stream is _DEFAULT else emg_stream,
                     imu() if imu_stream is _DEFAULT else imu_stream,
                     placement="deltoid (synthetic)", annotations=annotations)


def model_config(mode="multimodal", model_id="stub", normalization=None) -> ModelConfig:
    fc = FeatureConfig(mode, None if mode == "imu" else EmgSignalType.ENV)
    return ModelConfig(model_id, ModelFamily.RANDOM_FOREST, fc, feature_names(fc), WindowConfig(), 0.5,
                       normalization=normalization)


class StubScorePredictor:
    """TEST-ONLY: returns a fixed synthetic score. Not a model."""

    def __init__(self, config: ModelConfig, score: float = 0.9):
        self.config, self.score, self.calls, self.last_x = config, score, 0, None

    def predict_score(self, features: np.ndarray) -> float:
        self.calls += 1
        self.last_x = features
        return self.score


def write_edf(path, signals, record_s=1.0, variant="EDF+C", onsets=None):
    """Write a small 16-bit EDF+ file. signals: [(label, unit, rate_hz, values)]. SYNTHETIC."""
    n_rec = int(len(signals[0][3]) / (signals[0][2] * record_s))
    onsets = np.arange(n_rec) * record_s if onsets is None else np.asarray(onsets)
    spr = [int(round(rate * record_s)) for _, _, rate, _ in signals] + [30]
    digs, rows = [], []
    for label, unit, _, v in signals:
        lo, hi = float(np.min(v)) - 1e-6, float(np.max(v)) + 1e-6
        digs.append(np.round((np.asarray(v) - lo) / (hi - lo) * 65535 - 32768).astype("<i2"))
        rows.append((label, unit, f"{lo:.6g}"[:8], f"{hi:.6g}"[:8], "-32768", "32767"))
    rows.append(("EDF Annotations", "", "-1", "1", "-32768", "32767"))
    ns = len(rows)
    pad = lambda s, w: s.ljust(w)[:w].encode("latin-1")  # noqa: E731
    head = (pad("0", 8) + pad("X", 80) + pad("Startdate X", 80) + pad("01.01.00", 8) + pad("00.00.00", 8)
            + pad(str(256 * (ns + 1)), 8) + pad(variant, 44) + pad(str(n_rec), 8) + pad(f"{record_s:g}", 8)
            + pad(str(ns), 4))
    for col, width in ((0, 16), (None, 80), (1, 8), (2, 8), (3, 8), (4, 8), (5, 8), (None, 80)):
        head += b"".join(pad("" if col is None else r[col], width) for r in rows)
    head += b"".join(pad(str(s), 8) for s in spr) + pad("", 32) * ns
    body = bytearray()
    for k in range(n_rec):
        for d, n in zip(digs, spr[:-1]):
            body += d[k * n:(k + 1) * n].tobytes()
        body += f"+{onsets[k]:g}\x14\x14\x00".encode().ljust(60, b"\x00")
    with open(path, "wb") as f:
        f.write(head + bytes(body))


def window_quality(emg_state=QualityState.VALID, imu_state=QualityState.VALID) -> WindowQuality:
    per = {Modality.EMG: ModalityQuality(Modality.EMG, emg_state, (), 1.0),
           Modality.IMU: ModalityQuality(Modality.IMU, imu_state, (), 1.0)}
    worst = next(s for s in (QualityState.UNSYNCHRONIZED, QualityState.INVALID, QualityState.MISSING,
                             QualityState.DEGRADED, QualityState.VALID) if s in (emg_state, imu_state))
    return WindowQuality(per, (Modality.EMG, Modality.IMU), worst)


def pred(ts: float, score: float | None = None, status=PredictionStatus.VALID, quality=None) -> Prediction:
    q = quality or window_quality()
    if status is PredictionStatus.SUPPRESSED:
        return Prediction(ts, status, None, None, q.available_modalities, q, None, "synthetic suppressed")
    return Prediction(ts, status, score, int(score >= 0.5), q.available_modalities, q, "stub")
