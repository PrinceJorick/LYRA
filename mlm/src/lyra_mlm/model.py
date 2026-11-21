"""Model, training and embedded-export contracts. No model is implemented or trained here.

Model roles (planning, not results): logistic regression = baseline, random forest = primary
embedded candidate (initial research range 25-50 trees, depth 6-8), XGBoost = offline
comparison, SVM = optional comparison.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .evaluation import LeakageError, Partition
from .features import FeatureConfig, feature_names
from .preprocess import NormalizationParams
from .quality import WindowQuality
from .signals import EmgSignalType, Modality
from .windows import WindowConfig


class ModelFamily(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    SVM = "svm"


class PredictionStatus(StrEnum):
    VALID = "VALID"
    DEGRADED = "DEGRADED"  # produced under an explicit degraded-quality policy
    SUPPRESSED = "SUPPRESSED"  # no prediction; never a negative


@dataclass(frozen=True)
class Prediction:
    timestamp_s: float  # window end, recording/device clock
    status: PredictionStatus
    score: float | None
    predicted_class: int | None  # 1 = window resembles the trained possible-event pattern. Not a diagnosis.
    available_modalities: tuple[Modality, ...]
    quality: WindowQuality
    model_id: str | None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status is PredictionStatus.SUPPRESSED:
            if self.score is not None or self.predicted_class is not None:
                raise ValueError("a suppressed prediction has no score and no class")
        elif not (self.score is not None and math.isfinite(self.score) and 0 <= self.score <= 1
                  and self.predicted_class in (0, 1)):
            raise ValueError("a non-suppressed prediction needs a score in [0, 1] and a class in {0, 1}")


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    family: ModelFamily
    feature_config: FeatureConfig
    feature_names: tuple[str, ...]  # must equal feature_names(feature_config): fixed order
    window: WindowConfig
    decision_threshold: float  # chosen on training data only, in the training session
    normalization: NormalizationParams | None = None  # fitted on the training partition only
    hyperparameters: Mapping[str, Any] = field(default_factory=dict)
    trained: bool = False
    training_record: Mapping[str, Any] = field(default_factory=dict)  # dataset version, split, commit

    def __post_init__(self) -> None:
        if tuple(self.feature_names) != feature_names(self.feature_config):
            raise ValueError("feature_names do not match the feature schema for this feature_config")
        if not 0 <= self.decision_threshold <= 1:
            raise ValueError("decision_threshold must be in [0, 1]")
        if self.normalization is not None and self.normalization.feature_names != tuple(self.feature_names):
            raise ValueError("normalization was fitted on a different feature order")

    def to_json(self) -> str:
        d = asdict(self)
        d["normalization"] = self.normalization.to_dict() if self.normalization else None
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ModelConfig:
        d = json.loads(text)
        fc = d["feature_config"]
        t = fc["emg_signal_type"]
        return cls(
            model_id=d["model_id"], family=ModelFamily(d["family"]),
            feature_config=FeatureConfig(
                mode=fc["mode"], emg_signal_type=EmgSignalType(t) if t else None,
                movement_band_hz=tuple(fc["movement_band_hz"]), rhythm_lag_s=tuple(fc["rhythm_lag_s"]),
                max_xcorr_lag_s=fc["max_xcorr_lag_s"], min_paired_samples=fc["min_paired_samples"],
                gap_factor=fc["gap_factor"]),
            feature_names=tuple(d["feature_names"]), window=WindowConfig(**d["window"]),
            decision_threshold=d["decision_threshold"],
            normalization=NormalizationParams.from_dict(d["normalization"]) if d["normalization"] else None,
            hyperparameters=d["hyperparameters"], trained=d["trained"], training_record=d["training_record"],
        )


class ScorePredictor(Protocol):
    config: ModelConfig

    def predict_score(self, features: np.ndarray) -> float:
        """features: one row in config.feature_names order, every feature available (and
        normalized if config.normalization is set). Returns a score in [0, 1]."""
        ...


@dataclass(frozen=True, eq=False)
class FeatureDataset:
    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    participant_ids: tuple[str, ...]  # one per row, for participant-grouped cross-validation
    partition: Partition

    def __post_init__(self) -> None:
        if not (len(self.x) == len(self.y) == len(self.participant_ids)):
            raise ValueError("x, y and participant_ids must have one entry per window")


class Trainer(Protocol):
    def train(self, data: FeatureDataset, config: ModelConfig) -> ScorePredictor:
        """Deferred to the training session. Implementations must call require_training_partition."""
        ...


def require_training_partition(data: FeatureDataset) -> None:
    if data.partition is not Partition.TRAIN:
        raise LeakageError(f"refusing to train on the {data.partition} partition")


@dataclass(frozen=True)
class ExportManifest:
    model_id: str
    converter: str  # candidates to investigate: emlearn, micromlgen, m2cgen (family support varies)
    converter_version: str
    feature_names: tuple[str, ...]  # the C code must compute features in exactly this order
    normalization: NormalizationParams | None
    files: tuple[str, ...]
    c_standard: str = "C99"
    target: str = "ESP32-S3"


class EmbeddedExporter(Protocol):
    def export(self, predictor: ScorePredictor, out_dir: Path) -> ExportManifest: ...


def ensure_exportable(config: ModelConfig) -> None:
    if not config.trained:
        raise ValueError(f"refusing to export untrained model {config.model_id!r}")


@dataclass(frozen=True)
class EmbeddedBenchmark:
    """Measured on real ESP32-S3 hardware in a future session. None means not measured."""

    flash_bytes: int | None = None
    ram_bytes: int | None = None
    inference_latency_us: float | None = None
    feature_extraction_us: float | None = None
    total_processing_us: float | None = None
    battery_current_ma: float | None = None
    parity_max_abs_score_diff: float | None = None  # Python vs C on the same feature rows
    parity_rows: Sequence[int] = ()
