"""Quality-gated, window-by-window inference (the interface the ESP32-S3 port will mirror).

Routing:
- Required modalities usable -> primary model. DEGRADED follows degraded_policy and keeps the
  DEGRADED status.
- Multimodal primary with exactly one modality unusable -> missing_modality_policy: "suppress",
  or "fallback_unimodal" when a model for the remaining modality is configured.
- Cross-modality clock problems, unusable required modalities or unavailable features ->
  SUPPRESSED (score None, class None). Suppressed is never a negative prediction.
Sample buffering into windows is the firmware's job and is not implemented here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .features import MODE_MODALITIES, extract_features
from .model import Prediction, PredictionStatus, ScorePredictor
from .preprocess import apply_normalization
from .quality import USABLE, QualityState, WindowQuality
from .signals import Modality
from .windows import Window


@dataclass(frozen=True)
class InferenceConfig:
    degraded_policy: Literal["proceed", "suppress"]  # no default: must be chosen explicitly
    missing_modality_policy: Literal["suppress", "fallback_unimodal"]


class QualityGatedInference:
    def __init__(self, cfg: InferenceConfig, primary: ScorePredictor,
                 fallbacks: Mapping[Modality, ScorePredictor] | None = None):
        self.cfg, self.primary, self.fallbacks = cfg, primary, dict(fallbacks or {})
        if self.fallbacks and primary.config.feature_config.mode != "multimodal":
            raise ValueError("fallback models only apply to a multimodal primary model")
        for m, p in self.fallbacks.items():
            if p.config.feature_config.mode != m.value:
                raise ValueError(f"fallback for {m} must be a {m}-only model")

    def _route(self, q: WindowQuality) -> tuple[ScorePredictor | None, str]:
        required = MODE_MODALITIES[self.primary.config.feature_config.mode]
        if len(required) > 1 and q.overall is QualityState.UNSYNCHRONIZED and q.sync_reasons:
            return None, "modalities unsynchronized: " + "; ".join(q.sync_reasons)
        unusable = [m for m in required if q.per_modality[m].state not in USABLE]
        predictor, reason = self.primary, ""
        if unusable:
            detail = ", ".join(f"{m}: {q.per_modality[m].state}" for m in unusable)
            remaining = [m for m in required if m not in unusable]
            if (len(remaining) == 1 and self.cfg.missing_modality_policy == "fallback_unimodal"
                    and remaining[0] in self.fallbacks):
                predictor, reason = self.fallbacks[remaining[0]], f"fallback to {remaining[0]}-only model ({detail})"
            else:
                return None, detail
        used = MODE_MODALITIES[predictor.config.feature_config.mode]
        degraded = [m for m in used if q.per_modality[m].state is QualityState.DEGRADED]
        if degraded and self.cfg.degraded_policy == "suppress":
            return None, f"degraded {', '.join(degraded)} (degraded_policy=suppress)"
        return predictor, reason

    def process(self, window: Window) -> Prediction:
        q = window.quality

        def suppressed(reason: str) -> Prediction:
            return Prediction(window.end_s, PredictionStatus.SUPPRESSED, None, None,
                              q.available_modalities, q, None, reason)

        predictor, reason = self._route(q)
        if predictor is None:
            return suppressed(reason)
        cfg = predictor.config
        fv = extract_features(window, cfg.feature_config)
        if not fv.available.all():
            missing = [n for n, ok in zip(fv.names, fv.available) if not ok]
            return suppressed(f"features unavailable: {', '.join(missing)}")
        x = fv.values if cfg.normalization is None else apply_normalization(fv.values, fv.names, cfg.normalization)
        score = float(predictor.predict_score(x))
        used = MODE_MODALITIES[cfg.feature_config.mode]
        status = (PredictionStatus.DEGRADED if any(q.per_modality[m].state is QualityState.DEGRADED for m in used)
                  else PredictionStatus.VALID)
        return Prediction(window.end_s, status, score, int(score >= cfg.decision_threshold),
                          q.available_modalities, q, cfg.model_id, reason)
