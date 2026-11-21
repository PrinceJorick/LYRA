"""Stage 3 per-recording preparation: condition, segment, assess quality, extract, label.

Produces raw feature vectors with availability masks. No normalization or other statistic is
fitted here: that happens on training folds only, in Stage 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .annotations import AnnotationKind, LabelConfig, label_window, positive_overlap_s
from .features import FeatureConfig, extract_features, feature_names
from .preprocess import SimulatedEnvConfig, raw_to_simulated_env
<<<<<<< HEAD
from .quality import QualityConfig, QualityState, assess_modality, combine, inherit_source_quality
from .signals import EmgSignalType, Modality, Recording
from .sync import SyncReport
=======
from .quality import QualityConfig, QualityState, assess_modality, combine
from .signals import EmgSignalType, Modality, Recording
>>>>>>> 414a2140836e55fed760a18fff6c49fe2bc0082d
from .windows import WindowConfig, segment

QUALITY_CODE = {s: i for i, s in enumerate(QualityState)}  # stored as int8: VALID=0 .. UNSYNCHRONIZED=4


@dataclass(frozen=True)
class PrepConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    sim_env: SimulatedEnvConfig = field(default_factory=SimulatedEnvConfig)
    primary: FeatureConfig = field(  # Strategy D primary path
        default_factory=lambda: FeatureConfig("multimodal", EmgSignalType.SIMULATED_ENV_FROM_RAW))
    secondary: FeatureConfig = field(  # RAW research path, kept for side-by-side comparison
        default_factory=lambda: FeatureConfig("emg", EmgSignalType.RAW))


def prepare_recording(rec: Recording, cfg: PrepConfig) -> dict[str, np.ndarray]:
    """One row per window of the primary (SIMULATED_ENV_FROM_RAW + IMU) timeline. Every window is
    kept, including IGNORE-labelled and unusable ones, so the timeline stays continuous."""
    sim = raw_to_simulated_env(rec.emg, cfg.sim_env) if rec.emg is not None else None
    primary_rec = replace(rec, emg=sim) if sim is not None and sim.n else replace(rec, emg=None)
    artifacts = [a for a in rec.annotations if a.kind is AnnotationKind.ARTIFACT]
    cols: dict[str, list] = {k: [] for k in (
        "start_s", "end_s", "label", "fbtcs_overlap_s", "artifact_overlap", "overall", "emg_state", "imu_state",
        "raw_emg_state", "emg_coverage", "imu_coverage", "primary", "secondary")}
    both = (Modality.EMG, Modality.IMU)
    for w in segment(primary_rec, cfg.window, cfg.quality, both):
<<<<<<< HEAD
        raw_q = assess_modality(rec.emg, Modality.EMG, w.start_s, w.end_s, cfg.quality)
        q = dict(w.quality.per_modality)
        q[Modality.EMG] = inherit_source_quality(q[Modality.EMG], raw_q)  # simulated ENV never beats its RAW source
        w = replace(w, quality=combine(q, both, SyncReport(not w.quality.sync_reasons, w.quality.sync_reasons)))
=======
        q = w.quality.per_modality
        raw_q = assess_modality(rec.emg, Modality.EMG, w.start_s, w.end_s, cfg.quality)
>>>>>>> 414a2140836e55fed760a18fff6c49fe2bc0082d
        raw_w = replace(w, emg=rec.emg.slice_time(w.start_s, w.end_s) if rec.emg is not None else None,
                        quality=combine({Modality.EMG: raw_q, Modality.IMU: q[Modality.IMU]}, (Modality.EMG,), None))
        cols["start_s"].append(w.start_s)
        cols["end_s"].append(w.end_s)
        cols["label"].append(int(label_window(w.start_s, w.end_s, rec.annotations, cfg.labels)))
        cols["fbtcs_overlap_s"].append(positive_overlap_s(w.start_s, w.end_s, rec.annotations, cfg.labels))
        cols["artifact_overlap"].append(any(a.overlaps(w.start_s, w.end_s) for a in artifacts))
        cols["overall"].append(QUALITY_CODE[w.quality.overall])
        cols["emg_state"].append(QUALITY_CODE[q[Modality.EMG].state])
        cols["imu_state"].append(QUALITY_CODE[q[Modality.IMU].state])
        cols["raw_emg_state"].append(QUALITY_CODE[raw_q.state])
        cols["emg_coverage"].append(q[Modality.EMG].coverage)
        cols["imu_coverage"].append(q[Modality.IMU].coverage)
        cols["primary"].append(extract_features(w, cfg.primary).values)
        cols["secondary"].append(extract_features(raw_w, cfg.secondary).values)
    n_p, n_s = len(feature_names(cfg.primary)), len(feature_names(cfg.secondary))
    out = {
        "start_s": np.array(cols["start_s"], np.float64), "end_s": np.array(cols["end_s"], np.float64),
        "label": np.array(cols["label"], np.int8), "fbtcs_overlap_s": np.array(cols["fbtcs_overlap_s"], np.float32),
        "artifact_overlap": np.array(cols["artifact_overlap"], bool),
        **{k: np.array(cols[k], np.int8) for k in ("overall", "emg_state", "imu_state", "raw_emg_state")},
        "emg_coverage": np.array(cols["emg_coverage"], np.float32),
        "imu_coverage": np.array(cols["imu_coverage"], np.float32),
        "primary": np.array(cols["primary"], np.float32).reshape(-1, n_p),
        "secondary": np.array(cols["secondary"], np.float32).reshape(-1, n_s),
    }
    out["primary_available"] = ~np.isnan(out["primary"])
    out["secondary_available"] = ~np.isnan(out["secondary"])
    out.update(primary_emg_metadata(primary_rec))
    return out


def primary_emg_metadata(primary_rec: Recording) -> dict[str, np.ndarray]:
    """Signal type and full provenance of the EMG stream behind the primary features."""
    e = primary_rec.emg
    return {"primary_emg_signal_type": np.array(str(e.signal_type) if e is not None else ""),
            "primary_emg_provenance": np.array("\n".join(e.provenance) if e is not None else "")}
