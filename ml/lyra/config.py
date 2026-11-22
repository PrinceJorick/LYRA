"""Single source of truth for the LYRA ESP32-S3 pipeline (ENV path, locked decision D12).

Everything that changes a result is here and is hashed into cache keys and the experiment
manifest. The dataset is read-only; every generated file lives outside pj-training-data/.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
from lyra_mlm.preprocess import SimulatedEnvConfig
from lyra_mlm.quality import QualityConfig
from lyra_mlm.seizeit2 import ACC_SCALE_VALID, EMG_LABEL, GYRO_DPS_PER_UNIT, IMU_LABELS

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "pj-training-data" / "ds005873"  # OpenNeuro ds005873 v1.1.0, READ-ONLY
DATASET_ID, DATASET_VERSION = "ds005873", "1.1.0"
CACHE = REPO / "ml" / "lyra" / "cache"  # git-ignored: temporary EDFs, feature caches, fold predictions
RUNS = REPO / "ml" / "lyra" / "runs"  # git-ignored: experiment runs
REPORTS = REPO / "reports" / "lyra"
FIRMWARE = REPO / "firmware" / "model"
SPEC = json.loads((Path(__file__).with_name("feature_spec.json")).read_text())

# ---- Official SeizeIT2 partition: development pool and untouched final evaluation cohort ----
RID = re.compile(r"^sub-(\d{3})_ses-(\d{2})_task-([A-Za-z0-9]+)_run-(\d+)$")
DEV = range(1, 97)  # sub-001..sub-096
TEST = range(97, 126)  # sub-097..sub-125: never loaded by development code


def subject_number(recording_or_subject: str) -> int:
    return int(recording_or_subject.split("_")[0].removeprefix("sub-"))


def is_dev(recording_or_subject: str) -> bool:
    return subject_number(recording_or_subject) in DEV


# ---- Labels ----
TARGET_CODES = ("sz_foc_f2b", "sz_gen_m_tonicClonic", "sz_uo_m_tonicClonic")  # target convulsive seizures
ARTIFACT_CODES = ("impd",)  # impedance checks: windows overlapping them are excluded
BACKGROUND_CODES = ("bckg",)
POS_OVERLAP = 0.5  # positive if >= 50% of the window overlaps a target event; 0 < overlap < 50%: excluded

# ---- Signals (training side) ----
EMG_CHANNEL, IMU_CHANNELS = EMG_LABEL, IMU_LABELS
ENV = SimulatedEnvConfig()  # Kyle's MyoWare ENV model: 1st-order HPF 20.8 Hz, |x|, 1st-order LPF 3.6 Hz
ENV_RATE, IMU_RATE = 50, 25
GRAVITY = 9.80665  # m/s^2 per g
WARMUP_S = 1.0  # windows starting before this are excluded (filter state starts at zero)
MAX_SAT_FRACTION = QualityConfig().max_saturated_fraction  # lyra_mlm default: > 5% saturated -> invalid

# ---- Windows: (length s, hop s). 5 s uses 2.48 s, the nearest hop with integer samples at 25 and 50 Hz ----
WINDOWS = {"w2": (2.0, 1.0), "w4": (4.0, 2.0), "w5": (5.0, 2.48)}

FEATURES = tuple(f["name"] for f in SPEC["features"])
EPS = float(SPEC["eps"])
MODALITIES = {
    "emg": tuple(i for i, n in enumerate(FEATURES) if n.startswith("emg_")),
    "imu": tuple(i for i, n in enumerate(FEATURES) if n.startswith("imu_")),
    "fusion": tuple(range(len(FEATURES))),
}

# ---- Training / CV ----
SEED = 20261122
OUTER_FOLDS = 5  # GroupKFold by subject on the development pool
INNER_FOLDS = 3  # patient-wise inner CV inside each training part (calibration only)
NEG_FRACTION, NEG_MIN = 0.05, 50  # negatives subsampled per recording for FITTING only
SVC_MAX_TRAIN = 6000  # RBF SVC is O(n^2): fitted on a deterministic patient-stratified subsample

# ---- Event state machine and evaluation ----
THRESHOLDS_SPEC = tuple(float(x) for x in np.round(np.arange(0.20, 0.8001, 0.05), 2))  # task brief grid
# Platt calibration targets the true window prevalence (~1:6,500), so calibrated probabilities rarely exceed
# ~0.1 and the brief's 0.20-0.80 grid cannot fire (smoke run, 2026-11-22). The grid is extended downwards;
# the FAR/latency criteria are unchanged and the spec-grid result is reported separately.
THRESHOLDS_EXTENDED = (0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.15)
THRESHOLDS = tuple(sorted(THRESHOLDS_EXTENDED + THRESHOLDS_SPEC))
N_OPEN, M_CLOSE = (2, 3), (2, 3)
REFRACTORY_S = 60.0
MAX_GAP_S = 6.0  # D17: a gap longer than this between valid windows breaks "consecutive"
MATCH_PRE_S, MATCH_POST_S = 10.0, 30.0  # alert in [onset - 10 s, offset + 30 s] detects the event
MAX_FAR_PER_24H = 1.0  # owner decision (2026-11-22)
MAX_LATENCY_S = 20.0
FLASH_TARGET, SRAM_TARGET = 60 * 1024, 8 * 1024


def feature_config() -> dict:
    """Everything that determines cached windows/features (hashed into the cache key)."""
    return {"dataset": [DATASET_ID, DATASET_VERSION], "spec": SPEC["version"], "features": FEATURES,
            "env": {"hpf_hz": ENV.hpf_hz, "lpf_hz": ENV.lpf_hz, "order": ENV.order, "rate_hz": ENV_RATE,
                    "decimation": "causal mean over each 20 ms bin"},
            "imu": {"rate_hz": IMU_RATE, "acc_scale_valid": ACC_SCALE_VALID, "gyro_dps_per_unit": GYRO_DPS_PER_UNIT,
                    "gravity": GRAVITY},
            "windows": WINDOWS, "warmup_s": WARMUP_S, "max_sat_fraction": MAX_SAT_FRACTION,
            "labels": {"target": TARGET_CODES, "artifact": ARTIFACT_CODES, "pos_overlap": POS_OVERLAP}, "eps": EPS,
            "channels": [EMG_CHANNEL, *IMU_CHANNELS]}


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


FEATURE_HASH = config_hash(feature_config())
FEATURE_DIR = CACHE / "features" / FEATURE_HASH
