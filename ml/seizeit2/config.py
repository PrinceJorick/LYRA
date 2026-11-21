"""Single source of truth for the SeizeIT2 pipeline: paths, signal contract, labels, splits, models.

Everything that changes a result lives here and is copied into models/seizeit2/config.json
and metadata.json at export time, so every reported number can be traced to these values.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "pj-training-data" / "ds005873"  # BIDS SeizeIT2 (OpenNeuro ds005873), local only
CACHE = REPO / "pj-training-data" / "cache"  # derived features; inside the git-ignored data dir
RUNS_DIR = REPO / "ml" / "seizeit2" / "runs"  # experiment outputs and checkpoints (git-ignored)
MODEL_DIR = REPO / "models" / "seizeit2"  # deployable, git-tracked artefacts

# Annexed EDF content is fetched from the dataset's own public CC0 bucket and verified against
# the SHA256 in the git-annex key before use. Nothing is ever uploaded.
S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds005873"
DOWNLOAD_WORKERS = 8

# ---- Signal contract (what the model consumes; LYRA must deliver the same) ----
EMG_FS = 256  # Hz, SeizeIT2 "EMG SD" channel, left deltoid, µV
IMU_FS = 25  # Hz, SeizeIT2 "ECGEMG SD" module (left chest); ACC rescaled to g per run, GYR unit unknown
EMG_LABEL = "EMG SD"
ACC_LABELS = ("ECGEMG SD ACC X", "ECGEMG SD ACC Y", "ECGEMG SD ACC Z")
GYR_LABELS = ("ECGEMG SD GYR A", "ECGEMG SD GYR B", "ECGEMG SD GYR C")

HOP_S = 2  # one window every 2 s, right-aligned to whole seconds of the recording
WINDOWS_S = (5, 10)  # both are extracted so window length is a controlled experiment

# ---- Labels (see README: problem definition) ----
TARGET_LABELS = ("sz_foc_f2b",)  # focal-to-bilateral tonic-clonic = the convulsive target
POS_MIN_OVERLAP = 0.5  # window positive if >= 50% of it lies inside a target seizure
TARGET_GUARD_S = (30, 120)  # non-positive windows this close before/after a target seizure: ignored
OOS_GUARD_S = (30, 60)  # same for out-of-scope seizures (never used as negatives)

# ---- Quality ----
EMG_SAT_MAX_FRAC = 0.05  # window unusable if more than 5% of EMG samples sit at the ADC rail
FLAT_STD = 1e-6  # a modality with std below this (physical units) is treated as absent
# Median |acc| per run (= gravity in the file's unit). Valid runs sit at ~1/2 or ~1/6 (two storage
# scales); outside this range the channel is pinned at 32.768, zero, or of unknown scale -> run unusable.
ACC_SCALE_VALID = (0.12, 0.7)

# ---- Split ----
SEED = 20261121
TEST_FRACTION = 0.2  # of subjects, stratified by "has a target seizure"
CV_FOLDS = 5

# ---- Training ----
NEG_PER_RUN_FRACTION = 0.05  # negatives are subsampled for fitting only; evaluation uses every window
NEG_MIN_PER_RUN = 50

# ---- Event post-processing and evaluation ----
SMOOTH_K = (1, 3, 5)  # moving average over the last k window scores (searched on dev only)
REFRACTORY_S = 300  # after an alarm, no new alarm for 5 minutes (one event = one alarm)
MATCH_PRE_S = 10  # an alarm counts as a detection if it fires in [onset - 10 s, offset + 30 s]
MATCH_POST_S = 30
MAX_FAR_PER_24H = 1.0  # selection rule: max event sensitivity subject to FAR <= this on dev OOF
