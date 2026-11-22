"""Single source of truth for the SeizeIT2 v2 pipeline (follows locked decisions D9-D18).

Inputs come from lyra_mlm Stage 3 (ml/seizeit2/prepare.py): simulated MyoWare ENV at 50 Hz +
BMI270-like IMU at 25 Hz (D12/D15/D16), 4 s windows every 2 s (D13), D14 labels. Everything that
changes a result lives here and is copied into models/seizeit2/ at export time.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "pj-training-data" / "ds005873"  # BIDS SeizeIT2 (OpenNeuro ds005873 v1.1.0), local only
CACHE = REPO / "pj-training-data" / "cache"  # temporary downloads; inside the git-ignored data dir
PROCESSED = REPO / "pj-training-data" / "processed"  # lyra_mlm Stage 3 output (same format as mlm/data/processed)
RUNS_DIR = REPO / "ml" / "seizeit2" / "runs"  # experiment outputs and checkpoints (git-ignored)
MODEL_DIR = REPO / "models" / "seizeit2"  # deployable, git-tracked artefacts

# Annexed EDF content is fetched from the dataset's own public CC0 bucket and verified against
# the SHA256 in the git-annex key before use. Nothing is ever uploaded.
S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds005873"

WINDOW_S = 4.0  # D13 window length (lyra_mlm WindowConfig default)
HOP_S = 2.0  # D13 step; one prediction per window end

# ---- Cross-validation over ALL participants (owner decision, Session 5: no held-out set) ----
SEED = 20261121
CV_FOLDS = 5  # outer, participant-grouped; thresholds are chosen in an inner CV of each training part
INNER_FOLDS = 4

# ---- Training ----
NEG_PER_RECORDING_FRACTION = 0.05  # negatives are subsampled for fitting only; evaluation uses every window
NEG_MIN_PER_RECORDING = 50

# ---- Event confirmation (D17 structure; entry/exit thresholds chosen in training folds) ----
D17_DEFAULTS = {"entry_threshold": 0.65, "exit_threshold": 0.45}  # reported for comparison
EXIT_BELOW_ENTRY = 0.20  # D17 spacing between entry and exit
CONSECUTIVE = 3  # D17
MAX_GAP_S = 6.0  # D17
REFRACTORY_S = 60.0  # D17

# ---- Event matching and the predefined operating-point rule ----
MATCH_PRE_S = 10  # an alarm counts as a detection if it fires in [onset - 10 s, offset + 30 s]
MATCH_POST_S = 30
OOS_GUARD_S = (30, 60)  # alarms this close to an out-of-scope seizure are reported, not false alarms
MAX_FAR_PER_24H = 1.0  # rule: max event sensitivity subject to FAR <= this; ties -> lower FAR, latency
