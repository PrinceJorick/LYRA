# Stage 4 plan: controlled model training

**Requires explicit authorization from the project owner.** Nothing below has been done. Stage 3
produced `mlm/data/processed/` (gitignored): raw feature vectors, availability masks, quality states,
labels and split manifests. No model was trained and no statistic was fitted on real data.

<<<<<<< HEAD
## Stage 3 hand-over facts (see DATASETS.md)

- 20,296,484 windows from 2,806 recordings; 44 recordings excluded (no EMG and no ECGEMG IMU).
- Usable: 3,456 positive, 19,112,674 negative and 42,468 ignore windows; 1,137,886 signal-excluded.
- FBTCS with usable positive windows: **30 of 33 in Train/CV, 17 of 22 held-out**.
- Train/CV has only 2,070 positive windows (28 participants). Held-out holds 22 of the 55 events,
  11 of them from sub-115 and sub-116, so per-participant reporting is essential.
- sub-097 (held-out) has no usable recordings.
- Positive:negative among usable Train/CV windows is about 1:7,000 (2,070 vs 14,589,150). Plan class
  weighting or subsampling of negatives inside training folds only.
- `mlm/data/processed_superseded/` holds the 378 pre-fix files from the quality-inheritance defect.
  They can be deleted.

=======
>>>>>>> 414a2140836e55fed760a18fff6c49fe2bc0082d
## 0. Already decided (locked in Stage 3)

- **Labels:** POSITIVE ≥ 50% FBTCS overlap, NEGATIVE outside ±30 s of any seizure, IGNORE otherwise.
- **Primary path** `SIMULATED_ENV_FROM_RAW` (22-feature multimodal schema); **secondary** `RAW` (8 features).
- **Envelope rate** 50 Hz. Envelope modulation features cover 1.5–5.0 Hz.
- **Confirmation and inference defaults:** see ARCHITECTURE.md §7–8.

Still to decide before training: how to treat `IGNORE` and `artifact_overlap` windows in the loss
(the recommendation is to exclude both from training and keep them for event-level evaluation),
how events are matched for event-level metrics, whether the active-event gap rule in
ARCHITECTURE.md §8 stands, and how unavailable features are handled (suppress, or a model that
accepts missing values).

## 1. Inputs

- `processed/windows/<sub>/<recording>.npz`: `start_s`, `end_s`, `label`, `fbtcs_overlap_s`,
  `artifact_overlap`, quality codes (`overall`, `emg_state`, `imu_state`, `raw_emg_state`),
  coverages, `primary` + `primary_available`, `secondary` + `secondary_available`.
- `processed/feature_schema.json`: feature order, specs, quality and label codes, prep config.
- `processed/splits_manifest.json`: train/CV vs held-out participants, FBTCS counts, window label counts.

## 2. Leakage rules

1. Train/CV uses sub-001…sub-096 only. Cross-validation folds are grouped by participant
   (`check_participant_independence` on every fold).
2. Fit normalization (`fit_normalization(..., Partition.TRAIN, ...)`), feature selection, class
   weighting and threshold choice inside each training fold only. Apply them unchanged to the
   validation fold.
3. Do not open held-out data (sub-097…sub-125) for anything until the analysis plan is frozen.
   Then evaluate once.

## 3. Models (in this order)

1. Logistic regression baseline.
2. Random forest, the primary embedded candidate. Research range 25–50 trees, depth 6–8.
3. XGBoost for offline comparison, and SVM optionally.

Each `Trainer` calls `require_training_partition()`. Record the dataset version, split, git commit,
seed and hyperparameters in `ModelConfig.training_record`, and set `trained=True` only on a fitted
model. Run ablations: simulated-ENV-only, IMU-only, multimodal, RAW-only (secondary), and multimodal
with unimodal fallback.

## 4. Evaluation (held-out, once)

Fill `EvaluationReport`: event sensitivity, false alarms per 24 h (with valid monitored and
suppressed hours reported), detection latency, precision, AUPRC, specificity, and per-participant
results. Run `EventConfirmation` with the Decision 4 defaults over the continuous predictions,
including IGNORE windows, so event-level metrics reflect the deployed logic. Treat every number as a
research estimate on an in-hospital FBTCS cohort (55 events). It is not clinical validation and not
GTCS evidence.

## 5. Embedded

`ensure_exportable()` → `EmbeddedExporter` (investigate emlearn, micromlgen and m2cgen for the chosen
family) → C99 feature extraction in `feature_names` order → `EmbeddedBenchmark` on a real ESP32-S3,
including Python-vs-C score parity. Before any hardware claim, compare simulated ENV against real
MyoWare ENV recordings.
