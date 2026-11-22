# Stage 4 plan: controlled model training

**Authorized 2026-11-22** (after Stage 3.5, the IMU remediation). §6 fixes the protocol as
executed. It was written before any CV or held-out result was seen. Sections 0–5 are the original
plan, and §6 overrides them where they differ. Results are in §7 and PROGRESS.md.

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

## 6. Protocol as executed (frozen 2026-11-22, before results)

Code: `lyra_mlm.training`, `scripts/train_stage4.py`, `scripts/evaluate_heldout.py`, `scripts/export_c99.py`.

- **Data:** the Stage 3.5 windows (`processed/`). Train/CV is sub-001…sub-096; `training_rows()`
  raises `LeakageError` for any other participant. Held-out files are opened only by
  `evaluate_heldout.py`, once (it refuses a second run and refuses changed artifacts or data).
- **Tracks (no zero-imputation):**
  - multimodal: all 22 primary features, gated on `overall`;
  - SIMULATED_ENV-only: the 9 `emg_simenv_*` features, gated on `emg_state`;
  - IMU-only: the 9 `imu_*` features, gated on `imu_state`.

  A window is scored only when its gate is VALID or DEGRADED (DEGRADED allowed with a flag) and every
  track feature is available. Otherwise it is SUPPRESSED (`score=None`).
- **Training loss:** POSITIVE/NEGATIVE windows only, with no IGNORE and no unusable windows. Windows
  overlapping impedance-check (`impd`) annotations, which occur in sub-031 and sub-043, are excluded.
- **Negative subsampling, training folds only:** every positive is kept. Each training participant
  gives `round(f · its usable negatives)` negatives, choosing the lowest deterministic per-window keys
  (crc32 of the recording id). f = 50 × positives / negatives, so 1:50. Validation folds and the
  held-out set are never subsampled.
- **Models:**
  - Logistic regression: `StandardScaler` inside the pipeline (fold-fitted), `class_weight="balanced"`.
  - Random forest: `class_weight="balanced"`, `random_state=0`, grid n_estimators {25, 50} × max_depth
    {6, 8} × min_samples_leaf {5, 10}.
  - XGBoost (offline benchmark): 300 trees, depth 4, learning rate 0.05, subsample 0.8,
    colsample 0.8, `scale_pos_weight` = negatives / positives.
- **CV:** `StratifiedGroupKFold(5, shuffle=True, random_state=0)` over participants, stratified on
  each participant's labelled windows.
- **Event logic:** locked Decision 4 (`ConfirmationConfig()` defaults: entry 0.65, exit 0.45, 3
  consecutive, hold up to 6 s). A gap over 6 s ends an accumulating run to IDLE and an active event to
  REFRACTORY, the 60 s cooldown. `training.confirm()` is the array form of `EventConfirmation`, tested
  equal on random sequences. Thresholds are not tuned.
- **Matching:**
  - Detection: a POSSIBLE_EVENT entry (window end time) within [onset − 30 s, offset + 30 s] detects
    that FBTCS, at most one per event.
  - Latency: first matched entry − onset.
  - Alerts are classed as FBTCS-associated, non-FBTCS-seizure (within 30 s of another annotated
    seizure), or interictal.
- **Metrics (`evaluation.METRICS`):**
  - event sensitivity, pooled and per-participant macro;
  - interictal FA/24h, per valid (non-suppressed) monitored interictal hour, reported separately from
    the non-FBTCS seizure alert rate;
  - latency; precision (FBTCS-associated alerts / all alerts);
  - window AUPRC (usable labelled windows, 16,384-bin score histograms);
  - window specificity at the entry threshold;
  - per participant; valid and suppressed hours.

  CV results are reported pooled and by participant D group (D2 / D6 / mixed / no accepted IMU).
- **Selection:** per track, the RF config with the highest pooled CV AUPRC; ties go to fewer, then
  shallower, trees. The winners are retrained on all Train/CV and frozen in `mlm/models/`
  (gitignored) with sha256 values, the confirmation config and data-manifest hashes. LR and XGBoost
  are CV comparisons only.
- **Operating points (owner decision 2026-11-22, after CV and before any held-out run):** at the
  locked entry threshold 0.65, CV gave about 48 interictal false alarms per 24 h (multimodal RF). Each
  configuration therefore carries two pre-declared points:
  - the locked Decision 4 point (entry 0.65);
  - a CV-selected entry threshold: the lowest value in {0.65, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99}
    whose pooled CV interictal FA/24h is at most 1.

  The CV-selected point is primary. All other confirmation parameters stay at Decision 4. Both points
  are frozen in `frozen_pipeline.json` and reported from the single held-out run.
- **Held-out, once:** multimodal RF with `suppress`, multimodal RF with `fallback_unimodal`
  (SIMULATED_ENV-only RF when IMU is unusable), SIMULATED_ENV-only RF, and IMU-only RF. Reported
  pooled, macro, per participant, and with the two largest-FBTCS participants removed.
- **C99:** the multimodal and SIMULATED_ENV-only RFs are exported with m2cgen. emlearn 0.23.2 was
  rejected because its 8-bit leaf probabilities can't reach the 1e-5 tolerance. A float-input
  wrapper mirrors scikit-learn's float32 input cast. `tests/test_c99_export.py` compiles with
  `cc -std=c99 -O2 -Wall -Wextra -Werror` and requires 100% class agreement and
  max |p_C − p_sklearn| < 1e-5 on 1,200 threshold-boundary vectors per model.
