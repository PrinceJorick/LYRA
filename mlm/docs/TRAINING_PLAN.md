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

## 7. Results (2026-11-22)

Research estimates on an in-hospital focal-epilepsy cohort (SeizeIT2). They are not clinical
validation and not GTCS evidence. Frozen pipeline `1de1b13`; artifacts in `mlm/models/` (gitignored).

**Training sets (per training fold, 1:50):** about 1,660–1,710 positives and 83k–85k negatives per
track. RF winners, chosen by CV AUPRC, are the same for every track: 50 trees, depth 8,
min_samples_leaf 10.

**CV (Train/CV, 96 participants, 30 FBTCS events in processed recordings), locked entry 0.65:**

| Track | Model | Window AUPRC | Detected | Interictal FA/24h |
|---|---|---|---|---|
| multimodal | LR / **RF** / XGBoost | 0.019 / **0.160** / 0.201 | 27 / **28** / 28 | 89.2 / **48.1** / 85.5 |
| SIMULATED_ENV | LR / **RF** / XGBoost | 0.001 / **0.038** / 0.017 | 30 / **29** / 30 | 90.2 / **84.3** / 122.4 |
| IMU | LR / **RF** / XGBoost | 0.018 / **0.084** / 0.097 | 27 / **28** / 28 | 69.7 / **102.2** / 117.9 |

At 0.65, every model raises dozens of false alarms per day. Balanced class weights plus 1:50
subsampling inflate the forest's probabilities, so 0.65 sits far too low for a 1:7,000 prior. Hence
the CV-selected operating points (§6), all with the selected RFs:

| Configuration | CV-selected entry | CV detected | CV interictal FA/24h | CV median latency |
|---|---|---|---|---|
| multimodal, suppress | 0.95 | 27/30 | 0.49 | 54 s |
| **multimodal + SIMULATED_ENV fallback** | **0.97** | **27/30** | **0.38** | 54 s |
| SIMULATED_ENV-only | 0.97 | 21/30 | 0.67 | 62 s |
| IMU-only | 0.99 | 23/30 | 0.37 | 60 s |

**D2 vs D6 (CV, multimodal + fallback at 0.97):**
- D6: 59 participants, 25/27 detected, 0.11 FA/24h, AUPRC 0.203.
- D2: 24 participants, 1/2 detected, 0.34 FA/24h, AUPRC 0.048.

False-alarm rates are low in both groups, with no sign of a scale shortcut on the negative side. But
27 of the 30 Train/CV FBTCS events come from D6 participants, so D2 sensitivity can't be judged. This
cohort imbalance (D2 = mostly early participant IDs) is a confound, not something the scaling can fix.

**Held-out, single run (sub-097…sub-125: 28 participants with recordings, 756 recordings, 21
FBTCS events in 9 participants, 145 non-FBTCS seizures):**

| Configuration (entry) | Detected (pooled / macro) | Excl. sub-115, sub-116 | Interictal FA/24h (n, valid h) | Non-FBTCS alert rate | Median latency | Precision | Suppressed |
|---|---|---|---|---|---|---|---|
| **multimodal + fallback (0.97)** | **17/21, 81% / 83%** | 7/11 | **0.68** (78, 2,737 h) | 1.4% | 67 s | 17.5% | 0.4% |
| multimodal, suppress (0.95) | 15/21, 71% / 67% | 5/11 | 0.44 (42, 2,285 h) | 2.1% | 54 s | 24.6% | 16.8% |
| SIMULATED_ENV-only (0.97) | 15/21, 71% / 70% | 6/11 | 0.97 (111, 2,737 h) | 0.7% | 73 s | 11.7% | 0.4% |
| IMU-only (0.99) | 12/21, 57% / 53% | 3/11 | 0.17 (16, 2,286 h) | 1.4% | 65 s | 40.0% | 16.8% |
| multimodal + fallback (locked 0.65) | 19/21, 90% / 91% | 9/11 | 49.9 | 15.9% | 45 s | 0.4% | 0.4% |
| multimodal, suppress (locked 0.65) | 15/21, 71% / 67% | 5/11 | 50.3 | 13.8% | 36 s | 0.4% | 16.8% |

- Window AUPRC: multimodal 0.195, fallback combination 0.089, SIMULATED_ENV 0.013, IMU 0.102.
  Specificity at the entry threshold: 0.998 for fallback at 0.97.
- 61 of the 78 interictal false alarms of the primary configuration come from one participant (sub-114).
  sub-121 had 0/3 events detected.
- sub-115 and sub-116 hold 10 of the 21 events (all detected), so the result without them (7/11) matters
  as much as the pooled figure.
- Missing-modality policy: `fallback_unimodal` recovers 2 events that `suppress` misses and cuts
  suppressed time from 16.8% to 0.4%. The cost is a higher FA rate (0.68 vs 0.44 per 24 h).

**C99 export (`mlm/export/c99/`):**
- Both forests are 50 trees: multimodal 14,276 nodes, SIMULATED_ENV 13,470 nodes.
- `cc -std=c99 -O2 -Wall -Wextra -Werror` compiles both cleanly.
- On 1,200 threshold-boundary vectors each: 100% class agreement, max |Δp| = 3.3e-16, and identical
  decisions at 0.65 and 0.45.
- Host (arm64) object size is about 363 KB / 354 KB of code with no `.data`/`.bss`. That is a proxy
  only: **the ESP32-S3 Flash/RAM sizes were not measured.** The Espressif xtensa toolchain wasn't
  available, and downloading it was blocked. The forest uses double arithmetic, which the ESP32-S3
  (single-precision FPU only) runs in software. That is about 400 comparisons per 2 s window, so the
  cost is negligible.
