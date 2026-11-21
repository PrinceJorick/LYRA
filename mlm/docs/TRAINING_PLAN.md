# How the separate dataset-integration and training session should begin

Requires explicit authorization from the project owner first. Nothing below has been done.

## 0. Preconditions (decisions to record first)

- Window labelling convention: ≥50% overlap vs centre-in-event (context.md U11).
- ENV-vs-RAW strategy for SeizeIT2 EMG (DATASETS.md, difference 1).
- MyoWare ENV acquisition rate and analog bandwidth (set `SignalStream.bandwidth_hz`).
- `degraded_policy`, `missing_modality_policy`, `invalid_window_policy` choices for the experiments.
- Event-matching rule for event-based sensitivity and false alarms.

## 1. Data (see DATASETS.md, integration procedure steps 1–6)

Manifest + checksums → inspect annotations → `AnnotationMapper` → `DatasetAdapter` →
`validate_recording` report → reconcile counts → verify the split.

## 2. Windows and features

1. `segment()` every recording with the configured `WindowConfig`/`QualityConfig`. Keep **all**
   windows, including suppressed and unavailable ones, with their quality metadata.
2. `extract_features()` per window. Store the names, values, availability mask, quality and
   annotation refs.
3. Report per participant: window counts by quality state, and feature availability rates. A model
   cannot be compared fairly if suppression differs between configurations.

## 3. Splits and leakage

1. Assign partitions by participant with `SEIZEIT2_PLANNED_SPLIT.partition_of()`. Run
   `check_participant_independence()` on the result.
2. Cross-validation inside sub-001…096 only, grouped by participant.
3. Fit normalization (`fit_normalization(..., Partition.TRAIN, ...)`), feature selection and threshold
   choice inside each training fold only. Apply them unchanged to the validation fold.
4. Do not touch the held-out set (sub-097…125) until the analysis plan is frozen. Then evaluate once.

## 4. Models (in this order)

1. Logistic regression baseline.
2. Random forest, the primary embedded candidate. Starting search range 25–50 trees, depth 6–8
   (research bounds, not final values).
3. XGBoost for offline comparison, and SVM optionally.

Each `Trainer` must call `require_training_partition()`. Record the dataset version, split, git commit,
seed and hyperparameters in `ModelConfig.training_record`, and set `trained=True` only on a fitted model.
Run ablations: sEMG-only, IMU-only, multimodal, and multimodal with unimodal fallback.

## 5. Evaluation

Fill `EvaluationReport` on held-out participants: event sensitivity, false alarms per 24 h (with
`valid_monitored_hours` and `suppressed_hours` reported), detection latency, precision, AUPRC,
specificity, per-participant results, and false alarms by hard-negative activity group where labels
exist. Run `EventConfirmation` over the predictions so event-level metrics reflect the deployed logic.
Treat every number as a research estimate on an in-hospital FBTCS cohort. It is not clinical
validation and not GTCS evidence.

## 6. Embedded

`ensure_exportable()` → implement an `EmbeddedExporter` (investigate emlearn, micromlgen and m2cgen
support for the chosen family) → port the feature extraction to C99 in `feature_names` order → measure
`EmbeddedBenchmark` on a real ESP32-S3, including Python-vs-C score parity on shared feature rows.
