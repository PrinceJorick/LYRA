# lyra-seizeit2-fbtcs v1.0.0

> **Experimental research component.** Not a diagnostic device, not clinically validated, and not
> evaluated on LYRA hardware. The numbers below come from a public hospital dataset and do not
> describe how LYRA will perform on the pod or in daily life.

A gradient-boosted tree classifier (300 trees, ≤15 leaves each) that scores 5-second windows of
**one sEMG channel + one 6-axis IMU** for patterns resembling **focal-to-bilateral tonic-clonic
(convulsive) seizures**. It was trained and evaluated on SeizeIT2 (OpenNeuro ds005873 v1.1.0, CC0).
The training pipeline is in `ml/seizeit2/`. The app runtime is in `src/services/ml/seizeit2/`.

## Files

| File | Purpose |
|---|---|
| `model.json` | Trained tree ensemble (288 kB). Format `lyra-seizeit2-model/1`, evaluated in TypeScript |
| `config.json` | Input contract (rates, window, units), feature list, post-processing (k, threshold, refractory) |
| `labels.json` | Window classes and app event types |
| `metadata.json` | Version, training run, selection, export check |
| `split.json` | Subject-level test set and CV folds (pseudonymous SeizeIT2 IDs) |
| `dataset_summary.json` | Aggregate dataset counts and data-quality findings |
| `evaluation/metrics.json` | Held-out test metrics + every CV candidate (out-of-fold) |
| `evaluation/*.png` | Test confusion matrix, training history, CV operating curves |

## Why JSON and not `.tflite`

The app has no native inference runtime. It runs in Expo Go with no native ML module, and the
selected model is a small feature-based classifier. A JSON tree ensemble evaluated in TypeScript runs
unchanged on iOS, Android and web with no development build. The export is checked against sklearn on
200,000 real dev windows: the maximum score difference is 2.2e-16 and decisions agree 100%. Python/TS
parity (features, scores, detector) is checked by `node ml/seizeit2/parity_check.mjs`.

## Problem definition

- **Positive (class 1):** a window that overlaps an annotated `sz_foc_f2b` (FBTCS) by ≥ 50%. SeizeIT2
  has 55 FBTCS in 37 of 125 subjects, and no generalised tonic-clonic seizures.
- **Negative (class 0):** any other window, except the excluded ones below.
- **Excluded from training:**
  - windows within 30 s before or 120 s after an FBTCS that are not positive
  - windows within 30 s before or 60 s after any other seizure type ("out of scope": focal
    aware/impaired, hyperkinetic, etc.). These are never used as negatives.
  - impedance-check (`impd`) segments
- **Scope:** convulsive events only. Non-convulsive and most focal seizures are invisible to
  sEMG + IMU by design (see `context.md` §3.6). The target seizure type is still `[UNRESOLVED]` in
  `context.md` §4.1. This model adopts the recommended FBTCS/GTCS scope and needs owner sign-off.
- **App output:** `possible_event` or `no_event` after post-processing. The score itself is never
  shown. A `possible_event` goes through the existing alert, cancel countdown and caregiver workflow.

## Data and preprocessing (`ml/seizeit2/preprocessing.py`, mirrored in `features.ts`)

- **Recordings:** 2,850 runs, 11,626 h. 2,487 runs have both EMG and chest IMU.
- **Usable data after quality checks:** 2,082 runs, 10,034 h, 44 of 55 FBTCS.
- **Signals:**
  - EMG: `EMG SD` (left deltoid) at 256 Hz, in µV.
  - IMU: chest `ECGEMG SD` module at 25 Hz.
  - EDF+C files with 1-s records. All modalities share the run's time origin.
- **Accelerometer scale (data-quality finding):** files are labelled "g", but median |acc| per run is
  ~1/2 (245 runs) or ~1/6 (1,837 runs). It is pinned at 32.768 in 347 runs and invalid in 58 others.
  - Valid runs are divided by their own median |acc| (gravity), which puts them in g.
  - Invalid runs are excluded (`ACC_SCALE_VALID`).
- **Gyroscope:** the unit is undocumented and the scale differs between runs. Only scale-invariant
  gyro features are used.
- **Windows:** 5 s, every 2 s. The 10 s length was evaluated and not selected.
  - Windows with flat EMG or IMU are unusable.
  - Windows with > 5% of EMG samples at the ADC rail are unusable.
- **24 rotation-invariant features:**
  - sEMG Welch spectrum: 20–128 Hz power excluding 45–55 Hz, band fractions, low/mains ratios,
    median frequency, entropy, frame variability.
  - sEMG envelope: CV, peak ratio, 1–5 Hz rhythmicity, dominant frequency.
  - Accelerometer magnitude (in g): std, variance trace, jerk, 1–6 Hz fraction, dominant frequency,
    peak ratio.
  - Gyroscope magnitude: CV, 1–6 Hz fraction, dominant frequency, peak ratio.
  - Orientation change across the window.
  - EMG-envelope vs gyro dominant-frequency difference.
- **Corrupt annotations:** 5 `impd` rows with negative durations were dropped and logged.

## Validation protocol (leakage controls)

1. **Subject-level split, made once before any modelling (seed 20261121).** 25 test subjects
   (7 with FBTCS, 13 events) and 100 dev subjects. A subject's windows never cross splits.
2. **5-fold subject-grouped CV on dev.** Scalers and rule statistics are fitted inside each training
   fold.
3. **Negatives are subsampled only for fitting** (5% per run). Every window of held-out subjects is
   scored for evaluation.
4. **Predefined operating-point rule:** maximise out-of-fold event sensitivity subject to
   FAR ≤ 1/24 h. Ties go to lower FAR, then lower latency. The same rule picks the model and window
   length.
5. **The test set was evaluated once.** `evaluate.py` refuses to re-run.

**Event scoring:**
- An alarm fires when the mean of the last k=5 scores ≥ 0.9867 and no alarm fired in the previous
  300 s.
- **Detection:** an alarm in [onset − 10 s, offset + 30 s] of an FBTCS.
- **Out-of-scope:** alarms near other seizures are reported separately, not counted as false alarms.
- **False alarm:** every other alarm.

## Results

### Dev, out-of-fold (5 subject folds, 32 usable FBTCS, 8,049 h). Used for selection.

| Candidate | Sensitivity | FAR / 24 h | Median latency | PR-AUC (windows) |
|---|---|---|---|---|
| M0 rule: EMG power AND acc variability (baseline) | 16/32 (0.50) | 0.74 | 62 s | 0.02 |
| M1 logistic regression | 21/32 (0.66) | 0.92 | 66 s | 0.03 |
| M2 HGB, 100 trees | 30/32 (0.94) | 0.26 | 63 s | 0.22 |
| **M2 HGB, 300 trees (selected, W = 5 s)** | **31/32 (0.97)** | **0.23** | **59 s** | **0.27** |
| M2 HGB, 300 trees, W = 10 s | 31/32 (0.97) | 0.24 | 63 s | 0.25 |
| Ablation: HGB on EMG features only | 29/32 (0.91) | 0.52 | 73 s | 0.10 |
| Ablation: HGB on IMU features only | 31/32 (0.97) | 0.76 | 54 s | 0.19 |

Fusing sEMG and IMU cuts false alarms about 3× versus IMU alone at the same sensitivity. This is
consistent with LYRA's core hypothesis, but it is shown on hospital data only.

### Held-out test (25 subjects, 1,983 h). Evaluated once.

| Metric | Value |
|---|---|
| Event sensitivity | **11/12 = 0.92** (95% CI 0.62–1.00). 11/13 = 0.85 if the 1 FBTCS in an unusable run counts as missed |
| False alarms | 18 → **0.22 / 24 h** (95% CI 0.13–0.34). 7 of the 18 come from one subject |
| Detection latency from annotated onset | median **43 s** (IQR 26–77 s) |
| Window level (thresholded, smoothed) | precision 0.64, recall 0.39, specificity 0.99996, F1 0.49 |
| Window level (threshold-free) | ROC-AUC 0.86, PR-AUC 0.40 |
| Confusion matrix (windows) | TN 3,451,459 · FP 141 · FN 387 · TP 251 (`evaluation/confusion_matrix.png`) |

Window recall is low by design. The detector needs only a few high-scoring windows per event, and
event sensitivity is the metric that matters. Accuracy (0.9998) is meaningless at this class
imbalance and is not used.

## Limitations (read before using any number above)

- **Few events:**
  - 44 usable FBTCS in total and 12 in the test set. Confidence intervals are wide.
  - One test subject contributes 5 of the 12 events.
  - The single miss was the shortest event in the dataset (30 s).
- **Hospital data only (EMU, mostly in bed):**
  - False-alarm rates in daily life are expected to be higher. The literature reports about 9×
    (`context.md` R4).
  - There are no labelled daily-activity negatives.
- **Latency is measured from the annotated focal onset:** the convulsive phase starts later, so this
  is not a time-to-convulsion measure.
- **Probabilities are not calibrated:**
  - Held-out log-loss is lowest at about 50–100 trees and rises after that
    (`evaluation/training_history.png`).
  - The 300-tree model was chosen on event metrics, which depend only on ranking.
  - Do not read the score as a probability.
- **Sensor domain shift to the LYRA pod is untested:**
  - EMG must be a raw, single, differential channel in **µV at 256 Hz**. MyoWare 2.0 RECT/ENV
    outputs are not compatible, and RAW needs gain calibration to µV.
  - Deltoid placement follows SeizeIT2. Other muscles are untested.
  - IMU must be **25 Hz**, with the accelerometer in **g**. Chest placement follows SeizeIT2.
- **Missing metadata:** no age, centre or medication information, so performance cannot be checked
  across subgroups.
- **Data quality excluded runs:** 16% of runs were dropped for invalid accelerometer scale, which
  removed 3 FBTCS from evaluation.

## Reproduce

```bash
uv venv --python 3.14 ml/seizeit2/.venv
VIRTUAL_ENV=ml/seizeit2/.venv uv pip install -r ml/seizeit2/requirements.txt
P=ml/seizeit2/.venv/bin/python
$P -m ml.seizeit2.preprocessing          # fetch + verify EDFs, build feature cache (~4 GB, ~50 min)
$P -m ml.seizeit2.dataset_analysis       # -> dataset_summary.json
$P -m ml.seizeit2.train                  # split, CV, selection, final fit -> ml/seizeit2/runs/<id>
$P -m ml.seizeit2.evaluate --run ml/seizeit2/runs/<id>   # test set, once
$P -m ml.seizeit2.export   --run ml/seizeit2/runs/<id>   # -> models/seizeit2/
node ml/seizeit2/parity_check.mjs        # Python <-> TypeScript parity
```

The recordings stay in `pj-training-data/`, which is git-ignored. EDF content is downloaded from the
dataset's public bucket, checked against the git-annex SHA256, used, and deleted. Nothing is uploaded.
Trained on 2026-11-21: 12 CPU cores, about 14 min of training after a 50-min download-bound feature
build.
