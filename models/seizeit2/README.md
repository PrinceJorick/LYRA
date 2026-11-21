# lyra-seizeit2-fbtcs v2.0.0

> **Experimental research component.** Not a diagnostic device, not clinically validated, and never
> run on LYRA hardware. Every number below comes from a public in-hospital dataset. None of them
> describe how LYRA will perform on the pod or in daily life.

This model is a random forest of 50 trees, max depth 8. It scores 4-second windows of **simulated
MyoWare ENV (50 Hz) + a 6-axis IMU (25 Hz)** for patterns resembling **focal-to-bilateral
tonic-clonic (convulsive) seizures**. `lyra_mlm.EventConfirmation` (D17) turns the scores into
"possible event" states.

- **Training data:** all SeizeIT2 participants (OpenNeuro ds005873 v1.1.0, CC0), following locked
  decisions D9–D18.
- **Runtime:** the forest runs on the pod (D11: ESP32-S3) as a C99 header. The app does not run it.

Training pipeline: `ml/seizeit2/`. Data preparation: `lyra_mlm` Stage 3 (`mlm/`).

## Files

| File | Purpose |
|---|---|
| `lyra_seizeit2_model.h` | The forest as flat `const` arrays + `lyra_seizeit2_score(const float x[22])`, C99. Arrays ≈ 224 kB of flash; 17,230 nodes |
| `model.json` | Same arrays for any other runtime. `ml/seizeit2/export.score_json` is the reference evaluator |
| `model_config.json` | `lyra_mlm.model.ModelConfig` (feature config, names, window, threshold, training record) |
| `export_manifest.json` | `lyra_mlm.model.ExportManifest` |
| `config.json` | Input contract (D12 rates, units, IMU calibration), feature names, routing, confirmation parameters |
| `labels.json` | Window classes and confirmation states |
| `metadata.json` | Version, training run, hyperparameters, footprint, export check |
| `cv_folds.json` | Participant → outer CV fold (pseudonymous SeizeIT2 IDs) |
| `dataset_summary.json` | Aggregate dataset and Stage 3 counts |
| `evaluation/metrics.json` | Nested CV results for every candidate, per fold and per participant, and the final selection |
| `evaluation/*.png` | Operating curve, per-fold fit vs held-out, out-of-fold confusion matrix |
| `evaluation/gyro_calibration.json` | Evidence for the gyro unit (125 deg/s per calibrated unit) |

**Export check:** the C header is compiled with `gcc -std=c99 -Wall -Wextra -Werror -pedantic` and
compared with sklearn on 205,000 rows (200,000 real windows + 5,000 random in-range rows).
- Maximum score difference: 3.6e-08.
- Threshold decisions: 100% identical.
- Thresholds are stored in float32, rounded toward −∞. This makes every split identical to sklearn's
  for float32 inputs.

## What changed from v1

| | v1.0.0 (`ff1373d`) | v2.0.0 (this) |
|---|---|---|
| EMG input | RAW sEMG, 256 Hz, µV | **Simulated MyoWare ENV, 50 Hz** (D12/D15/D16) |
| Preprocessing and features | 24 own features (`preprocessing.py`) | **`lyra_mlm` Stage 3**, 22 primary features, quality states |
| Windows / labels | 5 s / own guards | **4 s, 2 s step (D13) / D14** |
| Event logic | Mean of 5 scores + 300 s refractory | **`EventConfirmation` (D17)** |
| Model | HGB 300 trees | **RF 50 × depth 8 (D18)** |
| Runtime | TypeScript in the app | **C99 on the pod (D11)** |
| Participants | 100 trained / 25 held out | **All 124 with usable data; nested CV estimate** |
| IMU units | Acc rescaled to g in own code; gyro scale-free | **Fixed in `lyra_mlm`**: acc in g, gyro in deg/s |

## Problem definition

- **Positive (class 1):** a window that overlaps an annotated `sz_foc_f2b` (FBTCS) by ≥ 50% (D14).
  SeizeIT2 has 55 FBTCS in 37 of 125 participants, and no GTCS.
- **Negative (class 0):** a window outside ±30 s of any annotated seizure (D14).
- **Ignored:** everything else, including all other seizure types. Non-convulsive seizures are out of
  scope by design (D9).
- **Output:** a score is passed to `EventConfirmation`, which raises a *possible event*, never a
  confirmed seizure. The cancel countdown and caregiver workflow are unchanged.

## Data (`dataset_summary.json`)

- **Dataset:** 125 participants, 2,850 recordings, 11,626 h.
- **Stage 3 windows:** 2,806 recordings (124 participants) have windows. The other 44 have no EMG and
  no chest IMU.
- **Window labels:** 20,296,484 windows: 3,716 positive, 20,244,889 negative, 47,879 ignored.
- **Evidence windows:** 17,973,050 windows have both modalities VALID/DEGRADED and all 22 features
  available. That is 9,985 h of valid monitoring. A further 1,291 h are suppressed: they are held,
  never scored as negative.
- **FBTCS usable for evaluation:** 44 of 55, from 31 participants.
  - 7 lie only in suppressed windows.
  - 4 are in recordings without windows.
- **Preparation:** `ml/seizeit2/prepare.py` streams each recording's EDFs from the dataset's public
  bucket and verifies their SHA256. It runs Kyle's unchanged `scripts/prepare_seizeit2.py` worker,
  then deletes the EDFs.
  - The worker computes D15: RAW → HPF 20.8 Hz → rectify → LPF 3.6 Hz → 50 Hz.
  - The run took 2.4 h with 10 workers and had 0 failures.

**IMU fix (Kyle's Stage 3 bug).** SeizeIT2 labels the IMU "g", but the per-recording scale varies.
Median |acc| is about 1/2 in 245 recordings and about 1/6 in 1,837. It is pinned at 32.768 in 347
and invalid in 58 more.
- `lyra_mlm.seizeit2.calibrate_imu` divides ACC and GYR by the recording's median |acc|, which puts
  ACC in g.
- It then multiplies GYR by 125 deg/s per unit. This factor was estimated from du/dt = −ω × u on 24
  recordings: median 125.4, IQR 109–136, with the identity axis map in 24 of 24 recordings.
- Recordings with an impossible scale have all IMU samples marked INVALID. Their windows are
  suppressed.

## Validation protocol (no held-out set)

By owner decision, every participant is used. Performance is estimated by **nested,
participant-grouped CV**:

1. **Outer split:** 124 participants go into 5 outer folds (seed 20261121). Participants with FBTCS
   are spread evenly across folds, giving 12/12/10/9/8 events. A participant never spans two folds,
   so overlapping windows cannot leak between them.
2. **Threshold selection:** for each outer fold, the entry threshold is chosen only from a 4-fold
   inner CV on the training participants. The exit threshold is entry − 0.20, as D17 spaces them.
   - The rule is fixed in advance: maximise event sensitivity subject to FAR ≤ 1 per 24 h.
   - Ties go to the lower FAR, then to the lower latency.
3. **Outer scoring:** the model is refitted on all training participants and scores every window of
   the outer fold once. The pooled outer folds are the estimate.
4. **Fitting data:** fits use all positives plus 5% of negatives per recording. Evaluation scores
   every evidence window.

**Event scoring:**
- **Detection:** a D17 *possible event* inside [onset − 10 s, offset + 30 s] of an FBTCS.
- **Reported separately, not counted as false alarms:**
  - alarms within −30/+60 s of an out-of-scope seizure
  - alarms during impedance checks
- **False alarm:** every other alarm.
- **FAR:** false alarms per 24 h of valid monitoring.

## Results: nested CV, all participants

| Candidate (D18) | Events detected | FAR / 24 h (95% CI) | Median latency | Window PR-AUC |
|---|---|---|---|---|
| LR (baseline) | 27/44 (0.61) | 0.79 (0.71–0.88) | 52 s | 0.02 |
| **RF 50 × 8 (primary, fixed in advance)** | **42/44 (0.95, CI 0.85–0.99)** | **0.32 (0.27–0.38)** | **54 s** (IQR 40–78) | 0.15 |
| RF 25 × 6 (smaller budget) | 41/44 (0.93) | 0.70 (0.62–0.78) | 52 s | 0.09 |
| HGB 300 × 15 (XGBoost stand-in, offline) | 41/44 (0.93) | 0.21 (0.17–0.26) | 54 s | 0.23 |
| Ablation: RF, simulated ENV features only | 37/44 (0.84) | 2.00 (1.86–2.14) | 72 s | 0.01 |
| Ablation: RF, IMU features only | 39/44 (0.89) | 0.71 (0.63–0.80) | 58 s | 0.06 |

**Primary model, in detail:**
- **Misses:** one event each in sub-015 and sub-096.
- **False alarms:**
  - 135 in total, from 35 of 124 participants. The top 5 participants produce 46% of them.
  - 16 more alarms fell near out-of-scope seizures and are reported separately.
- **Sensitivity if every annotated FBTCS counts:** 42/51 = 0.82 when unevaluable events in windowed
  recordings count as missed. It is 42/55 = 0.76 over all FBTCS.

**Fusion helps.** ENV + IMU detects more events than either modality alone and raises 2–6× fewer
false alarms: 0.32/24 h, against 0.71 for IMU only and 2.00 for ENV only.

**D17 default thresholds are not usable as-is.** At entry 0.65 / exit 0.45 the primary model detects
43/44 events but raises **37 false alarms/24 h**. The forest's scores are not calibrated
probabilities, so the thresholds must be tuned (see Deviations).

**Final model.** RF 50 × 8 is refitted on all 124 participants (901,619 windows, 2,979 positive).
- **Thresholds:** entry **0.9684**, exit **0.7684**, chosen by the same rule on the pooled
  out-of-fold scores.
- **Pooled out-of-fold performance at those thresholds:** 42/44 events and 0.14 FA/24 h. These
  thresholds were chosen on those same scores, so the figure is optimistic. **Quote the nested
  figure (0.32/24 h).**
- **Out-of-fold windows at that threshold:**
  - TN 17,926,162 · FP 2,999 · FN 2,219 · TP 760.
  - Precision 0.20, recall 0.26, specificity 0.99983, ROC-AUC 0.86.
  - Window recall is low by design: the event logic needs only 3 consecutive high windows.
  - Accuracy is meaningless at this imbalance and is not used.

## Deviations from the locked decisions (stated, not hidden)

1. **D17 entry/exit values.** The D17 structure is kept: 3 consecutive windows, max gap 6 s,
   refractory 60 s, hold. Entry/exit are tuned on training folds because the defaults give
   37 FA/24 h. The 0.20 entry–exit spacing is kept.
2. **D18 "XGBoost offline comparison"** is run with sklearn's `HistGradientBoostingClassifier`, so
   no new dependency is added. The optional SVM was skipped.
   - HGB had a lower FAR than the primary RF. The CIs do not overlap.
   - The RF stays primary because D18 fixed it in advance as the embedded model.
   - HGB is a candidate if the budget allows about 300 trees × 15 leaves on the ESP32-S3.
3. **No held-out participants.** This is an owner decision. `SEIZEIT2_PLANNED_SPLIT` (sub-097…125
   held out) is not used, so no untouched SeizeIT2 test set remains.

## Limitations (read before using any number above)

- **Few events:**
  - 44 evaluable FBTCS from 31 participants.
  - Two participants hold 10 of them (sub-115 and sub-116, 5 each).
  - Confidence intervals are wide.
- **Hospital data only (EMU, mostly in bed):** daily-life false-alarm rates are expected to be higher,
  and there are no labelled daily-activity negatives.
- **Simulated ENV, not a MyoWare:**
  - The envelope comes from SeizeIT2 RAW µV through the D15 chain, so amplitude features
    (`emg_simenv_rms`, `mav`, `var`, `std`, `wl_per_s`, `mod_band_power`) carry µV-derived units.
  - A real MyoWare ENV output needs a gain calibration to this scale before the model sees it.
  - The analog MyoWare chain has not been compared with the simulation.
- **IMU domain:**
  - Training IMU is a chest module at 25 Hz.
  - The gyro unit is an estimate (125 deg/s per unit, IQR 109–136).
  - The pod BMI270 placement (D10) is untested.
- **No on-device inference yet:**
  - The 22 features exist only in Python (`lyra_mlm.features`). A bit-faithful C99 port is still
    needed.
  - The header has not run on an ESP32-S3.
- **Latency** is measured from the annotated focal onset, not from the start of the convulsive phase.
- **Scores are not calibrated probabilities** and must not be shown as such.
- **Subgroups:** SeizeIT2 has no age, centre or medication metadata, so performance cannot be checked
  across subgroups.

## Reproduce

```bash
uv venv --python 3.14 ml/seizeit2/.venv
VIRTUAL_ENV=ml/seizeit2/.venv uv pip install -r ml/seizeit2/requirements.txt   # includes -e ./mlm
P=ml/seizeit2/.venv/bin/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $P -m ml.seizeit2.prepare --workers 10   # Stage 3, ~2.4 h
$P -m ml.seizeit2.dataset_analysis                                  # -> dataset_summary.json
$P -m ml.seizeit2.train                                             # nested CV + final fit, ~32 min
$P -m ml.seizeit2.export --run ml/seizeit2/runs/<id>                # -> models/seizeit2/ (gcc parity)
$P -m pytest -q ml/seizeit2/test_evaluate.py                        # confirm_alarms == EventConfirmation
(cd mlm && ../ml/seizeit2/.venv/bin/python -m pytest -q)            # lyra_mlm incl. IMU calibration
$P -m ml.seizeit2.gyro_calibration                                  # optional: re-derive the gyro unit
```

The recordings stay in `pj-training-data/`, which is git-ignored. EDF content is downloaded from the
dataset's public bucket, checked against the git-annex SHA256, used, and deleted. Nothing is uploaded.
Trained on 2026-11-22 (run `20261122-022006`), on CPU only: 12 threads, about 32 min. sklearn has no
GPU path.
