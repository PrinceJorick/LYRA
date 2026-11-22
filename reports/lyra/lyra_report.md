# LYRA convulsive-seizure detector: ESP32-S3 pipeline report

> Experimental research prototype. Not clinically validated, not a diagnostic device, and not a substitute for prescribed monitoring or medical care. A detection is a *possible* event, never a confirmed seizure. All evidence is from in-hospital SeizeIT2 recordings; nothing was run on LYRA hardware.

## 1. Dataset and split audit

| | Development (sub-001..096) | Final evaluation (sub-097..125) |
|---|---:|---:|
| Subjects | 96 | 29 |
| Subjects with usable windows | 96 | 28 |
| Recordings | 2072 | 778 |
| Recordings with windows | 2050 | 756 |
| Recordings without EMG | 24 | 22 |
| Recordings without MOV | 214 | 147 |
| Recordings with uncalibratable IMU | 253 | 153 |
| Recorded hours | 8714.70 | 2911.50 |
| Valid EMG+IMU hours (4 s windows) | 7644.30 | 2298.10 |
| Target seizures | 33 | 22 |
|   of which in recordings without data | 3 | 1 |
| Non-target seizures (hard negatives) | 669 | 159 |
| Impedance checks (excluded) | 382 | 77 |
| 4 s windows: fusion valid positive | 2,022 | 958 |
| 4 s windows: fusion valid negative | 13,757,654 | 4,135,656 |
| 4 s windows: fusion valid partial excluded | 31 | 15 |
| 4 s windows: fusion valid nontarget seizure | 15,826 | 2,819 |
| 4 s windows: artifact | 45,057 | 4,593 |
| 4 s windows: warmup | 2,050 | 756 |

Target codes present in the data: `sz_foc_f2b`: True, `sz_gen_m_tonicClonic`: False, `sz_uo_m_tonicClonic`: False. SeizeIT2 v1.1.0 contains no generalised or unknown-onset tonic-clonic codes, so every target event is a focal-to-bilateral tonic-clonic seizure.

- **EMG sampling rate:** EDF headers and JSON sidecars agree on 256 Hz for every EMG file. The 250 Hz stated in the task brief is not what the files contain; 256 Hz was used.
- **Synchronisation:** EMG and MOV EDF start times are de-identified differently (EMG 2000-01-01, MOV a real-looking date), so the absolute offset cannot be verified. Both come from the same ECGEMG Sensor Dot module and have identical durations (0 development and 0 final-evaluation recordings differ by more than 1 s). Offset 0 is assumed.
- **Split:** patient-wise. Development folds use GroupKFold(5) by subject; nothing in development code can load the final cohort (`ml/lyra/data.py` refuses until `deploy.py` writes the lock).
- **Caveat on the final cohort:** sub-097..125 were used to train earlier LYRA models (v1/v2, PR #1). They are held out for this pipeline only, not a pristine cohort.

## 2. Development results (out-of-fold, sub-001..096)

Selection rule: among (window, modality, classifier, threshold, N, M) with dev out-of-fold FAR <= MAX_FAR_PER_24H: 1) most detected events, 2) median latency < 20 s if any candidate achieves it, 3) smaller model, 4) lower FAR, 5) narrower between-subject sensitivity IQR. Best statistical = first candidate. Best deployable = first candidate whose exported package measures under the flash and SRAM targets on the ESP32-S3 toolchain. 

- FAR <= 1.0/24 h reachable on development data: **True**.
- Median latency < 20 s reachable together with it: **False**.
- Within the brief's threshold grid (0.20-0.80) alone: any detection **True**, FAR target met **True**. Calibrated probabilities target the true window prevalence, so they rarely exceed ~0.1; the grid was extended downwards (0.005-0.15) without changing the FAR or latency criteria.

| Role | Window | Modality | Classifier | Threshold | N | M | Detected | FAR/24 h | Median latency (s) | Precision |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Best statistical | w4 | fusion | XGB | 0.015 | 3 | 3 | 26/29 | 0.785 | 55.5 | 0.10 |
| Best deployable | w4 | fusion | XGB | 0.015 | 3 | 3 | 26/29 | 0.785 | 55.5 | 0.10 |
| Best emg | w5 | emg | LR | 0.0075 | 3 | 2 | 3/29 | 0.710 | 63.5 | 0.01 |
| Best imu | w4 | imu | XGB | 0.015 | 3 | 3 | 21/29 | 0.747 | 66.0 | 0.08 |
| Best fusion | w4 | fusion | XGB | 0.015 | 3 | 3 | 26/29 | 0.785 | 55.5 | 0.10 |

Full grid: `ablation_results.csv` (best configuration per window x modality x classifier) and `oof_metrics.json` (every threshold/N/M).

Tree models and feature scaling: standardising before a random forest changed predictions by at most 0.025 on 200,000 windows, so tree models are trained without normalisation; LR and SVC standardise inside each training fold.

## 3. Final evaluation (sub-097..125, evaluated once)

| | sEMG-only | IMU-only | sEMG + IMU |
|---|---:|---:|---:|
| Event-based sensitivity | 0.00 | 0.67 | 0.80 |
| Detected / evaluable | 0/15 | 10/15 | 12/15 |
| FAR/24 h | 0.010 | 0.063 | 0.240 |
| Median detection latency (s) | n/a | 51.0 | 47.5 |
| Event-level precision | 0.00 | 0.62 | 0.34 |
| Window-level AUPRC | 0.002 | 0.078 | 0.124 |
| Selected window / classifier | w5 / LR | w4 / XGB | w4 / XGB |

**Deployed model (exported C replica)** (w4, fusion, XGB, threshold 0.015, N=3, M=3):
- Event sensitivity 12/15 = 0.80 (patient-clustered bootstrap 95% CI [0.60, 1.00]); 6 target events had no valid window.
- 23 false alerts over 2297.6 valid target-negative hours = 0.240/24 h (CI [0.096, 0.429]); 1 of them during non-target seizures.
- Median latency 47.5 s from annotated onset; event precision 0.34; 35 alerts emitted, 2 suppressed by the 60 s refractory, 37 candidate events; window AUPRC 0.124.

**Best statistical model** (w4, fusion, XGB, threshold 0.015, N=3, M=3):
- Event sensitivity 12/15 = 0.80 (patient-clustered bootstrap 95% CI [0.60, 1.00]); 6 target events had no valid window.
- 23 false alerts over 2297.6 valid target-negative hours = 0.240/24 h (CI [0.096, 0.429]); 1 of them during non-target seizures.
- Median latency 47.5 s from annotated onset; event precision 0.34; 35 alerts emitted, 2 suppressed by the 60 s refractory, 37 candidate events; window AUPRC 0.124.

Per-patient breakdown (deployed model; subjects with target events or false alerts):

| Subject | Valid h | Events | Evaluable | Detected | Missed | Sensitivity | False alerts | FAR/24 h | Latency (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sub-098 | 0.0 | 1 | 0 | 0 | 0 | n/a | 0 | n/a | n/a |
| sub-100 | 112.3 | 0 | 0 | 0 | 0 | n/a | 4 | 0.85 | n/a |
| sub-101 | 74.2 | 0 | 0 | 0 | 0 | n/a | 6 | 1.94 | n/a |
| sub-103 | 89.1 | 1 | 1 | 1 | 0 | 1.00 | 0 | 0.00 | 41.0 |
| sub-104 | 98.0 | 0 | 0 | 0 | 0 | n/a | 1 | 0.25 | n/a |
| sub-105 | 109.1 | 0 | 0 | 0 | 0 | n/a | 1 | 0.22 | n/a |
| sub-106 | 99.1 | 0 | 0 | 0 | 0 | n/a | 4 | 0.97 | n/a |
| sub-107 | 105.6 | 0 | 0 | 0 | 0 | n/a | 2 | 0.45 | n/a |
| sub-109 | 113.2 | 1 | 1 | 1 | 0 | 1.00 | 2 | 0.42 | 47.0 |
| sub-110 | 98.9 | 2 | 2 | 1 | 1 | 0.50 | 0 | 0.00 | 40.0 |
| sub-115 | 269.2 | 5 | 5 | 5 | 0 | 1.00 | 3 | 0.27 | 30.0 |
| sub-116 | 99.7 | 5 | 5 | 3 | 2 | 0.60 | 0 | 0.00 | 85.0 |
| sub-121 | 0.0 | 3 | 0 | 0 | 0 | n/a | 0 | n/a | n/a |
| sub-124 | 66.6 | 1 | 1 | 1 | 0 | 1.00 | 0 | 0.00 | 285.0 |
| sub-125 | 1.5 | 2 | 0 | 0 | 0 | n/a | 0 | 0.00 | n/a |

## 4. Volunteer ADL negative cohort

Not provided: no pod recordings exist yet. `python -m ml.lyra.adl --lyra-negatives-dir PATH` implements the schema checks, the volunteer-wise calibration/evaluation split and the false-alarm report.

## 5. C99 export, parity and ESP32-S3 resources

- Exported: `XGB` via own exporter from XGBoost 3.4.1 JSON model (m2cgen 0.10.0 failed: TypeError on XGBoost 3.x base_score). Files: `firmware/model/` (`lyra_model.h`, `lyra_detector_config.h`, `lyra_feature_spec.h`, `lyra_features.[ch]`, `lyra_detector.[ch]`, `lyra_model_test.c`).
- Exported vs original estimator on 300,000 development windows: max |Δp| 1.49e-08, decision agreement 1.00000. The final evaluation scores the exported replica.
- Golden vectors (synthetic, `firmware/model/test/golden_vectors.bin`), host gcc C99: windows 43, features compared 312 (exact 312, max rel diff 0), probabilities 24 (max abs diff 0), detector steps 438, failures 0
- Real development data (local only): 31,206 windows from 2 recordings, 405,677/405,678 feature values bit-identical, max relative difference 2.2e-31 (passed: True); model on 200,000 feature vectors: max |Δp| 0, decision agreement 1.000000 (passed: True).
- Float32 pod inputs vs float64 training inputs: features differ by at most 1.1e-05 (relative).

| Measured with xtensa-esp-elf-gcc (crosstool-NG esp-14.2.0_20260121) 14.2.0 (-Os) | Bytes | Target |
|---|---:|---:|
| Flash: model | 8,851 | |
| Flash: feature extraction | 6,590 | |
| Flash: detector | 457 | |
| **Flash: LYRA package** | **15,898** | < 61,440 |
| SRAM: static (.data/.bss scratch) | 2,056 | |
| SRAM: peak stack (features + model) | 768 | |
| SRAM: one input window (ENV + 6-axis IMU, float32; requirement, computed) | 3,200 | |
| **SRAM: total** | **6,024** | < 8,192 |

ESP-IDF v5.5.6 application build for esp32s3 (`firmware/esp32s3_app`, size probe linking the package): per-object linked sizes `lyra_features.c.obj` flash 5,520 B, RAM 2,008 B; `lyra_detector.c.obj` flash 422 B, RAM 48 B; `main.c.obj` flash 9,185 B, RAM 3,200 B.

Scope: LYRA package only (model + features + detector + one input window); excludes RTOS, BLE/Wi-Fi, drivers and acquisition ring buffers beyond one window. Stack figures come from `-fstack-usage` (static, per function); no run on a board, so no runtime peak was measured.

## 6. Deviations from the task brief

- EMG is 256 Hz in every file (brief: 250 Hz); the files are authoritative.
- Signal path follows locked decision D12 (MyoWare **ENV** pin at 50 Hz), as the owner directed: the pod applies no digital 20-120 Hz band-pass or mains notch (the ENV output has neither). Training reproduces the MyoWare analog chain causally (1st-order HPF 20.8 Hz, rectification, 1st-order LPF 3.6 Hz, 20 ms bin averaging); no filtfilt anywhere.
- The 5 s configuration uses a 2.48 s hop: 2.5 s is not an integer number of 25 Hz samples.
- `sz_gen_m_tonicClonic` and `sz_uo_m_tonicClonic` do not occur in SeizeIT2 v1.1.0; all targets are `sz_foc_f2b`.
- Threshold grid extended below 0.20 (see section 2).
- m2cgen 0.10.0 cannot convert XGBoost 3.4 models (TypeError on base_score); XGBoost is exported by an own converter from XGBoost's JSON model. emlearn 0.23 (inline, 8-bit leaf probabilities) exports RF/ET; its quantisation is part of the deployed model and its deviation from scikit-learn is reported.
- RBF SVC is trained on a patient-stratified subsample of at most 6,000 windows per fit (O(n^2) kernel); its probabilities come from patient-wise inner-fold Platt calibration, not `probability=True`.
- No startup/recording-local normalisation is used: tree models need none and LR/SVC scalers are fitted per training fold.
- Final cohort sub-097..125 was used to train LYRA v1/v2 before this pipeline existed.

## 7. Limitations and hardware validation still required

- **Sensor domain:** training uses a software model of the MyoWare 2.0 ENV chain applied to SeizeIT2 Sensor Dot RAW EMG, not a MyoWare. Gain, offset, rectifier behaviour and the ADC must be calibrated on the pod so `env` arrives in the same uV-equivalent scale; amplitude features depend on it.
- **IMU:** SeizeIT2's chest module (25 Hz) stands in for the BMI270; its gyro unit is an estimate (125 deg/s per unit). Placement (D10) differs.
- **Labels and data:** in-hospital recordings only; latency is measured from the annotated focal onset, not from the convulsive phase; few target events; no daily-life negatives.
- **Hardware:** nothing has run on an ESP32-S3 board; acquisition, ring buffers, BLE and power are not part of the measured package.
- **Final cohort:** previously used to train LYRA v1/v2, so these numbers are not a pristine external test.
- **IMU scaling predates D19:** IMU is scaled by the per-recording median |acc| (`lyra_mlm.seizeit2.calibrate_imu`), not by the Stage 3.5 rule (D19, `assess_imu_scale`) that was merged afterwards. That rule also rejects recordings with a corrupt v1.1.0 Z axis, stuck axes or unstable resting gravity. Some recordings included here may fail it. Re-extract under D19 before relying on these numbers.
