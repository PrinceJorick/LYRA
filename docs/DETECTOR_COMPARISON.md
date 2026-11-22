# Which detector goes on the pod? A side-by-side

For Kyle and PrinceJorick, to decide which detector the ESP32-S3 pod runs. Written 2026-11-22 from the
code and reports in this repository. Descriptive only: no new training or evaluation was run for it.

> All three candidates are experimental research models trained on in-hospital SeizeIT2 data. None has
> run on pod hardware or on MyoWare / BMI270 signals. None is clinically validated. A detection is a
> *possible* event, never a confirmed seizure.

## 1. The candidates

| | **A. `mlm` Stage 4 RF** (Kyle) | **B. `ml/lyra` XGB** (PrinceJorick) | **C. `ml/seizeit2` v2 RF** (PrinceJorick) |
|---|---|---|---|
| Firmware code | `mlm/c99/` + `firmware/components/lyra_pod/` + generated `mlm/export/c99/` | `firmware/model/` | `models/seizeit2/lyra_seizeit2_model.h` (model only) |
| Model | Random forest, 50 trees, depth 8, class-balanced; plus a SIMULATED_ENV-only fallback forest | XGBoost, 40 trees, depth 4, Platt-calibrated | Random forest, 50 trees, depth 8 |
| Features | 22 `lyra_mlm` features (9 envelope, 9 IMU, 4 cross-modal) | 13 own features (5 envelope, 6 IMU, 2 cross-modal), `ml/lyra/feature_spec.json` | Same 22 `lyra_mlm` features as A |
| IMU scaling of the training data | Stage 3.5 rule **D19**: corrupt-Z, stuck-axis and unstable recordings rejected | Older median-\|acc\| calibration (`calibrate_imu`). His report says to re-extract under D19 before relying on the numbers | Older median-\|acc\| calibration |
| Training participants | sub-001…096 only | sub-001…096 (development) | **All 125**: no SeizeIT2 test set remains |
| Event logic | Decision 4 state machine (3 consecutive, entry/exit hysteresis, 6 s gap, 60 s cooldown), entry 0.97 chosen on CV | Own detector: threshold 0.015 on calibrated p, open after N=3, close after M=3, 60 s refractory | Decision 4 machine, entry 0.9684, exit 0.7684 (tuned) |
| Missing IMU | Falls back to the envelope-only model (fallback route) | The window is invalid and the detector holds | Suppressed (`missing_modality_policy: suppress`, `config.json`) |
| Pod inputs | ENV in the training envelope scale; ACC in g, GYR in deg/s | ENV in µV-equivalent, ACC in m/s², GYR in rad/s | ENV in the training envelope scale; g / deg/s |

## 2. Evidence (read the footnotes before comparing rows)

| | A (fallback, entry 0.97) | B (fusion XGB) | C |
|---|---|---|---|
| Development CV, sub-001…096 | 27 of 30 FBTCS detected; 0.38 false alarms per 24 h; median latency 54 s | 26 of 29 evaluable detected; 0.785 FA/24h; median latency 55.5 s | Nested CV over all participants: 42 of 44 evaluable; 0.32 FA/24h; median latency 54 s |
| Single evaluation on sub-097…125 | 17 of 21 detected (81%); 0.68 FA/24h; latency 67 s; precision 17.5% | 12 of 15 evaluable (80%); 0.24 FA/24h; latency 47.5 s; precision 0.34 | None possible (trained on them) |
| Probability calibration | None: uncalibrated forest scores, hence the high threshold | Platt calibration to the window prevalence, hence the threshold of 0.015 | None |

Footnotes:

1. **Denominators differ.**
   - A counts every FBTCS event in a processed recording; an event whose windows were all
     suppressed counts as missed.
   - B counts only *evaluable* events, those with at least one valid fusion window. On the
     sub-097…125 cohort, 6 of B's 21 events had no valid window and are left out of its 15.
   - Counted A's way, B would be 12 of 21.
   - The FA/24h denominators also differ: valid monitored hours, but with different validity rules.
2. **Do not choose by the sub-097…125 numbers.**
   - That cohort has now been used by A (once, clean), by B (once, but after v1/v2 had trained on
     it), and by C and v1 for training.
   - Picking whichever model looks best on it would be selecting on the test set, so those numbers
     would stop meaning anything.
   - Choose on development CV and engineering criteria. Then any future comparison needs new data:
     pod recordings.
3. **The data differ.** A uses D19-scaled IMU; B and C use the older calibration, which accepts some
   recordings that D19 rejects. Until B and C are rebuilt under D19, their numbers aren't on the same
   footing as A's.

## 3. On the ESP32-S3-DevKitC-1-N8R8 (8 MB flash, 512 KB SRAM)

| | A | B | C |
|---|---|---|---|
| Flash | ~701 KB for the two forests (m2cgen if/else, one function per tree, `-O2`); 646 KB linked with the pod runtime, libm and printf | **15.9 KB** for model + features + detector (`-Os`) | ~224 KB (flat arrays) for the model only |
| Static RAM | 54.7 KB, including the runtime's ring buffers (27 KB) and feature scratch (27 KB) | 6.0 KB, including one input window; no ring buffers | Not measured for a full pod build |
| Arithmetic | double (software-emulated on the S3); estimated ~30 ms per 2 s window | float32 | float32 inputs |
| What exists around the model | Streaming runtime (`lyra_pod`): ring buffers, 4 s / 2 s windows, per-window clock check, 5-state quality gate with fallback routing, confirmation, BLE event/status payloads, pod and phone cancel, data-collection logging. Parity-tested against Python. | Window validity check, 13 features, detector. Golden-vector parity. An ESP-IDF size-probe app (`firmware/esp32s3_app`). | A C header and JSON model only |

All three fit the board's flash and RAM, so footprint alone doesn't decide this. B is about 40×
smaller and cheaper to run. A's forests could shrink by about 2× with an array exporter like C's.

## 4. What's the same for all three

- They need a **per-device ENV calibration**: MyoWare mV to the training envelope scale. Every
  amplitude feature depends on it, and none of the three is valid on raw ADC values.
- Each was trained on a **chest IMU**, but the pod's IMU is on the upper arm.
- Each has **no pod data and no daily-life negatives**. The pod protocol (`mlm/docs/POD_DATA_PROTOCOL.md`,
  `scripts/pod_report.py`) is the next real evidence for any of them.

## 5. A suggested way to decide

1. **Agree on one data pipeline:** the D19-processed `lyra_mlm` windows. Otherwise the candidates
   aren't measured on the same data.
2. **Re-run B's selection on those windows**, with the same participant folds and the same event
   rules as A: development CV only, no sub-097…125. That gives a like-for-like development
   comparison of B's 13-feature XGB against A's 22-feature RF.
3. **Decide on that CV comparison plus engineering fit.** B is far smaller, uses float32 and is
   calibrated. A has fallback routing and the D19 data.
4. **Keep one runtime.** `lyra_pod` (windows, quality, BLE, cancel, logging) can host either model
   behind its scorer/step interface, so the runtime and the model decisions are separate.
5. **Revisit with pod data.** Run Phase 1 recordings through `pod_report.py` for the chosen model
   before any pod output is shown to a user as anything more than a test.
