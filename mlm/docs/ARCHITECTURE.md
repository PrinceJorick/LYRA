# LYRA MLM architecture

Experimental research software. Every threshold and default below is an unvalidated engineering
parameter, not a seizure-detection or clinical value. Stage 3 (dataset preparation) is implemented.
No model has been trained.

## 1. Where this fits

```
Wearable pod (ESP32-S3)                                         Phone (LYRA Companion)
 MyoWare 2.0 ENV (50 Hz) ─┐                                     pairing, battery, alert +
 BMI270 acc+gyro (25 Hz) ─┴─> validate → window → features        cancel countdown, caregiver
                              → fusion → model → EventConfirmation ─BLE─> flow, history, reports
```

Primary inference runs on the ESP32-S3. The phone does not run the detector, and the app's
`src/services/ml/` is legacy. Moving inference to the phone needs an explicit architecture review.
Wi-Fi CSI is a separate research track and is not an input to this pipeline.

This package is the Python reference for every pod stage. The C99 port must reproduce its feature
order and any fitted preprocessing exactly (`ExportManifest`, `EmbeddedBenchmark.parity_*`).

## 2. Data flow, quality and timestamps

```
Recording(participant, recording, origin, placement, annotations)
 ├─ emg: SignalStream  (native rate, signal_type RAW|RECT|ENV|SIMULATED_ENV_FROM_RAW, flags, provenance)
 └─ imu: SignalStream  (native rate, acc_x..gyr_z, flags, provenance)
        │   SeizeIT2: RAW 256 Hz ──raw_to_simulated_env──> SIMULATED_ENV_FROM_RAW 50 Hz (primary)
        │                     └── RAW kept as the secondary research path
        ▼ segment(window_cfg, quality_cfg, required)       check_sync() once per recording
Window(start_s, end_s, emg slice, imu slice, WindowQuality, annotation_refs)
        │   label_window(): POSITIVE / NEGATIVE / IGNORE (Decision 1)
        ▼ QualityGatedInference.process()
        │   route by quality → extract_features() → (normalize with stored params) → predict_score()
Prediction(timestamp_s, status VALID|DEGRADED|SUPPRESSED, score|None, class|None, ...)
        ▼ EventConfirmation.update()
TransitionRecord(state, reason, event_started, positive_count, per-modality quality)
        ▼ ble.EventStatusPayload (origin byte: HARDWARE | SIMULATED)
```

- **Timestamps:** seconds on each stream's own clock. They are never rewritten except by explicit,
  logged transforms (`resample`, `raw_to_simulated_env`), which record source and target rates in
  `provenance`.
- **Native rates:** a 4 s window holds 200 ENV samples at 50 Hz and 100 IMU samples at 25 Hz. That's
  an exact 2:1 ratio with no forced resampling. sEMG is never downsampled to the IMU's 25 Hz.
- **Missing vs invalid vs never acquired:** `SampleFlag.MISSING` (slot, no value),
  `SampleFlag.INVALID` (value rejected), and no slot at all (a timestamp gap).
- **Cross-modal alignment** is by shared window boundaries. `check_sync` only rules out gross rate
  disagreement. It does not prove physical synchronization.

## 3. Modules

| Module | Responsibility | Depth |
|---|---|---|
| `signals` | Stream/recording schemas, structural validation, timestamp report, O(log n) slicing | implemented |
| `annotations` | Annotation kinds, hard-negative taxonomy, `LabelConfig` + `label_window` (Decision 1) | implemented |
| `sync` | Clock check, explicit `resample(method=...)` | implemented |
| `preprocess` | `FilterConfig`, `apply_filter`, `raw_to_simulated_env` (Decision 2), fixed scale, normalization fit/apply | implemented |
| `quality` | Per-modality and per-window quality states | implemented |
| `windows` | Segmentation + window metadata | implemented |
| `features` | Feature math, specs, availability, fusion, bandwidth gating (Decision 3) | implemented |
| `model` | `Prediction`, `ModelConfig` (+JSON), predictor/trainer/exporter contracts | contracts |
| `inference` | Quality-gated routing (Decision 4 defaults) | implemented |
| `confirmation` | `EventConfirmation` (Decision 4 defaults) | implemented |
| `evaluation` | Partitions, `ParticipantSplit`, leakage check, `split_manifest`, metric definitions | implemented / contracts |
| `datasets` | Source registry, `RecordingInfo`, adapter contract, integrity checks | implemented |
| `edf` | Minimal 16-bit EDF/EDF+ reader (header-driven rates, EDF+D onsets) | implemented |
| `seizeit2` | SeizeIT2 BIDS adapter + label mapper | implemented |
| `prep` | Stage 3 per-recording preparation (condition, segment, label, extract) | implemented |
| `ble` | Draft payload layouts | implemented (draft) |

## 4. Signal-quality states

Assessed per modality, independent of any classifier (`quality.py` imports nothing from the model
side, and a test enforces this).

| State | Meaning | Examples |
|---|---|---|
| `VALID` | coverage ≥ 95%, no problems found | clean window |
| `DEGRADED` | usable, with recorded problems | some missing/invalid samples, a gap, slight saturation, some flat channels, irregular intervals |
| `INVALID` | samples present but untrustworthy | flatline on all channels, saturated > 5% |
| `MISSING` | modality not recorded, no samples, or coverage < 50% | disconnection, dropout, file absent |
| `UNSYNCHRONIZED` | duplicate/out-of-order timestamps, or cross-modality clock disagreement | swapped samples, misdeclared rate |

The overall window state is the worst state among the required modalities. The per-modality
breakdown and `available_modalities` are always kept, and poor quality is never turned into a
negative prediction. Saturation limits are only meaningful with known sensor full-scale values:
SeizeIT2's EDF physical ranges are fitted to each file's data, so no saturation limits are set for it.

## 5. Preprocessing, conditioning and leakage

- **`raw_to_simulated_env` (Decision 2, primary SeizeIT2 path):** RAW sEMG goes through a 1st-order
  Butterworth high-pass at 20.8 Hz, full-wave rectification (`abs`), and a 1st-order Butterworth
  low-pass at 3.6 Hz, then `resample_poly` to 50 Hz (FIR anti-aliasing, edge padding). The filters
  are causal (`sosfilt`) because the analog stages they model are causal. Each contiguous run is
  processed separately, so gaps are never bridged. Cutoffs are validated against the source Nyquist
  and invalid configurations raise `FilterConfigError`. The output is tagged
  `SIMULATED_ENV_FROM_RAW` and its provenance names every step. It is never labelled hardware ENV.
- **Known conditioning limitations:**
  1. **Digital rectification aliasing:** rectifying at the 256 Hz source rate creates harmonics
     that alias before the LPF (for a 60 Hz tone, the 240 Hz harmonic folds to 16 Hz). The analog
     MyoWare rectifies in continuous time. Oversampling before rectification would reduce this,
     and is a candidate refinement that needs a decision.
  2. Gain, rectifier non-idealities and output offset of the real MyoWare are not modelled.
  3. The default zero padding in `resample_poly` distorted the first and last ~0.5 s of every run
     by up to ~40%. Edge padding is used instead (found and fixed in Stage 3).
- **Sensor-side vs software-side filtering.** RECT, ENV and SIMULATED_ENV_FROM_RAW are already
  rectified/smoothed, so `FilterConfig` accepts only `lowpass` for them. RAW stays a separate path.
- **Filter validation:** cutoffs must lie strictly inside (0, fs/2), order 1–8, and a rationale is
  required. Invalid configurations are rejected and never adjusted.
- **Data-fitted transforms** (`fit_normalization`) refuse any partition except `TRAIN`.
  **No normalization has been fitted on real data.** Stage 3 stores raw feature values plus
  availability masks.

## 6. Features and fusion (Decision 3)

Order is fixed per `FeatureConfig`. EMG names carry the signal type (`emg_env_*`, `emg_simenv_*`,
`emg_raw_*`), so a model trained on simulated ENV can't silently receive hardware ENV or RAW features.
Transferring from simulated to hardware ENV is a deliberate, documented step. Unavailable means NaN
plus `available=False` plus a reason, while a measured 0.0 stays available.

- **Envelope types (ENV, SIMULATED_ENV_FROM_RAW):** time-domain features plus envelope-modulation
  features in the **1.5–5.0 Hz** clonic band, only when the envelope rate is **≥ 20 Hz**.
  High-frequency raw-EMG spectral features are never in an envelope schema.
- **RAW:** time-domain features plus 20–250 Hz spectral features (median/mean frequency, band
  power), with the band capped at the stream's bandwidth. For SeizeIT2 at 256 Hz that's 20–128 Hz.
- **Multimodal** needs an envelope type. RAW+IMU fusion is deferred research.

Stage 3 primary schema, `FeatureConfig("multimodal", SIMULATED_ENV_FROM_RAW)` (22 features):

| Feature | Unit | Requirements |
|---|---|---|
| `emg_simenv_rms` | sensor units | >=1 usable sample |
| `emg_simenv_mav` | sensor units | >=1 usable sample |
| `emg_simenv_var` | sensor units^2 | >=2 usable samples |
| `emg_simenv_std` | sensor units | >=2 usable samples |
| `emg_simenv_wl_per_s` | sensor units/s | contiguous runs of >=2 samples; waveform length per second |
| `emg_simenv_autocorr_peak` | unitless | envelope ≥ 20 Hz; run ≥ 2 cycles of 1.5 Hz; peak autocorrelation at lags 1/5..1/1.5 s |
| `emg_simenv_mod_band_power` | units^2 | envelope ≥ 20 Hz; power in 1.5–5.0 Hz |
| `emg_simenv_mod_band_ratio` | fraction | envelope ≥ 20 Hz; 1.5–5.0 Hz power / total envelope power |
| `emg_simenv_mod_peak_freq_hz` | Hz | envelope ≥ 20 Hz; spectral peak in 1.5–5.0 Hz |
| `imu_acc_mag_mean` | g* | >=2 usable samples; includes gravity |
| `imu_acc_mag_std` | g* | >=2 usable samples |
| `imu_gyr_mag_mean` | deg/s* | >=2 usable samples |
| `imu_gyr_mag_std` | deg/s* | >=2 usable samples |
| `imu_gyr_mag_rms` | deg/s* | >=2 usable samples |
| `imu_acc_jerk_rms` | g/s* | contiguous runs of >=2 samples |
| `imu_acc_dominant_freq_hz` | Hz | IMU bandwidth ≥ 8 Hz; run ≥ 2 cycles of 1 Hz; energy in 1–8 Hz |
| `imu_acc_band_power` | g^2* | IMU bandwidth ≥ 8 Hz; run ≥ 2 cycles of 1 Hz |
| `imu_orientation_change_deg` | deg | >=4 usable samples; first vs last quarter gravity estimate |
| `xmod_env_accmag_corr` | unitless | envelope + IMU usable; ≥ 20 envelope bins on the IMU timeline |
| `xmod_energy_log_ratio` | log10 | non-zero energy in both; same-setup comparisons only |
| `xmod_lag_s` | s | gap-free IMU timeline fully covered by the envelope; not proof of physical sync |
| `xmod_coherence_band` | unitless | envelope ≥ 20 Hz; IMU bandwidth ≥ 5 Hz; mean coherence in 1.5–5.0 Hz; few segments bias it high |

\* Hardware units. For SeizeIT2 the IMU is in **source units** (see DATASETS.md): the EDF labels
the gyro "g", and the accelerometer reads |a| ≈ 0.5 at rest, so its scale is unverified. Scale-free
features (frequencies, ratios, correlations, angles) are unaffected.

Secondary schema, `FeatureConfig("emg", RAW)` (8 features): `emg_raw_rms`, `_mav`, `_var`, `_std`,
`_wl_per_s`, `_median_freq_hz`, `_mean_freq_hz`, `_high_band_power`.

## 7. Quality-gated inference (Decision 4)

`InferenceConfig()` defaults: `degraded_window_policy="allow_with_flag"` (infer, keep the `DEGRADED`
status) and `missing_modality_policy="suppress"`. Required modalities that are `INVALID`, `MISSING`
or `UNSYNCHRONIZED`, cross-modality clock problems, and unavailable features all produce
`SUPPRESSED` with `score=None, predicted_class=None`. `"fallback_unimodal"` stays available as an
explicit option.

## 8. Event confirmation (Decision 4)

States `IDLE → ACCUMULATING → POSSIBLE_EVENT → REFRACTORY → IDLE`. `event_started` marks the single
notification point. The phone then shows **"Possible convulsive seizure detected. This is not a
diagnosis."**

| Parameter | Default | Meaning |
|---|---|---|
| `entry_threshold` | 0.65 | score ≥ entry is a positive window |
| `exit_threshold` | 0.45 | an active possible event continues while score ≥ exit |
| `consecutive_positive_required` | 3 | ≥ 8 s confirmation latency at 4 s / 2 s |
| `invalid_window_policy` | `"hold"` | SUPPRESSED windows neither add evidence nor reset progress |
| `max_timestamp_gap_s` | 6.0 | up to two consecutive missing windows are held; longer gaps reset |
| `degraded_window_policy` | `"allow_with_flag"` | DEGRADED predictions count as evidence |
| `refractory_cooldown_s` | 60.0 | cooldown from the end of a possible event |
| `max_latency_s` | 10.0 | predictions older than this at `now_s` are rejected as stale |

Interpretation note: a gap longer than 6 s resets an accumulating run to `IDLE`. If a possible
event is already active, the gap ends it in `REFRACTORY` rather than `IDLE`, so a data dropout
mid-episode cannot trigger a second notification. Confirm or change this in Stage 4.

## 9. Reconciliation with the rest of the repo

- **Companion app:** `src/services/ml/` is legacy (phone-side detection, equal channel lengths).
  The app should consume `EventStatusPayload`/`DeviceStatusPayload` when BLE lands. It was not
  modified in Stage 3.
- **Alert wording:** the required text above differs from the app's "Possible Event". Align it in an
  app session.
- **Countdown:** the app defaults to 8 s and the pod spec mentions 10 s. Still unresolved.
- **context.md** was updated on 2026-11-21 with the hardware and decisions D9–D18.

## 10. Known limitations

- SeizeIT2 is a focal-epilepsy, in-hospital cohort. FBTCS windows are not GTCS evidence and not
  clinical validation.
- SeizeIT2's IMU is on the chest (ECGEMG module) while LYRA's is on the upper arm, and its IMU scale
  and gyro units are unverified (§6).
- SeizeIT2 EMG/MOV header start times are anonymised differently, so both are assumed to start at t = 0.
- Quality thresholds and feature parameters are placeholders. Coherence on 4 s windows is biased high.
- Window buffering of live sample chunks is left to firmware.

## 11. Deferred

Model training or fitting, normalization fitting on real data, hyperparameter search, threshold
optimisation, evaluation metrics, C99 export, on-device benchmarks, firmware, phone BLE transport and
Wi-Fi CSI. See [TRAINING_PLAN.md](TRAINING_PLAN.md).
