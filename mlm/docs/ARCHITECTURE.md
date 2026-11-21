# LYRA MLM architecture (pre-training foundation)

Experimental research software. Every threshold and default below is an engineering placeholder,
not a validated seizure-detection value.

## 1. Where this fits

```
Wearable pod (ESP32-S3)                                         Phone (LYRA Companion)
 MyoWare 2.0 ENV ─┐                                             pairing, battery, alert +
 BMI270 acc+gyro ─┴─> validate → preprocess → window → features   cancel countdown, caregiver
                      → fusion → model → EventConfirmation ──BLE──> flow, history, observations
```

Primary inference runs on the ESP32-S3. The phone does not run the detector. Moving inference to the
phone needs an explicit architecture review. Wi-Fi CSI is a separate research track and is not an input
to this pipeline.

This package is the Python reference for every pod stage. The C99 port must reproduce its feature
order and fitted preprocessing exactly (`ExportManifest`, `EmbeddedBenchmark.parity_*`).

## 2. Data flow, quality and timestamps

```
Recording(participant, recording, origin, placement, annotations)
 ├─ emg: SignalStream  (native rate, signal_type RAW|RECT|ENV, bandwidth_hz, flags, provenance)
 └─ imu: SignalStream  (native rate, acc_x..gyr_z in g and deg/s, flags, provenance)
        │
        ▼ segment(window_cfg, quality_cfg, required)       check_sync() once per recording
Window(start_s, end_s, emg slice, imu slice, WindowQuality, annotation_refs)
        │   WindowQuality = per-modality ModalityQuality + overall state + sync reasons
        ▼ QualityGatedInference.process()
        │   route by quality → extract_features() → (normalize with stored params) → predict_score()
Prediction(timestamp_s, status VALID|DEGRADED|SUPPRESSED, score|None, class|None,
           available_modalities, quality, model_id, reason)
        ▼ EventConfirmation.update()
TransitionRecord(state, reason, event_started, positive_count, per-modality quality)
        ▼ ble.EventStatusPayload (origin byte: HARDWARE | SIMULATED)
```

- **Timestamps:** seconds on each stream's own clock. They are never rewritten except by explicit
  `resample()`, which records `"resampled X Hz -> Y Hz (method)"` in `provenance`.
- **Native rates:** a 4 s window holds 4·fs_emg sEMG samples and 4·fs_imu IMU samples. sEMG is never
  downsampled to the IMU's 25 Hz. Windows align modalities by shared boundaries, not a shared grid.
- **Missing vs invalid vs never acquired:** `SampleFlag.MISSING` (slot exists, value NaN),
  `SampleFlag.INVALID` (value present but rejected: non-finite, or source-flagged corrupt). Samples
  that were never acquired have no slot and appear as timestamp gaps.
- **Cross-modal alignment:** `check_sync` compares each stream's measured rate with its nominal rate.
  Passing it does not prove physical synchronization.

## 3. Modules

| Module | Responsibility | Depth |
|---|---|---|
| `signals` | Stream/recording schemas, structural validation, timestamp irregularity report | implemented |
| `annotations` | Annotation kinds, hard-negative taxonomy, `AnnotationMapper` | schema + contract |
| `sync` | Clock check, explicit `resample(method=...)` (linear upsample / polyphase) | implemented |
| `preprocess` | `FilterConfig` (Nyquist-checked), `apply_filter`, fixed scale, IMU magnitudes, normalization fit/apply | implemented |
| `quality` | Per-modality and per-window quality states | implemented |
| `windows` | Segmentation + window metadata | implemented |
| `features` | Feature math, specs, availability, fusion | implemented |
| `model` | `Prediction`, `ModelConfig` (+JSON), `ScorePredictor`, `Trainer`, `EmbeddedExporter`, `EmbeddedBenchmark` | contracts |
| `inference` | Quality-gated routing | implemented |
| `confirmation` | `EventConfirmation` | implemented |
| `evaluation` | `Partition`, `ParticipantSplit`, leakage check, `METRICS`, `EvaluationReport`, `Evaluator` | contracts |
| `datasets` | `SOURCES`, `RecordingInfo`, `DatasetAdapter`, `sha256_file`, `validate_recording` | registry + contract |
| `ble` | Draft payload layouts | implemented (draft) |

## 4. Signal-quality states

Assessed per modality, independent of any classifier (`quality.py` imports nothing from the model side,
and a test enforces this).

| State | Meaning | Examples |
|---|---|---|
| `VALID` | coverage ≥ 95%, no problems found | clean window |
| `DEGRADED` | usable, with recorded problems | some missing/invalid samples, a gap, slight saturation, some flat channels, irregular intervals |
| `INVALID` | samples present but untrustworthy | flatline on all channels, saturated > 5% (e.g. output pinned at a rail) |
| `MISSING` | modality not recorded, no samples, or coverage < 50% | disconnection, dropout |
| `UNSYNCHRONIZED` | duplicate/out-of-order timestamps in the modality, or cross-modality clock disagreement | swapped samples, IMU declared 25 Hz but running at 26 Hz |

The overall window state is the worst state among the required modalities
(UNSYNCHRONIZED > INVALID > MISSING > DEGRADED > VALID). Cross-modality clock problems make it
UNSYNCHRONIZED. The per-modality breakdown is always kept (e.g. `emg: MISSING, imu: VALID`), and
`available_modalities` lists the usable ones. Poor quality is never converted into a negative prediction.

Real MyoWare disconnection signatures (float vs rail) must be characterised on hardware. Today they are
caught as flatline, saturation or missing samples.

## 5. Preprocessing and leakage

- **Sensor-side vs software-side filtering.** MyoWare RECT/ENV are rectified and smoothed in analog
  hardware, and the BMI270 filters at its configured ODR. `FilterConfig` covers only additional
  software filtering. For RECT/ENV only `lowpass` is accepted: a high-pass or band-pass would strip the
  envelope level that is the measurement. The RAW research path stays separate.
- **Filter validation.** Each `FilterConfig` names modality, signal type, sample rate, Butterworth kind,
  order (1–8), cutoffs and a required rationale. Cutoffs must lie strictly inside (0, fs/2). Invalid
  configs raise `FilterConfigError` and are never adjusted. `apply_filter` also rejects a stream whose
  rate or signal type differs from the config. Each contiguous run is filtered separately, so gaps
  are never bridged. `causal=True` (sosfilt) is the embedded-compatible mode, and zero-phase is
  offline only.
- **Fixed transforms** (`apply_fixed_scale`) estimate nothing from data.
- **Data-fitted transforms** (`fit_normalization`) refuse any partition except `TRAIN` (`LeakageError`).
  `NormalizationParams` is frozen and refuses `fitted_on != TRAIN`. `apply_normalization` never changes
  it. The params are stored in `ModelConfig.normalization` so the pod can apply the same transform.
  NaN (unavailable) values are ignored during fitting and stay NaN when applied.
- **Bandwidth.** `SignalStream.effective_bandwidth_hz` = min(Nyquist, declared bandwidth). For RECT/ENV
  without a declared bandwidth it is `None`, and frequency-dependent features are then unavailable.
  **The MyoWare ENV bandwidth is not yet specified.** Set `bandwidth_hz` from the datasheet or a bench
  characterisation before relying on `emg_env_autocorr_peak` or `xmod_coherence_band`.

## 6. Features and fusion

Order is fixed per `FeatureConfig` (`feature_names(cfg)`). EMG names carry the signal type, so a model
trained on ENV features cannot receive RAW ones. Unavailable means NaN plus `available=False` plus a
reason, while a measured 0.0 stays available. If a modality is unusable in the window, its whole block
is unavailable with the reason `quality: emg MISSING`. Fusion concatenates the EMG, IMU and cross-modal
blocks and keeps the window's `WindowQuality`.

Multimodal ENV schema (19 features). RAW/RECT `emg` mode drops `autocorr_peak` (5 EMG features).
Multimodal mode currently supports ENV only. RAW fusion is deferred research.

| Feature | Unit | Requirements |
|---|---|---|
| `emg_env_rms` | sensor units | >=1 usable sample |
| `emg_env_mav` | sensor units | >=1 usable sample |
| `emg_env_var` | sensor units^2 | >=2 usable samples |
| `emg_env_std` | sensor units | >=2 usable samples |
| `emg_env_wl_per_s` | sensor units/s | contiguous runs of >=2 samples; waveform length per second |
| `emg_env_autocorr_peak` | unitless | ENV only; declared bandwidth >= 1/min lag; contiguous run >= 2x max lag |
| `imu_acc_mag_mean` | g | >=2 usable samples; includes gravity |
| `imu_acc_mag_std` | g | >=2 usable samples |
| `imu_gyr_mag_mean` | deg/s | >=2 usable samples |
| `imu_gyr_mag_std` | deg/s | >=2 usable samples |
| `imu_gyr_mag_rms` | deg/s | >=2 usable samples |
| `imu_acc_jerk_rms` | g/s | contiguous runs of >=2 samples |
| `imu_acc_dominant_freq_hz` | Hz | bandwidth >= band top; run >= 2 cycles of band bottom; energy in band |
| `imu_acc_band_power` | g^2 | bandwidth >= band top; run >= 2 cycles of band bottom |
| `imu_orientation_change_deg` | deg | >=4 usable samples; first vs last quarter gravity estimate (biased by dynamic acceleration) |
| `xmod_env_accmag_corr` | unitless | ENV + IMU usable; >= min_paired_samples ENV bins on the IMU timeline |
| `xmod_energy_log_ratio` | log10(units/g) | non-zero energy in both; same-setup comparisons only |
| `xmod_lag_s` | s | gap-free IMU timeline fully covered by ENV; positive = IMU follows EMG; not proof of physical sync |
| `xmod_coherence_band` | unitless | both bandwidths >= band top; >= 2 coherence segments; few segments bias it high |

Cross-modal features need ENV on the IMU timeline. They use a centred boxcar average of ENV samples
within ±½ IMU period of each IMU sample. This is local to those features: the streams themselves are
not resampled.

## 7. Quality-gated inference

`InferenceConfig(degraded_policy, missing_modality_policy)` has no defaults: both policies must be chosen.

1. Multimodal model and cross-modality clock disagreement → `SUPPRESSED`.
2. Every required modality usable → primary model.
3. Multimodal model with exactly one modality unusable → `"suppress"`, or `"fallback_unimodal"` using
   a configured model for the remaining modality (the reason records the fallback).
4. Any used modality `DEGRADED` → `degraded_policy` decides: `"proceed"` gives status `DEGRADED`, and
   `"suppress"` gives `SUPPRESSED`.
5. Any feature unavailable → `SUPPRESSED`, with the feature names in the reason.

`SUPPRESSED` always has `score=None, predicted_class=None`, and the `Prediction` schema enforces this.

## 8. Event confirmation

States: `IDLE → ACCUMULATING → POSSIBLE_EVENT → REFRACTORY → IDLE`. Each update appends a
`TransitionRecord` with its reason. `event_started` is True only on entry to `POSSIBLE_EVENT`, which is
the single notification point. On that event the phone shows
**"Possible convulsive seizure detected. This is not a diagnosis."**

| Parameter | Default (placeholder) | Meaning |
|---|---|---|
| `entry_threshold` | 0.5 | classification threshold: score ≥ entry is a positive window |
| `exit_threshold` | 0.4 | hysteresis: an active possible event continues while score ≥ exit |
| `consecutive_required` | 2 | valid positive windows needed |
| `max_gap_s` | 4.0 | tolerates one isolated missing window at the 2 s step; longer gaps reset |
| `refractory_s` | 60 | cooldown from the end of a possible event |
| `max_latency_s` | 10 | predictions older than this at `now_s` are rejected as stale |
| `degraded_policy` | `"use"` | DEGRADED predictions count as evidence (`"ignore"` treats them as no evidence) |
| `invalid_window_policy` | `"hold"` | SUPPRESSED windows keep the positive count (`"reset"` clears it) |

Out-of-order and duplicate timestamps are rejected without a state change. A per-modality quality
change is logged as its own history record. `reset()` handles user cancellation and recovery.

## 9. Reconciliation with the rest of the repo

- **Companion app (`src/services/ml`).** Its `MlService` assumes phone-side detection, and its
  `validateWindow` rejects channels of different lengths. Both conflict with this architecture
  (detection on the ESP32-S3, native rates kept). When BLE is integrated, the app should consume
  `EventStatusPayload`/`DeviceStatusPayload` rather than run detection. The app was not changed here.
- **Alert wording.** The required alert text above differs from the app's current "Possible Event"
  wording. The app was not changed. Align it in an app session.
- **Countdown.** The app defaults to 8 s and the pod spec mentions 10 s. This is unresolved, and
  `DeviceConfigPayload.countdown_s` accepts 7–10.
- **context.md.** It still lists hardware, placement, inference location and window parameters as
  undecided (§5, §14 U1–U10). This brief resolves several of them: GTCS/FBTCS scope, deltoid placement,
  25 Hz IMU, on-device inference, 4 s/50% windows, RF as the primary candidate. context.md was not edited.

## 10. Known limitations

- MyoWare ENV acquisition rate and analog bandwidth are not specified. The tests use a synthetic 100 Hz
  stream with a declared 20 Hz bandwidth.
- Quality thresholds, filter choices, feature parameters and confirmation parameters are placeholders.
- `check_sync` detects rate disagreement only. Without shared sync markers, fixed clock offsets between
  the sensors cannot be detected.
- Coherence on a 4 s, 25 Hz window has about 3 segments and is biased high.
- Orientation change assumes low dynamic acceleration.
- Window buffering of live sample chunks is left to firmware. `QualityGatedInference` works per window.
- Python model-weight serialization depends on the chosen library and is deferred. `ModelConfig`
  JSON is implemented.

## 11. Deferred (explicitly not done)

Model training or fitting, hyperparameter search, threshold optimisation, dataset download or
processing, annotation mapping, real windows and splits, evaluation and metrics, C99 export, on-device
benchmarks, firmware, phone BLE transport, and Wi-Fi CSI. See [TRAINING_PLAN.md](TRAINING_PLAN.md).
