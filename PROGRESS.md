# LYRA Companion: Progress Log

Working notes for future sessions. This file records what was built and what was decided. It is not a product spec and does not replace `context.md`.

## Session 1 (2026-11-20): first full build of the app

### What exists
Expo SDK 58 / React Native 0.88 / Expo Router / TypeScript (strict) app, iOS and Android. In-memory state only. No real BLE, SMS or ML.

- **Tabs:** Home, History, Insights, Reports, Care.
- **Other routes:** pairing, pod details, event details, possible-event alert, observation form, clinician summary preview, context log, caregiver form, settings, about.
- **Before/During/After flow:**
  - Before: context notes, triggers, user-entered medication log (no advice).
  - During: full-screen alert with a 7–10 s cancel countdown (default 8) and a simulated caregiver notification.
  - After: observation form, then a clinician summary preview.
- **Theme:** lavender/white, deep-purple buttons, amber warnings. Light by default, dark mode in Settings.

### Where things live
| Area | Path |
|---|---|
| Routes | `src/app/` |
| Shared UI and graphics | `src/components/ui.tsx`, `graphics.tsx`, `EventCard.tsx`, `NotificationStatus.tsx`, `HardwareList.tsx` |
| State (reducer + context) | `src/state/AppState.tsx` |
| Theme / palettes | `src/theme/colors.ts`, `useTheme.ts` |
| Mock data (central) | `src/data/` |
| Service seams | `src/services/{device,ml,events,notifications,reports}/` |
| Types | `src/types/index.ts` |

### Integration seams (where real code replaces the mocks)
- `DeviceService` (`src/services/device`): swap the mock for real BLE.
- `MlService` (`src/services/ml`): swap the mock for a real model. The contracts are `SensorWindow`, `ModelMetadata` and `DetectionResult`. The mock `detect` always returns `no_event`, and `demoTrigger()` produces the demo alert.
- `NotificationService` (`src/services/notifications`): swap the mock for real push/SMS. Sending, delivery and acknowledgment are separate statuses; acknowledgment is "not simulated".

### Decisions worth remembering
- **Never** use the labels "Seizure Confirmed" or "Seizure Detected". The UI says "Possible Event".
- No fake confidence scores or risk percentages anywhere. The ML screen says "ML model not integrated".
- All demo data is labelled ("Demo data", "Simulated", "Illustrative Sample Data"). Events carry `origin` (`demo_sample` | `simulated_alert`) and `outcome` (`sample_record` | `cancelled` | `notification_simulated`).
- The device is a **pod**, not a smartwatch or wristband. Say "LYRA Pod" / "LYRA Device" / "Connected Pod".
- `context.md` is **not** a source of truth for this app. Don't create or update it unless asked.
- Status text colours (`successText` #1B7452, `warningText` #8F5A0A) are darkened from the brief for AA contrast. Fill colours still follow the brief.
- Insights windows are anchored to the most recent record, so the fixed Nov-2026 sample dates stay meaningful.
- Alert countdown uses a wall-clock deadline (not tick counting) and ignores preference changes made mid-alert. Android back is blocked during the countdown; use the Cancel Alert button.
- `useStyles(factory)` needs a module-level factory (memoised on the palette only).
- Hardware parts shown in the app (ESP32-S3, MyoWare 2.0, BMI270, LiPo, BQ24074, TPS63031, MAX17048) come from the user's brief. They are descriptive only, with no hardware claims.

### Gotchas hit
- `tsconfig.json` **must** exclude `dist` or `tsc` crashes on exported bundles. Expo also rewrites `include` to `["src","*.d.ts"]` on start; that's fine.
- Web tab labels clip unless the tab bar has an explicit web height (set in `(tabs)/_layout.tsx`).
- SVG gradient ids collide when several pods are mounted, so `PodGraphic` uses a `useId`-based id.
- React 19 lint rules (`react-hooks` refs/purity) shaped `alert.tsx`: deadline kept in a ref set inside the effect, callbacks kept in a ref updated in an effect.

### Verified
- `npx tsc --noEmit` and `npx eslint src` are clean.
- Web build driven with headless Chrome (puppeteer-core), no console errors: alert cancel and timeout, pod disconnect/pair/reconnect, observation → summary ready → summary preview, caregiver add/edit/remove with confirm, countdown cap at 10 s, alert with no caregivers, dark mode.

### Not verified / known limits
- **Never run on iOS or Android.** No simulators were installed. Back gesture, screen-reader announcements and the native Share sheet are untested.
- **No hardware testing or real-device validation of any kind.**
- State is in memory and resets on app restart.
- The pod graphic is an abstract SVG, not a real render.
- Default countdown is 8 s; the hardware spec mentions a 10 s pod cancel window. Confirm which is intended.
- Only the Ponytail skill is installed. Superpowers, UIUX Promax and Caveman were not available and were not used.
- (Superseded: session 1 work was later committed as `5c0e664` and `e97cd8f`.)

### How to run
```bash
npm install
npx expo start          # scan QR with Expo Go, or press w for web
npx tsc --noEmit
npx eslint src
```
The dev server was last left running on `http://localhost:8081` (web). Stop it with `pkill -f "expo start"`.

### Suggested next steps
1. Run on a real phone via Expo Go and fix whatever native-only issues appear.
2. Persist state (e.g. `expo-sqlite` or AsyncStorage) behind the existing reducer.
3. Decide the 8 s vs 10 s cancel window.
4. Replace `mockDeviceService` with real BLE once the pod firmware's GATT profile is defined (needs a development build, not Expo Go).
5. Replace `mockMlService` once a model and its `ModelMetadata` exist.
6. Real notification channel, with caregiver consent flow.

## Session 2 (2026-11-21): MLM software foundation, pre-training phase

### What exists
A Python package in `mlm/` (separate from the Expo app; `src/` stays JS-only). Python >= 3.11, numpy, scipy, pytest. Venv at `mlm/.venv` (uv, Python 3.12).
- Modules in `mlm/src/lyra_mlm/`: signals, annotations, sync, preprocess, quality, windows, features, model, inference, confirmation, evaluation, datasets, ble.
- 104 pytest tests on synthetic fixtures (`mlm/tests/`), all passing. `StubScorePredictor` exists in `tests/synthetic.py` only.
- Docs: `mlm/README.md`, `mlm/docs/{ARCHITECTURE,DATASETS,TRAINING_PLAN,BLE_PROTOCOL}.md`.

### Hard boundaries kept
No model trained (`.fit()` never called; a test greps for it). No dataset downloaded. No metrics. No export. Wi-Fi CSI not integrated.

### Decisions worth remembering
- Native sampling rates are preserved end to end. `resample()` needs an explicit method and is never called by the pipeline.
- Quality states (VALID/DEGRADED/INVALID/MISSING/UNSYNCHRONIZED) live in `quality.py`, which must not import model code.
- SUPPRESSED predictions have `score=None, class=None` and never count as negatives. `invalid_window_policy` defaults to "hold".
- EMG feature names include the signal type (`emg_env_*` vs `emg_raw_*`). Multimodal currently supports ENV only.
- RECT/ENV filters may only be lowpass. ENV frequency features stay unavailable until `bandwidth_hz` is declared.
- Confirmation states are IDLE/ACCUMULATING/POSSIBLE_EVENT/REFRACTORY. There is no "confirmed seizure" state.

### Open items
- MyoWare ENV sample rate and analog bandwidth: not specified.
- Required alert text "Possible convulsive seizure detected. This is not a diagnosis." differs from the app's "Possible Event" wording. The app was not changed.
- The app's `MlService` assumes phone-side detection and equal channel lengths, which conflicts with on-pod inference and native rates. Align when BLE lands.
- context.md §5/§14 is out of date against the finalized hardware brief. Not edited (policy: only when asked).
- Superpowers and Research Superpowers are not installed. Installing them needs the user's authorization.
- `node_modules` was absent at session start, so the app's tsc/eslint were not re-run. The app code was not touched.

### How to run
```bash
cd mlm && .venv/bin/python -m pytest -q
```

## Session 3 (2026-11-21): Stage 3, dataset preparation (complete)

### What exists
- SeizeIT2 wearable subset in `mlm/data/seizeit2/` (gitignored): 13,568 files, 52.8 GB, size-checked, SHA-256 for every file. EEG and ECG were not downloaded.
- `mlm/data/processed/` (gitignored): 2,806 window files (20,296,484 windows), plus `inventory.json`, `feature_schema.json`, `windows_manifest.json`, `splits_manifest.json` and `audit.json`.
- New code: `edf.py` (EDF reader), `seizeit2.py` (adapter + mapper), `prep.py`, `raw_to_simulated_env`, `label_window`, `split_manifest`, `inherit_source_quality`. Scripts: `download_seizeit2.py`, `prepare_seizeit2.py`, `audit_stage3.py`.
- 134 tests pass. The guard test still forbids `.fit(`, ML imports and model files. sklearn and the like are not installed.
- context.md updated (v1.2: hardware, decisions D9–D18, status). `src/services/ml/` is marked legacy; the app was not modified.

### Verified facts
125 participants, 2,850 recordings, 11,626 h, 883 seizures, **55 FBTCS (37 participants), 0 GTCS**. EMG 256 Hz and MOV 25 Hz on every file. 44 recordings have no EMG and no chest IMU and are excluded. sub-097 has no usable recordings. 5 negative-duration `impd` rows were skipped (sub-031 run-07, sub-043 run-02). FBTCS with usable windows: 30/33 Train/CV, 17/22 held-out.

### Defects found and fixed in the Stage 3 audit
- `resample_poly` zero padding distorted run edges by up to ~40% → `padtype="edge"`.
- Checksums missed 4 top-level files → now hashed from the download listing (13,568).
- Window files lacked the SIMULATED_ENV_FROM_RAW provenance → stored, and 299 old files backfilled.
- The simulated ENV of flatlined RAW was rated VALID (92,382 windows, 378 recordings, no positives) → derived quality now inherits the source quality. Affected recordings were reprocessed, and an old-vs-new diff showed only the EMG-quality fields changed.

### Gotchas
- 8 GB RAM: 5 workers thrashed (Chrome + VS Code use ~6 GB). 3 workers fit: about 2,400 windows/s, ~2.5 h for everything.
- macOS uses spawn: pass paths to workers explicitly. Background runs from the Claude session die with it unless detached (PPID 1).
- SeizeIT2 IMU: the EDF labels the gyro "g" and resting |acc| ≈ 0.5, so units are kept as given (unverified).

### Git incident after Stage 3 (fixed)
- A rebase followed by a pull produced merge `0a50a6e` ("Initial Commit"), which was pushed with unresolved conflict markers in 9 files. `lyra_mlm` stopped importing and pytest collected nothing.
- Fixed in `5919031` (pushed): the 9 files were restored from `1c1a9a6`. The tree now matches `1c1a9a6` exactly, and 134 tests pass on a clean checkout of `origin/main`.
- History now shows each Stage 2–3 commit twice (identical trees, different commit dates). That's cosmetic; it was deliberately left alone, with no force-push.
- A local `.git/hooks/pre-commit` runs `git diff --cached --check` and blocks commits with conflict markers or trailing whitespace. It isn't versioned, so other clones need it added by hand.

### State at end of session
`main` = `origin/main`, working tree clean, 134 tests passing. Stage 3 processing and audit are done, but two blockers (below) were found afterwards, so **Stage 4 must not start yet**.

### Stage 4 blockers (found after the Stage 3 audit)
1. **SeizeIT2 accelerometer scale is inconsistent between recordings.** This is a Stage 3 data defect, verified on Train/CV recordings only. The median |acc| per recording falls into clusters instead of one value: ~0.167 (1,403 recordings, 62 participants; likely ±6 g scaled to ±1), ~0.5 (203; ±2 g), 32.768 (203, 6 participants: **ACC Z corrupted**, swinging between the ±32.768 rails) and other (49, e.g. sub-083 ~0.07).
   - Stage 3 missed it: there are no known SeizeIT2 saturation limits, and "resting |acc| ≈ 0.5" came from a single recording. Corrupted-Z windows were rated DEGRADED/VALID.
   - The positive rate differs ~2.7× between clusters (176 vs 65 positives per 1M negatives), so a model could learn device configuration instead of seizures.
   - Affected: `imu_acc_mag_mean/std`, `imu_acc_jerk_rms`, `imu_acc_band_power`, `xmod_energy_log_ratio`, and anything using ACC Z. EMG is not affected. Gyro scale is also unverified.
   - Proposed fix (needs sign-off): mark ACC samples at the ±32.768 rails INVALID; calibrate each recording so gravity at rest = 1 g (that recording's data only, same rule for all, nothing fitted across participants); add an audit check; first check `github.com/biomedepi/seizeit2` for official scaling. Then rerun the IMU features only.
2. **Branch `origin/seizeit2-model` (PrinceJorick) uses a different participant split.** It adds a second pipeline (`ml/seizeit2/`), a trained HGB model (`models/seizeit2/`, 5 s windows) and phone-side app integration (`src/services/ml/seizeit2/`).
   - 22 of our 29 locked held-out participants (sub-097..sub-125) are in its training folds and the other 7 are in its test set. 18 of its 25 test participants come from our Train/CV.
   - Using its window length, hyperparameters, threshold, findings or metrics in our Stage 4 would leak held-out information, and its model cannot be fairly scored on our held-out set.
   - Phone-side inference conflicts with D11 (inference on the ESP32-S3).
   - Its fix of the `0a50a6e` conflicts matches ours.
   - Recommendation: keep the locked split, tune Stage 4 only by CV within sub-001..sub-096, and don't merge `seizeit2-model` into `main` until the split and the inference location are agreed.

### Next
1. Decide on blocker 1 (accelerometer fix) and blocker 2 (keep the locked split; the branch stays independent).
2. Decide the open training choices in `mlm/docs/TRAINING_PLAN.md`: IGNORE/artifact windows in the loss, the event-matching rule, unavailable-feature handling, and the active-event gap rule.
3. Only then authorize Stage 4.

`mlm/data/processed_superseded/` (384 MB) can be deleted.

## Session 4 (2026-11-22): Step 1 IMU investigation, Stage 3.5 remediation, Stage 4

### Step 1 (read-only, Train/CV only)
- The authors' loader (`biomedepi/seizeit2`, which redirects to `mbhagubai/seizeit2`, commit
  `fcde7a7`, pre-v1.1.0) only calls `pyedflib.readSignal()`, which returns EDF physical units. It
  does no movement scaling, and the authors' model uses EEG only. EDF physical min/max are taken
  from the data, so they carry no sensor full-scale information.
- v1.0.0 exported ACC Z as an exact copy of Y. v1.1.0 ("Corrected accelerometer data") re-exported
  Z only. In 221 Train/CV recordings (sub-066/067/068/084/086/088) the new Z is garbage.
- Stored ACC/GYR = sensor / D, with D = 6 or 2 per recording:
  - both sensors clip at exactly 8/D;
  - both modules of a recording share D;
  - the ACC axes share one scale;
  - GYR ≈ counts/4096 at ±1000 deg/s (125·D deg/s per unit, medium confidence).
- Device faults: sub-083 runs 01–20, the sub-074 Y offset progressing to a stuck axis, sub-075 Y
  stuck, sub-084 Z stuck.

### Stage 3.5 (approved rule, D19)
- Code:
  - `seizeit2.assess_imu_scale` / `convert_imu` / `rest_gravity`.
  - `SignalStream.untrusted`: a rejected stream, which quality rates INVALID.
  - Saturation limits ±8 g / ±1000 deg/s.
  - `imu_scale_manifest.json`.
  - Audit `--baseline`.
  - `tests/test_stage35.py`.
- Train/CV: 1,552 accepted (1,376 D6, 176 D2) and 306 rejected. The 21 rejections beyond the Step 1
  list come from the approved axis-disagreement and post-conversion checks.
- Held-out (aggregate only): 474 accepted, 157 rejected (153 of them corrupted Z).
- Full recompute: 2 h 13 min, 3 workers. The old files are in `mlm/data/processed_pre35/`, which can
  be deleted once Stage 4 is accepted.
- Audit: 0 problems.
  - Non-IMU arrays are bit-identical in 2,806 of 2,806 recordings.
  - All 2,026 accepted IMU rest at 0.93–1.08 g.
  - Usable positives (multimodal) fell 3,456 → 2,980 (Train/CV 2,022).
  - Usable FBTCS (overall / SIMULATED_ENV): Train/CV 29 / 30 of 33, held-out 15 / 21 of 22.
- Environment: `brew install libomp` (xgboost); scikit-learn, xgboost and m2cgen are in the `train`
  extra. emlearn was evaluated and rejected: its 8-bit leaf probabilities miss the 1e-5 parity
  tolerance.
- Blocked: ESP32-S3 Flash/RAM measurement needs Espressif's xtensa toolchain. Auto mode denied
  downloading it, so the user must install it (ESP-IDF) or allow it.

### Stage 4 (authorized 2026-11-22; protocol in `mlm/docs/TRAINING_PLAN.md` §6, results in §7)
- Code commit `1de1b13`: `lyra_mlm.training`, `scripts/train_stage4.py`, `evaluate_heldout.py`,
  `export_c99.py`, the replaced guard test, the `train` extra and `uv.lock`. Training takes about 6 min.
- At the locked entry threshold 0.65, CV gave about 48 interictal FA/24h (multimodal RF), so that
  threshold is unusable as is. The owner chose two pre-declared points per configuration: locked 0.65
  and a CV-selected entry threshold (lowest grid value with CV interictal FA ≤ 1/24h). Selected:
  multimodal 0.95, multimodal + fallback 0.97, SIMULATED_ENV 0.97, IMU 0.99.
- Held-out, run once (`mlm/models/heldout_report.json`; a second run is refused):
  - Primary, multimodal + SIMULATED_ENV fallback at 0.97: 17/21 FBTCS detected (81%, macro 83%).
    Without sub-115 and sub-116, 7/11.
  - Interictal FA 0.68/24h, of which 61 of the 78 alarms come from sub-114. Non-FBTCS seizure alert
    rate 1.4%; median latency 67 s.
  - At locked 0.65: 19/21 detected but 49.9 FA/24h.
- D2 vs D6 (CV): FA rates are low in both groups, but 27 of the 30 Train/CV events are in D6
  participants, so D2 sensitivity is unmeasurable (cohort confound).
- C99 (`mlm/export/c99/`, untracked, about 1.7 MB per model):
  - m2cgen; exact (max |Δp| = 3.3e-16; 100% agreement on 1,200 boundary vectors per model).
  - ESP32-S3 Flash/RAM not measured, because the xtensa toolchain is unavailable. The host arm64 proxy
    is about 360 KB of code per model, with no RAM data.

### After the push (same session)
- `mlm/export/` is gitignored: regenerate it with `scripts/export_c99.py` from the frozen models.
  `mlm/data/processed_pre35/` and `processed_superseded/` were deleted (2.3 GB).
- **Item 3 started: C99 port for the pod** (`mlm/c99/`, `tests/test_c99_port.py`).
  - `lyra_features.c/.h`: all 22 features in training order, mirroring scipy's periodogram/Welch
    defaults, using a plain DFT and static buffers (no malloc).
  - `lyra_confirm.c/.h`: `EventConfirmation` with the locked policies.
  - Parity on 116 synthetic windows cut by the real `segment()`: max absolute difference 2.8e-13, with
    peak frequencies and lag identical. NaN patterns match on edge cases (flat envelope, still IMU,
    no IMU). Alerts are identical to the Python reference on 10 random sequences.
  - Host arm64 size: features about 11 KB of code + 22.6 KB of static RAM; confirmation about 0.5 KB.
  - Precondition: gap-free windows whose quality gate passed.
  - Quality gate (`lyra_quality.c/.h`, `tests/test_c99_quality.py`):
    - `lyra_assess` is `assess_modality` with `check_timestamps`, and uses the same state codes as
      the window files. `lyra_combine` is `combine`, where the caller passes the clock check, because
      Python's `check_sync` is recording-level.
    - `lyra_route_window` picks multimodal, SIMULATED_ENV fallback or SUPPRESSED, as in
      `training.track_status` / `fallback_status`.
    - Parity on 59 synthetic windows with injected dropouts, outages, inf samples, flat channels, a
      gap, jitter, a duplicate timestamp and light/heavy saturation: every state matches, coverage
      matches to 1e-12, and all three routes are exercised.
    - Host size: about 4.5 KB of code, no static RAM.
  - `lyra_step()` (`lyra_pipeline.c/.h`, `tests/test_c99_step.py`): one call per window chains the
    quality gate → features → RF → confirmation.
    - RF scorers are function pointers with the generated `lyra_*_rf_predict` signature. A NULL
      fallback means policy "suppress".
    - Pod rule, stricter than Python: a modality's features are computed only from a clean window
      (every sample OK, regular, no gap). Python may still score a usable but gappy window.
    - Parity against a Python reference chain (`segment` quality → `extract_features` → training
      routing → same scorer → `training.confirm`) on a bursty and a defect-injected recording, with and
      without fallback: routes, statuses, scores, features and alerts are all identical.
    - With the real exported forests linked in, scores are within 1e-5 of scikit-learn.
    - The fix found by this test: `xmod_env_accmag_corr` uses the covered IMU bins (≥ 20) as Python
      does; only lag and coherence need full coverage.
  - Host total: hand-written code about 18 KB of code + 22.6 KB of static RAM; forests about 360 KB
    of code each.
  - Still firmware-side: window buffering (4 s / 2 s ring buffers), the running clock check
    (`clocks_ok`), BLE notification.
- **Step 2: ESP32-S3 build** (`mlm/scripts/size_esp32s3.py`).
  - Toolchain: Espressif crosstool-NG `esp-14.4.0_20261030` (xtensa GCC 14.4.0), verified against its
    published SHA-256 (checksum file kept next to it). It is installed in
    `~/.espressif/tools/xtensa-esp-elf/esp-14.4.0_20261030/` (ESP-IDF layout, outside the repo), which
    is the size script's default.
  - Found on the S3 only: each m2cgen forest was a single ~370 KB function. Xtensa can't build that
    (L32R literal reach is 256 KB and J reach is 128 KB), and neither `-mtext-section-literals` nor
    `-mauto-litpools` fixes it. `export_c99.py` now emits one static function per tree (50 × ~7 KB)
    plus an averaging wrapper. C parity is unchanged: the boundary-vector and `lyra_step` real-forest
    tests pass.
  - The DFT now uses a twiddle table, so the trig calls happen once per length instead of per sample
    × bin. Results are bit-identical.
  - Sizes (`-mlongcalls -O2`, objects and a test link from `lyra_step` with newlib libm and libgcc
    soft-float): 593 KB of Flash and 26.7 KB of static RAM in total. The forests are about 350 KB
    each; the hand-written code is about 13 KB plus the 26.7 KB of scratch RAM. The largest stack
    frame is 240 B.
  - That excludes FreeRTOS, drivers and BLE. ESP-IDF's default 1 MB app partition will be tight with
    NimBLE, so use a larger app partition (the board has 8 MB of flash).
  - Not measured: per-window latency, which needs the board. A rough estimate is ~30 ms per window,
    because doubles are software-emulated on the S3's single-precision FPU. Compare with the 2 s step.

### Item 3: pod data-collection plan and tooling
- `mlm/docs/POD_DATA_PROTOCOL.md` (draft v1) covers:
  - governance: consent approval first, pseudonyms, data never in git;
  - questions Q1–Q4: ENV mapping, domain gap, hard negatives and false alarms per 24 h, engineering
    checks;
  - hardware setup: data-collection firmware records RAW 1000 Hz + ENV 50 Hz in mV and IMU 25 Hz on
    one clock;
  - the Phase 1 lab script (about 75 min, using the existing activity vocabulary) and Phase 2
    (24 h free living);
  - session format schema v1, the report, decision gates and rules.
- New kind `AnnotationKind.SIMULATED_CONVULSION` (scripted movement by healthy volunteers). It blocks
  NEGATIVE labels and is excluded from every seizure metric.
- `lyra_mlm.pod`: `PodAdapter` with `HARDWARE` origin, plus `check_session`.
  - It refuses sessions without consent, without a pseudonym, with a wrong schema, or with
    wrong units.
  - Events are mapped onto the taxonomy, and unknown labels become UNMAPPED.
- `lyra_mlm.domain` + `scripts/pod_report.py`: per session, quality, the `env` path (hardware ENV) and
  the `simenv` path (pod RAW → trained schema) against the SeizeIT2 Train/CV reference.
  - The comparison gives percentiles, the share outside the 1st–99th percentile range, KS values,
    simulated convulsions vs FBTCS, and MyoWare ENV vs simulated ENV (r, ratio, line).
  - The reference is cached in `data/processed/reference_traincv.npz`: 261,232 negatives and 2,021
    positives from the 96 Train/CV participants.
- `tests/test_pod.py` (9 tests) uses synthetic sessions. The script was also run end to end on a
  synthetic session against the real reference, and its output was deleted.
- Not built yet: the data-collection firmware mode and the laptop/phone logger that writes this format.
  They depend on the firmware (step 3 of the roadmap).

### State at end of session
- `main` has `9584404` (Stage 3.5) and `1de1b13` (Stage 4 code), plus the results-docs commit. None
  of these are pushed.
- `mlm/models/` and `mlm/export/` are gitignored.

### Next
1. Firmware: data-collection mode (RAW 1 kHz + ENV 50 Hz + IMU 25 Hz in the POD_DATA_PROTOCOL format) and the deployed `lyra_step` loop. An on-board test app measures latency and checks parity (needs ESP-IDF + the DevKit).
2. Ethics/consent approval for Phase 1 pod recordings (owner action) before any session.
2. Install the ESP-IDF / xtensa toolchain (or allow the download) to measure ESP32-S3 Flash/RAM.
   Then an on-device benchmark and Python-vs-C parity on hardware.
3. Port feature extraction to C99, in `feature_names` order.
4. Record real MyoWare ENV + BMI270 data from the pod, and compare simulated ENV with real ENV before
   any hardware claim.
5. Optional research: probability calibration instead of threshold-only operating points; sub-114
   false alarms; why sub-121's three events were missed.

## Branch sessions: `seizeit2-model` and `lyra-esp32-ml` (PrinceJorick)

These sessions ran on branches in parallel with the main-line sessions above, so their numbers restart at 4. Merged into `main` with Kyle's Stage 3.5/4 work on 2026-11-22.

### Session 4 (2026-11-21): SeizeIT2 model trained and integrated (branch `seizeit2-model`)

Developed in parallel with Sessions 2–3 (`mlm/`), from the older base `9ef25ed`. How the two pipelines differ and how to merge them: `docs/MLM_MERGE_GUIDE.md`.

#### What exists
- `ml/seizeit2/`: reproducible pipeline (fetch + SHA256-verify EDFs, features, subject-level split, 5-fold grouped CV, one-time test evaluation, JSON export, Python↔TS parity check). Reuses `lyra_mlm.annotations` for label mapping.
- `models/seizeit2/`: `lyra-seizeit2-fbtcs` v1.0.0 (HGB, 300 trees, 5 s windows), config, labels, metadata, metrics, plots, model card (`README.md`).
- `src/services/ml/seizeit2/`: TS feature extractor, tree evaluator + streaming detector, `MlService` implementation. `services/ml/index.ts` now serves it (falls back to the mock if the bundled assets are incompatible).
- Insights/About wording updated: "experimental, not clinically validated, no live pod data".

#### Decisions worth remembering
- Target = FBTCS (`sz_foc_f2b`); other seizures are excluded from training negatives and from the false-alarm count. `context.md` U1 still needs owner sign-off.
- Model shipped as JSON evaluated in TypeScript, not `.tflite`: no native runtime in the app, keeps Expo Go working.
- SeizeIT2 accelerometer files use per-file scales (~1/2, ~1/6, or pinned at 32.768): runs are rescaled to g by their median |acc|; invalid runs excluded. Gyro unit unknown → only scale-invariant gyro features.
- Held-out test (25 subjects, 1,983 h): 11/12 FBTCS (CI 0.62–1.00), 0.22 false alarms/24 h, median latency 43 s from annotated onset. Hospital data only.
- `detect()` is not called anywhere yet (no real sensor stream); the alert flow is unchanged.

#### Gotchas
- `pj-training-data/ds005873` is a DataLad clone without EDF content; `preprocessing.py` downloads from the public bucket. Don't `git annex`/`datalad` inside it.
- Run Python modules from the repo root (`python -m ml.seizeit2.<module>`); pickled models need `ml.seizeit2.train` importable.
- `evaluate.py` refuses to evaluate the test set twice for the same run.

#### Not verified
- No run on a phone; no LYRA pod data; MyoWare/BMI270 units and placement not matched to SeizeIT2.

### Session 5 (2026-11-22): v2 retrained on all participants under D9–D18 (branch `seizeit2-model`)

Owner instruction: retrain on all participants and all seizures, follow Kyle's locked hardware decisions, fix the bug in Kyle's prepared data, push and open a PR (do not merge).

#### What exists
- **`lyra_mlm` IMU fix (`218a656`):** `seizeit2.calibrate_imu` puts ACC in g (per-recording median |acc|) and GYR in deg/s (125 deg/s per unit, estimated by `ml/seizeit2/gyro_calibration.py`). Impossible scales are marked INVALID. 7 new tests; 141 pass.
- **App follows D11 (`93a3f24`):** `src/services/ml/seizeit2/` is removed and `services/ml/index.ts` is the mock again. Insights/About wording is restored.
- **`ml/seizeit2/` v2 pipeline:**
  - `prepare.py`: streams and verifies EDFs, then runs Kyle's Stage 3 worker. All 2,850 recordings were regenerated, 0 failed.
  - `train.py`: nested participant-grouped CV over all 124 usable participants. Thresholds are chosen in inner CV only.
  - `evaluate.py`: `confirm_alarms` is an exact, fast replica of `EventConfirmation`, checked by `test_evaluate.py` on 300 streams.
  - `export.py`: writes the C99 header and checks it with gcc parity.
  - `preprocessing.py` and the TS parity files are deleted.
- **`models/seizeit2/` v2.0.0:** RF 50 × 8 (D18), with `lyra_seizeit2_model.h`, `model_config.json`/`export_manifest.json` (Kyle's contracts) and the model card.

#### Results (nested 5-fold CV, no held-out set)
- **Primary RF:**
  - 42/44 evaluable FBTCS (CI 0.85–0.99).
  - 0.32 FA/24 h (CI 0.27–0.38) over 9,985 valid hours.
  - Median latency 54 s.
- **Fusion vs single modality:** ENV-only reaches 2.0 FA/24 h and IMU-only 0.71.
- **HGB:** 0.21 FA/24 h at 41/44.
- **D17 default thresholds (0.65/0.45):** 37 FA/24 h. Final thresholds are entry 0.9684, exit 0.7684.

#### Decisions worth remembering
- All participants are used, so no SeizeIT2 test set remains. The next independent check needs new data.
- D17 entry/exit are tuned, because the defaults are unusable with uncalibrated forest scores. HGB stands in for XGBoost.
- Python 3.14 uses forkserver. Set `OMP_NUM_THREADS=1` for the Stage 3 worker pool, or each worker spawns about 34 threads and runs about 15× slower.

#### Not verified
- The 22 features have no C99 port.
- The header has never run on an ESP32-S3.
- The MyoWare ENV gain has not been calibrated against the simulated envelope.
- The gyro unit is an estimate.
- There is no pod data.

### Session 6 (2026-11-22): ENV-path pipeline and ESP32-S3 package (branch `lyra-esp32-ml`)

Owner brief: an end-to-end, patient-wise, cached SeizeIT2 pipeline with a C99 export for the ESP32-S3. The owner ruled that D12 (MyoWare ENV pin at 50 Hz) is the deployable signal path, set the FAR cap to ≤ 1 per 24 h, and approved installing ESP-IDF.

#### What exists
- **`ml/lyra/`** (see its README for the commands):
  - Extraction, benchmark, deploy, parity, final evaluation, ADL loader and report.
  - 224 tests, all passing.
- **`firmware/model/`:** C99 features, XGB model, detector state machine, golden vectors and host test.
- **`firmware/esp32s3_app/`:** an ESP-IDF size probe, not product firmware.
- **`reports/lyra/`:** `lyra_report.md`/`.json`, dataset and experiment manifests, `ablation_results.csv`, `oof_metrics.json`, `resource_report.json`.
- **ESP-IDF v5.5.6** for esp32s3 is installed in `~/esp/esp-idf`.

#### Results
- **Development (sub-001..096, subject-wise 5-fold OOF):**
  - Selected model: w4 (4 s window, 2 s hop), fusion, XGB 40 × 4, threshold 0.015, N=3, M=3.
  - 26/29 events at 0.79 FA/24 h, median latency 55.5 s.
  - No configuration reaches a median latency below 20 s.
- **Final evaluation (sub-097..125, run once):**
  - 12/15 events (bootstrap CI 0.60–1.00) at 0.24 FA/24 h (CI 0.10–0.43), median latency 47.5 s.
  - IMU-only: 10/15. sEMG-only: 0/15.
- **ESP32-S3 (measured with xtensa gcc 14.2 -Os):**
  - 15,898 B flash and 6,024 B SRAM, against targets of 60 KB and 8 KB.
  - Python/C parity is bit-exact on the golden vectors and on 31,206 real windows.

#### Decisions worth remembering
- **EMG rate:** EMG is 256 Hz in every file, not 250.
- **ENV simulation:** a 1st-order 20.8 Hz HPF, rectification, a 1st-order 3.6 Hz LPF and 20 ms bin means, all causal.
- **Threshold grid:** calibrated probabilities follow the true prevalence, so the grid is extended down to 0.005. Within the brief's 0.20–0.80 grid, the best result is 1/29 events.
- **Dominant frequency:** uses a tie-tolerant argmax (relative 1e-6). Quantised still signals give exact periodogram ties, and a plain argmax broke Python/C parity.
- **XGB export:** m2cgen 0.10 fails on XGBoost 3.x, so XGB uses its own exporter. emlearn 0.23 sets `numpy.seterr(all='raise')` on import, and `export_c.py` restores the previous state.

#### Not verified
- **Hardware:** nothing has run on a board.
- **Calibration:** the MyoWare ENV gain and the BMI270 units are not calibrated against the simulation. The SeizeIT2 gyro unit is an estimate.
- **Daily-life data:** no ADL or pod data exists yet.
- **Test cohort:** the final cohort was used to train v1/v2, so it is not pristine.
- **Latency:** measured from the annotated focal onset, not from the convulsive phase.
- **IMU rule predates D19.** `ml/lyra` scales the IMU with the legacy `calibrate_imu` (median |acc|), not Kyle's Stage 3.5 rule (D19, merged afterwards). The legacy rule lacks the Z-corruption, ceiling and rest-spread rejections. Some included recordings may carry a corrupt v1.1.0 Z axis. Re-extract under D19 before relying on these numbers.
- **`ml/seizeit2` after the merge.** The adapter now applies D19. Rerunning `ml/seizeit2/prepare.py` will not reproduce the data that `models/seizeit2` v2 was trained on.
