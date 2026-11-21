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
