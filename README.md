# LYRA Companion

Expo + React Native + TypeScript companion app for the **LYRA Pod**, an experimental sensor pod
(sEMG + 6-axis IMU) researching possible convulsive seizure events.

> LYRA is an experimental research project. It is not a diagnostic or treatment device and must not
> replace professional medical evaluation or emergency services.

**Everything in this app is simulated demo data:** pod pairing and battery, the possible-event alert
and caregiver notification. No Bluetooth, SMS or calls. An experimental model trained on the public
SeizeIT2 dataset is bundled (`models/seizeit2/`, see its README), but no live pod data reaches it yet
and it is not clinically validated.

## Run

```bash
npm install
npx expo start        # then press i / a, or scan the QR code with Expo Go
npx tsc --noEmit      # typecheck
npx eslint src        # lint
```

## Structure

```
src/app/         Expo Router routes: 5 tabs (Home, History, Insights, Reports, Care) + pairing,
                 device, event, alert, observation, summary, context-log, caregiver, settings, about
src/data/        All mock data (device, events, signals, caregivers, reports)
src/services/    device/ (BLE seam), ml/ (model seam), events/, notifications/, reports/
src/state/       In-memory app state (resets on restart)
src/components/  Reusable UI + brand graphics (logo from assets/lyra-mark.png, abstract pod SVG)
ml/seizeit2/     Reproducible SeizeIT2 training pipeline (Python); dataset stays in pj-training-data/
models/seizeit2/ Trained model, inference config, metrics and model card (git-tracked)
mlm/             lyra_mlm: multimodal ML foundation (contracts, quality, features, Stage 3 prep)
```

Two ML pipelines currently coexist and are planned to merge into one (see
[`docs/MLM_MERGE_GUIDE.md`](docs/MLM_MERGE_GUIDE.md)):
[`mlm/`](mlm/README.md) is the multimodal ML foundation (Stage 3 dataset preparation, no trained
model), and `ml/seizeit2/` + `models/seizeit2/` hold a trained SeizeIT2 baseline.

To integrate real parts, replace the exports in `services/device/index.ts` (BLE) and
`services/notifications/index.ts` (SMS/push); screens only depend on those interfaces.
`services/ml/index.ts` already serves the SeizeIT2 model (`services/ml/seizeit2/`); a BLE stream
must call `mlService.detect()` with the window contract in `models/seizeit2/config.json`.
