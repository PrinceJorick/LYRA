# LYRA Companion

Expo + React Native + TypeScript companion app for the **LYRA Pod**, an experimental sensor pod
(sEMG + 6-axis IMU) researching possible convulsive seizure events.

> LYRA is an experimental research project. It is not a diagnostic or treatment device and must not
> replace professional medical evaluation or emergency services.

**Everything in this app is simulated demo data:** pod pairing and battery, the possible-event alert
and caregiver notification. No Bluetooth, SMS or calls. An experimental model trained on the public
SeizeIT2 dataset is exported for the pod firmware (`models/seizeit2/`, see its README). Per decision
D11 it runs on the pod, not in the app, so the app does not use it yet. It is not clinically validated.

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
models/seizeit2/ Trained model (C99 header for the pod), inference config, metrics, model card
mlm/             lyra_mlm: multimodal ML foundation (contracts, quality, features, Stage 3 prep)
```

[`mlm/`](mlm/README.md) (`lyra_mlm`) prepares the data: quality states, simulated MyoWare ENV,
features, labels and event confirmation. `ml/seizeit2/` trains and exports a model on its Stage 3
windows into `models/seizeit2/`. The merge status of the two pipelines is in
[`docs/MLM_MERGE_GUIDE.md`](docs/MLM_MERGE_GUIDE.md).

To integrate real parts, replace the exports in `services/device/index.ts` (BLE) and
`services/notifications/index.ts` (SMS/push); screens only depend on those interfaces.
`services/ml/index.ts` stays the mock: the detector runs on the pod (D11), and the app will receive
possible events over BLE.
