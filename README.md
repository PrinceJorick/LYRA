# LYRA Companion

Expo + React Native + TypeScript companion app for the **LYRA Pod**, an experimental sensor pod
(sEMG + 6-axis IMU) researching possible convulsive seizure events.

> LYRA is an experimental research project. It is not a diagnostic or treatment device and must not
> replace professional medical evaluation or emergency services.

**Everything in this app is simulated demo data:** pod pairing and battery, the possible-event alert,
caregiver notification, and the ML model ("ML model not integrated"). No Bluetooth, SMS or calls.

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
```

To integrate real parts, replace the exports in `services/device/index.ts` (BLE),
`services/ml/index.ts` (model) and `services/notifications/index.ts` (SMS/push); screens only
depend on those interfaces.
