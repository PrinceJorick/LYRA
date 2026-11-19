import type { DeviceStatus } from '../types';

export const DEMO_DEVICE_ID = 'LYRA-DEMO-001';

export const initialDevice = (): DeviceStatus => ({
  connection: 'connected',
  deviceId: DEMO_DEVICE_ID,
  name: 'LYRA Pod',
  batteryPercent: 82, // mock value; MAX17048 is not connected
  lastSyncAt: Date.now(),
});

/** Phase 2 prototype parts, per the Hardware & 3D Printing Specification. */
export interface HardwarePart {
  component: string;
  part: string;
  purpose: string;
}

export const HARDWARE_PARTS: HardwarePart[] = [
  { component: 'MCU', part: 'ESP32-S3-DevKitC-1-N8R8', purpose: 'Main processing and connectivity (Wi-Fi / BLE 5)' },
  { component: 'sEMG sensor', part: 'MyoWare 2.0 Muscle Sensor v2.0.4', purpose: 'Muscle electrical activity acquisition' },
  { component: 'IMU', part: 'Bosch BMI270', purpose: '3-axis accelerometer + 3-axis gyroscope' },
  { component: 'Battery', part: '3.7 V, 500 mAh protected LiPo', purpose: 'Power supply' },
  { component: 'Charging IC', part: 'TI BQ24074', purpose: 'Battery charging (USB-C)' },
  { component: 'Voltage regulator', part: 'TI TPS63031', purpose: '3.3 V power regulation' },
  { component: 'Fuel gauge', part: 'MAX17048', purpose: 'Battery-level monitoring' },
];

export const HARDWARE_STATUS_NOTE =
  'Selected Phase 2 prototype components. This app has not verified that any of them are assembled, tested, or connected.';

export const CSI_NOTE =
  'A separate ESP32-based Wi-Fi CSI transmitter/receiver node is planned for Phase 6 research. It is an off-pod component and is not part of the LYRA Pod.';
