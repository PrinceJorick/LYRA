// UI imports only from here. Per locked decision D11 the detector runs on the pod (ESP32-S3,
// models/seizeit2/lyra_seizeit2_model.h), not on the phone; the app will receive possible events
// over BLE. Until that link exists this stays the mock.
export { mockMlService as mlService, demoTrigger } from './mockMlService';
export type * from './types';
