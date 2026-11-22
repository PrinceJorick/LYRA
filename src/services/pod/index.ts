// Swap this export for a BLE-backed link when it exists (needs GATT UUIDs and a native BLE module).
export { simulatedPodLink as podLink } from './simulatedPodLink';
export * from './protocol';
export type * from './types';
