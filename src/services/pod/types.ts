/**
 * Transport for pod notifications. A link only delivers raw payload bytes; the app decodes them
 * with ./protocol so the simulator and a future BLE link share one validated path. Swap
 * simulatedPodLink for a BLE-backed link once GATT UUIDs and a native BLE module exist.
 */
export interface PodLink {
  /** Subscribes to DeviceStatus and EventStatus notifications; returns the unsubscribe function. */
  subscribe(onDeviceStatus: (bytes: Uint8Array) => void, onEventStatus: (bytes: Uint8Array) => void): () => void;
  /** Writes a phone -> pod command (encodePhoneCommand) to the phone_command characteristic. */
  sendCommand(bytes: Uint8Array): Promise<void>;
}
