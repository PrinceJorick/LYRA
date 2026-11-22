// Pod -> phone BLE payloads (draft protocol v1, mlm/docs/BLE_PROTOCOL.md). Byte-exact with
// mlm/src/lyra_mlm/ble.py and firmware/components/lyra_pod/lyra_ble.c (parity checked by
// mlm/tests/test_app_pod_protocol.py). Little-endian. Unknown versions, origins, codes and wrong
// lengths are rejected, never guessed. Self-contained so Node can run it for that test.

export const PROTOCOL_VERSION = 1;
export const DEVICE_STATUS_LENGTH = 13; // <BBBHBBHI
export const EVENT_STATUS_LENGTH = 11; // <BBHBBIB

/** Every payload says whether it came from pod hardware or a simulation; show simulated ones as simulated. */
export type PayloadOrigin = 'hardware' | 'simulated';
export type QualityCode = 'VALID' | 'DEGRADED' | 'INVALID' | 'MISSING' | 'UNSYNCHRONIZED';
export type EventState = 'idle' | 'accumulating' | 'possible_event' | 'refractory';
export type Cancellation = 'none' | 'cancelled_on_pod' | 'cancelled_on_phone';

const ORIGINS: Record<number, PayloadOrigin> = { 1: 'hardware', 2: 'simulated' };
const QUALITY: QualityCode[] = ['VALID', 'DEGRADED', 'INVALID', 'MISSING', 'UNSYNCHRONIZED'];
const STATES: EventState[] = ['idle', 'accumulating', 'possible_event', 'refractory'];
const CANCELS: Cancellation[] = ['none', 'cancelled_on_pod', 'cancelled_on_phone'];

export interface DeviceStatusPayload {
  origin: PayloadOrigin;
  batteryPercent: number;
  batteryMv: number;
  emgQuality: QualityCode;
  imuQuality: QualityCode;
  heartbeatSeq: number;
  uptimeS: number;
}

export interface EventStatusPayload {
  origin: PayloadOrigin;
  eventId: number;
  state: EventState;
  cancellation: Cancellation;
  /** Pod uptime (s) at the state change: pod clock, not phone time. */
  timestampS: number;
  emgAvailable: boolean;
  imuAvailable: boolean;
}

export class PayloadError extends Error {}

function view(bytes: Uint8Array, length: number, name: string): DataView {
  if (bytes.length !== length) throw new PayloadError(`${name}: expected ${length} bytes, got ${bytes.length}`);
  if (bytes[0] !== PROTOCOL_VERSION) throw new PayloadError(`${name}: unsupported protocol version ${bytes[0]}`);
  if (!(bytes[1] in ORIGINS)) throw new PayloadError(`${name}: unknown origin ${bytes[1]}`);
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function code<T>(table: T[], value: number, field: string): T {
  if (value >= table.length) throw new PayloadError(`${field}: unknown code ${value}`);
  return table[value];
}

export function decodeDeviceStatus(bytes: Uint8Array): DeviceStatusPayload {
  const v = view(bytes, DEVICE_STATUS_LENGTH, 'DeviceStatus');
  const batteryPercent = v.getUint8(2);
  if (batteryPercent > 100) throw new PayloadError(`battery_percent ${batteryPercent} outside 0..100`);
  return {
    origin: ORIGINS[v.getUint8(1)],
    batteryPercent,
    batteryMv: v.getUint16(3, true),
    emgQuality: code(QUALITY, v.getUint8(5), 'emg_quality'),
    imuQuality: code(QUALITY, v.getUint8(6), 'imu_quality'),
    heartbeatSeq: v.getUint16(7, true),
    uptimeS: v.getUint32(9, true),
  };
}

export function decodeEventStatus(bytes: Uint8Array): EventStatusPayload {
  const v = view(bytes, EVENT_STATUS_LENGTH, 'EventStatus');
  const available = v.getUint8(10);
  if (available > 3) throw new PayloadError(`available_modalities ${available} outside 0..3`);
  return {
    origin: ORIGINS[v.getUint8(1)],
    eventId: v.getUint16(2, true),
    state: code(STATES, v.getUint8(4), 'state'),
    cancellation: code(CANCELS, v.getUint8(5), 'cancellation'),
    timestampS: v.getUint32(6, true),
    emgAvailable: (available & 1) !== 0,
    imuAvailable: (available & 2) !== 0,
  };
}

const originCode = (o: PayloadOrigin) => (o === 'hardware' ? 1 : 2);

/** For the simulated pod: the bytes real firmware would send. */
export function encodeDeviceStatus(p: DeviceStatusPayload): Uint8Array {
  const b = new Uint8Array(DEVICE_STATUS_LENGTH);
  const v = new DataView(b.buffer);
  v.setUint8(0, PROTOCOL_VERSION);
  v.setUint8(1, originCode(p.origin));
  v.setUint8(2, p.batteryPercent);
  v.setUint16(3, p.batteryMv, true);
  v.setUint8(5, QUALITY.indexOf(p.emgQuality));
  v.setUint8(6, QUALITY.indexOf(p.imuQuality));
  v.setUint16(7, p.heartbeatSeq, true);
  v.setUint32(9, p.uptimeS, true);
  return b;
}

export function encodeEventStatus(p: EventStatusPayload): Uint8Array {
  const b = new Uint8Array(EVENT_STATUS_LENGTH);
  const v = new DataView(b.buffer);
  v.setUint8(0, PROTOCOL_VERSION);
  v.setUint8(1, originCode(p.origin));
  v.setUint16(2, p.eventId, true);
  v.setUint8(4, STATES.indexOf(p.state));
  v.setUint8(5, CANCELS.indexOf(p.cancellation));
  v.setUint32(6, p.timestampS, true);
  v.setUint8(10, (p.emgAvailable ? 1 : 0) | (p.imuAvailable ? 2 : 0));
  return b;
}
