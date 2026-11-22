import { encodeDeviceStatus, encodeEventStatus, type Cancellation, type EventState } from './protocol';
import type { PodLink } from './types';

// No Bluetooth involved: emits the exact bytes the pod firmware sends, every payload marked origin 'simulated'.
type Listener = { status: (b: Uint8Array) => void; event: (b: Uint8Array) => void };
const listeners = new Set<Listener>();
const started = Date.now();
const uptimeS = () => Math.floor((Date.now() - started) / 1000);
let eventId = 0;
let seq = 0;
let timer: ReturnType<typeof setInterval> | null = null;

function heartbeat() {
  seq = (seq + 1) & 0xffff;
  const b = encodeDeviceStatus({
    origin: 'simulated', batteryPercent: 82, batteryMv: 3990, emgQuality: 'VALID', imuQuality: 'VALID',
    heartbeatSeq: seq, uptimeS: uptimeS(),
  });
  listeners.forEach((l) => l.status(b));
}

function emitEvent(state: EventState, cancellation: Cancellation) {
  const b = encodeEventStatus({
    origin: 'simulated', eventId, state, cancellation, timestampS: uptimeS(), emgAvailable: true, imuAvailable: true,
  });
  listeners.forEach((l) => l.event(b));
}

export const simulatedPodLink: PodLink & { simulatePossibleEvent(): void; simulatePodCancel(): void } = {
  subscribe(onDeviceStatus, onEventStatus) {
    const l = { status: onDeviceStatus, event: onEventStatus };
    listeners.add(l);
    timer ??= setInterval(heartbeat, 15000);
    return () => {
      listeners.delete(l);
      if (listeners.size === 0 && timer) {
        clearInterval(timer);
        timer = null;
      }
    };
  },
  /** Demo: the pod confirming a possible event (POSSIBLE_EVENT entry). */
  simulatePossibleEvent() {
    eventId = (eventId + 1) & 0xffff;
    emitEvent('possible_event', 'none');
  },
  /** Demo: the wearer pressing the pod's cancel button. */
  simulatePodCancel() {
    emitEvent('idle', 'cancelled_on_pod');
  },
};
