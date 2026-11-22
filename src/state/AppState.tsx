import { router } from 'expo-router';
import React, { createContext, useCallback, useContext, useEffect, useMemo, useReducer } from 'react';
import { defaultPrefs, mockCaregivers, mockContextLogs } from '../data/mockCaregivers';
import { initialDevice } from '../data/mockDevice';
import { mockEvents } from '../data/mockEvents';
import { nextEventId } from '../services/events';
import { notificationService } from '../services/notifications';
import { decodeDeviceStatus, decodeEventStatus, PayloadError, podLink, type PayloadOrigin } from '../services/pod';
import type {
  Caregiver, ContextLog, DeviceStatus, EventOutcome, NotificationPrefs, NotificationResult,
  Observation, PossibleEvent, ReportStatus, ReviewStatus, ThemeMode,
} from '../types';
import { newId } from '../utils/format';

/** The possible event the pod last reported (POSSIBLE_EVENT entry), shown by the alert screen. */
export interface PodAlert {
  eventId: number;
  origin: PayloadOrigin;
  receivedAt: number; // phone time; the pod's own clock is uptime, not wall time
  cancelledOnPod: boolean;
}

// In-memory only: restarting the app resets everything to the demo data.
interface State {
  podAlert: PodAlert | null;
  device: DeviceStatus;
  events: PossibleEvent[];
  caregivers: Caregiver[];
  prefs: NotificationPrefs;
  contextLogs: ContextLog[];
  themeMode: ThemeMode;
}

type Action =
  | { type: 'device'; device: DeviceStatus }
  | { type: 'podStatus'; batteryPercent: number; at: number }
  | { type: 'podAlert'; alert: PodAlert }
  | { type: 'podCancel'; eventId: number }
  | { type: 'addEvent'; event: PossibleEvent }
  | { type: 'patchEvent'; id: string; patch: Partial<PossibleEvent> }
  | { type: 'upsertCaregiver'; caregiver: Caregiver }
  | { type: 'removeCaregiver'; id: string }
  | { type: 'prefs'; patch: Partial<NotificationPrefs> }
  | { type: 'addContext'; log: ContextLog }
  | { type: 'theme'; mode: ThemeMode };

const init = (): State => ({
  podAlert: null,
  device: initialDevice(),
  events: mockEvents(),
  caregivers: mockCaregivers(),
  prefs: defaultPrefs(),
  contextLogs: mockContextLogs(),
  themeMode: 'light',
});

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'device':
      return { ...s, device: a.device };
    case 'podStatus': // only a paired pod updates the device card
      return s.device.connection === 'connected'
        ? { ...s, device: { ...s.device, batteryPercent: a.batteryPercent, lastSyncAt: a.at } }
        : s;
    case 'podAlert':
      return { ...s, podAlert: a.alert };
    case 'podCancel':
      return s.podAlert?.eventId === a.eventId ? { ...s, podAlert: { ...s.podAlert, cancelledOnPod: true } } : s;
    case 'addEvent':
      return { ...s, events: [a.event, ...s.events] };
    case 'patchEvent':
      return { ...s, events: s.events.map((e) => (e.id === a.id ? { ...e, ...a.patch } : e)) };
    case 'upsertCaregiver': {
      const exists = s.caregivers.some((c) => c.id === a.caregiver.id);
      return {
        ...s,
        caregivers: exists
          ? s.caregivers.map((c) => (c.id === a.caregiver.id ? a.caregiver : c))
          : [...s.caregivers, a.caregiver],
      };
    }
    case 'removeCaregiver':
      return { ...s, caregivers: s.caregivers.filter((c) => c.id !== a.id) };
    case 'prefs': {
      const next = { ...s.prefs, ...a.patch };
      next.countdownSeconds = Math.min(10, Math.max(7, next.countdownSeconds));
      return { ...s, prefs: next };
    }
    case 'addContext':
      return { ...s, contextLogs: [a.log, ...s.contextLogs] };
    case 'theme':
      return { ...s, themeMode: a.mode };
  }
}

const Ctx = createContext<ReturnType<typeof useValue> | null>(null);

function useValue() {
  const [state, dispatch] = useReducer(reducer, undefined, init);

  // Pod notifications: decoded and validated here; malformed payloads are dropped, never guessed.
  useEffect(
    () =>
      podLink.subscribe(
        (bytes) => {
          try {
            dispatch({ type: 'podStatus', batteryPercent: decodeDeviceStatus(bytes).batteryPercent, at: Date.now() });
          } catch (e) {
            if (!(e instanceof PayloadError)) throw e;
            console.warn('Dropped pod status payload:', e.message);
          }
        },
        (bytes) => {
          try {
            const ev = decodeEventStatus(bytes);
            if (ev.state === 'possible_event' && ev.cancellation === 'none') {
              dispatch({ type: 'podAlert', alert: { eventId: ev.eventId, origin: ev.origin, receivedAt: Date.now(), cancelledOnPod: false } });
              router.push('/alert');
            } else if (ev.cancellation === 'cancelled_on_pod') {
              dispatch({ type: 'podCancel', eventId: ev.eventId });
            }
          } catch (e) {
            if (!(e instanceof PayloadError)) throw e;
            console.warn('Dropped pod event payload:', e.message);
          }
        },
      ),
    [],
  );

  const actions = useMemo(
    () => ({
      setDevice: (device: DeviceStatus) => dispatch({ type: 'device', device }),
      setReview: (id: string, review: ReviewStatus) => dispatch({ type: 'patchEvent', id, patch: { review } }),
      setReportStatus: (id: string, reportStatus: ReportStatus) =>
        dispatch({ type: 'patchEvent', id, patch: { reportStatus } }),
      saveObservation: (id: string, observation: Observation) =>
        dispatch({ type: 'patchEvent', id, patch: { observation, reportStatus: 'ready' } }),
      upsertCaregiver: (caregiver: Caregiver) => dispatch({ type: 'upsertCaregiver', caregiver }),
      removeCaregiver: (id: string) => dispatch({ type: 'removeCaregiver', id }),
      setPrefs: (patch: Partial<NotificationPrefs>) => dispatch({ type: 'prefs', patch }),
      addContextLog: (log: Omit<ContextLog, 'id' | 'at'>) =>
        dispatch({ type: 'addContext', log: { ...log, id: newId('ctx'), at: Date.now() } }),
      setThemeMode: (mode: ThemeMode) => dispatch({ type: 'theme', mode }),
    }),
    [],
  );

  /** Records an alert outcome (cancel / timeout) in History and returns the new event. */
  const recordAlert = useCallback(
    (outcome: Exclude<EventOutcome, 'sample_record'>, timestamp: number, origin: PayloadOrigin,
      notification?: NotificationResult) => {
      const event: PossibleEvent = {
        id: nextEventId(state.events, timestamp),
        timestamp,
        origin: origin === 'hardware' ? 'pod_hardware' : 'simulated_alert',
        outcome,
        review: 'needs_review',
        reportStatus: 'not_started',
        detectionSource: origin === 'hardware' ? 'ml_model' : 'mock',
        notification,
        signalSeed: timestamp % 97,
      };
      dispatch({ type: 'addEvent', event });
      return event;
    },
    [state.events],
  );

  /** Runs the (mock) caregiver notification and logs the event. */
  const completeAlert = useCallback(
    async (timestamp: number, origin: PayloadOrigin) => {
      const notification = state.prefs.notifyOnTimeout
        ? await notificationService.notifyCaregivers(state.caregivers)
        : undefined;
      return recordAlert('notification_simulated', timestamp, origin, notification);
    },
    [recordAlert, state.caregivers, state.prefs.notifyOnTimeout],
  );

  return { ...state, ...actions, recordAlert, completeAlert };
}

export function AppProvider({ children }: { children: React.ReactNode }) {
  const value = useValue();
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useApp() {
  const v = useContext(Ctx);
  if (!v) throw new Error('useApp must be used inside AppProvider');
  return v;
}
