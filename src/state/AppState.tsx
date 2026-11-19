import React, { createContext, useCallback, useContext, useMemo, useReducer } from 'react';
import { defaultPrefs, mockCaregivers, mockContextLogs } from '../data/mockCaregivers';
import { initialDevice } from '../data/mockDevice';
import { mockEvents } from '../data/mockEvents';
import { nextEventId } from '../services/events';
import { notificationService } from '../services/notifications';
import type {
  Caregiver, ContextLog, DeviceStatus, EventOutcome, NotificationPrefs, NotificationResult,
  Observation, PossibleEvent, ReportStatus, ReviewStatus, ThemeMode,
} from '../types';
import { newId } from '../utils/format';

// In-memory only: restarting the app resets everything to the demo data.
interface State {
  device: DeviceStatus;
  events: PossibleEvent[];
  caregivers: Caregiver[];
  prefs: NotificationPrefs;
  contextLogs: ContextLog[];
  themeMode: ThemeMode;
}

type Action =
  | { type: 'device'; device: DeviceStatus }
  | { type: 'addEvent'; event: PossibleEvent }
  | { type: 'patchEvent'; id: string; patch: Partial<PossibleEvent> }
  | { type: 'upsertCaregiver'; caregiver: Caregiver }
  | { type: 'removeCaregiver'; id: string }
  | { type: 'prefs'; patch: Partial<NotificationPrefs> }
  | { type: 'addContext'; log: ContextLog }
  | { type: 'theme'; mode: ThemeMode };

const init = (): State => ({
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
    (outcome: Exclude<EventOutcome, 'sample_record'>, timestamp: number, notification?: NotificationResult) => {
      const event: PossibleEvent = {
        id: nextEventId(state.events, timestamp),
        timestamp,
        origin: 'simulated_alert',
        outcome,
        review: 'needs_review',
        reportStatus: 'not_started',
        detectionSource: 'mock',
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
    async (timestamp: number) => {
      const notification = state.prefs.notifyOnTimeout
        ? await notificationService.notifyCaregivers(state.caregivers)
        : undefined;
      return recordAlert('notification_simulated', timestamp, notification);
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
