import type { Caregiver, ContextLog, NotificationPrefs } from '../types';

export const mockCaregivers = (): Caregiver[] => [
  { id: 'cg-sample-1', name: 'Sample Caregiver', relationship: 'Family (sample)', phone: '+63 900 000 0000' },
];

export const defaultPrefs = (): NotificationPrefs => ({ notifyOnTimeout: true, countdownSeconds: 8 });

export const mockContextLogs = (): ContextLog[] => [
  {
    id: 'ctx-sample-1',
    at: new Date(2026, 10, 18, 9, 0).getTime(),
    triggers: ['Poor sleep'],
    medicationNote: 'Sample entry: morning medication taken as usual.',
    notes: 'Sample note for demonstration only.',
  },
];
