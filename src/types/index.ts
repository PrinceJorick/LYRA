// Shared domain types. ML contracts live in services/ml/types.ts.

export type ReviewStatus = 'needs_review' | 'reviewed';
export type ReportStatus = 'not_started' | 'ready' | 'exported';
/** sample_record = bundled demo data; the other two come from the in-app simulation. */
export type EventOutcome = 'sample_record' | 'cancelled' | 'notification_simulated';
export type EventOrigin = 'demo_sample' | 'simulated_alert';

export type Responsiveness = 'responsive' | 'partly_responsive' | 'unresponsive' | 'unsure';
export type Reporter = 'self' | 'witness';

export interface Observation {
  reporter: Reporter;
  durationApprox: string; // free text, e.g. "about 1 minute"
  movements: string[];
  responsiveness: Responsiveness;
  recovery: string;
  notes: string;
  recordedAt: number;
}

export interface NotificationRecipientStatus {
  caregiverName: string;
  sending: 'simulated';
  delivery: 'simulated';
  acknowledgment: 'not_simulated';
}

export interface NotificationResult {
  simulated: true;
  at: number;
  recipients: NotificationRecipientStatus[];
}

export interface PossibleEvent {
  id: string;
  timestamp: number;
  origin: EventOrigin;
  outcome: EventOutcome;
  review: ReviewStatus;
  reportStatus: ReportStatus;
  /** Who/what raised the event: the mock detector, or (later) a real model. */
  detectionSource: 'mock' | 'ml_model';
  observation?: Observation;
  notification?: NotificationResult;
  /** Seed for the illustrative waveforms shown on the detail screen. */
  signalSeed: number;
}

export interface Caregiver {
  id: string;
  name: string;
  relationship: string;
  phone: string;
}

export interface NotificationPrefs {
  notifyOnTimeout: boolean;
  countdownSeconds: number; // 7–10
}

export interface ContextLog {
  id: string;
  at: number;
  triggers: string[];
  medicationNote: string; // user-entered only; LYRA gives no medication advice
  notes: string;
}

export type ConnectionState = 'disconnected' | 'connected';

export interface DeviceStatus {
  connection: ConnectionState;
  deviceId: string;
  name: string;
  batteryPercent: number;
  lastSyncAt: number | null;
}

export type ThemeMode = 'light' | 'dark' | 'system';
