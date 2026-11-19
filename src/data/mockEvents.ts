import type { PossibleEvent } from '../types';
import { SAMPLE_OBSERVATION } from './mockReports';

const at = (m: number, d: number, h: number, min: number) => new Date(2026, m, d, h, min).getTime();

const base = {
  origin: 'demo_sample',
  outcome: 'sample_record',
  detectionSource: 'mock',
  reportStatus: 'not_started',
} as const;

/** Bundled demo records. Never presented as real detections. */
export const mockEvents = (): PossibleEvent[] => [
  { ...base, id: 'EVT-20261119-001', timestamp: at(10, 19, 20, 42), review: 'needs_review', signalSeed: 11 },
  {
    ...base,
    id: 'EVT-20261118-001',
    timestamp: at(10, 18, 14, 16),
    review: 'reviewed',
    reportStatus: 'ready',
    observation: SAMPLE_OBSERVATION,
    signalSeed: 23,
  },
  { ...base, id: 'EVT-20261116-001', timestamp: at(10, 16, 11, 5), review: 'needs_review', signalSeed: 37 },
  { ...base, id: 'EVT-20261109-001', timestamp: at(10, 9, 6, 30), review: 'reviewed', signalSeed: 41 },
  { ...base, id: 'EVT-20261102-001', timestamp: at(10, 2, 22, 10), review: 'reviewed', signalSeed: 53 },
];
