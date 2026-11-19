import type { Observation } from '../types';

/** Sample witness observation attached to one demo record. Illustrative only. */
export const SAMPLE_OBSERVATION: Observation = {
  reporter: 'witness',
  durationApprox: 'About 1 minute (sample)',
  movements: ['Stiffening', 'Rhythmic jerking'],
  responsiveness: 'unresponsive',
  recovery: 'Drowsy and confused for several minutes (sample).',
  notes: 'Sample note for demonstration only.',
  recordedAt: new Date(2026, 10, 18, 14, 40).getTime(),
};

export const MOVEMENT_OPTIONS = [
  'Stiffening',
  'Rhythmic jerking',
  'Limp / loss of tone',
  'Eyes open, staring',
  'Fell down',
  'Other',
];

export const RESPONSIVENESS_OPTIONS = [
  { value: 'responsive', label: 'Responsive' },
  { value: 'partly_responsive', label: 'Partly responsive' },
  { value: 'unresponsive', label: 'Not responsive' },
  { value: 'unsure', label: 'Not sure' },
] as const;

export const TRIGGER_OPTIONS = ['Poor sleep', 'Stress', 'Illness', 'Missed meal', 'Flashing lights', 'Other'];
