import type { DetectionResult, MlService, SensorWindow } from './types';

const channels = (w: SensorWindow) => [
  w.emg, w.accelerometer.x, w.accelerometer.y, w.accelerometer.z,
  w.gyroscope.x, w.gyroscope.y, w.gyroscope.z,
];

/**
 * Placeholder: no model is integrated. detect() never invents a prediction, it always
 * reports no_event. Possible events in the demo come only from demoTrigger().
 */
export const mockMlService: MlService = {
  getMetadata: () => ({ modelId: 'none', version: '0.0.0', status: 'mock' }),

  validateWindow(w) {
    const errors: string[] = [];
    if (!Number.isFinite(w.timestamp)) errors.push('Missing timestamp');
    if (w.emg.length === 0) errors.push('Empty sEMG channel');
    const lens = new Set(channels(w).map((c) => c.length));
    if (lens.size > 1) errors.push('Channels have different lengths');
    return errors;
  },

  async detect(w) {
    return { eventType: 'no_event', timestamp: w.timestamp, source: 'mock' };
  },

  getInsights: () => [],
};

/** Explicit, user-initiated demo trigger. Labelled source: 'mock', never 'ml_model'. */
export const demoTrigger = (): DetectionResult => ({
  eventType: 'possible_event',
  timestamp: Date.now(),
  source: 'mock',
});
