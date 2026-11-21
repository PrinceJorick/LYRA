// MlService backed by the SeizeIT2-trained model in models/seizeit2/ (see its README).
// Experimental research component, not clinically validated. A `possible_event` from detect() only
// means the score stream crossed the threshold; the app's alert, cancel countdown and caregiver
// workflow decide what happens next, exactly as for the demo trigger.

import config from '../../../../models/seizeit2/config.json';
import metadata from '../../../../models/seizeit2/metadata.json';
import model from '../../../../models/seizeit2/model.json';
import type { DetectionResult, MlService, SensorWindow } from '../types';
import { EMG_FS, FEATURE_NAMES, IMU_FS, windowFeatures } from './features';
import { checkModel, createDetector, scoreModel, type ModelJson } from './model';

const { window_s: windowS } = config.input;
const { smoothing_k: k, threshold, refractory_s: refractoryS, gap_reset_s: gapResetS } = config.postprocessing;

/** Asset problems found at load time; non-empty means the model must not be used. */
export const assetProblems: string[] = [
  ...checkModel(model, FEATURE_NAMES),
  ...(config.format !== (model as { format?: string }).format ? ['config.json and model.json formats differ'] : []),
  ...(config.input.emg.rate_hz !== EMG_FS || config.input.accelerometer.rate_hz !== IMU_FS
    ? ['model sample rates differ from this app\'s feature extractor'] : []),
  ...(JSON.stringify(config.features) !== JSON.stringify(FEATURE_NAMES) ? ['config feature list mismatch'] : []),
];

const flat = (a: number[]) => {
  let m = 0; for (const v of a) m += v; m /= a.length;
  let s = 0; for (const v of a) s += (v - m) ** 2;
  return Math.sqrt(s / a.length) <= config.quality.flat_std;
};

export function createSeizeit2Service(): MlService {
  const push = createDetector(k, threshold, refractoryS, gapResetS);

  const service: MlService = {
    getMetadata: () => ({ modelId: metadata.model_id, version: metadata.version, status: 'integrated' }),

    validateWindow(w) {
      const errors = [...assetProblems];
      if (!Number.isFinite(w.timestamp)) errors.push('Missing timestamp');
      if (w.emg.length !== EMG_FS * windowS) errors.push(`sEMG needs ${EMG_FS * windowS} samples (${windowS} s at ${EMG_FS} Hz)`);
      const imu = [w.accelerometer.x, w.accelerometer.y, w.accelerometer.z, w.gyroscope.x, w.gyroscope.y, w.gyroscope.z];
      if (imu.some((a) => a.length !== IMU_FS * windowS)) errors.push(`each IMU axis needs ${IMU_FS * windowS} samples (${windowS} s at ${IMU_FS} Hz)`);
      if ([w.emg, ...imu].some((a) => a.some((v) => !Number.isFinite(v)))) errors.push('Non-finite sample');
      return errors;
    },

    async detect(w: SensorWindow): Promise<DetectionResult> {
      const errors = service.validateWindow(w);
      if (errors.length) throw new Error(`SeizeIT2 model cannot score this window: ${errors.join('; ')}`);
      const unusable = flat(w.emg) || [w.accelerometer, w.gyroscope].some((a) => flat(a.x) && flat(a.y) && flat(a.z));
      const score = unusable ? 0 : scoreModel(model as ModelJson, windowFeatures(w.emg, w.accelerometer, w.gyroscope, windowS));
      const alarm = push(w.timestamp, score);
      return { eventType: alarm ? 'possible_event' : 'no_event', timestamp: w.timestamp, source: 'ml_model', modelVersion: metadata.version };
    },

    getInsights: () => [],
  };
  return service;
}

export const seizeit2Service = createSeizeit2Service();
