// Evaluator for models/seizeit2/model.json (mirror of ml/seizeit2/export.py `score_json`).
// No imports on purpose, so ml/seizeit2/parity_check.mjs can run it under Node.

export const MODEL_FORMAT = 'lyra-seizeit2-model/1';

interface Tree { feature: number[]; threshold: number[]; left: number[]; right: number[]; value: number[] }
interface Common { format: string; feature_names: string[]; columns: number[] }
export type ModelJson =
  | (Common & { kind: 'gbt'; baseline: number; trees: Tree[] })
  | (Common & { kind: 'logreg'; mean: number[]; scale: number[]; coef: number[]; intercept: number });

/** Problems that make the asset unusable with this app build; empty means compatible. */
export function checkModel(m: unknown, featureNames: readonly string[]): string[] {
  const j = m as Partial<ModelJson> | null;
  if (!j || j.format !== MODEL_FORMAT) return [`model format must be ${MODEL_FORMAT}`];
  const errors: string[] = [];
  if (JSON.stringify(j.feature_names) !== JSON.stringify(featureNames)) {
    errors.push('model feature list does not match this app\'s feature extractor');
  }
  if (!Array.isArray(j.columns) || j.columns.some((c) => !(c >= 0 && c < featureNames.length))) {
    errors.push('model columns out of range');
  }
  if (j.kind === 'gbt' ? !Array.isArray(j.trees) || !j.trees.length : j.kind !== 'logreg') {
    errors.push('unsupported or empty model');
  }
  return errors;
}

/** Probability-like score in [0, 1] for one feature vector (full FEATURE_NAMES order). */
export function scoreModel(m: ModelJson, features: number[]): number {
  const z = m.columns.map((c) => features[c]);
  let raw: number;
  if (m.kind === 'logreg') {
    raw = m.intercept;
    for (let i = 0; i < z.length; i++) raw += ((z[i] - m.mean[i]) / m.scale[i]) * m.coef[i];
  } else {
    raw = m.baseline;
    for (const t of m.trees) {
      let n = 0;
      while (t.feature[n] >= 0) n = z[t.feature[n]] <= t.threshold[n] ? t.left[n] : t.right[n];
      raw += t.value[n];
    }
  }
  return 1 / (1 + Math.exp(-raw));
}

/**
 * Post-processing (mirror of export.py `stream_alarms`): moving average of the last k scores, alarm
 * when it reaches the threshold and no alarm fired in the previous refractoryS. A gap longer than
 * gapResetS, or time going backwards, restarts the average.
 */
export function createDetector(k: number, threshold: number, refractoryS: number, gapResetS: number) {
  let scores: number[] = [];
  let prevMs: number | undefined;
  let lastAlarmMs: number | undefined;
  return (timestampMs: number, score: number): boolean => {
    if (prevMs !== undefined && (timestampMs - prevMs > gapResetS * 1000 || timestampMs < prevMs)) {
      scores = [];
      if (timestampMs < prevMs) lastAlarmMs = undefined;
    }
    prevMs = timestampMs;
    scores = [...scores, score].slice(-k);
    const smoothed = scores.reduce((a, b) => a + b, 0) / scores.length;
    const alarm = smoothed >= threshold && (lastAlarmMs === undefined || timestampMs - lastAlarmMs >= refractoryS * 1000);
    if (alarm) lastAlarmMs = timestampMs;
    return alarm;
  };
}
