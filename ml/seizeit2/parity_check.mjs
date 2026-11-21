// Python <-> TypeScript parity for the deployed SeizeIT2 model. Run from the repo root after export:
//   node ml/seizeit2/parity_check.mjs
// Needs Node >= 22.18 (runs the .ts sources directly via type stripping). Exits 1 on any mismatch.
import { readFileSync } from 'node:fs';
import { FEATURE_NAMES, windowFeatures } from '../../src/services/ml/seizeit2/features.ts';
import { checkModel, createDetector, scoreModel } from '../../src/services/ml/seizeit2/model.ts';

const read = (p) => JSON.parse(readFileSync(new URL(p, import.meta.url)));
const model = read('../../models/seizeit2/model.json');
const config = read('../../models/seizeit2/config.json');
const pv = read('./parity_vectors.json');
let failed = 0;
const check = (name, ok, detail) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? `  ${detail}` : ''}`); if (!ok) failed++; };

check('model asset compatible', checkModel(model, FEATURE_NAMES).length === 0, checkModel(model, FEATURE_NAMES).join('; '));
check('config matches parity vectors', config.input.window_s === pv.window_s && config.postprocessing.smoothing_k === pv.detector.k
  && config.postprocessing.threshold === pv.detector.threshold);

for (const w of pv.windows) {
  const f = windowFeatures(w.emg, w.acc, w.gyr, pv.window_s);
  // relative tolerance: summation order differs between numpy and JS
  const worst = Math.max(...f.map((v, i) => Math.abs(v - w.features[i]) / Math.max(1, Math.abs(w.features[i]))));
  check(`features ${w.name}`, worst < 1e-7, `max rel diff ${worst.toExponential(2)}`);
  const s = scoreModel(model, f);
  check(`score ${w.name}`, Math.abs(s - w.score) < 1e-6, `ts ${s.toFixed(6)} py ${w.score.toFixed(6)}`);
}

const diffs = pv.feature_vectors.map((x, i) => Math.abs(scoreModel(model, x) - pv.feature_vector_scores[i]));
check(`scores on ${diffs.length} feature vectors`, Math.max(...diffs) < 1e-12, `max diff ${Math.max(...diffs).toExponential(2)}`);

const d = pv.detector;
const push = createDetector(d.k, d.threshold, d.refractory_s, config.postprocessing.gap_reset_s);
const got = d.t_end_s.filter((t, i) => push(t * 1000, d.scores[i]));
check('detector alarms', JSON.stringify(got) === JSON.stringify(d.alarm_t_end_s), `ts [${got}] py [${d.alarm_t_end_s}]`);

process.exit(failed ? 1 : 0);
