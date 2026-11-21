// UI imports only from here. The SeizeIT2-trained model (models/seizeit2/) is used when its bundled
// assets match this app's feature extractor; otherwise the app falls back to the mock (and says so).
import { mockMlService } from './mockMlService';
import { assetProblems, seizeit2Service } from './seizeit2/service';

if (assetProblems.length) console.error(`SeizeIT2 model unavailable: ${assetProblems.join('; ')}`);

export const mlService = assetProblems.length ? mockMlService : seizeit2Service;
export { demoTrigger } from './mockMlService';
export type * from './types';
