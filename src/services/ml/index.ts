// Swap mockMlService for the real model service when integrated. UI imports only from here.
export { mockMlService as mlService, demoTrigger } from './mockMlService';
export type * from './types';
