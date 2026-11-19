// Deterministic, illustrative waveforms. These are NOT physiological readings.

function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}

/** Burst-like noise to look like muscle activity. */
export function emgSample(seed: number, n = 64): number[] {
  const r = rng(seed);
  return Array.from({ length: n }, (_, i) => {
    const env = 0.25 + 0.75 * Math.abs(Math.sin(i / 7 + seed));
    return (r() * 2 - 1) * env;
  });
}

/** Smooth wave to look like motion. `phase` offsets between channels. */
export function imuSample(seed: number, phase = 0, n = 64): number[] {
  const r = rng(seed + 7);
  return Array.from({ length: n }, (_, i) =>
    0.6 * Math.sin(i / 5 + phase + seed) + 0.25 * Math.sin(i / 2.3 + phase) + (r() - 0.5) * 0.12);
}

export const ILLUSTRATIVE_LABEL = 'Illustrative Sample Data';
