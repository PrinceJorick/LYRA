// Port of ml/seizeit2/preprocessing.py `window_features` for one window. Keep the two in lock-step:
// ml/seizeit2/export.py writes golden vectors and `node ml/seizeit2/parity_check.mjs` compares.
// No imports on purpose, so the parity check can run this file directly under Node.

export const EMG_FS = 256;
export const IMU_FS = 25;
const ENV_BIN = 16;
const ORIENT_S = 2;

export const FEATURE_NAMES = [
  'emg_hf_logpow', 'emg_frac_20_45', 'emg_frac_55_90', 'emg_low_ratio', 'emg_mains_ratio',
  'emg_median_freq', 'emg_spectral_entropy', 'emg_frame_logpow_std',
  'env_cv', 'env_peak', 'env_rhythm_1_5', 'env_dom_freq',
  'acc_mag_std', 'acc_var_trace', 'acc_jerk', 'acc_frac_1_6', 'acc_dom_freq', 'acc_peak',
  'gyr_cv', 'gyr_frac_1_6', 'gyr_dom_freq', 'gyr_peak',
  'orient_change_deg', 'xmod_freq_diff',
] as const;

export interface Axes { x: ArrayLike<number>; y: ArrayLike<number>; z: ArrayLike<number> }

const nextPow2 = (n: number) => 2 ** Math.ceil(Math.log2(n));
const hanning = (n: number) =>
  Float64Array.from({ length: n }, (_, i) => (n === 1 ? 1 : 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (n - 1))));
const mean = (a: ArrayLike<number>) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / a.length; };
const variance = (a: ArrayLike<number>) => {
  const m = mean(a); let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2;
  return s / a.length;
};
const std = (a: ArrayLike<number>) => Math.sqrt(variance(a));
const median = (a: ArrayLike<number>) => {
  const s = Float64Array.from(a).sort(); const h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
};
const sum = (a: ArrayLike<number>, lo = 0, hi = a.length) => { let s = 0; for (let i = lo; i < hi; i++) s += a[i]; return s; };

/** |FFT|^2 for bins 0..n/2 of a real signal zero-padded to n (n a power of two). */
function powerSpectrum(x: ArrayLike<number>, n: number): Float64Array {
  const re = new Float64Array(n); const im = new Float64Array(n);
  for (let i = 0; i < x.length; i++) re[i] = x[i];
  for (let i = 1, j = 0; i < n; i++) { // bit-reversal permutation
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < len / 2; k++) {
        const wr = Math.cos(ang * k); const wi = Math.sin(ang * k);
        const a = i + k; const b = a + len / 2;
        const tr = re[b] * wr - im[b] * wi; const ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
  return Float64Array.from({ length: n / 2 + 1 }, (_, k) => re[k] * re[k] + im[k] * im[k]);
}

/** Mean-removed, Hann-windowed, zero-padded power spectrum (preprocessing._spectrum). */
function spectrum(x: ArrayLike<number>) {
  const m = mean(x); const w = hanning(x.length);
  const n = nextPow2(x.length);
  return { p: powerSpectrum(Float64Array.from(x, (v, i) => (v - m) * w[i]), n), n };
}

/** [fraction in band, dominant frequency, log10 peak/mean] over lo <= f < hi (preprocessing._band_stats). */
function bandStats(p: Float64Array, df: number, lo: number, hi: number, band: [number, number]) {
  let total = 0; let inside = 0; let peak = -Infinity; let dom = 0; let count = 0;
  for (let k = 0; k < p.length; k++) {
    const f = k * df;
    if (f < lo || f >= hi) continue;
    total += p[k]; count++;
    if (f >= band[0] && f < band[1]) inside += p[k];
    if (p[k] > peak) { peak = p[k]; dom = f; }
  }
  return [inside / (total + 1e-12), dom, Math.log10((peak + 1e-12) / (total / count + 1e-12))];
}

/**
 * Features of one window. emg: EMG_FS * windowS samples in µV; acc: IMU_FS * windowS samples per axis
 * in g; gyr: same length, any fixed unit (only scale-invariant gyro features are used). Throws on wrong
 * lengths (callers validate first).
 */
export function windowFeatures(emg: ArrayLike<number>, acc: Axes, gyr: Axes, windowS: number): number[] {
  const nE = EMG_FS * windowS; const nI = IMU_FS * windowS;
  if (emg.length !== nE || [acc.x, acc.y, acc.z, gyr.x, gyr.y, gyr.z].some((a) => a.length !== nI)) {
    throw new Error(`window must hold ${nE} EMG and ${nI} IMU samples per axis`);
  }
  const out: number[] = new Array(FEATURE_NAMES.length);

  // sEMG PSD: mean of one-sided periodograms of 1 s Hann frames, 1 Hz bins
  const w = hanning(EMG_FS); const w2 = sum(w.map((v) => v * v));
  const psd = new Float64Array(EMG_FS / 2 + 1);
  const frameHf: number[] = [];
  const isHf = (k: number) => (k >= 20 && k < 45) || (k >= 55 && k < 128);
  for (let fr = 0; fr < windowS; fr++) {
    const seg = Array.prototype.slice.call(emg, fr * EMG_FS, (fr + 1) * EMG_FS) as number[];
    const m = mean(seg);
    const per = powerSpectrum(seg.map((v, i) => (v - m) * w[i]), EMG_FS);
    let hf = 0;
    for (let k = 0; k < per.length; k++) {
      per[k] /= EMG_FS * w2;
      if (k > 0 && k < per.length - 1) per[k] *= 2;
      psd[k] += per[k] / windowS;
      if (isHf(k)) hf += per[k];
    }
    frameHf.push(Math.log10(hf + 1e-6));
  }
  const hfBins: number[] = [];
  for (let k = 0; k < psd.length; k++) if (isHf(k)) hfBins.push(k);
  const pHf = hfBins.reduce((s, k) => s + psd[k], 0);
  out[0] = Math.log10(pHf + 1e-6);
  out[1] = sum(psd, 20, 45) / (pHf + 1e-12);
  out[2] = sum(psd, 55, 90) / (pHf + 1e-12);
  out[3] = Math.log10(sum(psd, 1, 20) + 1e-6) - out[0];
  out[4] = Math.log10(sum(psd, 45, 55) + 1e-6) - out[0];
  let cum = 0; out[5] = hfBins[0];
  for (const k of hfBins) { cum += psd[k]; if (cum >= 0.5 * pHf) { out[5] = k; break; } }
  let h = 0;
  for (const k of hfBins) { const p = psd[k] / (pHf + 1e-12); h -= p * Math.log(p + 1e-12); }
  out[6] = h / Math.log(hfBins.length);
  out[7] = std(frameHf);

  // sEMG envelope: RMS of the first difference in 1/16 s bins
  const env = new Float64Array(nE / ENV_BIN);
  for (let j = 0; j < env.length; j++) {
    let s = 0;
    for (let i = j * ENV_BIN; i < (j + 1) * ENV_BIN; i++) { const d = i === 0 ? 0 : emg[i] - emg[i - 1]; s += d * d; }
    env[j] = Math.sqrt(s / ENV_BIN);
  }
  const em = mean(env);
  out[8] = std(env) / (em + 1e-9);
  out[9] = Math.log10((Math.max(...env) + 1e-9) / (median(env) + 1e-9));
  const fsEnv = EMG_FS / ENV_BIN;
  const se = spectrum(env);
  [out[10], out[11]] = bandStats(se.p, fsEnv / se.n, 0.5, fsEnv / 2, [1, 5]);

  // IMU
  const mag = (a: Axes) => Float64Array.from({ length: nI }, (_, i) => Math.hypot(a.x[i], a.y[i], a.z[i]));
  const amag = mag(acc);
  out[12] = Math.log10(std(amag) + 1e-4);
  out[13] = Math.log10(variance(acc.x) + variance(acc.y) + variance(acc.z) + 1e-8);
  let jerk = 0;
  for (let i = 1; i < nI; i++) jerk += Math.abs(amag[i] - amag[i - 1]);
  out[14] = Math.log10((jerk / (nI - 1)) * IMU_FS + 1e-4);
  const sa = spectrum(amag);
  [out[15], out[16], out[17]] = bandStats(sa.p, IMU_FS / sa.n, 0.5, IMU_FS / 2, [1, 6]);
  const gmag = mag(gyr);
  out[18] = std(gmag) / (mean(gmag) + 1e-9);
  const sg = spectrum(gmag);
  [out[19], out[20], out[21]] = bandStats(sg.p, IMU_FS / sg.n, 0.5, IMU_FS / 2, [1, 6]);

  const k = IMU_FS * ORIENT_S;
  const avg = (a: ArrayLike<number>, lo: number, hi: number) => sum(a, lo, hi) / (hi - lo);
  const u = [acc.x, acc.y, acc.z].map((a) => avg(a, 0, k));
  const v = [acc.x, acc.y, acc.z].map((a) => avg(a, nI - k, nI));
  const cos = (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (Math.hypot(...u) * Math.hypot(...v) + 1e-12);
  out[22] = (Math.acos(Math.min(1, Math.max(-1, cos))) * 180) / Math.PI;
  out[23] = Math.abs(out[11] - out[20]);
  return out;
}
