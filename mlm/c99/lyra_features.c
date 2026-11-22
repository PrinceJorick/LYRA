/* lyra_features.c: see lyra_features.h. Each function mirrors the Python function named in its
 * comment (mlm/src/lyra_mlm/features.py), including scipy defaults: periodogram/Welch use constant
 * detrend, a periodic Hann window, density scaling, one-sided spectra and rfftfreq bins.
 * Python round() is round-half-even; rint() matches it in the default rounding mode. */
#include "lyra_features.h"

#include <math.h>
#include <stddef.h>

#define LYRA_PI 3.14159265358979323846
#define MAXN LYRA_MAX_SAMPLES
#define MAXBINS (MAXN / 2 + 1)

static const double ENV_LO = 1.5, ENV_HI = 5.0;  /* envelope_band_hz (Decision 3) */
static const double MOV_LO = 1.0, MOV_HI = 8.0;  /* movement_band_hz */
static const double MIN_ENV_RATE = 20.0, MAX_LAG_S = 0.5;
static const int MIN_PAIRED = 20;

static double s_a[MAXN], s_b[MAXN], s_c[MAXN], s_w[MAXN], s_f[MAXBINS], s_p[MAXBINS];
static double s_seg_a[MAXN], s_seg_b[MAXN];  /* coherence segments (inputs live in s_a / s_b) */
static double s_xr[MAXBINS], s_xi[MAXBINS], s_yr[MAXBINS], s_yi[MAXBINS];
static double s_sxx[MAXBINS], s_syy[MAXBINS], s_sxr[MAXBINS], s_sxi[MAXBINS];

static double mean(const double *x, int n) {
    double s = 0.0;
    int i;
    for (i = 0; i < n; ++i) s += x[i];
    return s / n;
}

static double pop_var(const double *x, int n) {  /* np.var (ddof 0) */
    double m = mean(x, n), s = 0.0;
    int i;
    for (i = 0; i < n; ++i) s += (x[i] - m) * (x[i] - m);
    return s / n;
}

static double trapezoid(const double *y, const double *x, int n) {  /* np.trapezoid */
    double s = 0.0;
    int i;
    for (i = 0; i + 1 < n; ++i) s += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2.0;
    return s;
}

static void hann_periodic(double *w, int n) {  /* scipy get_window('hann', n), fftbins=True */
    int i;
    for (i = 0; i < n; ++i) w[i] = 0.5 - 0.5 * cos(2.0 * LYRA_PI * i / n);
}

static double s_cos[MAXN], s_sin[MAXN];  /* twiddles cos/sin(2*pi*m/n), rebuilt when n changes */
static int s_twiddle_n = 0;

static void dft_onesided(const double *y, int n, double *re, double *im) {  /* rfft, nb = n/2 + 1 bins */
    int k, i, m, nb = n / 2 + 1;
    if (n != s_twiddle_n) {  /* the trig calls happen here only, not per sample and bin */
        for (m = 0; m < n; ++m) {
            double ang = 2.0 * LYRA_PI * (double)m / n;
            s_cos[m] = cos(ang);
            s_sin[m] = sin(ang);
        }
        s_twiddle_n = n;
    }
    for (k = 0; k < nb; ++k) {
        double r = 0.0, s = 0.0;
        for (i = 0, m = 0; i < n; ++i, m = (m + k) % n) {  /* m = k * i mod n */
            r += y[i] * s_cos[m];
            s -= y[i] * s_sin[m];
        }
        re[k] = r;
        im[k] = s;
    }
}

static void rfftfreq(double *f, int n, double fs) {  /* scipy.fft.rfftfreq(n, 1/fs) */
    double val = 1.0 / (n * (1.0 / fs));
    int k;
    for (k = 0; k < n / 2 + 1; ++k) f[k] = k * val;
}

/* scipy.signal.periodogram(x, fs, window="hann"): one-sided density into s_f/s_p; returns #bins */
static int periodogram(const double *x, int n, double fs) {
    int i, k, nb = n / 2 + 1;
    double m = mean(x, n), w2 = 0.0, scale;
    hann_periodic(s_w, n);
    for (i = 0; i < n; ++i) {
        s_c[i] = (x[i] - m) * s_w[i];
        w2 += s_w[i] * s_w[i];
    }
    dft_onesided(s_c, n, s_xr, s_xi);
    scale = 1.0 / (fs * w2);
    for (k = 0; k < nb; ++k) {
        s_p[k] = (s_xr[k] * s_xr[k] + s_xi[k] * s_xi[k]) * scale;
        if (k > 0 && (n % 2 == 1 || k < nb - 1)) s_p[k] *= 2.0;
    }
    rfftfreq(s_f, n, fs);
    return nb;
}

/* band [lo, hi] of the periodogram in s_f/s_p: first index and count */
static int band(int nb, double lo, double hi, int *first) {
    int k, cnt = 0;
    *first = -1;
    for (k = 0; k < nb; ++k) {
        if (s_f[k] >= lo && s_f[k] <= hi) {
            if (*first < 0) *first = k;
            ++cnt;
        }
    }
    return cnt;
}

static int argmax(const double *x, int n) {  /* first maximum, as np.argmax */
    int i, best = 0;
    for (i = 1; i < n; ++i)
        if (x[i] > x[best]) best = i;
    return best;
}

/* _env_on_imu_grid: mean envelope within [t_j - half, t_j + half) of each IMU sample, into s_a */
static void env_on_imu_grid(const double *env_t, const double *env, int n_env, const double *imu_t, int n_imu,
                            double imu_rate) {
    double half = 0.5 / imu_rate;
    int j, lo = 0, hi = 0;
    for (j = 0; j < n_imu; ++j) {
        double a = imu_t[j] - half, b = imu_t[j] + half, s = 0.0;
        int k;
        while (lo < n_env && env_t[lo] < a) ++lo;
        if (hi < lo) hi = lo;
        while (hi < n_env && env_t[hi] < b) ++hi;
        for (k = lo; k < hi; ++k) s += env[k];
        s_a[j] = hi > lo ? s / (hi - lo) : NAN;
    }
}

static double coherence_band(const double *a, const double *b, int n, double fs) {  /* _xmod_coherence */
    int nperseg = (int)rint(2.0 * fs / ENV_LO), noverlap, step, nseg, nb, s, i, k, cnt = 0;
    double total = 0.0, val;
    if (nperseg > MAXN || n < 2 * nperseg) return NAN;
    noverlap = nperseg / 2;
    step = nperseg - noverlap;
    nseg = (n - noverlap) / step;
    nb = nperseg / 2 + 1;
    hann_periodic(s_w, nperseg);
    for (k = 0; k < nb; ++k) s_sxx[k] = s_syy[k] = s_sxr[k] = s_sxi[k] = 0.0;
    for (s = 0; s < nseg; ++s) {
        double ma = mean(a + s * step, nperseg), mb = mean(b + s * step, nperseg);
        for (i = 0; i < nperseg; ++i) {
            s_seg_a[i] = (a[s * step + i] - ma) * s_w[i];
            s_seg_b[i] = (b[s * step + i] - mb) * s_w[i];
        }
        dft_onesided(s_seg_a, nperseg, s_xr, s_xi);
        dft_onesided(s_seg_b, nperseg, s_yr, s_yi);
        for (k = 0; k < nb; ++k) {
            s_sxx[k] += s_xr[k] * s_xr[k] + s_xi[k] * s_xi[k];
            s_syy[k] += s_yr[k] * s_yr[k] + s_yi[k] * s_yi[k];
            s_sxr[k] += s_xr[k] * s_yr[k] + s_xi[k] * s_yi[k];  /* conj(X) * Y */
            s_sxi[k] += s_xr[k] * s_yi[k] - s_xi[k] * s_yr[k];
        }
    }
    rfftfreq(s_f, nperseg, fs);
    for (k = 0; k < nb; ++k) {
        if (s_f[k] >= ENV_LO && s_f[k] <= ENV_HI) {
            total += (s_sxr[k] * s_sxr[k] + s_sxi[k] * s_sxi[k]) / (s_sxx[k] * s_syy[k]);
            ++cnt;
        }
    }
    if (cnt == 0) return NAN;
    val = total / cnt;
    return val;
}

static void emg_block(const double *x, int n, double rate, double *out) {
    int i, k, nb, first, cnt;
    double m, var, wl = 0.0, energy = 0.0, best = -INFINITY, total;
    if (n >= 1) {
        double sq = 0.0, ab = 0.0;
        for (i = 0; i < n; ++i) {
            sq += x[i] * x[i];
            ab += fabs(x[i]);
        }
        out[0] = sqrt(sq / n);  /* _emg_rms */
        out[1] = ab / n;        /* _emg_mav */
    }
    if (n >= 2) {
        var = pop_var(x, n);
        out[2] = var;        /* _emg_var */
        out[3] = sqrt(var);  /* _emg_std */
        for (i = 0; i + 1 < n; ++i) wl += fabs(x[i + 1] - x[i]);
        out[4] = wl / ((n - 1) / rate);  /* _emg_wl */
    }
    /* _envelope_run */
    if (!(rate >= MIN_ENV_RATE && ENV_HI < rate / 2) || n / rate < 2.0 / ENV_LO) return;
    m = mean(x, n);
    for (i = 0; i < n; ++i) {
        s_a[i] = x[i] - m;
        energy += s_a[i] * s_a[i];
    }
    if (energy > 0) {  /* _emg_autocorr */
        int k0 = (int)rint(rate / ENV_HI), k1 = (int)rint(rate / ENV_LO);
        for (k = k0 < 1 ? 1 : k0; k <= k1 && k < n; ++k) {
            double d = 0.0;
            for (i = 0; i + k < n; ++i) d += s_a[i] * s_a[i + k];
            if (d / energy > best) best = d / energy;
        }
        out[5] = best;
    }
    nb = periodogram(s_a, n, rate);  /* _env_spectrum (periodogram detrends again, as scipy does) */
    cnt = band(nb, ENV_LO, ENV_HI, &first);
    if (cnt < 2) return;
    total = trapezoid(s_p + 1, s_f + 1, nb - 1);
    out[6] = trapezoid(s_p + first, s_f + first, cnt);  /* _env_mod_power */
    if (total > 0) out[7] = out[6] / total;             /* _env_mod_ratio */
    k = first + argmax(s_p + first, cnt);
    if (s_p[k] > 0) out[8] = s_f[k];                    /* _env_mod_peak */
}

static void imu_block(const double *imu, int n, double rate, double *out) {
    int i, j, nb, first, cnt, q;
    double jsum = 0.0, g1[3] = {0, 0, 0}, g2[3] = {0, 0, 0}, n1, n2, dot;
    if (n < 2) return;
    for (i = 0; i < n; ++i) {
        const double *r = imu + 6 * i;
        s_b[i] = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);  /* acc_mag */
        s_c[i] = sqrt(r[3] * r[3] + r[4] * r[4] + r[5] * r[5]);  /* gyr_mag */
    }
    out[0] = mean(s_b, n);
    out[1] = sqrt(pop_var(s_b, n));
    out[2] = mean(s_c, n);
    out[3] = sqrt(pop_var(s_c, n));
    {
        double sq = 0.0;
        for (i = 0; i < n; ++i) sq += s_c[i] * s_c[i];
        out[4] = sqrt(sq / n);
    }
    for (i = 0; i + 1 < n; ++i) {  /* _acc_jerk_rms */
        const double *r0 = imu + 6 * i, *r1 = imu + 6 * (i + 1);
        double dx = r1[0] - r0[0], dy = r1[1] - r0[1], dz = r1[2] - r0[2];
        double jv = sqrt(dx * dx + dy * dy + dz * dz) * rate;
        jsum += jv * jv;
    }
    out[5] = sqrt(jsum / (n - 1));
    if (rate / 2 >= MOV_HI && n / rate >= 2.0 / MOV_LO) {  /* _acc_spectrum */
        nb = periodogram(s_b, n, rate);
        cnt = band(nb, MOV_LO, MOV_HI, &first);
        if (cnt >= 2) {
            j = first + argmax(s_p + first, cnt);
            if (s_p[j] > 0) out[6] = s_f[j];                   /* _acc_dominant_freq */
            out[7] = trapezoid(s_p + first, s_f + first, cnt);  /* _acc_band_power */
        }
    }
    q = n / 4;  /* _orientation_change */
    if (q >= 1) {
        for (i = 0; i < q; ++i)
            for (j = 0; j < 3; ++j) {
                g1[j] += imu[6 * i + j];
                g2[j] += imu[6 * (n - q + i) + j];
            }
        for (j = 0; j < 3; ++j) {
            g1[j] /= q;
            g2[j] /= q;
        }
        n1 = sqrt(g1[0] * g1[0] + g1[1] * g1[1] + g1[2] * g1[2]);
        n2 = sqrt(g2[0] * g2[0] + g2[1] * g2[1] + g2[2] * g2[2]);
        if (n1 > 0 && n2 > 0) {
            dot = (g1[0] * g2[0] + g1[1] * g2[1] + g1[2] * g2[2]) / (n1 * n2);
            dot = dot > 1.0 ? 1.0 : dot < -1.0 ? -1.0 : dot;
            out[8] = acos(dot) * 180.0 / LYRA_PI;
        }
    }
}

/* cross-modal block; needs s_b = acc_mag (from imu_block) */
static void xmod_block(const double *env_t, const double *env, int n_env, double env_rate, const double *imu_t,
                       int n_imu, double imu_rate, double emg_rms, double acc_std, double *out) {
    int j, ok = 1, max_k, m, best;
    double ma, mb, saa = 0.0, sbb = 0.0, sab = 0.0, bestv = -INFINITY;
    if (emg_rms > 0 && acc_std > 0) out[1] = log10(emg_rms / acc_std);  /* _xmod_energy_ratio */
    env_on_imu_grid(env_t, env, n_env, imu_t, n_imu, imu_rate);
    {   /* _xmod_corr: np.corrcoef over the IMU bins that have envelope samples */
        int n_ok = 0;
        double sa = 0.0, sb = 0.0;
        for (j = 0; j < n_imu; ++j) {
            if (isnan(s_a[j])) {
                ok = 0;
                continue;
            }
            sa += s_a[j];
            sb += s_b[j];
            ++n_ok;
        }
        if (n_ok >= MIN_PAIRED) {
            double ca = sa / n_ok, cb = sb / n_ok, r;
            for (j = 0; j < n_imu; ++j) {
                if (isnan(s_a[j])) continue;
                saa += (s_a[j] - ca) * (s_a[j] - ca);
                sbb += (s_b[j] - cb) * (s_b[j] - cb);
                sab += (s_a[j] - ca) * (s_b[j] - cb);
            }
            if (saa > 0 && sbb > 0) {
                r = (sab / (n_ok - 1)) / sqrt(saa / (n_ok - 1)) / sqrt(sbb / (n_ok - 1));
                out[0] = r > 1.0 ? 1.0 : r < -1.0 ? -1.0 : r;
            }
        }
    }
    if (!ok || n_imu < MIN_PAIRED) return;  /* _paired_regular: every IMU bin covered by the envelope */
    ma = mean(s_a, n_imu);
    mb = mean(s_b, n_imu);
    saa = sbb = 0.0;
    for (j = 0; j < n_imu; ++j) {
        s_a[j] -= ma;
        s_b[j] -= mb;
        saa += s_a[j] * s_a[j];
        sbb += s_b[j] * s_b[j];
    }
    if (!(saa > 0 && sbb > 0)) return;
    max_k = (int)rint(MAX_LAG_S * imu_rate);  /* _xmod_lag: scipy.signal.correlate(b, a)[mid + m] */
    if (max_k > n_imu - 1) max_k = n_imu - 1;
    best = 0;
    for (m = -max_k; m <= max_k; ++m) {
        double z = 0.0;
        int l;
        for (l = 0; l < n_imu; ++l)
            if (l - m >= 0 && l - m < n_imu) z += s_b[l] * s_a[l - m];
        if (z > bestv) {
            bestv = z;
            best = m;
        }
    }
    out[2] = best / imu_rate;
    if (env_rate >= MIN_ENV_RATE && imu_rate / 2 >= ENV_HI)
        out[3] = coherence_band(s_a, s_b, n_imu, imu_rate);
}

int lyra_features(const double *env_t, const double *env, int n_env, double env_rate_hz,
                  const double *imu_t, const double *imu, int n_imu, double imu_rate_hz,
                  double out[LYRA_N_FEATURES]) {
    int i;
    if (out == NULL || n_env < 0 || n_imu < 0 || n_env > MAXN || n_imu > MAXN ||
        (n_env > 0 && (env == NULL || env_t == NULL || !(env_rate_hz > 0))) ||
        (n_imu > 0 && (imu == NULL || imu_t == NULL || !(imu_rate_hz > 0))))
        return -1;
    for (i = 0; i < LYRA_N_FEATURES; ++i) out[i] = NAN;
    if (n_env > 0) emg_block(env, n_env, env_rate_hz, out);
    if (n_imu > 0) imu_block(imu, n_imu, imu_rate_hz, out + LYRA_N_EMG_FEATURES);
    if (n_env > 0 && n_imu > 0)
        xmod_block(env_t, env, n_env, env_rate_hz, imu_t, n_imu, imu_rate_hz, out[0], out[10], out + 18);
    for (i = 0; i < LYRA_N_FEATURES; ++i)
        if (!isfinite(out[i])) out[i] = NAN;  /* non-finite results are unavailable, as in Python */
    return 0;
}
