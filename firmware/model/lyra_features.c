/* See lyra_features.h. Mirrors ml/lyra/features.py; double accumulators, float inputs/outputs. */
#include "lyra_features.h"

#include <math.h>

#define LYRA_PI 3.14159265358979323846
#define COH_NPERSEG 16
#define COH_STEP 8
#define COH_MIN_SEGMENTS 4
#define DOM_TIE_REL 1e-6

static double s_mag[LYRA_MAX_IMU_SAMPLES]; /* ||a|| */
static double s_env25[LYRA_MAX_IMU_SAMPLES];
static double s_pow[LYRA_MAX_IMU_SAMPLES / 2 + 1];

static int all_finite(const float *v, int n) {
    for (int i = 0; i < n; i++)
        if (!isfinite(v[i])) return 0;
    return 1;
}

static int check_sizes(int n_env, int n_imu) {
    if (n_imu < LYRA_MIN_IMU_SAMPLES || n_imu > LYRA_MAX_IMU_SAMPLES || n_env != 2 * n_imu) return 0;
    return (n_imu - COH_NPERSEG) / COH_STEP + 1 >= COH_MIN_SEGMENTS;
}

int lyra_window_valid(const float *env, int n_env, const float *acc, const float *gyr, int n_imu,
                      float env_saturation_level) {
    if (!check_sizes(n_env, n_imu)) return LYRA_ERR_SIZE;
    if (!all_finite(env, n_env) || !all_finite(acc, 3 * n_imu) || !all_finite(gyr, 3 * n_imu)) return LYRA_ERR_NONFINITE;
    float lo = env[0], hi = env[0];
    int sat = 0;
    for (int i = 0; i < n_env; i++) {
        if (env[i] < lo) lo = env[i];
        if (env[i] > hi) hi = env[i];
        if (env[i] >= env_saturation_level) sat++;
    }
    if (!(hi > lo)) return LYRA_ERR_FLAT;
    if ((double)sat / n_env > LYRA_MAX_SATURATED_FRACTION) return LYRA_ERR_SATURATED;
    int moving = 0;
    for (int j = 1; j < n_imu && !moving; j++)
        for (int c = 0; c < 3; c++)
            if (acc[3 * j + c] != acc[3 * (j - 1) + c] || gyr[3 * j + c] != gyr[3 * (j - 1) + c]) moving = 1;
    return moving ? LYRA_OK : LYRA_ERR_FLAT;
}

static double hann(int i, int n) { return 0.5 - 0.5 * cos(2.0 * LYRA_PI * (double)i / (double)n); }

/* X_k = sum_j d_j exp(-2 pi i j k / n); twiddle angle reduced modulo n for accuracy */
static void dft_bin(const double *d, int n, int k, double *re, double *im) {
    double r = 0.0, s = 0.0;
    for (int j = 0; j < n; j++) {
        const double a = 2.0 * LYRA_PI * (double)((j * k) % n) / (double)n;
        r += d[j] * cos(a);
        s -= d[j] * sin(a);
    }
    *re = r;
    *im = s;
}

static double safe_div(double num, double den) { return den > LYRA_EPS ? num / den : 0.0; }

int lyra_features_compute(const float *env, int n_env, const float *acc, const float *gyr, int n_imu,
                          float out[LYRA_N_FEATURES]) {
    if (!check_sizes(n_env, n_imu)) return LYRA_ERR_SIZE;
    if (!all_finite(env, n_env) || !all_finite(acc, 3 * n_imu) || !all_finite(gyr, 3 * n_imu)) return LYRA_ERR_NONFINITE;

    /* ---- sEMG (ENV) ---- */
    double sum = 0.0, sq = 0.0, mav = 0.0, wl = 0.0;
    for (int i = 0; i < n_env; i++) {
        const double e = env[i];
        sum += e;
        sq += e * e;
        mav += fabs(e);
        if (i) wl += fabs(e - (double)env[i - 1]);
    }
    const double mean = sum / n_env;
    double var_sum = 0.0;
    for (int i = 0; i < n_env; i++) var_sum += (env[i] - mean) * (env[i] - mean);
    double acf = -INFINITY;
    for (int lag = LYRA_ENV_RATE_HZ / 5; lag <= LYRA_ENV_RATE_HZ && lag < n_env; lag++) {
        double s = 0.0;
        for (int t = 0; t + lag < n_env; t++) s += (env[t] - mean) * (env[t + lag] - mean);
        if (s > acf) acf = s;
    }

    /* ---- IMU ---- */
    double msum = 0.0, gsum = 0.0, jerk = 0.0;
    for (int j = 0; j < n_imu; j++) {
        const double ax = acc[3 * j], ay = acc[3 * j + 1], az = acc[3 * j + 2];
        const double gx = gyr[3 * j], gy = gyr[3 * j + 1], gz = gyr[3 * j + 2];
        s_mag[j] = sqrt(ax * ax + ay * ay + az * az);
        msum += s_mag[j];
        gsum += sqrt(gx * gx + gy * gy + gz * gz);
        if (j) {
            const double dj = (s_mag[j] - s_mag[j - 1]) * LYRA_IMU_RATE_HZ;
            jerk += dj * dj;
        }
    }
    const double mmean = msum / n_imu;
    double band = 0.0, total = 0.0, best_p = -1.0, best_f = 0.0;
    /* periodogram of periodic-Hann * (|a| - mean): reuse s_env25 as scratch before it is filled;
     * s_pow keeps the bins so the dominant frequency can use the tie-tolerant rule of the spec */
    for (int j = 0; j < n_imu; j++) s_env25[j] = (s_mag[j] - mmean) * hann(j, n_imu);
    for (int k = 0; k <= n_imu / 2; k++) {
        double re, im;
        dft_bin(s_env25, n_imu, k, &re, &im);
        const double p = re * re + im * im, f = (double)k * LYRA_IMU_RATE_HZ / n_imu;
        s_pow[k] = p;
        if (k >= 1) total += p;
        if (f >= 1.0 && f <= 5.0) band += p;
        if (f >= 0.5 && f <= 8.0 && p > best_p) best_p = p;
    }
    for (int k = 0; k <= n_imu / 2; k++) { /* lowest bin within DOM_TIE_REL of the maximum */
        const double f = (double)k * LYRA_IMU_RATE_HZ / n_imu;
        if (f >= 0.5 && f <= 8.0 && s_pow[k] >= best_p * (1.0 - DOM_TIE_REL)) {
            best_f = f;
            break;
        }
    }
    double g1[3] = {0, 0, 0}, g2[3] = {0, 0, 0};
    for (int j = 0; j < LYRA_IMU_RATE_HZ; j++)
        for (int c = 0; c < 3; c++) {
            g1[c] += acc[3 * j + c];
            g2[c] += acc[3 * (n_imu - LYRA_IMU_RATE_HZ + j) + c];
        }
    for (int c = 0; c < 3; c++) {
        g1[c] /= LYRA_IMU_RATE_HZ;
        g2[c] /= LYRA_IMU_RATE_HZ;
    }
    const double cx = g1[1] * g2[2] - g1[2] * g2[1], cy = g1[2] * g2[0] - g1[0] * g2[2], cz = g1[0] * g2[1] - g1[1] * g2[0];
    const double n1 = sqrt(g1[0] * g1[0] + g1[1] * g1[1] + g1[2] * g1[2]);
    const double n2 = sqrt(g2[0] * g2[0] + g2[1] * g2[1] + g2[2] * g2[2]);
    const double tilt = (n1 > LYRA_EPS && n2 > LYRA_EPS)
                            ? atan2(sqrt(cx * cx + cy * cy + cz * cz), g1[0] * g2[0] + g1[1] * g2[1] + g1[2] * g2[2]) * 180.0 / LYRA_PI
                            : 0.0;

    /* ---- cross-modal: env at 25 Hz vs |a| ---- */
    double xs = 0.0;
    for (int j = 0; j < n_imu; j++) {
        s_env25[j] = 0.5 * ((double)env[2 * j] + (double)env[2 * j + 1]);
        xs += s_env25[j];
    }
    const double xm = xs / n_imu;
    double sxy = 0.0, sxx = 0.0, syy = 0.0;
    for (int j = 0; j < n_imu; j++) {
        const double dx = s_env25[j] - xm, dy = s_mag[j] - mmean;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    const double r = (sxx > LYRA_EPS && syy > LYRA_EPS) ? sxy / sqrt(sxx * syy) : 0.0;
    double pxx[3] = {0, 0, 0}, pyy[3] = {0, 0, 0}, pxr[3] = {0, 0, 0}, pxi[3] = {0, 0, 0};
    const int nseg = (n_imu - COH_NPERSEG) / COH_STEP + 1;
    for (int s = 0; s < nseg; s++) {
        double dx[COH_NPERSEG], dy[COH_NPERSEG], mx = 0.0, my = 0.0;
        for (int i = 0; i < COH_NPERSEG; i++) {
            mx += s_env25[s * COH_STEP + i];
            my += s_mag[s * COH_STEP + i];
        }
        mx /= COH_NPERSEG;
        my /= COH_NPERSEG;
        for (int i = 0; i < COH_NPERSEG; i++) {
            dx[i] = (s_env25[s * COH_STEP + i] - mx) * hann(i, COH_NPERSEG);
            dy[i] = (s_mag[s * COH_STEP + i] - my) * hann(i, COH_NPERSEG);
        }
        for (int b = 0; b < 3; b++) { /* bins k = 1, 2, 3: 1.5625, 3.125, 4.6875 Hz */
            double xr, xi, yr, yi;
            dft_bin(dx, COH_NPERSEG, b + 1, &xr, &xi);
            dft_bin(dy, COH_NPERSEG, b + 1, &yr, &yi);
            pxx[b] += xr * xr + xi * xi;
            pyy[b] += yr * yr + yi * yi;
            pxr[b] += xr * yr + xi * yi; /* x * conj(y) */
            pxi[b] += xi * yr - xr * yi;
        }
    }
    double coh = 0.0;
    for (int b = 0; b < 3; b++) coh += safe_div(pxr[b] * pxr[b] + pxi[b] * pxi[b], pxx[b] * pyy[b]);

    out[LYRA_F_EMG_RMS] = (float)sqrt(sq / n_env);
    out[LYRA_F_EMG_MAV] = (float)(mav / n_env);
    out[LYRA_F_EMG_WL] = (float)wl;
    out[LYRA_F_EMG_ENV_STD] = (float)sqrt(var_sum / n_env);
    out[LYRA_F_EMG_ACF_PEAK] = (float)safe_div(acf, var_sum);
    out[LYRA_F_IMU_ACC_MAG_MEAN] = (float)mmean;
    out[LYRA_F_IMU_GYR_MAG_MEAN] = (float)(gsum / n_imu);
    out[LYRA_F_IMU_JERK_RMS] = (float)sqrt(jerk / (n_imu - 1));
    out[LYRA_F_IMU_ACC_BAND_RATIO_1_5HZ] = (float)safe_div(band, total);
    out[LYRA_F_IMU_ACC_DOM_FREQ_HZ] = (float)(best_p > LYRA_EPS ? best_f : 0.0);
    out[LYRA_F_IMU_TILT_DEG] = (float)tilt;
    out[LYRA_F_XMOD_CORR_ENV_ACCMAG] = (float)r;
    out[LYRA_F_XMOD_COH_1_5HZ] = (float)(coh / 3.0);
    return LYRA_OK;
}
