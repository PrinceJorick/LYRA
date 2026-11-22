/* lyra_quality.c: see lyra_quality.h. Checks run in the same order as assess_modality. */
#include "lyra_quality.h"

#include <math.h>
#include <stddef.h>

#include "lyra_features.h"

static int usable(lyra_quality q) { return q == LYRA_Q_VALID || q == LYRA_Q_DEGRADED; }

static unsigned char flag_of(const double *values, const unsigned char *flags, int i, int n_ch) {
    unsigned char f = LYRA_FLAG_OK;
    int c;
    for (c = 0; c < n_ch; ++c) {
        double v = values[i * n_ch + c];
        if (isnan(v)) f = LYRA_FLAG_MISSING;
    }
    for (c = 0; c < n_ch; ++c)
        if (isinf(values[i * n_ch + c])) f = LYRA_FLAG_INVALID;
    if (flags != NULL && flags[i] > f) f = flags[i];
    return f;
}

lyra_quality lyra_assess(const double *t, const double *values, const unsigned char *flags, int n, int n_ch,
                         double rate_hz, double start_s, double end_s, const double *sat_lo, const double *sat_hi,
                         const lyra_quality_cfg *cfg, double *coverage_out) {
    double period = 1.0 / rate_hz, coverage, prev = 0.0;
    int i, c, n_ok = 0, n_flat = 0, degraded = 0, irregular = 0, gaps = 0, have_prev = 0;
    if (coverage_out != NULL) *coverage_out = 0.0;
    if (n <= 0) return LYRA_Q_MISSING;
    for (i = 0; i < n; ++i) {  /* check_timestamps on finite timestamps */
        double d;
        if (!isfinite(t[i])) continue;
        if (have_prev) {
            d = t[i] - prev;
            if (d <= 0) return LYRA_Q_UNSYNCHRONIZED; /* duplicate or out-of-order */
            if (d > cfg->gap_factor * period) ++gaps;
            else if (fabs(d - period) > cfg->interval_tolerance * period) ++irregular;
        }
        prev = t[i];
        have_prev = 1;
    }
    for (i = 0; i < n; ++i) {
        unsigned char f = flag_of(values, flags, i, n_ch);
        if (f == LYRA_FLAG_OK) ++n_ok;
        else degraded = 1; /* missing / invalid samples are noted */
    }
    coverage = n_ok / ((end_s - start_s) * rate_hz);
    if (coverage > 1.0) coverage = 1.0;
    if (coverage_out != NULL) *coverage_out = coverage;
    if (coverage < cfg->min_coverage_usable) return LYRA_Q_MISSING;
    for (c = 0; c < n_ch; ++c) {  /* flatline: peak-to-peak over OK samples */
        double lo = INFINITY, hi = -INFINITY;
        for (i = 0; i < n; ++i) {
            if (flag_of(values, flags, i, n_ch) != LYRA_FLAG_OK) continue;
            if (values[i * n_ch + c] < lo) lo = values[i * n_ch + c];
            if (values[i * n_ch + c] > hi) hi = values[i * n_ch + c];
        }
        if (hi - lo <= cfg->flatline_ptp) ++n_flat;
    }
    if (n_flat == n_ch) return LYRA_Q_INVALID;
    if (n_flat > 0) degraded = 1;
    if (sat_lo != NULL && sat_hi != NULL) {
        for (c = 0; c < n_ch; ++c) {
            int sat = 0;
            for (i = 0; i < n; ++i) {
                double v;
                if (flag_of(values, flags, i, n_ch) != LYRA_FLAG_OK) continue;
                v = values[i * n_ch + c];
                sat += v <= sat_lo[c] || v >= sat_hi[c];
            }
            if ((double)sat / n_ok > cfg->max_saturated_fraction) return LYRA_Q_INVALID;
            if (sat > 0) degraded = 1;
        }
    }
    if (coverage < cfg->min_coverage_valid || gaps > 0 ||
        irregular > cfg->max_irregular_fraction * (n - 1 > 1 ? n - 1 : 1))
        degraded = 1;
    return degraded ? LYRA_Q_DEGRADED : LYRA_Q_VALID;
}

lyra_quality lyra_combine(const lyra_quality *required, int n_required, int clocks_ok) {
    static const lyra_quality worst_first[] = {LYRA_Q_UNSYNCHRONIZED, LYRA_Q_INVALID, LYRA_Q_MISSING,
                                               LYRA_Q_DEGRADED, LYRA_Q_VALID};
    int w, i;
    if (n_required > 1 && !clocks_ok) return LYRA_Q_UNSYNCHRONIZED;
    for (w = 0; w < 5; ++w)
        for (i = 0; i < n_required; ++i)
            if (required[i] == worst_first[w]) return worst_first[w];
    return LYRA_Q_VALID;
}

static int all_finite(const double *x, int n) {
    int i;
    for (i = 0; i < n; ++i)
        if (!isfinite(x[i])) return 0;
    return 1;
}

lyra_route lyra_route_window(lyra_quality emg, lyra_quality imu, lyra_quality overall, const double *features,
                             int fallback_unimodal, lyra_status *status_out) {
    *status_out = LYRA_SUPPRESSED;
    if (usable(overall) && all_finite(features, LYRA_N_FEATURES)) {
        *status_out = overall == LYRA_Q_DEGRADED ? LYRA_DEGRADED : LYRA_VALID;
        return LYRA_ROUTE_MULTIMODAL;
    }
    /* clock mismatch (UNSYNCHRONIZED overall not caused by the IMU itself) suppresses, as in Python */
    if (fallback_unimodal && !usable(imu) && (overall != LYRA_Q_UNSYNCHRONIZED || imu == LYRA_Q_UNSYNCHRONIZED) &&
        usable(emg) && all_finite(features, LYRA_N_EMG_FEATURES)) {
        *status_out = emg == LYRA_Q_DEGRADED ? LYRA_DEGRADED : LYRA_VALID;
        return LYRA_ROUTE_SIMENV_FALLBACK;
    }
    return LYRA_ROUTE_NONE;
}
