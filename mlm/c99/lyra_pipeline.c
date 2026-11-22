/* lyra_pipeline.c: see lyra_pipeline.h. */
#include "lyra_pipeline.h"

#include <math.h>
#include <stddef.h>

static const lyra_quality_cfg QUALITY_DEFAULTS = LYRA_QUALITY_DEFAULTS;

int lyra_pipeline_init(lyra_pipeline *p, const lyra_confirm_cfg *confirm_cfg, lyra_scorer multimodal,
                       lyra_scorer simenv_fallback) {
    static const double lo[6] = {-8.0, -8.0, -8.0, -1000.0, -1000.0, -1000.0};
    int i;
    if (p == NULL || multimodal == NULL || lyra_confirm_init(&p->confirm, confirm_cfg) != 0) return -1;
    p->quality = QUALITY_DEFAULTS;
    for (i = 0; i < 6; ++i) {
        p->sat_lo[i] = lo[i];
        p->sat_hi[i] = -lo[i];
    }
    p->multimodal = multimodal;
    p->simenv_fallback = simenv_fallback;
    return 0;
}

/* every sample OK, timestamps strictly increasing with no gap and no irregular step */
static int clean(const double *t, const double *v, const unsigned char *flags, int n, int n_ch, double rate,
                 const lyra_quality_cfg *cfg) {
    double period = 1.0 / rate;
    int i, c;
    if (n <= 0) return 0;
    for (i = 0; i < n; ++i) {
        if (!isfinite(t[i]) || (flags != NULL && flags[i] != LYRA_FLAG_OK)) return 0;
        for (c = 0; c < n_ch; ++c)
            if (!isfinite(v[i * n_ch + c])) return 0;
        if (i > 0) {
            double d = t[i] - t[i - 1];
            if (d <= 0 || d > cfg->gap_factor * period || fabs(d - period) > cfg->interval_tolerance * period)
                return 0;
        }
    }
    return 1;
}

int lyra_step(lyra_pipeline *p, const lyra_window *w, lyra_step_result *out) {
    lyra_quality req[2];
    float x[LYRA_N_FEATURES];
    double proba[2];
    int i, env_ok, imu_ok, ev;
    if (p == NULL || w == NULL || out == NULL || (p->confirm.has_ts && w->end_s <= p->confirm.last_ts)) return -1;
    out->emg = lyra_assess(w->env_t, w->env, w->env_flags, w->n_env, 1, w->env_rate_hz, w->start_s, w->end_s, NULL,
                           NULL, &p->quality, NULL);
    out->imu = lyra_assess(w->imu_t, w->imu, w->imu_flags, w->n_imu, 6, w->imu_rate_hz, w->start_s, w->end_s,
                           p->sat_lo, p->sat_hi, &p->quality, NULL);
    req[0] = out->emg;
    req[1] = out->imu;
    out->overall = lyra_combine(req, 2, w->clocks_ok);
    /* a block is computed only for a usable modality (as extract_features) with a clean window (pod rule) */
    env_ok = (out->emg == LYRA_Q_VALID || out->emg == LYRA_Q_DEGRADED) &&
             clean(w->env_t, w->env, w->env_flags, w->n_env, 1, w->env_rate_hz, &p->quality);
    imu_ok = (out->imu == LYRA_Q_VALID || out->imu == LYRA_Q_DEGRADED) &&
             clean(w->imu_t, w->imu, w->imu_flags, w->n_imu, 6, w->imu_rate_hz, &p->quality);
    if (lyra_features(w->env_t, w->env, env_ok ? w->n_env : 0, w->env_rate_hz, w->imu_t, w->imu,
                      imu_ok ? w->n_imu : 0, w->imu_rate_hz, out->features) != 0)
        return -1;
    out->route = lyra_route_window(out->emg, out->imu, out->overall, out->features, p->simenv_fallback != NULL,
                                   &out->status);
    out->score = NAN;
    if (out->route != LYRA_ROUTE_NONE) {
        int n = out->route == LYRA_ROUTE_MULTIMODAL ? LYRA_N_FEATURES : LYRA_N_EMG_FEATURES;
        lyra_scorer f = out->route == LYRA_ROUTE_MULTIMODAL ? p->multimodal : p->simenv_fallback;
        for (i = 0; i < n; ++i) x[i] = (float)out->features[i]; /* training features were float32 */
        if (f(x, n, proba) != 0) return -1;
        out->score = proba[1];
    }
    ev = lyra_confirm_update(&p->confirm, w->end_s, out->route != LYRA_ROUTE_NONE ? out->score : 0.0, out->status);
    if (ev < 0) return -1;
    out->event_started = ev;
    out->state = p->confirm.state;
    return 0;
}
