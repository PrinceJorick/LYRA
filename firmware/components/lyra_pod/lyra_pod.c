/* lyra_pod.c: see lyra_pod.h. */
#include "lyra_pod.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

/* window scratch (not reentrant; one pod per firmware) */
static double w_env_t[LYRA_POD_ENV_CAP], w_env[LYRA_POD_ENV_CAP];
static uint8_t w_env_f[LYRA_POD_ENV_CAP];
static double w_imu_t[LYRA_POD_IMU_CAP], w_imu[LYRA_POD_IMU_CAP * 6];
static uint8_t w_imu_f[LYRA_POD_IMU_CAP];
static double w_dt[LYRA_POD_ENV_CAP];

int lyra_pod_init(lyra_pod *p, const lyra_confirm_cfg *confirm_cfg, lyra_scorer multimodal,
                  lyra_scorer simenv_fallback, uint8_t origin) {
    if (p == NULL || (origin != LYRA_ORIGIN_HARDWARE && origin != LYRA_ORIGIN_SIMULATED) ||
        lyra_pipeline_init(&p->pipe, confirm_cfg, multimodal, simenv_fallback) != 0)
        return -1;
    p->window_s = 4.0;
    p->step_s = 2.0;
    p->margin_s = 0.25;
    p->env_rate_hz = 50.0;
    p->imu_rate_hz = 25.0;
    p->max_rate_error_ppm = 5000.0;
    p->next_k = 0;
    p->env_head = p->env_n = p->env_dropped = 0;
    p->imu_head = p->imu_n = p->imu_dropped = 0;
    p->event_id = 0;
    p->origin = origin;
    p->last_state = LYRA_IDLE;
    p->last_emg = p->last_imu = LYRA_Q_MISSING;
    return 0;
}

static double next_start(const lyra_pod *p) { return p->next_k * p->step_s; }

int lyra_pod_push_env(lyra_pod *p, double t_s, double mv, uint8_t flag) {
    int full = p->env_n == LYRA_POD_ENV_CAP, i;
    if (t_s < next_start(p)) return 0; /* too old for any future window */
    if (full) {
        p->env_head = (p->env_head + 1) % LYRA_POD_ENV_CAP;
        --p->env_n;
        ++p->env_dropped;
    }
    i = (p->env_head + p->env_n++) % LYRA_POD_ENV_CAP;
    p->env_t[i] = t_s;
    p->env_v[i] = mv;
    p->env_f[i] = flag;
    return full;
}

int lyra_pod_push_imu(lyra_pod *p, double t_s, const double v[6], uint8_t flag) {
    int full = p->imu_n == LYRA_POD_IMU_CAP, i, c;
    if (t_s < next_start(p)) return 0;
    if (full) {
        p->imu_head = (p->imu_head + 1) % LYRA_POD_IMU_CAP;
        --p->imu_n;
        ++p->imu_dropped;
    }
    i = (p->imu_head + p->imu_n++) % LYRA_POD_IMU_CAP;
    p->imu_t[i] = t_s;
    for (c = 0; c < 6; ++c) p->imu_v[i * 6 + c] = v[c];
    p->imu_f[i] = flag;
    return full;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* 1 if the median forward interval matches the nominal rate (or there is nothing to measure) */
static int rate_ok(const double *t, int n, double rate, double max_ppm) {
    int i, m = 0;
    double med;
    for (i = 1; i < n; ++i)
        if (t[i] > t[i - 1]) w_dt[m++] = t[i] - t[i - 1];
    if (m == 0) return 1;
    qsort(w_dt, (size_t)m, sizeof(double), cmp_double);
    med = m % 2 ? w_dt[m / 2] : 0.5 * (w_dt[m / 2 - 1] + w_dt[m / 2]);
    return fabs((1.0 / med) / rate - 1.0) * 1e6 <= max_ppm;
}

/* drop samples older than the start of the next window (they belong to no future window) */
static void prune(lyra_pod *p) {
    double s = next_start(p);
    while (p->env_n > 0 && p->env_t[p->env_head] < s) {
        p->env_head = (p->env_head + 1) % LYRA_POD_ENV_CAP;
        --p->env_n;
    }
    while (p->imu_n > 0 && p->imu_t[p->imu_head] < s) {
        p->imu_head = (p->imu_head + 1) % LYRA_POD_IMU_CAP;
        --p->imu_n;
    }
}

static int usable(lyra_quality q) { return q == LYRA_Q_VALID || q == LYRA_Q_DEGRADED; }

int lyra_pod_tick(lyra_pod *p, double now_s, lyra_pod_window_cb cb, void *ctx) {
    int emitted = 0;
    while (next_start(p) + p->window_s + p->margin_s <= now_s) {
        lyra_pod_window out;
        lyra_window w;
        int i, k, c, ne = 0, ni = 0;
        out.start_s = next_start(p);
        out.end_s = out.start_s + p->window_s;
        for (k = 0; k < p->env_n; ++k) { /* arrival order kept, so disorder stays detectable */
            i = (p->env_head + k) % LYRA_POD_ENV_CAP;
            if (p->env_t[i] >= out.start_s && p->env_t[i] < out.end_s) {
                w_env_t[ne] = p->env_t[i];
                w_env[ne] = p->env_v[i];
                w_env_f[ne++] = p->env_f[i];
            }
        }
        for (k = 0; k < p->imu_n; ++k) {
            i = (p->imu_head + k) % LYRA_POD_IMU_CAP;
            if (p->imu_t[i] >= out.start_s && p->imu_t[i] < out.end_s) {
                w_imu_t[ni] = p->imu_t[i];
                for (c = 0; c < 6; ++c) w_imu[ni * 6 + c] = p->imu_v[i * 6 + c];
                w_imu_f[ni++] = p->imu_f[i];
            }
        }
        out.clocks_ok = rate_ok(w_env_t, ne, p->env_rate_hz, p->max_rate_error_ppm) &&
                        rate_ok(w_imu_t, ni, p->imu_rate_hz, p->max_rate_error_ppm);
        w.start_s = out.start_s;
        w.end_s = out.end_s;
        w.env_t = w_env_t;
        w.env = w_env;
        w.env_flags = w_env_f;
        w.n_env = ne;
        w.env_rate_hz = p->env_rate_hz;
        w.imu_t = w_imu_t;
        w.imu = w_imu;
        w.imu_flags = w_imu_f;
        w.n_imu = ni;
        w.imu_rate_hz = p->imu_rate_hz;
        w.clocks_ok = out.clocks_ok;
        if (lyra_step(&p->pipe, &w, &out.step) != 0) return -1;
        p->last_emg = out.step.emg;
        p->last_imu = out.step.imu;
        out.has_event_payload = 0;
        if (out.step.event_started) ++p->event_id;
        if (out.step.event_started || (p->last_state == LYRA_POSSIBLE_EVENT && out.step.state != LYRA_POSSIBLE_EVENT)) {
            uint8_t avail = (uint8_t)(usable(out.step.emg) | (usable(out.step.imu) << 1));
            if (lyra_ble_pack_event_status(out.event_payload, p->origin, p->event_id, (uint8_t)out.step.state,
                                           LYRA_CANCEL_NONE, (uint32_t)out.end_s, avail) != 0)
                return -1;
            out.has_event_payload = 1;
        }
        out.event_id = p->event_id;
        p->last_state = out.step.state;
        ++p->next_k;
        prune(p);
        if (cb != NULL) cb(ctx, &out);
        ++emitted;
    }
    return emitted;
}

int lyra_pod_cancel(lyra_pod *p, double now_s, uint8_t cancellation, uint8_t out[LYRA_BLE_EVENT_STATUS_LEN]) {
    if (p == NULL || (cancellation != LYRA_CANCEL_ON_POD && cancellation != LYRA_CANCEL_ON_PHONE)) return -1;
    lyra_confirm_reset(&p->pipe.confirm);
    p->last_state = LYRA_IDLE;
    return lyra_ble_pack_event_status(out, p->origin, p->event_id, LYRA_IDLE, cancellation, (uint32_t)now_s,
                                      (uint8_t)(usable(p->last_emg) | (usable(p->last_imu) << 1)));
}

int lyra_pod_on_phone_command(lyra_pod *p, double now_s, const uint8_t *in, int len, uint8_t out[LYRA_BLE_EVENT_STATUS_LEN]) {
    lyra_ble_command cmd;
    if (p == NULL || lyra_ble_unpack_phone_command(in, len, &cmd) != 0) return -1;
    if (p->event_id == 0 || cmd.event_id != p->event_id) return 1;
    return lyra_pod_cancel(p, now_s, LYRA_CANCEL_ON_PHONE, out);
}

int lyra_pod_device_status(const lyra_pod *p, uint8_t battery_percent, uint16_t battery_mv, uint16_t heartbeat_seq,
                           uint32_t uptime_s, uint8_t out[LYRA_BLE_DEVICE_STATUS_LEN]) {
    if (p == NULL) return -1;
    return lyra_ble_pack_device_status(out, p->origin, battery_percent, battery_mv, (uint8_t)p->last_emg,
                                       (uint8_t)p->last_imu, heartbeat_seq, uptime_s);
}
