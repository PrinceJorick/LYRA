/* lyra_pod.h: hardware-independent pod runtime around lyra_step (mlm/c99/lyra_pipeline.h).
 *
 * The platform layer (ESP-IDF: ADC, BMI270, timers, BLE) pushes samples as they arrive, timestamped
 * on the pod clock in seconds from session start, and calls lyra_pod_tick() periodically. The
 * runtime keeps per-stream ring buffers and cuts 4 s windows every 2 s from t = 0, the grid of
 * lyra_mlm.windows.segment. It runs lyra_step per window and reports results and BLE event payloads
 * through a callback. Static allocation only, single-threaded (call from one task).
 *
 * Clock check: per window, each stream's median sample interval must match its nominal rate within
 * max_rate_error_ppm. A stream with fewer than 2 samples in the window is not measured: its own
 * quality is MISSING. This is a live, per-window form of lyra_mlm.sync.check_sync, which is
 * recording-level. EXPERIMENTAL research code, not a medical device.
 */
#ifndef LYRA_POD_H
#define LYRA_POD_H

#include <stdint.h>

#include "lyra_ble.h"
#include "lyra_pipeline.h"

#define LYRA_POD_ENV_CAP 512 /* >= (window + tick latency) x 50 Hz */
#define LYRA_POD_IMU_CAP 256 /* >= (window + tick latency) x 25 Hz */

typedef struct {
    double start_s, end_s;
    lyra_step_result step;
    int clocks_ok;
    int has_event_payload;   /* 1: send event_payload to the phone (event started or ended) */
    uint8_t event_payload[LYRA_BLE_EVENT_STATUS_LEN];
    uint16_t event_id;
} lyra_pod_window;

typedef void (*lyra_pod_window_cb)(void *ctx, const lyra_pod_window *w);

typedef struct {
    lyra_pipeline pipe;
    double window_s, step_s, margin_s, env_rate_hz, imu_rate_hz, max_rate_error_ppm;
    long next_k;  /* next window starts at next_k * step_s */
    double env_t[LYRA_POD_ENV_CAP], env_v[LYRA_POD_ENV_CAP];
    uint8_t env_f[LYRA_POD_ENV_CAP];
    int env_head, env_n, env_dropped;
    double imu_t[LYRA_POD_IMU_CAP], imu_v[LYRA_POD_IMU_CAP * 6];
    uint8_t imu_f[LYRA_POD_IMU_CAP];
    int imu_head, imu_n, imu_dropped;
    uint16_t event_id;
    uint8_t origin;
    lyra_state last_state;
    lyra_quality last_emg, last_imu;
} lyra_pod;

/* confirm_cfg: the frozen operating point; simenv_fallback NULL = policy "suppress".
 * Defaults: 4 s windows, 2 s step, 0.25 s tick margin, ENV 50 Hz, IMU 25 Hz, 5000 ppm.
 * origin: LYRA_ORIGIN_HARDWARE, or LYRA_ORIGIN_SIMULATED for bench tests with generated signals.
 * Returns 0 or -1. */
int lyra_pod_init(lyra_pod *p, const lyra_confirm_cfg *confirm_cfg, lyra_scorer multimodal,
                  lyra_scorer simenv_fallback, uint8_t origin);

/* flag: LYRA_FLAG_OK / MISSING / INVALID. Samples older than the next window are discarded.
 * Returns 0, or 1 when the ring was full and the oldest sample was dropped (counted). */
int lyra_pod_push_env(lyra_pod *p, double t_s, double mv, uint8_t flag);
int lyra_pod_push_imu(lyra_pod *p, double t_s, const double v[6], uint8_t flag);

/* Emits, in order, every window whose end + margin <= now_s. Returns the number emitted, or -1 on
 * an internal error (lyra_step / scorer failure). */
int lyra_pod_tick(lyra_pod *p, double now_s, lyra_pod_window_cb cb, void *ctx);

/* User cancellation (pod button or phone): resets the event confirmation (EventConfirmation.reset)
 * and fills an event payload (state IDLE, the cancellation code) for the phone. Returns 0 or -1. */
int lyra_pod_cancel(lyra_pod *p, double now_s, uint8_t cancellation, uint8_t out[LYRA_BLE_EVENT_STATUS_LEN]);

/* Device status payload with the latest window's quality codes. Returns 0 or -1. */
int lyra_pod_device_status(const lyra_pod *p, uint8_t battery_percent, uint16_t battery_mv, uint16_t heartbeat_seq,
                           uint32_t uptime_s, uint8_t out[LYRA_BLE_DEVICE_STATUS_LEN]);

#endif
