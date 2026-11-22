/* lyra_pipeline.h: one call per window, chaining quality gate -> features -> random forest ->
 * event confirmation (the C99 counterpart of QualityGatedInference + EventConfirmation).
 *
 * The firmware buffers each window (4 s, 2 s step) and calls lyra_step() once per window, in time
 * order. A returned event_started = 1 means the pod notifies the phone once:
 * "Possible convulsive seizure detected. This is not a diagnosis."
 * EXPERIMENTAL research code, not a medical device. Parity: mlm/tests/test_c99_step.py.
 *
 * Stricter than the Python reference on one point: a modality's features are computed only from a
 * clean window (every sample OK, timestamps regular and gap-free). A usable-but-gappy or jittered
 * window is suppressed on the pod, where Python may still score its usable samples.
 */
#ifndef LYRA_PIPELINE_H
#define LYRA_PIPELINE_H

#include "lyra_confirm.h"
#include "lyra_features.h"
#include "lyra_quality.h"

/* Same signature as the generated lyra_<track>_rf_predict (mlm/export/c99/). Returns 0 on success. */
typedef int (*lyra_scorer)(const float *features, int n_features, double proba_out[2]);

typedef struct {
    double start_s, end_s;                        /* window [start_s, end_s) */
    const double *env_t, *env;                    /* envelope samples in the window */
    const unsigned char *env_flags;               /* optional source flags (NULL = none) */
    int n_env;
    double env_rate_hz;                           /* 50 */
    const double *imu_t, *imu;                    /* row-major [ax ay az gx gy gz], g and deg/s */
    const unsigned char *imu_flags;
    int n_imu;
    double imu_rate_hz;                           /* 25 */
    int clocks_ok;                                /* firmware's running clock check */
} lyra_window;

typedef struct {
    lyra_quality_cfg quality;
    double sat_lo[6], sat_hi[6];                  /* IMU full scale: -+8 g, -+1000 deg/s */
    lyra_scorer multimodal;                       /* required: 22 features */
    lyra_scorer simenv_fallback;                  /* NULL = missing_modality_policy "suppress" */
    lyra_confirm confirm;
} lyra_pipeline;

typedef struct {
    lyra_quality emg, imu, overall;
    lyra_route route;
    lyra_status status;                           /* SUPPRESSED: no score, never a negative */
    double score;                                 /* P(event-like); NAN when suppressed */
    double features[LYRA_N_FEATURES];             /* NaN = unavailable */
    int event_started;                            /* 1 exactly once per possible event */
    lyra_state state;
} lyra_step_result;

/* confirm_cfg: the frozen operating point (e.g. entry 0.97 for multimodal + fallback; see
 * frozen_pipeline.json). Uses LYRA_QUALITY_DEFAULTS and the trained IMU full scale.
 * Returns 0, or -1 for a missing multimodal scorer or an invalid confirmation config. */
int lyra_pipeline_init(lyra_pipeline *p, const lyra_confirm_cfg *confirm_cfg, lyra_scorer multimodal,
                       lyra_scorer simenv_fallback);

/* Returns 0; -1 for invalid arguments, a scorer error, or a window out of time order (state unchanged). */
int lyra_step(lyra_pipeline *p, const lyra_window *w, lyra_step_result *out);

#endif
