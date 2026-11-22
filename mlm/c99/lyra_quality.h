/* lyra_quality.h: C99 port of the quality gate (lyra_mlm.quality.assess_modality / combine,
 * signals.check_timestamps) and of the inference routing (training.track_status /
 * fallback_status, mirroring QualityGatedInference).
 *
 * A window that fails the gate is SUPPRESSED: no score. It is never treated as a negative window.
 * EXPERIMENTAL research code, not a medical device. Parity: mlm/tests/test_c99_quality.py.
 */
#ifndef LYRA_QUALITY_H
#define LYRA_QUALITY_H

#include "lyra_confirm.h" /* lyra_status */

/* Same codes as the stored window files (prep.QUALITY_CODE). */
typedef enum {
    LYRA_Q_VALID = 0,
    LYRA_Q_DEGRADED = 1,
    LYRA_Q_INVALID = 2,
    LYRA_Q_MISSING = 3,
    LYRA_Q_UNSYNCHRONIZED = 4
} lyra_quality;

/* Per-sample source flags (signals.SampleFlag). NaN values count as MISSING and +-inf as INVALID. */
enum { LYRA_FLAG_OK = 0, LYRA_FLAG_MISSING = 1, LYRA_FLAG_INVALID = 2 };

typedef struct {
    double min_coverage_valid;     /* 0.95: usable / expected samples needed for VALID */
    double min_coverage_usable;    /* 0.5: below this the modality is MISSING */
    double flatline_ptp;           /* 0.0: channel peak-to-peak at or below this is flat */
    double max_saturated_fraction; /* 0.05: above -> INVALID; any saturation -> DEGRADED */
    double max_irregular_fraction; /* 0.1 */
    double interval_tolerance;     /* 0.25 periods */
    double gap_factor;             /* 3.0 periods */
} lyra_quality_cfg;

#define LYRA_QUALITY_DEFAULTS {0.95, 0.5, 0.0, 0.05, 0.1, 0.25, 3.0}

/* Assess one modality over the window [start_s, end_s).
 * t, values: the n samples that fall in the window (values row-major, n_ch per sample). n may be 0.
 * flags: optional source flags per sample (NULL = none); merged with the flags derived from values.
 * sat_lo, sat_hi: per-channel full-scale limits, or NULL for no saturation check
 *   (pod: +-8 g ACC and +-1000 deg/s GYR in the trained configuration).
 * coverage_out (optional): OK samples / expected samples, capped at 1. */
lyra_quality lyra_assess(const double *t, const double *values, const unsigned char *flags, int n, int n_ch,
                         double rate_hz, double start_s, double end_s, const double *sat_lo, const double *sat_hi,
                         const lyra_quality_cfg *cfg, double *coverage_out);

/* Window quality over the required modalities: UNSYNCHRONIZED if more than one is required and the
 * clocks disagree (clocks_ok = 0), else the worst state (UNSYNCHRONIZED > INVALID > MISSING > DEGRADED > VALID). */
lyra_quality lyra_combine(const lyra_quality *required, int n_required, int clocks_ok);

typedef enum { LYRA_ROUTE_NONE = 0, LYRA_ROUTE_MULTIMODAL = 1, LYRA_ROUTE_SIMENV_FALLBACK = 2 } lyra_route;

/* Which model scores this window, and with which status (VALID / DEGRADED, or SUPPRESSED with
 * LYRA_ROUTE_NONE). features: the 22 lyra_features() outputs.
 * Multimodal when the window quality is usable and all 22 features are available. Otherwise, if
 * fallback_unimodal is set, the SIMULATED_ENV-only model is used when IMU is unusable, the envelope
 * is usable with its 9 features available, and there is no clock mismatch. */
lyra_route lyra_route_window(lyra_quality emg, lyra_quality imu, lyra_quality overall, const double *features,
                             int fallback_unimodal, lyra_status *status_out);

#endif
