/* See lyra_detector.h. */
#include "lyra_detector.h"

#include "lyra_detector_config.h"

void lyra_detector_reset(lyra_detector_t *d) {
    const lyra_detector_cfg_t cfg = d->cfg;
    *d = (lyra_detector_t){0};
    d->cfg = cfg;
}

void lyra_detector_init(lyra_detector_t *d, const lyra_detector_cfg_t *cfg) {
    d->cfg = *cfg;
    lyra_detector_reset(d);
}

unsigned lyra_detector_update(lyra_detector_t *d, float prob, uint32_t t) {
    if (!(prob >= 0.0f && prob <= 1.0f)) return LYRA_EV_INVALID; /* also false for NaN */
    unsigned flags = LYRA_EV_NONE;
    if (d->has_valid && (uint32_t)(t - d->last_valid_ms) > d->cfg.max_gap_ms) {
        d->run_pos = d->run_neg = 0;
        if (d->open) {
            d->open = 0;
            flags |= LYRA_EV_CLOSED;
        }
    }
    d->has_valid = 1;
    d->last_valid_ms = t;
    if (prob >= d->cfg.threshold) {
        if (d->run_pos < UINT16_MAX) d->run_pos++;
        d->run_neg = 0;
        if (!d->open && d->run_pos >= d->cfg.n_open) {
            d->open = 1;
            d->n_opened++;
            flags |= LYRA_EV_OPENED;
            if (!d->has_alert || (uint32_t)(t - d->last_alert_ms) >= d->cfg.refractory_ms) {
                d->has_alert = 1;
                d->last_alert_ms = t;
                d->n_alerts++;
                flags |= LYRA_EV_ALERT;
            } else {
                d->n_suppressed++;
                flags |= LYRA_EV_SUPPRESSED;
            }
        }
    } else {
        d->run_pos = 0;
        if (d->open && ++d->run_neg >= d->cfg.m_close) {
            d->open = 0;
            d->run_neg = 0;
            flags |= LYRA_EV_CLOSED;
        }
    }
    return flags;
}

static lyra_detector_t g_detector;
static uint32_t g_fixed_ms;

void lyra_init_event_state(void) {
    const lyra_detector_cfg_t cfg = {LYRA_THRESHOLD, LYRA_N_OPEN, LYRA_M_CLOSE, LYRA_REFRACTORY_MS, LYRA_MAX_GAP_MS};
    lyra_detector_init(&g_detector, &cfg);
    g_fixed_ms = 0;
}

unsigned lyra_update_event_state(float prob, uint32_t window_end_ms) {
    return lyra_detector_update(&g_detector, prob, window_end_ms);
}

unsigned lyra_update_event_state_fixed_hop(float prob) {
    g_fixed_ms += LYRA_HOP_MS;
    return lyra_detector_update(&g_detector, prob, g_fixed_ms);
}
