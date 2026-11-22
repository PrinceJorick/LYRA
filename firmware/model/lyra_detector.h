/* LYRA candidate-event state machine (C99). Reference: ml/lyra/events.py (Detector).
 *
 * Experimental research component. An opened candidate event is a *possible* convulsive event,
 * never a confirmed seizure or a diagnosis; the app's cancel countdown and caregiver flow apply.
 *
 * Per completed window (timestamp = window end, monotonic milliseconds; uint32 wrap-around safe):
 *  - prob NaN or outside [0, 1] (invalid window): hold, returns LYRA_EV_INVALID;
 *  - a gap > max_gap_ms since the previous valid window resets the positive run and closes an open event;
 *  - prob >= threshold: run += 1; when a closed detector's run reaches n_open a candidate event opens and
 *    a caregiver alert is emitted unless one was emitted less than refractory_ms earlier (then suppressed);
 *  - prob < threshold: run = 0; an open event closes after m_close consecutive such windows.
 * Call lyra_detector_reset() at every recording/session boundary.
 */
#ifndef LYRA_DETECTOR_H
#define LYRA_DETECTOR_H

#include <stdint.h>

#define LYRA_EV_NONE 0u
#define LYRA_EV_OPENED 1u     /* candidate event opened at this window */
#define LYRA_EV_ALERT 2u      /* caregiver notification emitted */
#define LYRA_EV_SUPPRESSED 4u /* opened inside the refractory interval: tracked, not notified */
#define LYRA_EV_CLOSED 8u
#define LYRA_EV_INVALID 16u

typedef struct {
    float threshold;
    uint8_t n_open;
    uint8_t m_close;
    uint32_t refractory_ms;
    uint32_t max_gap_ms;
} lyra_detector_cfg_t;

typedef struct {
    lyra_detector_cfg_t cfg;
    uint16_t run_pos, run_neg;
    uint8_t open, has_valid, has_alert;
    uint32_t last_valid_ms, last_alert_ms;
    uint32_t n_opened, n_alerts, n_suppressed; /* counters since the last reset */
} lyra_detector_t;

void lyra_detector_init(lyra_detector_t *d, const lyra_detector_cfg_t *cfg);
void lyra_detector_reset(lyra_detector_t *d);
unsigned lyra_detector_update(lyra_detector_t *d, float prob, uint32_t window_end_ms);

/* Single-instance API configured from lyra_detector_config.h. */
void lyra_init_event_state(void);
unsigned lyra_update_event_state(float prob, uint32_t window_end_ms);
/* Compatibility wrapper: assumes exactly one window every LYRA_HOP_MS with no gaps. */
unsigned lyra_update_event_state_fixed_hop(float prob);

#endif
