/* lyra_confirm.h: C99 port of lyra_mlm.confirmation.EventConfirmation with the locked policies
 * (degraded windows are evidence "with a flag"; suppressed windows "hold" up to the gap limit).
 *
 * POSSIBLE_EVENT only means the configured evidence rule was met. It is never a diagnosis.
 * The phone shows "Possible convulsive seizure detected. This is not a diagnosis."
 * EXPERIMENTAL research code, not a medical device. Parity: mlm/tests/test_c99_port.py.
 */
#ifndef LYRA_CONFIRM_H
#define LYRA_CONFIRM_H

typedef enum { LYRA_IDLE = 0, LYRA_ACCUMULATING = 1, LYRA_POSSIBLE_EVENT = 2, LYRA_REFRACTORY = 3 } lyra_state;
typedef enum { LYRA_SUPPRESSED = 0, LYRA_VALID = 1, LYRA_DEGRADED = 2 } lyra_status;

typedef struct {
    double entry_threshold;     /* Decision 4: 0.65; the CV-selected operating point is in frozen_pipeline.json */
    double exit_threshold;      /* 0.45 */
    int consecutive_required;   /* 3 */
    double max_gap_s;           /* 6.0: longer without evidence ends a run (ACCUMULATING -> IDLE, event -> REFRACTORY) */
    double refractory_s;        /* 60.0 cooldown after a possible event ends */
} lyra_confirm_cfg;

#define LYRA_CONFIRM_DECISION4 {0.65, 0.45, 3, 6.0, 60.0}

typedef struct {
    lyra_confirm_cfg cfg;
    lyra_state state;
    int count;
    int has_ts, has_evidence;
    double last_ts, last_evidence_ts, refractory_until;
} lyra_confirm;

/* Returns 0, or -1 for an invalid configuration (needs 0 <= exit <= entry <= 1, count >= 1, gap > 0). */
int lyra_confirm_init(lyra_confirm *c, const lyra_confirm_cfg *cfg);

/* One window prediction: ts = window end time (s), score in [0, 1] (ignored when suppressed).
 * Returns 1 exactly once per possible event (on entry: notify), 0 otherwise, or -1 when the
 * prediction is rejected (out-of-order or duplicate timestamp; state unchanged). */
int lyra_confirm_update(lyra_confirm *c, double ts, double score, lyra_status status);

#endif
