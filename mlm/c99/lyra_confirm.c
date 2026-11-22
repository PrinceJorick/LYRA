/* lyra_confirm.c: see lyra_confirm.h. Same transitions as EventConfirmation.update(). */
#include "lyra_confirm.h"

#include <stddef.h>

int lyra_confirm_init(lyra_confirm *c, const lyra_confirm_cfg *cfg) {
    if (c == NULL || cfg == NULL || !(0.0 <= cfg->exit_threshold && cfg->exit_threshold <= cfg->entry_threshold &&
                                      cfg->entry_threshold <= 1.0) ||
        cfg->consecutive_required < 1 || !(cfg->max_gap_s > 0) || cfg->refractory_s < 0)
        return -1;
    c->cfg = *cfg;
    c->state = LYRA_IDLE;
    c->count = 0;
    c->has_ts = c->has_evidence = 0;
    c->last_ts = c->last_evidence_ts = 0.0;
    c->refractory_until = -1e300;
    return 0;
}

static void end_run(lyra_confirm *c, double ts) {
    c->count = 0;
    if (c->state == LYRA_POSSIBLE_EVENT) {
        c->refractory_until = ts + c->cfg.refractory_s;
        c->state = LYRA_REFRACTORY;
    } else if (c->state == LYRA_ACCUMULATING) {
        c->state = LYRA_IDLE;
    }
}

int lyra_confirm_update(lyra_confirm *c, double ts, double score, lyra_status status) {
    if (c->has_ts && ts <= c->last_ts) return -1;
    c->has_ts = 1;
    c->last_ts = ts;
    if (c->state == LYRA_REFRACTORY && ts >= c->refractory_until) c->state = LYRA_IDLE;
    if (c->has_evidence && ts - c->last_evidence_ts > c->cfg.max_gap_s) end_run(c, ts);
    if (status == LYRA_SUPPRESSED) return 0;  /* never negative evidence: hold */
    c->has_evidence = 1;
    c->last_evidence_ts = ts;
    if (c->state == LYRA_REFRACTORY) return 0;
    if (c->state == LYRA_POSSIBLE_EVENT) {
        if (score < c->cfg.exit_threshold) end_run(c, ts);
        return 0;
    }
    if (score >= c->cfg.entry_threshold) {
        if (++c->count >= c->cfg.consecutive_required) {
            c->state = LYRA_POSSIBLE_EVENT;
            return 1;
        }
        c->state = LYRA_ACCUMULATING;
        return 0;
    }
    c->count = 0;
    c->state = LYRA_IDLE;
    return 0;
}

void lyra_confirm_reset(lyra_confirm *c) {
    c->count = 0;
    c->has_evidence = 0;
    c->refractory_until = -1e300;
    c->state = LYRA_IDLE;
}
