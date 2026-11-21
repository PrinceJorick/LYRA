"""EventConfirmation state handling, with deterministic synthetic scores only.
These tests verify software behaviour. They are not evidence of detection performance.
Unless a test says otherwise it uses the Decision 4 defaults: entry 0.65, exit 0.45,
3 consecutive positives, 6 s max gap, hold, allow_with_flag, 60 s refractory."""

import pytest
from synthetic import StubScorePredictor, model_config, pred, recording, window_quality

from lyra_mlm.confirmation import ConfirmationConfig, ConfirmationState, EventConfirmation
from lyra_mlm.inference import InferenceConfig, QualityGatedInference
from lyra_mlm.model import PredictionStatus
from lyra_mlm.quality import QualityConfig, QualityState
from lyra_mlm.signals import Modality
from lyra_mlm.windows import WindowConfig, segment

S = ConfirmationState
SUP, DEG = PredictionStatus.SUPPRESSED, PredictionStatus.DEGRADED


def run(ec, items):
    out = []
    for item in items:
        ts, score, *status = item
        out.append(ec.update(pred(ts, score, *status)))
    return out


def started(ec):
    return sum(r.event_started for r in ec.history)


def test_decision4_defaults():
    c = ConfirmationConfig()
    assert (c.entry_threshold, c.exit_threshold, c.consecutive_positive_required, c.max_timestamp_gap_s,
            c.refractory_cooldown_s, c.invalid_window_policy, c.degraded_window_policy) == (
        0.65, 0.45, 3, 6.0, 60.0, "hold", "allow_with_flag")


def test_three_consecutive_valid_positives_start_exactly_one_event():
    ec = EventConfirmation()
    recs = run(ec, [(2, .9), (4, .9), (6, .9), (8, .9)])
    assert [r.state for r in recs] == [S.ACCUMULATING, S.ACCUMULATING, S.POSSIBLE_EVENT, S.POSSIBLE_EVENT]
    assert started(ec) == 1 and recs[2].event_started  # 3rd window ends at 6 s: >= 8 s after first onset


def test_consecutive_valid_negatives_stay_idle():
    ec = EventConfirmation()
    assert {r.state for r in run(ec, [(2, .1), (4, .2), (6, .0)])} == {S.IDLE}


def test_score_between_old_and_new_entry_threshold_is_negative():
    ec = EventConfirmation()
    assert run(ec, [(2, .6)])[0].state is S.IDLE  # 0.6 < 0.65


def test_alternating_predictions_never_confirm():
    ec = EventConfirmation()
    run(ec, [(2, .9), (4, .1), (6, .9), (8, .9), (10, .1), (12, .9)])
    assert started(ec) == 0 and ec.state is S.ACCUMULATING


def test_isolated_invalid_window_hold_policy():
    ec = EventConfirmation(ConfirmationConfig(invalid_window_policy="hold"))
    recs = run(ec, [(2, .9), (4, None, SUP), (6, .9), (8, .9)])
    assert recs[1].state is S.ACCUMULATING and recs[1].positive_count == 1
    assert recs[3].state is S.POSSIBLE_EVENT


def test_isolated_invalid_window_reset_policy():
    ec = EventConfirmation(ConfirmationConfig(invalid_window_policy="reset"))
    recs = run(ec, [(2, .9), (4, None, SUP), (6, .9), (8, .9)])
    assert recs[1].state is S.IDLE and recs[3].state is S.ACCUMULATING and started(ec) == 0


def test_hold_spans_up_to_the_6s_gap_then_resets():
    two_missing = EventConfirmation()
    run(two_missing, [(2, .9), (4, None, SUP), (6, None, SUP), (8, .9)])  # 6 s since last evidence
    assert two_missing.positive_count == 2
    three_missing = EventConfirmation()
    run(three_missing, [(2, .9), (4, None, SUP), (6, None, SUP), (8, None, SUP), (10, .9)])  # 8 s
    assert three_missing.positive_count == 1
    assert any("exceeds max_timestamp_gap_s" in r.reason for r in three_missing.history)


def test_suppressed_is_not_a_negative_prediction():
    held = EventConfirmation()
    run(held, [(2, .9), (4, None, SUP)])
    negative = EventConfirmation()
    run(negative, [(2, .9), (4, .1)])
    assert (held.state, held.positive_count) == (S.ACCUMULATING, 1)
    assert (negative.state, negative.positive_count) == (S.IDLE, 0)


def test_invalid_windows_never_accumulate_evidence():
    ec = EventConfirmation(ConfirmationConfig(consecutive_positive_required=1))
    run(ec, [(t, None, SUP) for t in range(2, 20, 2)])
    assert ec.state is S.IDLE and ec.positive_count == 0 and started(ec) == 0


def test_long_gap_is_not_continuous_evidence():
    ec = EventConfirmation()
    recs = run(ec, [(2, .9), (4, .9), (12, .9)])
    assert recs[-1].state is S.ACCUMULATING and recs[-1].positive_count == 1 and started(ec) == 0


def test_missing_windows_end_an_active_event_without_renotifying():
    ec = EventConfirmation()
    run(ec, [(2, .9), (4, .9), (6, .9), (30, .9)])
    assert ec.state is S.REFRACTORY and started(ec) == 1


def test_out_of_order_and_duplicate_predictions_are_rejected():
    ec = EventConfirmation()
    run(ec, [(4, .9)])
    for ts in (2, 4):
        r = ec.update(pred(ts, .9))
        assert not r.accepted and r.state is S.ACCUMULATING and ec.positive_count == 1


def test_stale_predictions_are_rejected():
    ec = EventConfirmation(ConfirmationConfig(max_latency_s=10.0))
    r = ec.update(pred(2, .9), now_s=30.0)
    assert not r.accepted and "stale" in r.reason and ec.positive_count == 0
    assert ec.update(pred(4, .9), now_s=5.0).accepted


def test_degraded_window_policy():
    allow = EventConfirmation(ConfirmationConfig(degraded_window_policy="allow_with_flag"))
    run(allow, [(2, .9, DEG), (4, .9, DEG), (6, .9, DEG)])
    ignore = EventConfirmation(ConfirmationConfig(degraded_window_policy="ignore"))
    run(ignore, [(2, .9, DEG), (4, .9, DEG), (6, .9, DEG)])
    assert allow.state is S.POSSIBLE_EVENT and ignore.state is S.IDLE


def test_hysteresis_entry_and_exit_thresholds():
    ec = EventConfirmation()
    assert run(ec, [(2, .5)])[0].state is S.IDLE  # below entry 0.65
    recs = run(ec, [(4, .9), (6, .9), (8, .9), (10, .5), (12, .4)])
    assert recs[3].state is S.POSSIBLE_EVENT  # 0.5 is between exit 0.45 and entry 0.65: continues
    assert recs[4].state is S.REFRACTORY  # 0.4 < exit


def test_refractory_prevents_duplicate_notifications_then_recovers():
    ec = EventConfirmation(ConfirmationConfig(refractory_cooldown_s=10.0))
    run(ec, [(2, .9), (4, .9), (6, .9), (8, .1)])  # event ends at 8, refractory until 18
    run(ec, [(10, .9), (12, .9), (14, .9), (16, .9)])
    assert started(ec) == 1 and ec.state is S.REFRACTORY
    run(ec, [(18, .9), (20, .9), (22, .9)])
    assert started(ec) == 2


def test_reset_and_recovery():
    ec = EventConfirmation()
    run(ec, [(2, .9), (4, .9), (6, .9)])
    r = ec.reset(7.0, "user cancelled on the pod button")
    assert r.state is S.IDLE and ec.positive_count == 0
    run(ec, [(8, .9), (10, .9), (12, .9)])
    assert ec.state is S.POSSIBLE_EVENT and started(ec) == 2


def test_quality_changes_are_kept_in_history():
    ec = EventConfirmation()
    ec.update(pred(2, .9))
    ec.update(pred(4, None, SUP, window_quality(emg_state=QualityState.MISSING)))
    assert any(r.reason == "quality changed: emg VALID->MISSING" for r in ec.history)
    assert all(r.reason for r in ec.history)


def test_config_validation():
    with pytest.raises(ValueError):
        ConfirmationConfig(entry_threshold=.3, exit_threshold=.4)


def test_end_to_end_wiring_with_synthetic_stub():
    """Pipeline wiring only: a fixed stub score on synthetic signals, not a detection result."""
    inf = QualityGatedInference(InferenceConfig(), StubScorePredictor(model_config(), 0.9))
    ec = EventConfirmation()
    for w in segment(recording(), WindowConfig(), QualityConfig(), (Modality.EMG, Modality.IMU)):
        ec.update(inf.process(w))
    assert started(ec) == 1 and ec.state is S.POSSIBLE_EVENT
