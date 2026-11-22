"""Stage 4 helpers, on SYNTHETIC data only."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from synthetic import pred

from lyra_mlm.annotations import Annotation, AnnotationKind
from lyra_mlm.confirmation import EventConfirmation
from lyra_mlm.evaluation import LeakageError
from lyra_mlm.model import PredictionStatus
from lyra_mlm.training import (BINS, DEGRADED_STATUS, SUPPRESSED, VALID_STATUS, binned_average_precision, confirm,
                               make_model, score_recording, summarise, training_rows)

STATUS = {VALID_STATUS: PredictionStatus.VALID, DEGRADED_STATUS: PredictionStatus.DEGRADED,
          SUPPRESSED: PredictionStatus.SUPPRESSED}


@pytest.mark.parametrize("seed", range(20))
def test_offline_confirmation_matches_event_confirmation(seed):
    rng = np.random.default_rng(seed)
    n = 600
    ts = np.cumsum(np.where(rng.random(n) < 0.03, rng.uniform(4, 20, n), 2.0))
    status = rng.choice([VALID_STATUS, DEGRADED_STATUS, SUPPRESSED], n, p=[0.7, 0.15, 0.15])
    burst = np.repeat(rng.random(n // 20) < 0.3, 20)[:n]
    score = np.clip(np.where(burst, rng.uniform(0.4, 1.0, n), rng.uniform(0.0, 0.7, n)), 0, 1)
    ref = EventConfirmation()
    for t, s, st in zip(ts, score, status):
        ref.update(pred(float(t), None if st == SUPPRESSED else float(s), STATUS[int(st)]))
    expected = [r.timestamp_s for r in ref.history if r.event_started]
    np.testing.assert_array_equal(confirm(ts, score, status), expected)


def test_event_matching_splits_alerts_into_fbtcs_non_fbtcs_and_interictal():
    ts = np.arange(4.0, 2000.0, 2.0)
    score = np.zeros(ts.size)
    for start in (100.0, 700.0, 1500.0):  # FBTCS, focal seizure, nothing annotated
        score[(ts >= start) & (ts < start + 20)] = 0.9
    anns = (Annotation("a", 110.0, 60.0, AnnotationKind.SEIZURE_FBTCS, "sz_foc_f2b"),
            Annotation("b", 720.0, 30.0, AnnotationKind.SEIZURE_OUT_OF_SCOPE, "sz_foc_ia"),
            Annotation("c", 1200.0, 30.0, AnnotationKind.SEIZURE_FBTCS, "sz_foc_f2b"))
    status = np.full(ts.size, VALID_STATUS)
    status[(ts > 1800) & (ts < 1900)] = SUPPRESSED
    t = score_recording(ts, np.zeros(ts.size, np.int8), status, score, anns)
    s = summarise(t)
    assert (t.alerts, t.fbtcs, t.detected, t.fbtcs_alerts, t.other_alerts, t.interictal_fa) == (3, 2, 1, 1, 1, 1)
    assert t.latencies == [104.0 - 110.0]  # third window >= 0.65 ends at 104 s, 6 s before onset
    assert s["event_sensitivity"] == 0.5 and s["non_fbtcs_seizure_alert_rate"] == 1.0 and s["precision"] == 1 / 3
    assert s["suppressed_h"] == round(sum(status == SUPPRESSED) * 2 / 3600, 2)


def test_binned_average_precision_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.random(5000) < 0.05
    s = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
    s = (np.floor(s * BINS) + 0.5) / BINS  # one value per bin, so binning loses nothing
    b = np.minimum((s * BINS).astype(int), BINS - 1)
    ap = binned_average_precision(np.bincount(b[y], minlength=BINS), np.bincount(b[~y], minlength=BINS))
    assert ap == pytest.approx(average_precision_score(y, s), abs=1e-12)


def test_training_rows_keep_all_positives_and_stratify_negatives_per_participant():
    part = np.array(["sub-001"] * 1000 + ["sub-002"] * 3000)
    label = np.zeros(4000, np.int8)
    label[[5, 6, 2500]] = 1
    key = np.random.default_rng(0).random(4000)
    rows = training_rows(part, label, key, {"sub-001": 998, "sub-002": 2999}, ratio=50)
    assert set(np.flatnonzero(label == 1)) <= set(rows)
    neg = rows[label[rows] == 0]
    assert (np.sum(part[neg] == "sub-001"), np.sum(part[neg] == "sub-002")) == (37, 113)  # 150 negatives, 1:3
    np.testing.assert_array_equal(rows, training_rows(part, label, key, {"sub-001": 998, "sub-002": 2999}, 50))


def test_training_rows_refuse_held_out_participants():
    with pytest.raises(LeakageError, match="sub-097"):
        training_rows(np.array(["sub-001", "sub-097"]), np.array([1, 0]), np.zeros(2), {"sub-001": 1, "sub-097": 1}, 50)


def test_logistic_regression_scaler_is_fitted_on_the_training_fold_only():
    rng = np.random.default_rng(0)
    x_train, x_val = rng.normal(5.0, 2.0, (400, 3)), rng.normal(-50.0, 9.0, (400, 3))
    y = (x_train[:, 0] > 5).astype(int)
    m = make_model("lr", {}, int(y.sum()), int((1 - y).sum())).fit(x_train, y)
    m.predict_proba(x_val)
    np.testing.assert_allclose(m[0].mean_, x_train.mean(axis=0))
