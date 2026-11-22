"""The fast confirmation replica must equal lyra_mlm's EventConfirmation exactly.

    ml/seizeit2/.venv/bin/python -m pytest -q ml/seizeit2/test_evaluate.py
"""

import numpy as np
import pytest
from lyra_mlm.confirmation import ConfirmationConfig, EventConfirmation
from lyra_mlm.model import Prediction, PredictionStatus
from lyra_mlm.quality import ModalityQuality, QualityState, WindowQuality
from lyra_mlm.signals import Modality

from ml.seizeit2.evaluate import confirm_alarms

Q = WindowQuality({m: ModalityQuality(m, QualityState.VALID, (), 1.0) for m in (Modality.EMG, Modality.IMU)},
                  (Modality.EMG, Modality.IMU), QualityState.VALID)


def reference(t, evidence, score, cfg: ConfirmationConfig) -> list[float]:
    ec, out = EventConfirmation(cfg), []
    for ts, ev, s in zip(t, evidence, score):
        p = (Prediction(float(ts), PredictionStatus.VALID, float(s), int(s >= cfg.entry_threshold),
                        (Modality.EMG, Modality.IMU), Q, "m") if ev else
             Prediction(float(ts), PredictionStatus.SUPPRESSED, None, None, (), Q, None))
        if ec.update(p).event_started:
            out.append(float(ts))
    return out


@pytest.mark.parametrize("seed", range(300))
def test_replica_matches_event_confirmation(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(50, 600))
    steps = np.where(rng.random(n) < 0.03, rng.uniform(3, 20, n), 2.0)  # mostly 2 s, some gaps
    t = 4.0 + np.cumsum(steps)
    evidence = rng.random(n) > rng.uniform(0, 0.3)
    bursts = np.convolve(rng.random(n) < 0.05, np.ones(int(rng.integers(1, 30))), "same") > 0
    score = np.clip(np.where(bursts, rng.uniform(0.4, 1.0, n), rng.uniform(0, 0.7, n)), 0, 1)
    entry = float(rng.uniform(0.5, 0.95))
    exit_ = float(rng.uniform(0, entry))
    k = int(rng.integers(1, 5))
    gap, refr = float(rng.choice([4.0, 6.0, 10.0])), float(rng.choice([0.0, 20.0, 60.0]))
    cfg = ConfirmationConfig(entry_threshold=entry, exit_threshold=exit_, consecutive_positive_required=k,
                             max_timestamp_gap_s=gap, refractory_cooldown_s=refr, max_latency_s=None)
    assert confirm_alarms(t, evidence, score, entry, exit_, k, gap, refr) == reference(t, evidence, score, cfg)
