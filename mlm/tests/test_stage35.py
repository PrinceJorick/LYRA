"""Stage 3.5 IMU scale rule, on SYNTHETIC signals only (not physiological data)."""

from dataclasses import replace

import numpy as np
import pytest
from synthetic import imu, write_edf

from lyra_mlm.prep import QUALITY_CODE, PrepConfig, prepare_recording
from lyra_mlm.quality import QualityConfig, QualityState, assess_modality
from lyra_mlm.seizeit2 import (IMU_LABELS, IMU_UNITS, SATURATION_LIMITS, SeizeIT2Adapter, assess_imu_scale,
                               convert_imu)
from lyra_mlm.signals import DataOrigin, EmgSignalType, Modality, Recording, SignalStream

ORIENTATIONS = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, -1.0]])


def stored(d=6, seconds=600, gains=(1.0, 1.0, 1.0), rest=1.0, spread=0.0, seed=0):
    """ACC/GYR as SeizeIT2 v1.1.0 stores them: gravity / D in three orientations, 0.001 resolution."""
    rng = np.random.default_rng(seed)
    n = seconds * 25
    g = np.repeat(ORIENTATIONS, n // 3 + 1, axis=0)[:n] * rest / d
    g *= 1 + spread * rng.uniform(-1, 1, (n // 25 + 1, 1)).repeat(25, axis=0)[:n]
    acc = np.round(g * np.array(gains) + 0.002 * rng.standard_normal((n, 3)), 3)
    gyr = np.round(0.004 * rng.standard_normal((n, 3)), 3)
    return acc, gyr


def decide(acc, gyr, units=("g",) * 3, z_bounds=(-1.0, 1.0)):
    return assess_imu_scale(acc, gyr, units, z_bounds, 25.0)


@pytest.mark.parametrize("d", [6, 2])
def test_divisor_is_found_and_conversion_rests_at_one_g(d):
    acc, gyr = stored(d)
    s = decide(acc, gyr)
    assert (s.d, s.category) == (d, "ACCEPTED") and 0.98 < s.evidence["rest_g_converted"] < 1.02
    out = convert_imu(acc, gyr, d)
    np.testing.assert_allclose(out[:, :3], acc * d)
    np.testing.assert_allclose(out[:, 3:], gyr * 125 * d)


@pytest.mark.parametrize("d", [6, 2])
def test_ceiling_samples_become_exactly_full_scale(d):
    acc, gyr = stored(d)
    acc[10, 0], gyr[11, 2] = -round(8 / d, 3), round(8 / d, 3)
    out = convert_imu(acc, gyr, d)
    assert out[10, 0] == -8.0 and out[11, 5] == 1000.0
    assert np.abs(np.delete(out[:, :3], 10, axis=0)).max() < 8.0


def test_rejections_name_the_failed_check():
    acc, gyr = stored(6)
    assert decide(acc, gyr, units=("10E-3 g",) * 3).category == "UNIT"
    assert decide(acc, gyr, z_bounds=(-32.768, 32.767)).category == "Z_CORRUPT"
    rails = acc.copy()
    rails[::20, 2] = -32.768
    assert decide(rails, gyr).category == "Z_CORRUPT"
    stuck = acc.copy()
    stuck[: int(0.6 * len(stuck)), 1] = 1.333
    assert decide(stuck, gyr).category == "STUCK_AXIS"
    assert decide(np.round(acc * 0.42, 3), gyr).category == "D_TEST"  # rests at ~0.07, fits no divisor
    over = acc.copy()
    over[5, 0] = 1.5
    assert decide(over, gyr).category == "OVER_CEILING"
    assert decide(*stored(6, gains=(1.0, 1.0, 0.4))).category == "AXIS_DISAGREEMENT"
    assert decide(*stored(6, spread=0.09)).category == "REST_SPREAD"
    assert decide(*stored(2, rest=0.91)).category == "POST_CONVERSION"  # D=2 window, but 0.91 g after x2
    assert decide(acc[:250], gyr[:250]).category == "TOO_LITTLE_DATA"


def test_untrusted_stream_rates_invalid_but_an_empty_window_stays_missing():
    s = SignalStream(Modality.IMU, np.arange(100) / 25, np.zeros((100, 6)), 25.0, imu().channels, IMU_UNITS,
                     DataOrigin.SYNTHETIC, untrusted="rejected (Z_CORRUPT)")
    q = assess_modality(s, Modality.IMU, 0.0, 4.0, QualityConfig())
    assert q.state is QualityState.INVALID and q.reasons == ("rejected (Z_CORRUPT)",)
    assert assess_modality(s, Modality.IMU, 10.0, 14.0, QualityConfig()).state is QualityState.MISSING


def test_saturation_at_full_scale_degrades_then_invalidates_the_window():
    acc, gyr = stored(6, seconds=12)
    vals = convert_imu(acc, gyr, 6)
    cfg = QualityConfig(saturation_limits=SATURATION_LIMITS)
    stream = lambda v: SignalStream(Modality.IMU, np.arange(len(v)) / 25, v, 25.0, imu().channels, IMU_UNITS,  # noqa: E731
                                    DataOrigin.SYNTHETIC)
    few, many = vals.copy(), vals.copy()
    few[3, 0] = 8.0
    many[:10, 3] = -1000.0  # 10 of 100 samples
    assert assess_modality(stream(vals), Modality.IMU, 0, 4, cfg).state is QualityState.VALID
    assert assess_modality(stream(few), Modality.IMU, 0, 4, cfg).state is QualityState.DEGRADED
    assert assess_modality(stream(many), Modality.IMU, 0, 4, cfg).state is QualityState.INVALID


def test_rejected_imu_blanks_imu_and_cross_modal_features_and_leaves_emg_identical():
    t = np.arange(256 * 40) / 256.0
    v = 100 * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 80 * t)
    raw = SignalStream(Modality.EMG, t, v, 256.0, ("raw",), ("uV",), DataOrigin.SYNTHETIC, EmgSignalType.RAW)
    good = imu(40.0)
    bad = replace(good, untrusted="IMU rejected for the recording (D_TEST)")
    out = {k: prepare_recording(Recording("SYN-P01", "SYN-R01", DataOrigin.SYNTHETIC, raw, s, placement="synthetic"),
                                PrepConfig()) for k, s in (("good", good), ("bad", bad))}
    assert (out["bad"]["imu_state"] == QUALITY_CODE[QualityState.INVALID]).all()
    assert (out["bad"]["overall"] == QUALITY_CODE[QualityState.INVALID]).all()
    assert not out["bad"]["primary_available"][:, 9:].any()  # 9 IMU + 4 cross-modal features
    for k in ("emg_state", "raw_emg_state", "secondary", "label"):
        assert out["bad"][k].tobytes() == out["good"][k].tobytes()
    assert out["bad"]["primary"][:, :9].tobytes() == out["good"]["primary"][:, :9].tobytes()


def bids_mov(root, acc, gyr, z_bounds_value=None):
    stem = "sub-001_ses-01_task-szMonitoring_run-01"
    base = root / "sub-001" / "ses-01"
    for d in ("mov", "eeg"):
        (base / d).mkdir(parents=True, exist_ok=True)
    cols = [*acc.T, *gyr.T]
    if z_bounds_value is not None:
        cols[2] = cols[2].copy()
        cols[2][:2] = (-z_bounds_value, z_bounds_value)
    write_edf(base / "mov" / f"{stem}_mov.edf", [(lb, "g", 25.0, c) for lb, c in zip(IMU_LABELS, cols)])
    (base / "eeg" / f"{stem}_events.tsv").write_text(f"onset\tduration\teventType\trecordingDuration\n0\t{len(acc) / 25}\tbckg\t{len(acc) / 25}\n")
    return stem


def test_adapter_converts_accepted_imu_and_marks_rejected_imu_untrusted(tmp_path):
    acc, gyr = stored(6)
    stem = bids_mov(tmp_path / "ok", acc, gyr)
    rec = SeizeIT2Adapter(tmp_path / "ok").load_recording(stem)
    assert rec.imu.units == IMU_UNITS and not rec.imu.untrusted
    np.testing.assert_allclose(rec.imu.values[:, :3], acc * 6, atol=1e-9)
    assert "x D, D=6" in rec.imu.provenance[-1] and "medium confidence" in rec.imu.provenance[-1]
    stem = bids_mov(tmp_path / "bad", acc, gyr, z_bounds_value=32.768)
    rec = SeizeIT2Adapter(tmp_path / "bad").load_recording(stem)
    assert "Z_CORRUPT" in rec.imu.untrusted and rec.imu.units != IMU_UNITS
    assert SeizeIT2Adapter(tmp_path / "bad").imu_scale(stem).category == "Z_CORRUPT"
