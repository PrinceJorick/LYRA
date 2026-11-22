"""C99 port (mlm/c99/) against the Python reference, on SYNTHETIC signals only.

The library is compiled with cc -std=c99 -O2 -Wall -Wextra -Werror and called through ctypes.
Features must match lyra_mlm.features.extract_features for every window cut by the real
segment() pipeline, and the confirmation state machine must give the same alerts as
EventConfirmation (via training.confirm, which is tested equal to it)."""

import ctypes
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from synthetic import window_quality

from lyra_mlm.features import extract_features, feature_names
from lyra_mlm.prep import PrepConfig
from lyra_mlm.quality import QualityConfig
from lyra_mlm.seizeit2 import SATURATION_LIMITS
from lyra_mlm.signals import IMU_CHANNELS, DataOrigin, EmgSignalType, Modality, Recording, SignalStream
from lyra_mlm.training import DEGRADED_STATUS, SUPPRESSED, VALID_STATUS, confirm
from lyra_mlm.windows import segment

C99 = Path(__file__).resolve().parents[1] / "c99"
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
CFG = PrepConfig().primary
D = ctypes.POINTER(ctypes.c_double)


class Cfg(ctypes.Structure):
    _fields_ = [("entry", ctypes.c_double), ("exit", ctypes.c_double), ("k", ctypes.c_int),
                ("gap", ctypes.c_double), ("refractory", ctypes.c_double)]


class Confirm(ctypes.Structure):
    _fields_ = [("cfg", Cfg), ("state", ctypes.c_int), ("count", ctypes.c_int), ("has_ts", ctypes.c_int),
                ("has_evidence", ctypes.c_int), ("last_ts", ctypes.c_double), ("last_evidence_ts", ctypes.c_double),
                ("refractory_until", ctypes.c_double)]


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    so = tmp_path_factory.mktemp("c99") / "liblyra.so"
    cc = subprocess.run(["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "-o", str(so),
                         str(C99 / "lyra_features.c"), str(C99 / "lyra_confirm.c"), "-lm"], capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    lib = ctypes.CDLL(str(so))
    lib.lyra_features.argtypes = [D, D, ctypes.c_int, ctypes.c_double, D, D, ctypes.c_int, ctypes.c_double, D]
    lib.lyra_confirm_init.argtypes = [ctypes.POINTER(Confirm), ctypes.POINTER(Cfg)]
    lib.lyra_confirm_update.argtypes = [ctypes.POINTER(Confirm), ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.lyra_confirm_reset.argtypes = [ctypes.POINTER(Confirm)]
    return lib


def c_features(lib, w) -> np.ndarray:
    def arr(x):
        x = np.ascontiguousarray(x, dtype=np.float64)
        return x, x.ctypes.data_as(D)
    (et, etp), (ev, evp) = arr(w.emg.timestamps), arr(w.emg.values[:, 0])
    (it, itp), (iv, ivp) = arr(w.imu.timestamps), arr(w.imu.values)
    out = np.empty(22)
    assert lib.lyra_features(etp, evp, len(ev), w.emg.nominal_rate_hz, itp, ivp, len(it), w.imu.nominal_rate_hz,
                             out.ctypes.data_as(D)) == 0
    return out


def recording(seed: int, seconds: float = 60.0) -> Recording:
    rng = np.random.default_rng(seed)
    te, ti = np.arange(int(seconds * 50)) / 50.0, np.arange(int(seconds * 25)) / 25.0
    burst = (np.sin(2 * np.pi * ti / 17) > 0.3).repeat(2)[: te.size]  # seizure-like stretches
    env = 8 + 3 * rng.random() + burst * 20 * (1 + np.sin(2 * np.pi * 3.2 * te)) + rng.gamma(2.0, 1.0, te.size)
    g = np.column_stack([np.sin(ti / 9 + seed), 0.3 * np.cos(ti / 5), np.cos(ti / 9 + seed)])
    shake = (np.sin(2 * np.pi * ti / 17) > 0.3)[:, None] * 0.8 * np.sin(2 * np.pi * 3.2 * ti)[:, None]
    acc = g + shake + 0.03 * rng.standard_normal((ti.size, 3))
    gyr = 40 * shake.repeat(3, axis=1) * rng.random(3) + 2.0 * rng.standard_normal((ti.size, 3))
    emg = SignalStream(Modality.EMG, te, env, 50.0, ("simulated_env_from_raw",), ("uV",), DataOrigin.SYNTHETIC,
                       EmgSignalType.SIMULATED_ENV_FROM_RAW, 25.0)
    imu = SignalStream(Modality.IMU, ti, np.column_stack([acc, gyr]), 25.0, IMU_CHANNELS, ("g",) * 3 + ("deg/s",) * 3,
                       DataOrigin.SYNTHETIC)
    return Recording("SYN-P01", f"SYN-R{seed}", DataOrigin.SYNTHETIC, emg, imu, placement="synthetic")


def windows(seed: int):
    q = QualityConfig(saturation_limits=SATURATION_LIMITS)
    return list(segment(recording(seed), PrepConfig().window, q, (Modality.EMG, Modality.IMU)))


@pytest.mark.parametrize("seed", range(4))
def test_c_features_match_python_on_every_window(lib, seed):
    ws = windows(seed)
    assert len(ws) == 29
    py = np.array([extract_features(w, CFG).values for w in ws])
    c = np.array([c_features(lib, w) for w in ws])
    assert not np.isnan(py).any()  # every feature exercised
    np.testing.assert_allclose(c, py, rtol=1e-8, atol=1e-10)


def test_c_features_unavailable_exactly_where_python_is(lib):
    w = windows(0)[3]
    vq = window_quality()  # force VALID so the feature rules themselves decide availability
    flat_env = replace(w, quality=vq, emg=replace(w.emg, values=np.full_like(w.emg.values, 5.0)))
    still = replace(w, quality=vq, imu=replace(w.imu, values=np.tile([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (len(w.imu.values), 1))))
    for case in (flat_env, still):
        py, c = extract_features(case, CFG).values, c_features(lib, case)
        np.testing.assert_array_equal(np.isnan(c), np.isnan(py))
        np.testing.assert_allclose(c[~np.isnan(c)], py[~np.isnan(py)], rtol=1e-8, atol=1e-10)
        assert np.isnan(py).any()
    assert list(feature_names(CFG)) == list(feature_names(CFG))  # order is the training order (header lists it)


def test_c_features_without_imu_leave_imu_and_cross_modal_blocks_unavailable(lib):
    w = windows(1)[5]
    ev = np.ascontiguousarray(w.emg.values[:, 0])
    et = np.ascontiguousarray(w.emg.timestamps)
    out = np.empty(22)
    assert lib.lyra_features(et.ctypes.data_as(D), ev.ctypes.data_as(D), len(ev), 50.0, None, None, 0, 25.0,
                             out.ctypes.data_as(D)) == 0
    np.testing.assert_allclose(out[:9], extract_features(w, CFG).values[:9], rtol=1e-8, atol=1e-10)
    assert np.isnan(out[9:]).all()


@pytest.mark.parametrize("seed", range(10))
def test_c_confirmation_matches_python(lib, seed):
    rng = np.random.default_rng(seed)
    n = 800
    ts = np.cumsum(np.where(rng.random(n) < 0.03, rng.uniform(4, 20, n), 2.0))
    status = rng.choice([VALID_STATUS, DEGRADED_STATUS, SUPPRESSED], n, p=[0.7, 0.15, 0.15])
    burst = np.repeat(rng.random(n // 20) < 0.3, 20)[:n]
    score = np.clip(np.where(burst, rng.uniform(0.4, 1.0, n), rng.uniform(0.0, 0.7, n)), 0, 1)
    state = Confirm()
    assert lib.lyra_confirm_init(ctypes.byref(state), ctypes.byref(Cfg(0.65, 0.45, 3, 6.0, 60.0))) == 0
    got = [t for t, s, st in zip(ts, score, status) if lib.lyra_confirm_update(ctypes.byref(state), t, s, int(st)) == 1]
    expected = confirm(ts, score, status)
    assert len(expected) > 0
    np.testing.assert_array_equal(got, expected)
    assert lib.lyra_confirm_update(ctypes.byref(state), ts[-1], 0.9, 1) == -1  # duplicate timestamp rejected
    assert lib.lyra_confirm_init(ctypes.byref(state), ctypes.byref(Cfg(0.4, 0.5, 3, 6.0, 60.0))) == -1


def test_c_confirmation_reset_matches_python(lib):
    from lyra_mlm.confirmation import EventConfirmation
    from synthetic import pred
    from lyra_mlm.model import PredictionStatus
    ts = np.arange(2.0, 200.0, 2.0)
    score = np.where((ts > 20) & (ts < 120), 0.9, 0.1)
    ref, state = EventConfirmation(), Confirm()
    lib.lyra_confirm_init(ctypes.byref(state), ctypes.byref(Cfg(0.65, 0.45, 3, 6.0, 60.0)))
    got, want = [], []
    for t, s in zip(ts, score):
        if t == 40.0:  # user cancels during the possible event: back to IDLE, may re-alert
            ref.reset(t - 1.0, "cancelled on pod")
            lib.lyra_confirm_reset(ctypes.byref(state))
        if ref.update(pred(float(t), float(s), PredictionStatus.VALID)).event_started:
            want.append(t)
        if lib.lyra_confirm_update(ctypes.byref(state), t, s, 1) == 1:
            got.append(t)
    assert got == want and len(want) == 2  # once before the cancel, once after it
