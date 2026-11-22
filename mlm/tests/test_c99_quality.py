"""C99 quality gate and routing (mlm/c99/lyra_quality) against the Python reference, on SYNTHETIC
signals with injected defects: dropouts, outages, infinite values, flat channels, gaps, jitter,
a duplicate timestamp, and light and heavy saturation."""

import ctypes
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from lyra_mlm.features import extract_features
from lyra_mlm.prep import QUALITY_CODE, PrepConfig
from lyra_mlm.quality import QualityConfig, QualityState
from lyra_mlm.seizeit2 import SATURATION_LIMITS
from lyra_mlm.signals import IMU_CHANNELS, DataOrigin, EmgSignalType, Modality, Recording, SignalStream
from lyra_mlm.sync import SyncConfig, check_sync
from lyra_mlm.training import SUPPRESSED, TRACKS, fallback_status, track_status
from lyra_mlm.windows import segment

C99 = Path(__file__).resolve().parents[1] / "c99"
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
D = ctypes.POINTER(ctypes.c_double)
QCFG = QualityConfig(saturation_limits=SATURATION_LIMITS)
SAT_LO = np.array([SATURATION_LIMITS[c][0] for c in IMU_CHANNELS])
SAT_HI = np.array([SATURATION_LIMITS[c][1] for c in IMU_CHANNELS])


class QCfg(ctypes.Structure):
    _fields_ = [(k, ctypes.c_double) for k in ("valid", "usable", "flat", "sat", "irregular", "tol", "gap")]


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    so = tmp_path_factory.mktemp("c99q") / "liblyraq.so"
    cc = subprocess.run(["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", "-o", str(so),
                         *(str(C99 / f) for f in ("lyra_quality.c", "lyra_features.c", "lyra_confirm.c")), "-lm"],
                        capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    lib = ctypes.CDLL(str(so))
    lib.lyra_assess.argtypes = [D, D, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
                                ctypes.c_double, D, D, ctypes.POINTER(QCfg), D]
    lib.lyra_combine.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    lib.lyra_route_window.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, D, ctypes.c_int,
                                      ctypes.POINTER(ctypes.c_int)]
    return lib


def p(a):
    return np.ascontiguousarray(a, dtype=np.float64).ctypes.data_as(D) if a is not None else None


def c_assess(lib, s, start, end, sat=False, flags=None):
    cfg = QCfg(QCFG.min_coverage_valid, QCFG.min_coverage_usable, QCFG.flatline_ptp, QCFG.max_saturated_fraction,
               QCFG.max_irregular_fraction, QCFG.interval_tolerance, QCFG.gap_factor)
    t, v = np.ascontiguousarray(s.timestamps), np.ascontiguousarray(s.values)
    cov = ctypes.c_double()
    fl = None if flags is None else np.ascontiguousarray(flags, dtype=np.uint8).tobytes()
    q = lib.lyra_assess(t.ctypes.data_as(D), v.ctypes.data_as(D), fl, len(t), v.shape[1], s.nominal_rate_hz,
                        start, end, p(SAT_LO) if sat else None, p(SAT_HI) if sat else None, ctypes.byref(cfg),
                        ctypes.byref(cov))
    return q, cov.value


def defective_recording(seed: int = 0, seconds: float = 120.0) -> Recording:
    rng = np.random.default_rng(seed)
    te, ti = np.arange(int(seconds * 50)) / 50.0, np.arange(int(seconds * 25)) / 25.0
    env = 10 + 5 * np.sin(2 * np.pi * 0.3 * te) + rng.gamma(2.0, 1.0, te.size)
    span = lambda t, a, b: (t >= a) & (t < b)  # noqa: E731
    env[span(te, 10, 14) & (rng.random(te.size) < 0.3)] = np.nan  # dropouts -> DEGRADED
    env[span(te, 20, 26)] = np.nan  # outage -> MISSING
    env[[1550, 1551]] = np.inf  # invalid samples -> DEGRADED
    env[span(te, 40, 50)] = 7.0  # flat -> INVALID
    jitter = span(te, 70, 80)
    te = te + np.where(jitter, rng.uniform(-0.4, 0.4, te.size) / 50, 0.0)  # irregular -> DEGRADED
    te[4501] = te[4500]  # duplicate timestamp -> UNSYNCHRONIZED
    keep = ~span(te, 60, 64)  # gap -> DEGRADED / MISSING
    imu = np.column_stack([0.1 * rng.standard_normal((ti.size, 3)) + [0, 0, 1], 5 * rng.standard_normal((ti.size, 3))])
    imu[np.flatnonzero(span(ti, 15, 19))[:3], 0] = 8.0  # light saturation -> DEGRADED
    imu[np.flatnonzero(span(ti, 33, 37))[:20], 0] = -8.0  # heavy saturation -> INVALID
    imu[span(ti, 50, 58), 5] = 0.0  # one flat channel -> DEGRADED
    imu[span(ti, 100, 110)] = [0, 0, 1, 0, 0, 0]  # all flat -> INVALID
    imu[span(ti, 112, 116)] = np.nan  # outage -> MISSING
    emg = SignalStream(Modality.EMG, te[keep], env[keep], 50.0, ("simulated_env_from_raw",), ("uV",),
                       DataOrigin.SYNTHETIC, EmgSignalType.SIMULATED_ENV_FROM_RAW, 25.0)
    imu_s = SignalStream(Modality.IMU, ti, imu, 25.0, IMU_CHANNELS, ("g",) * 3 + ("deg/s",) * 3, DataOrigin.SYNTHETIC)
    return Recording("SYN-P01", "SYN-QR", DataOrigin.SYNTHETIC, emg, imu_s, placement="synthetic")


@pytest.fixture(scope="module")
def windows():
    rec = defective_recording()
    sync = check_sync([rec.emg, rec.imu], SyncConfig())
    return rec, sync, list(segment(rec, PrepConfig().window, QCFG, (Modality.EMG, Modality.IMU)))


def test_c_assess_matches_python_per_modality(lib, windows):
    _, _, ws = windows
    seen = Counter()
    for w in ws:
        for m, s, sat in ((Modality.EMG, w.emg, False), (Modality.IMU, w.imu, True)):
            py = w.quality.per_modality[m]
            q, cov = c_assess(lib, s, w.start_s, w.end_s, sat)
            assert q == QUALITY_CODE[py.state], (m, w.start_s, py.reasons)
            assert cov == pytest.approx(py.coverage, abs=1e-12)
            seen[py.state] += 1
    assert set(seen) == set(QualityState), seen  # every state exercised


def test_c_combine_matches_python_window_quality(lib, windows):
    _, sync, ws = windows
    for w in ws:
        req = (ctypes.c_int * 2)(QUALITY_CODE[w.quality.per_modality[Modality.EMG].state],
                                 QUALITY_CODE[w.quality.per_modality[Modality.IMU].state])
        assert lib.lyra_combine(req, 2, int(sync.synchronized)) == QUALITY_CODE[w.quality.overall]
    assert lib.lyra_combine((ctypes.c_int * 2)(0, 0), 2, 0) == QUALITY_CODE[QualityState.UNSYNCHRONIZED]
    assert lib.lyra_combine((ctypes.c_int * 1)(1), 1, 0) == QUALITY_CODE[QualityState.DEGRADED]


def test_c_source_flags_are_merged_like_signal_stream(lib, windows):
    rec, _, _ = windows
    s = rec.imu.slice_time(0.0, 4.0)
    flags = np.zeros(len(s.timestamps), np.uint8)
    flags[:60] = 2  # source says INVALID for 60 % of the window -> coverage below usable -> MISSING
    q, cov = c_assess(lib, s, 0.0, 4.0, True, flags)
    from lyra_mlm.quality import assess_modality
    from dataclasses import replace
    py = assess_modality(replace(rec.imu, flags=np.where(np.arange(len(rec.imu.timestamps)) < 60, 2, 0)),
                         Modality.IMU, 0.0, 4.0, QCFG)
    assert (q, cov) == (QUALITY_CODE[py.state], pytest.approx(py.coverage))


def test_c_routing_matches_training_track_and_fallback_status(lib, windows):
    _, _, ws = windows
    feats = np.array([extract_features(w, PrepConfig().primary).values for w in ws])
    z = {"overall": np.array([QUALITY_CODE[w.quality.overall] for w in ws], np.int8),
         "emg_state": np.array([QUALITY_CODE[w.quality.per_modality[Modality.EMG].state] for w in ws], np.int8),
         "imu_state": np.array([QUALITY_CODE[w.quality.per_modality[Modality.IMU].state] for w in ws], np.int8),
         "primary_available": ~np.isnan(feats)}
    mm, fb = track_status(z, TRACKS["multimodal"]), fallback_status(z)
    routes = Counter()
    for i in range(len(ws)):
        for fallback in (0, 1):
            st = ctypes.c_int()
            route = lib.lyra_route_window(int(z["emg_state"][i]), int(z["imu_state"][i]), int(z["overall"][i]),
                                          p(feats[i]), fallback, ctypes.byref(st))
            expected = mm[i] if mm[i] != SUPPRESSED else (fb[i] if fallback else SUPPRESSED)
            assert st.value == expected
            assert route == (1 if mm[i] != SUPPRESSED else 2 if fallback and fb[i] != SUPPRESSED else 0)
            routes[route] += 1
    assert set(routes) == {0, 1, 2}, routes


def test_c_routing_suppresses_clock_mismatch_but_not_imu_disorder(lib):
    feats = np.ones(22)
    st = ctypes.c_int()
    v, i_, u = (QUALITY_CODE[s] for s in (QualityState.VALID, QualityState.INVALID, QualityState.UNSYNCHRONIZED))
    assert lib.lyra_route_window(v, i_, u, p(feats), 1, ctypes.byref(st)) == 0 and st.value == SUPPRESSED
    assert lib.lyra_route_window(v, u, u, p(feats), 1, ctypes.byref(st)) == 2 and st.value == 1
