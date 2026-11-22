"""lyra_step() (mlm/c99/lyra_pipeline) against a Python reference chain, on SYNTHETIC recordings:
segment() quality -> extract_features (with the pod's clean-window rule) -> training track /
fallback routing -> the same scorer -> training.confirm. With the frozen models and the generated
export present, a second test links the real forests and checks the scores against scikit-learn."""

import ctypes
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_c99_port import recording as bursty_recording
from test_c99_quality import QCFG, defective_recording

from lyra_mlm.confirmation import ConfirmationConfig
from lyra_mlm.features import extract_features
from lyra_mlm.prep import QUALITY_CODE, PrepConfig
from lyra_mlm.signals import Modality, check_timestamps
from lyra_mlm.sync import SyncConfig, check_sync
from lyra_mlm.training import SUPPRESSED, TRACKS, confirm, fallback_status, track_status
from lyra_mlm.windows import segment

ROOT = Path(__file__).resolve().parents[1]
C99, EXPORT = ROOT / "c99", ROOT / "export" / "c99"
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
D, U8 = ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint8)
SCORER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int, D)
SOURCES = [C99 / f for f in ("lyra_pipeline.c", "lyra_quality.c", "lyra_features.c", "lyra_confirm.c")]


class Cfg(ctypes.Structure):
    _fields_ = [("entry", ctypes.c_double), ("exit", ctypes.c_double), ("k", ctypes.c_int),
                ("gap", ctypes.c_double), ("refractory", ctypes.c_double)]


class Win(ctypes.Structure):
    _fields_ = [("start_s", ctypes.c_double), ("end_s", ctypes.c_double), ("env_t", D), ("env", D),
                ("env_flags", U8), ("n_env", ctypes.c_int), ("env_rate_hz", ctypes.c_double), ("imu_t", D),
                ("imu", D), ("imu_flags", U8), ("n_imu", ctypes.c_int), ("imu_rate_hz", ctypes.c_double),
                ("clocks_ok", ctypes.c_int)]


class Res(ctypes.Structure):
    _fields_ = [("emg", ctypes.c_int), ("imu", ctypes.c_int), ("overall", ctypes.c_int), ("route", ctypes.c_int),
                ("status", ctypes.c_int), ("score", ctypes.c_double), ("features", ctypes.c_double * 22),
                ("event_started", ctypes.c_int), ("state", ctypes.c_int)]


def build(tmp_path, extra=()):
    so = tmp_path / "liblyrastep.so"
    cc = subprocess.run(["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", f"-I{C99}",
                         f"-I{EXPORT}", "-o", str(so), *map(str, SOURCES), *map(str, extra), "-lm"],
                        capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    lib = ctypes.CDLL(str(so))
    lib.lyra_pipeline_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(Cfg), ctypes.c_void_p, ctypes.c_void_p]
    lib.lyra_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(Win), ctypes.POINTER(Res)]
    return lib


def synthetic_score(x: np.ndarray, centre: float) -> float:
    return float(1 / (1 + np.exp(-(float(x[0]) - centre) / 3)))  # rises with envelope RMS


@SCORER
def mm_scorer(x, n, out):
    p = synthetic_score(np.ctypeslib.as_array(x, (n,)), 25.0)
    out[0], out[1] = 1 - p, p
    return 0


@SCORER
def sim_scorer(x, n, out):
    p = synthetic_score(np.ctypeslib.as_array(x, (n,)), 27.0)
    out[0], out[1] = 1 - p, p
    return 0


def is_clean(s, rate) -> bool:
    return s.n > 0 and bool(s.usable.all()) and check_timestamps(s.timestamps, rate).is_clean


def run_c(lib, rec, ws, clocks_ok, cfg, mm, sim):
    state = ctypes.create_string_buffer(4096)  # opaque lyra_pipeline
    assert lib.lyra_pipeline_init(state, ctypes.byref(cfg), ctypes.cast(mm, ctypes.c_void_p),
                                  ctypes.cast(sim, ctypes.c_void_p) if sim is not None else None) == 0
    out, keep = [], []
    for w in ws:
        a = [np.ascontiguousarray(x, np.float64) for x in (w.emg.timestamps, w.emg.values[:, 0], w.imu.timestamps,
                                                           w.imu.values)]
        keep.append(a)
        win = Win(w.start_s, w.end_s, a[0].ctypes.data_as(D), a[1].ctypes.data_as(D), None, len(a[0]), 50.0,
                  a[2].ctypes.data_as(D), a[3].ctypes.data_as(D), None, len(a[2]), 25.0, int(clocks_ok))
        r = Res()
        assert lib.lyra_step(state, ctypes.byref(win), ctypes.byref(r)) == 0
        out.append((r.route, r.status, r.score, np.array(r.features), r.event_started, r.overall))
    return out


def reference(ws, fallback: bool, cfg: ConfirmationConfig):
    """Python chain with the pod's clean-window rule; scores from the same synthetic scorer."""
    feats = []
    for w in ws:
        f = extract_features(w, PrepConfig().primary).values.copy()
        e_ok, i_ok = is_clean(w.emg, 50.0), is_clean(w.imu, 25.0)
        if not e_ok:
            f[:9] = np.nan
        if not i_ok:
            f[9:18] = np.nan
        if not (e_ok and i_ok):
            f[18:] = np.nan
        feats.append(f)
    feats = np.array(feats)
    z = {"overall": np.array([QUALITY_CODE[w.quality.overall] for w in ws]),
         "emg_state": np.array([QUALITY_CODE[w.quality.per_modality[Modality.EMG].state] for w in ws]),
         "imu_state": np.array([QUALITY_CODE[w.quality.per_modality[Modality.IMU].state] for w in ws]),
         "primary_available": ~np.isnan(feats)}
    mm, fb = track_status(z, TRACKS["multimodal"]), fallback_status(z) if fallback else np.zeros(len(ws), np.int8)
    status = np.where(mm != SUPPRESSED, mm, fb)
    route = np.where(mm != SUPPRESSED, 1, np.where(fb != SUPPRESSED, 2, 0))
    x32 = feats.astype(np.float32)
    score = np.array([synthetic_score(x32[i], 25.0) if route[i] == 1 else synthetic_score(x32[i, :9], 27.0)
                      if route[i] == 2 else np.nan for i in range(len(ws))])
    alerts = confirm(np.array([w.end_s for w in ws]), score, status, cfg)
    return route, status, score, feats, alerts


@pytest.mark.parametrize("which", ["bursty", "defective"])
@pytest.mark.parametrize("fallback", [True, False])
def test_lyra_step_matches_python_chain(tmp_path, which, fallback):
    rec = bursty_recording(7, 240.0) if which == "bursty" else defective_recording()
    ws = list(segment(rec, PrepConfig().window, QCFG, (Modality.EMG, Modality.IMU)))
    sync = check_sync([rec.emg, rec.imu], SyncConfig())
    entry = ConfirmationConfig(entry_threshold=0.9)
    c = run_c(build(tmp_path), rec, ws, sync.synchronized, Cfg(0.9, 0.45, 3, 6.0, 60.0), mm_scorer,
              sim_scorer if fallback else None)
    route, status, score, feats, alerts = reference(ws, fallback, entry)
    assert [r[0] for r in c] == route.tolist()
    assert [r[1] for r in c] == status.tolist()
    np.testing.assert_array_equal(np.array([r[2] for r in c]), score)  # same float32 inputs, same scorer
    np.testing.assert_allclose(np.array([r[3] for r in c]), feats, rtol=1e-8, atol=1e-10)
    assert [w.end_s for w, r in zip(ws, c) if r[4]] == alerts.tolist()
    assert [r[5] for r in c] == [QUALITY_CODE[w.quality.overall] for w in ws]
    if which == "bursty":
        assert len(alerts) > 0
    else:
        assert set(route.tolist()) >= ({0, 1, 2} if fallback else {0, 1})


@pytest.mark.skipif(not (EXPORT / "export_manifest.json").exists() or not (ROOT / "models" / "frozen_pipeline.json").exists(),
                    reason="needs the frozen models and the generated export")
def test_lyra_step_with_the_exported_forests_matches_sklearn(tmp_path):
    import joblib
    pipe = json.loads((ROOT / "models" / "frozen_pipeline.json").read_text())
    entry = pipe["operating_points"]["by_configuration"]["multimodal_fallback_unimodal"]["cv_selected"]
    lib = build(tmp_path, [EXPORT / "lyra_multimodal_rf.c", EXPORT / "lyra_simenv_rf.c"])
    mm_fn = ctypes.cast(lib.lyra_multimodal_rf_predict, ctypes.c_void_p)
    sim_fn = ctypes.cast(lib.lyra_simenv_rf_predict, ctypes.c_void_p)
    rec = defective_recording()
    ws = list(segment(rec, PrepConfig().window, QCFG, (Modality.EMG, Modality.IMU)))
    state = ctypes.create_string_buffer(4096)
    assert lib.lyra_pipeline_init(state, ctypes.byref(Cfg(entry, 0.45, 3, 6.0, 60.0)), mm_fn, sim_fn) == 0
    models = {t: joblib.load(ROOT / "models" / pipe["models"][t]["file"]) for t in ("multimodal", "simenv")}
    scored = 0
    for w in ws:
        a = [np.ascontiguousarray(x, np.float64) for x in (w.emg.timestamps, w.emg.values[:, 0], w.imu.timestamps,
                                                           w.imu.values)]
        win = Win(w.start_s, w.end_s, a[0].ctypes.data_as(D), a[1].ctypes.data_as(D), None, len(a[0]), 50.0,
                  a[2].ctypes.data_as(D), a[3].ctypes.data_as(D), None, len(a[2]), 25.0, 1)
        r = Res()
        assert lib.lyra_step(state, ctypes.byref(win), ctypes.byref(r)) == 0
        if r.route:
            x = np.array(r.features, np.float32)[None, : 22 if r.route == 1 else 9]
            p = models["multimodal" if r.route == 1 else "simenv"].predict_proba(x)[0, 1]
            assert abs(r.score - p) < 1e-5
            scored += 1
    assert scored > 0
