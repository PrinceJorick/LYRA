"""Hardware-independent firmware core (firmware/components/lyra_pod) on SYNTHETIC data: BLE codec vs
lyra_mlm.ble, the streaming runtime vs the Python chain, event/cancel/status payloads, and the
data-collection line format through scripts/pod_logger.py into a protocol session."""

import ctypes
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from test_c99_port import recording as bursty_recording
from test_c99_quality import QCFG, defective_recording
from test_c99_step import SCORER, Res, mm_scorer, reference, sim_scorer

from lyra_mlm.ble import DeviceConfigPayload, DeviceStatusPayload, EventStatusPayload
from lyra_mlm.confirmation import ConfirmationConfig
from lyra_mlm.pod import PodAdapter, check_session
from lyra_mlm.prep import PrepConfig
from lyra_mlm.quality import combine
from lyra_mlm.signals import Modality
from lyra_mlm.sync import SyncReport
from lyra_mlm.windows import segment

ROOT = Path(__file__).resolve().parents[1]
FW, C99 = ROOT.parent / "firmware" / "components" / "lyra_pod", ROOT / "c99"
pytestmark = pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
D, U8 = ctypes.POINTER(ctypes.c_double), ctypes.c_uint8


class Cfg(ctypes.Structure):
    _fields_ = [("entry", ctypes.c_double), ("exit", ctypes.c_double), ("k", ctypes.c_int),
                ("gap", ctypes.c_double), ("refractory", ctypes.c_double)]


class PodWindow(ctypes.Structure):
    _fields_ = [("start_s", ctypes.c_double), ("end_s", ctypes.c_double), ("step", Res), ("clocks_ok", ctypes.c_int),
                ("has_event_payload", ctypes.c_int), ("event_payload", U8 * 11), ("event_id", ctypes.c_uint16)]


CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(PodWindow))


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    so = tmp_path_factory.mktemp("fw") / "liblyrafw.so"
    srcs = [*FW.glob("*.c"), *(C99 / f for f in ("lyra_pipeline.c", "lyra_quality.c", "lyra_features.c", "lyra_confirm.c"))]
    cc = subprocess.run(["cc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC", f"-I{FW}", f"-I{C99}",
                         "-o", str(so), *map(str, srcs), "-lm"], capture_output=True, text=True)
    assert cc.returncode == 0, cc.stderr
    lib = ctypes.CDLL(str(so))
    lib.lyra_pod_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(Cfg), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint8]
    lib.lyra_pod_push_env.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_uint8]
    lib.lyra_pod_push_imu.argtypes = [ctypes.c_void_p, ctypes.c_double, D, ctypes.c_uint8]
    lib.lyra_pod_tick.argtypes = [ctypes.c_void_p, ctypes.c_double, CB, ctypes.c_void_p]
    lib.lyra_pod_cancel.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_uint8, ctypes.c_char_p]
    lib.lyra_pod_device_status.argtypes = [ctypes.c_void_p, ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint16,
                                           ctypes.c_uint32, ctypes.c_char_p]
    lib.lyra_ble_pack_device_status.argtypes = [ctypes.c_char_p] + [ctypes.c_uint8] * 2 + [ctypes.c_uint16] + \
        [ctypes.c_uint8] * 2 + [ctypes.c_uint16, ctypes.c_uint32]
    lib.lyra_ble_pack_event_status.argtypes = [ctypes.c_char_p, ctypes.c_uint8, ctypes.c_uint16, ctypes.c_uint8,
                                               ctypes.c_uint8, ctypes.c_uint32, ctypes.c_uint8]
    lib.lyra_ble_unpack_device_config.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
    for f in ("lyra_log_emg", "lyra_log_imu", "lyra_log_marker", "lyra_log_header"):
        getattr(lib, f).restype = ctypes.c_int
    lib.lyra_log_emg.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char, ctypes.c_double, ctypes.c_double, ctypes.c_uint8]
    lib.lyra_log_imu.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_double, D, ctypes.c_uint8]
    lib.lyra_log_marker.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_double, ctypes.c_char_p]
    lib.lyra_log_header.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_char_p] + \
        [ctypes.c_double] * 4 + [ctypes.c_int] * 2
    return lib


# --- BLE codec ------------------------------------------------------------------------------------

def test_ble_payloads_are_byte_exact_with_python(lib):
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = dict(origin=int(rng.integers(1, 3)), battery_percent=int(rng.integers(0, 101)), battery_mv=int(rng.integers(0, 65536)),
                 emg_quality=int(rng.integers(0, 5)), imu_quality=int(rng.integers(0, 5)),
                 heartbeat_seq=int(rng.integers(0, 65536)), uptime_s=int(rng.integers(0, 2**32)))
        buf = ctypes.create_string_buffer(13)
        assert lib.lyra_ble_pack_device_status(buf, *s.values()) == 0
        assert buf.raw == DeviceStatusPayload(**s).pack()
        e = dict(origin=int(rng.integers(1, 3)), event_id=int(rng.integers(0, 65536)), state=int(rng.integers(0, 4)),
                 cancellation=int(rng.integers(0, 3)), timestamp_s=int(rng.integers(0, 2**32)),
                 available_modalities=int(rng.integers(0, 4)))
        buf = ctypes.create_string_buffer(11)
        assert lib.lyra_ble_pack_event_status(buf, *e.values()) == 0
        assert buf.raw == EventStatusPayload(**e).pack()
    bad = ctypes.create_string_buffer(13)
    assert lib.lyra_ble_pack_device_status(bad, 3, 50, 3700, 0, 0, 1, 1) == -1  # unknown origin
    assert lib.lyra_ble_pack_device_status(bad, 1, 101, 3700, 0, 0, 1, 1) == -1  # battery > 100 %
    assert lib.lyra_ble_pack_event_status(bad, 1, 1, 4, 0, 1, 0) == -1  # unknown state


def test_ble_config_unpacking_matches_python(lib):
    class Conf(ctypes.Structure):
        _fields_ = [("origin", ctypes.c_uint8), ("emg_rate_hz", ctypes.c_uint16), ("imu_odr_hz", ctypes.c_uint8),
                    ("window_ms", ctypes.c_uint16), ("step_ms", ctypes.c_uint16), ("countdown_s", ctypes.c_uint8)]
    good = DeviceConfigPayload(1, 50, 25, 4000, 2000, 8).pack()
    c = Conf()
    assert lib.lyra_ble_unpack_device_config(good, 10, ctypes.byref(c)) == 0
    assert (c.origin, c.emg_rate_hz, c.imu_odr_hz, c.window_ms, c.step_ms, c.countdown_s) == (1, 50, 25, 4000, 2000, 8)
    for raw in (good[:9], b"\x02" + good[1:], good[:4] + b"\x1e" + good[5:], good[:9] + b"\x0b",
                good[:5] + (3000).to_bytes(2, "little") + (4000).to_bytes(2, "little") + good[9:]):
        assert lib.lyra_ble_unpack_device_config(raw, len(raw), ctypes.byref(c)) == -1
        with pytest.raises(ValueError):
            DeviceConfigPayload.unpack(raw)


# --- streaming runtime ----------------------------------------------------------------------------

def stream_through_pod(lib, rec, fallback=True, entry=0.9, cancel_after_first_event=False):
    pod = ctypes.create_string_buffer(300_000)  # opaque lyra_pod
    assert lib.lyra_pod_init(pod, ctypes.byref(Cfg(entry, 0.45, 3, 6.0, 60.0)), ctypes.cast(mm_scorer, ctypes.c_void_p),
                             ctypes.cast(sim_scorer, ctypes.c_void_p) if fallback else None, 1) == 0
    out, cancels = [], []

    @CB
    def collect(_, w):
        w = w.contents
        out.append((w.start_s, w.step.route, w.step.status, w.step.score, w.step.event_started, w.clocks_ok,
                    bytes(w.event_payload) if w.has_event_payload else None, w.step.overall, w.step.emg, w.step.imu))

    samples = sorted([(t, 0, i) for i, t in enumerate(rec.emg.timestamps)] +
                     [(t, 1, i) for i, t in enumerate(rec.imu.timestamps)], key=lambda x: (x[0], x[1]))
    imu = np.ascontiguousarray(rec.imu.values)
    for t, kind, i in samples:
        if kind == 0:
            lib.lyra_pod_push_env(pod, t, rec.emg.values[i, 0], 0)
        else:
            lib.lyra_pod_push_imu(pod, t, imu[i].ctypes.data_as(D), 0)
        assert lib.lyra_pod_tick(pod, t, collect, None) >= 0
        if cancel_after_first_event and not cancels and any(o[4] for o in out):
            buf = ctypes.create_string_buffer(11)
            assert lib.lyra_pod_cancel(pod, t, 1, buf) == 0
            cancels.append(buf.raw)
    t_stop = max(rec.emg.timestamps[-1] + 1 / 50, rec.imu.timestamps[-1] + 1 / 25)
    assert lib.lyra_pod_tick(pod, t_stop + 0.25, collect, None) >= 0
    status = ctypes.create_string_buffer(13)
    assert lib.lyra_pod_device_status(pod, 87, 3900, 5, 3600, status) == 0
    return out, cancels, status.raw


def median_rate_ok(t, rate):
    d = np.diff(t)
    return d[d > 0].size == 0 or abs((1 / np.median(d[d > 0])) / rate - 1) * 1e6 <= 5000


@pytest.mark.parametrize("which", ["bursty", "defective"])
def test_streaming_runtime_matches_python_chain(lib, which):
    rec = bursty_recording(7, 240.0) if which == "bursty" else defective_recording()
    out, _, status = stream_through_pod(lib, rec)
    ws = list(segment(rec, PrepConfig().window, QCFG, (Modality.EMG, Modality.IMU)))
    clocks = [median_rate_ok(w.emg.timestamps, 50) and median_rate_ok(w.imu.timestamps, 25) for w in ws]
    ws = [replace(w, quality=combine(w.quality.per_modality, (Modality.EMG, Modality.IMU), SyncReport(ok, ())))
          for w, ok in zip(ws, clocks)]  # the pod's per-window clock check
    route, st, score, _, alerts = reference(ws, True, ConfirmationConfig(entry_threshold=0.9))
    assert [o[0] for o in out] == [w.start_s for w in ws]
    assert [o[5] for o in out] == [int(c) for c in clocks]
    assert [o[1] for o in out] == route.tolist() and [o[2] for o in out] == st.tolist()
    np.testing.assert_array_equal([o[3] for o in out], score)
    assert [w.end_s for w, o in zip(ws, out) if o[4]] == alerts.tolist()
    if which == "bursty":
        assert len(alerts) > 0
        payloads = [EventStatusPayload.unpack(o[6]) for o in out if o[6] is not None]
        starts = [p for p in payloads if p.state == 2]
        assert [p.event_id for p in starts] == list(range(1, len(alerts) + 1))
        assert [p.timestamp_s for p in starts] == [int(a) for a in alerts]
        assert all(p.cancellation == 0 and p.origin == 1 for p in payloads)
        assert len(payloads) > len(starts)  # each event also reports its end
    s = DeviceStatusPayload.unpack(status)
    assert (s.battery_percent, s.battery_mv, s.heartbeat_seq, s.uptime_s) == (87, 3900, 5, 3600)
    assert (int(s.emg_quality), int(s.imu_quality)) == (out[-1][8], out[-1][9])  # latest window's states


def test_cancel_resets_confirmation_and_reports_to_the_phone(lib):
    rec = bursty_recording(7, 240.0)
    plain, _, _ = stream_through_pod(lib, rec)
    out, cancels, _ = stream_through_pod(lib, rec, cancel_after_first_event=True)
    p = EventStatusPayload.unpack(cancels[0])
    assert (int(p.state), int(p.cancellation), p.event_id) == (0, 1, 1)
    assert sum(o[4] for o in out) >= sum(o[4] for o in plain)  # back to IDLE: no cooldown after a cancel


# --- data-collection line format -> logger -> session ------------------------------------------------

def test_logline_capture_round_trips_through_the_logger(lib, tmp_path):
    rng = np.random.default_rng(3)
    te, tr, ti = np.arange(3000) / 50, np.arange(60000) / 1000, np.arange(1500) / 25
    env, raw = 40 + rng.gamma(2, 2, te.size), 1650 + 30 * rng.standard_normal(tr.size)
    imu = np.column_stack([0.02 * rng.standard_normal((ti.size, 3)) + [0, 0, 1], 3 * rng.standard_normal((ti.size, 3))])
    buf = ctypes.create_string_buffer(160)
    lines = []
    assert lib.lyra_log_header(buf, 160, b"pod-dev-0", b"0.1.0", 50.0, 1000.0, 25.0, 3300.0, 8, 1000) > 0
    lines.append(buf.value.decode())
    events = sorted([(t, "E", i) for i, t in enumerate(te)] + [(t, "R", i) for i, t in enumerate(tr)] +
                    [(t, "I", i) for i, t in enumerate(ti)])
    for t, k, i in events:
        if k == "I":
            row = np.ascontiguousarray(imu[i])
            assert lib.lyra_log_imu(buf, 160, t, row.ctypes.data_as(D), 0) > 0
        else:
            assert lib.lyra_log_emg(buf, 160, k.encode(), t, (env if k == "E" else raw)[i], 0) > 0
        lines.append(buf.value.decode())
    assert lib.lyra_log_marker(buf, 160, 12.5, b"button") > 0
    lines.append(buf.value.decode())
    assert lib.lyra_log_marker(buf, 160, 1.0, b"bad,label") == -1  # separators never reach the stream
    lines.insert(500, "E,not-a-number,1.0,0\n")  # corrupted line on the link
    cap = tmp_path / "capture.txt"
    cap.write_text("".join(lines))
    run = subprocess.run([sys.executable, str(ROOT / "scripts" / "pod_logger.py"), str(cap), "--participant", "pod-p001",
                          "--session", "s01", "--consent", "SYNTHETIC-TEST", "--side", "left", "--out", str(tmp_path / "pod")],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert "malformed lines dropped 1" in run.stdout and "0 integrity problems" in run.stdout
    ad = PodAdapter(tmp_path / "pod")
    rec, rawst = ad.load_recording("pod-p001_s01"), ad.load_raw("pod-p001_s01")
    np.testing.assert_allclose(rec.emg.values[:, 0], env, atol=6e-4)
    np.testing.assert_allclose(rawst.values[:, 0], raw, atol=6e-4)
    np.testing.assert_allclose(rec.imu.values[:, :3], imu[:, :3], atol=6e-6)
    np.testing.assert_allclose(rec.imu.values[:, 3:], imu[:, 3:], atol=6e-5)
    assert check_session(ad, "pod-p001_s01") == []
    assert (tmp_path / "pod" / "pod-p001" / "s01" / "markers.tsv").read_text() == "onset\tlabel\n12.500000\tbutton\n"
    again = subprocess.run(run.args, capture_output=True, text=True)
    assert again.returncode != 0 and "refusing to overwrite" in again.stderr
