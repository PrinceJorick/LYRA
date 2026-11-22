"""Synthetic golden vectors for Python/C parity (no dataset samples are written to the repo).

Windows cover: background, non-target-seizure-like movement, target-like rhythmic convulsive
activity, constant and low-variance signals, NaN/Inf inputs, a too-short window, and values that put
the probability near the threshold. The detector stream covers threshold boundaries, NaN and
out-of-range probabilities, gaps longer than MAX_GAP_S, refractory edges, recording resets and
uint32 millisecond wrap-around. Expected values come from the Python reference.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from . import config as C
from . import events as Ev
from . import features as F

MAGIC = b"LYRAGV01"
ERR_SIZE, ERR_NONFINITE, ERR_FLAT = -1, -2, -3


def _window(kind: str, n: int, rng: np.random.Generator):
    t = np.arange(n) / C.IMU_RATE
    te = np.arange(2 * n) / C.ENV_RATE
    g = np.array([0.3, -0.5, 9.78])
    acc = g + rng.normal(0, 0.02, (n, 3))
    gyr = rng.normal(0, 0.01, (n, 3))
    env = np.abs(rng.normal(12, 3, 2 * n))
    if kind == "target":
        f = 3.2
        env = 40 + 120 * np.maximum(0, np.sin(2 * np.pi * f * te)) + rng.normal(0, 5, 2 * n).clip(-30, 30)
        acc = g + np.outer(np.sin(2 * np.pi * f * t), [4.0, 2.5, 1.5]) + rng.normal(0, 0.3, (n, 3))
        gyr = np.outer(np.cos(2 * np.pi * f * t), [3.0, 1.0, 2.0]) + rng.normal(0, 0.1, (n, 3))
    elif kind == "nontarget":
        env = np.abs(rng.normal(30, 15, 2 * n))
        acc = g + np.cumsum(rng.normal(0, 0.1, (n, 3)), axis=0)
        gyr = rng.normal(0, 0.4, (n, 3))
    elif kind == "constant":
        env = np.full(2 * n, 5.0)
        acc = np.tile(g, (n, 1))
        gyr = np.zeros((n, 3))
    elif kind == "lowvar":
        env = 5.0 + rng.normal(0, 1e-6, 2 * n)
        acc = g + rng.normal(0, 1e-7, (n, 3))
        gyr = rng.normal(0, 1e-7, (n, 3))
    elif kind == "tilt":
        ang = np.linspace(0, np.pi / 2, n)[:, None]
        acc = 9.81 * np.hstack([np.sin(ang), np.zeros((n, 1)), np.cos(ang)]) + rng.normal(0, 0.05, (n, 3))
    return env.astype(np.float32), acc.astype(np.float32), gyr.astype(np.float32)


def windows(rng: np.random.Generator) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    out = []
    for name, (w, _) in C.WINDOWS.items():
        n = int(w * C.IMU_RATE)
        for kind in ("background", "nontarget", "target", "constant", "lowvar", "tilt"):
            for rep in range(3 if kind in ("background", "target", "nontarget") else 1):
                out.append((f"{name}_{kind}_{rep}", *_window(kind, n, rng)))
        e, a, g = _window("background", n, rng)
        e2 = e.copy()
        e2[7] = np.nan
        out.append((f"{name}_nan_env", e2, a, g))
        a2 = a.copy()
        a2[3, 1] = np.inf
        out.append((f"{name}_inf_acc", e, a2, g))
    e, a, g = _window("background", 30, rng)
    out.append(("too_short", e, a, g))
    return out


def py_valid(env, acc, gyr) -> int:
    n = len(acc)
    if n < int(min(w for w, _ in C.WINDOWS.values()) * C.IMU_RATE) or len(env) != 2 * n:
        return ERR_SIZE
    if not (np.isfinite(env).all() and np.isfinite(acc).all() and np.isfinite(gyr).all()):
        return ERR_NONFINITE
    if not env.max() > env.min():
        return ERR_FLAT
    moving = (np.diff(acc, axis=0) != 0).any() or (np.diff(gyr, axis=0) != 0).any()
    return 0 if moving else ERR_FLAT


def py_features(env, acc, gyr) -> tuple[int, np.ndarray]:
    n = len(acc)
    if n < int(min(w for w, _ in C.WINDOWS.values()) * C.IMU_RATE) or len(env) != 2 * n:
        return ERR_SIZE, np.full(len(C.FEATURES), np.nan, np.float32)
    if not (np.isfinite(env).all() and np.isfinite(acc).all() and np.isfinite(gyr).all()):
        return ERR_NONFINITE, np.full(len(C.FEATURES), np.nan, np.float32)
    x = F.compute(env[None].astype(np.float64), acc[None].astype(np.float64), gyr[None].astype(np.float64), 1)
    return 0, x[0].astype(np.float32)


def detector_stream(threshold: float, rng: np.random.Generator) -> list[tuple[int, float, int]]:
    """(op, prob, t_ms): op 0 update, 1 reset. Hop 1000 ms unless stated."""
    thr = float(np.float32(threshold))
    below, at = float(np.nextafter(np.float32(thr), np.float32(0))), thr
    s: list[tuple[int, float, int]] = []
    t = 1000
    seq = [0.1, at, at, at, 0.1, 0.1, 0.1, below, at, at, float("nan"), at, -0.1, 1.5, 0.9, 0.9, 0.9, 0.05, 0.05]
    for p in seq:
        s.append((0, p, t))
        t += 1000
    t += int(C.MAX_GAP_S * 1000) + 1000  # gap: resets the run and closes an open event
    for p in (0.9, 0.9, 0.9, 0.9):
        s.append((0, p, t))
        t += 1000
    t += int(C.MAX_GAP_S * 1000)  # exactly MAX_GAP: still consecutive
    for p in (0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9):
        s.append((0, p, t))
        t += 1000
    s.append((1, 0.0, 0))  # recording boundary
    t = 2**32 - 3500  # uint32 wrap-around inside an event
    for _ in range(6):
        s.append((0, 0.95, t % 2**32))
        t += 1000
    s.append((1, 0.0, 0))
    t = 0
    for p in rng.uniform(0, 1, 400):  # random stream with refractory edges
        s.append((0, float(np.float32(p)), t))
        t += int(rng.choice([1000, 2000, 2480, 7000]))
    return s


def write(path: Path, model_prob, threshold: float, n_open: int, m_close: int, seed: int = C.SEED) -> dict:
    rng = np.random.default_rng(seed)
    ws = windows(rng)
    stream = detector_stream(threshold, rng)
    det = Ev.Detector(float(np.float32(threshold)), n_open, m_close)
    buf = bytearray(MAGIC)
    buf += struct.pack("<i", len(ws))
    n_valid = 0
    for _, env, acc, gyr in ws:
        code_v = py_valid(env, acc, gyr)
        code_f, x = py_features(env, acc, gyr)
        prob = np.float32(model_prob(x[None])[0]) if code_f == 0 else np.float32(np.nan)
        n_valid += code_f == 0
        buf += struct.pack("<i", len(acc)) + env.tobytes() + acc.tobytes() + gyr.tobytes()
        buf += struct.pack("<ii", code_v, code_f) + x.astype(np.float32).tobytes() + struct.pack("<f", prob)
    buf += struct.pack("<ifiiII", len(stream), np.float32(threshold), n_open, m_close, int(C.REFRACTORY_S * 1000),
                       int(C.MAX_GAP_S * 1000))
    for op, p, t in stream:
        if op == 1:
            det.reset()
            flags = 0
        else:
            flags = det.update(float(np.float32(p)), t)
        buf += struct.pack("<BfII", op, np.float32(p), t % 2**32, flags)
    path.write_bytes(bytes(buf))
    return {"windows": len(ws), "valid_windows": int(n_valid), "detector_steps": len(stream)}
