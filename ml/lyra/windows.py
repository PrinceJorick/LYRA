"""Windows, window quality and labels for one recording (all three window configurations).

Window k of a configuration ends at T_k = (W*25 + k*hop*25) / 25 s: it holds IMU samples
[T-W, T) at 25 Hz and ENV samples [T-W, T) at 50 Hz (integer counts; never across a recording
boundary). The decision for a window is available at T, which is also its event timestamp.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import config as C
from . import features as F
from .bids import Intervals


def _ints(w: float, hop: float) -> tuple[int, int]:
    wi, hi = w * C.IMU_RATE, hop * C.IMU_RATE
    if not (float(wi).is_integer() and float(hi).is_integer()):
        raise ValueError(f"window {w} s / hop {hop} s is not an integer number of IMU samples")
    return int(wi), int(hi)


def grid(duration_imu_samples: int, w: float, hop: float) -> np.ndarray:
    """End index (exclusive, in 25 Hz samples) of every complete window."""
    wi, hi = _ints(w, hop)
    if duration_imu_samples < wi:
        return np.empty(0, np.int64)
    return wi + hi * np.arange((duration_imu_samples - wi) // hi + 1, dtype=np.int64)


def _range_counts(cum: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Number of True values in [a, b) given cum = concatenate([0], cumsum(mask))."""
    return cum[b] - cum[a]


def _cum(mask: np.ndarray) -> np.ndarray:
    return np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])


def signal_ok(bad: np.ndarray, changes: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Window [a, b) of a source stream is usable: <= MAX_SAT_FRACTION bad samples and not flat.
    bad: (n,) saturated/non-finite samples; changes: (n-1,) True where any channel changes."""
    frac = _range_counts(_cum(bad), a, b) / np.maximum(b - a, 1)
    moving = _range_counts(_cum(changes), a, np.maximum(b - 1, a)) > 0
    return (frac <= C.MAX_SAT_FRACTION) & moving & (b > a)


def overlap(t0: np.ndarray, t1: np.ndarray, ivs: list[tuple[float, float, str]]) -> np.ndarray:
    """Seconds of [t0, t1) covered by the union of intervals."""
    if not ivs:
        return np.zeros(t0.shape)
    merged = []
    for on, off, _ in sorted(ivs):
        if merged and on <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], off)
        else:
            merged.append([on, off])
    out = np.zeros(t0.shape)
    for on, off in merged:
        out += np.clip(np.minimum(t1, off) - np.maximum(t0, on), 0, None)
    return out


def build(env: np.ndarray | None, env_src: tuple[np.ndarray, np.ndarray, int] | None,
          acc: np.ndarray | None, gyr: np.ndarray | None, imu_src: tuple[np.ndarray, np.ndarray] | None,
          imu_calibrated: bool, events: Intervals) -> dict[str, np.ndarray]:
    """All window configurations for one recording.

    env: ENV at 50 Hz; env_src: (raw bad-sample mask, raw any-change mask, raw rate) for quality,
    acc/gyr: SI units at 25 Hz; imu_src: (bad-sample mask, any-change mask) of the 6 IMU channels.
    """
    lengths = []
    if env is not None:
        lengths.append(len(env) // 2)  # in 25 Hz samples
    if acc is not None:
        lengths.append(len(acc))
    n_imu = min(lengths)
    out: dict[str, np.ndarray] = {}
    for name, (w, hop) in C.WINDOWS.items():
        wi, _ = _ints(w, hop)
        end = grid(n_imu, w, hop)
        n = end.size
        t_end = end / C.IMU_RATE
        t0 = t_end - w
        e_win = a_win = g_win = None
        emg_ok = np.zeros(n, bool)
        imu_ok = np.zeros(n, bool)
        if env is not None and n:
            e_win = sliding_window_view(env, 2 * wi)[2 * (end - wi)]
            bad, changes, fs = env_src
            emg_ok = signal_ok(bad, changes, (end - wi) * fs // C.IMU_RATE, end * fs // C.IMU_RATE)
            emg_ok &= np.isfinite(e_win).all(axis=1)
        if acc is not None and n:
            a_win = sliding_window_view(acc, wi, axis=0)[end - wi].transpose(0, 2, 1)
            g_win = sliding_window_view(gyr, wi, axis=0)[end - wi].transpose(0, 2, 1)
            bad, changes = imu_src
            imu_ok = signal_ok(bad, changes, end - wi, end) & imu_calibrated
            imu_ok &= np.isfinite(a_win).all(axis=(1, 2)) & np.isfinite(g_win).all(axis=(1, 2))
        x = F.compute(e_win, a_win, g_win, n) if n else np.empty((0, len(C.FEATURES)))
        out |= {f"{name}_t_end": t_end, f"{name}_X": x.astype(np.float32),
                f"{name}_emg_ok": emg_ok, f"{name}_imu_ok": imu_ok,
                f"{name}_artifact": overlap(t0, t_end, events.artifact) > 0,
                f"{name}_warmup": t0 < C.WARMUP_S,
                f"{name}_tfrac": (overlap(t0, t_end, events.target) / w).astype(np.float32),
                f"{name}_nontarget": overlap(t0, t_end, events.nontarget) > 0}
    for k in ("target", "nontarget", "artifact"):
        ivs = getattr(events, k)
        out[f"iv_{k}"] = np.array([(a, b) for a, b, _ in ivs], np.float64).reshape(-1, 2)
    return out


def labels(tfrac: np.ndarray) -> np.ndarray:
    """1 target (>= POS_OVERLAP of the window), 0 no target overlap, -1 partial overlap (excluded)."""
    return np.where(tfrac >= C.POS_OVERLAP, 1, np.where(tfrac > 0, -1, 0)).astype(np.int8)
