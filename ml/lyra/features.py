"""The 13 LYRA features (feature_spec.json), vectorised over windows. This is the Python reference
that firmware/model/lyra_features.c reproduces; every formula and edge case is defined there.

Inputs per window: env (n, 50*W) MyoWare-ENV samples at 50 Hz, acc (n, 25*W, 3) in m/s^2 and
gyr (n, 25*W, 3) in rad/s. Only samples inside the completed window are used (no future data).
A missing modality (None) gives NaN for its features and for the cross-modal ones.
"""

from __future__ import annotations

import numpy as np

from . import config as C

ACF_LAGS = range(C.ENV_RATE // 5, C.ENV_RATE // 1 + 1)  # 1-5 Hz periodicity at 50 Hz: lags 10..50
COH_NPERSEG, COH_STEP = 16, 8
COH_BINS = [k for k in range(COH_NPERSEG // 2 + 1) if 1.0 <= k * C.IMU_RATE / COH_NPERSEG <= 5.0]  # k = 1, 2, 3
COH_MIN_SEGMENTS = 4
DOM_TIE_REL = 1e-6


def hann_periodic(n: int) -> np.ndarray:
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros(np.broadcast(num, den).shape)
    ok = den > C.EPS
    np.divide(num, den, out=out, where=ok)
    return out


def emg_features(env: np.ndarray) -> np.ndarray:
    e = np.asarray(env, np.float64)
    d = e - e.mean(axis=1, keepdims=True)
    var_sum = (d * d).sum(axis=1)
    acf = np.full(e.shape[0], -np.inf)
    for lag in ACF_LAGS:
        acf = np.maximum(acf, (d[:, :-lag] * d[:, lag:]).sum(axis=1))
    return np.column_stack([
        np.sqrt((e * e).mean(axis=1)), np.abs(e).mean(axis=1), np.abs(np.diff(e, axis=1)).sum(axis=1),
        np.sqrt(var_sum / e.shape[1]), _safe_div(acf, var_sum)])


def _power(x: np.ndarray) -> np.ndarray:
    """One-sided |DFT|^2 of periodic-Hann * (x - mean), bins k = 0..N/2."""
    d = (x - x.mean(axis=1, keepdims=True)) * hann_periodic(x.shape[1])
    return np.abs(np.fft.rfft(d, axis=1)) ** 2


def imu_features(acc: np.ndarray, gyr: np.ndarray) -> np.ndarray:
    a, g = np.asarray(acc, np.float64), np.asarray(gyr, np.float64)
    m = np.sqrt((a * a).sum(axis=2))
    n = m.shape[1]
    f = np.arange(n // 2 + 1) * C.IMU_RATE / n
    p = _power(m)
    band = p[:, (f >= 1.0) & (f <= 5.0)].sum(axis=1)
    total = p[:, 1:].sum(axis=1)
    dom_sel = np.flatnonzero((f >= 0.5) & (f <= 8.0))
    dom_p = p[:, dom_sel]
    mx = dom_p.max(axis=1)
    # lowest bin within DOM_TIE_REL of the maximum: exact ties (e.g. quantised still signals) are common,
    # and a plain argmax would let rounding noise pick the bin
    first = np.argmax(dom_p >= mx[:, None] * (1.0 - DOM_TIE_REL), axis=1)
    dom = np.where(mx > C.EPS, f[dom_sel][first], 0.0)
    k = C.IMU_RATE  # 1 s of samples for the start/end gravity estimate
    g1, g2 = a[:, :k].mean(axis=1), a[:, -k:].mean(axis=1)
    cross = np.sqrt((np.cross(g1, g2) ** 2).sum(axis=1))
    dot = (g1 * g2).sum(axis=1)
    ok = (np.sqrt((g1 * g1).sum(axis=1)) > C.EPS) & (np.sqrt((g2 * g2).sum(axis=1)) > C.EPS)
    tilt = np.where(ok, np.degrees(np.arctan2(cross, dot)), 0.0)
    jerk = np.diff(m, axis=1) * C.IMU_RATE
    return np.column_stack([m.mean(axis=1), np.sqrt((g * g).sum(axis=2)).mean(axis=1),
                            np.sqrt((jerk * jerk).mean(axis=1)), _safe_div(band, total), dom, tilt])


def xmod_features(env: np.ndarray, acc: np.ndarray) -> np.ndarray:
    e = np.asarray(env, np.float64)
    x = 0.5 * (e[:, 0::2] + e[:, 1::2])  # env at 25 Hz: 2-sample mean (envelope is band-limited to ~3.6 Hz)
    a = np.asarray(acc, np.float64)
    y = np.sqrt((a * a).sum(axis=2))
    dx, dy = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    sxx, syy = (dx * dx).sum(axis=1), (dy * dy).sum(axis=1)
    ok = (sxx > C.EPS) & (syy > C.EPS)
    r = np.zeros(x.shape[0])
    r[ok] = (dx * dy).sum(axis=1)[ok] / np.sqrt(sxx[ok] * syy[ok])
    n_seg = (x.shape[1] - COH_NPERSEG) // COH_STEP + 1
    if n_seg < COH_MIN_SEGMENTS:
        raise ValueError(f"window too short for coherence: {n_seg} segments < {COH_MIN_SEGMENTS}")
    w = hann_periodic(COH_NPERSEG)
    idx = np.arange(n_seg)[:, None] * COH_STEP + np.arange(COH_NPERSEG)
    sx, sy = x[:, idx], y[:, idx]  # (n, n_seg, nperseg)
    fx = np.fft.rfft((sx - sx.mean(axis=2, keepdims=True)) * w, axis=2)[:, :, COH_BINS]
    fy = np.fft.rfft((sy - sy.mean(axis=2, keepdims=True)) * w, axis=2)[:, :, COH_BINS]
    pxx, pyy = (np.abs(fx) ** 2).sum(axis=1), (np.abs(fy) ** 2).sum(axis=1)
    pxy = (fx * np.conj(fy)).sum(axis=1)
    coh = _safe_div(np.abs(pxy) ** 2, pxx * pyy)
    return np.column_stack([r, coh.mean(axis=1)])


def compute(env: np.ndarray | None, acc: np.ndarray | None, gyr: np.ndarray | None, n: int) -> np.ndarray:
    """(n, 13) float64 features in feature_spec order; NaN where a modality is absent."""
    out = np.full((n, len(C.FEATURES)), np.nan)
    if env is not None:
        out[:, list(C.MODALITIES["emg"])] = emg_features(env)
    if acc is not None:
        out[:, list(C.MODALITIES["imu"])] = imu_features(acc, gyr)
    if env is not None and acc is not None:
        out[:, [C.FEATURES.index("xmod_corr_env_accmag"), C.FEATURES.index("xmod_coh_1_5hz")]] = xmod_features(env, acc)
    return out
