"""Reference preprocessing: windowing, quality checks, labels and the 24 window features.

This file is the specification of the model input. src/services/ml/seizeit2/features.ts is a
line-by-line port and must stay numerically identical (checked by export.py golden vectors).
Every feature is rotation invariant (magnitudes, variance traces, angles), so the IMU's
mounting orientation does not matter. Units: sEMG in µV; accelerometer in g (SeizeIT2 files
store it in per-file scales, so each run is divided by its median |acc|, i.e. by gravity);
gyroscope features are scale invariant only, because the files' gyro unit is undocumented and
differs between recordings.

Run as a module to build the feature cache for every run:
    python -m ml.seizeit2.preprocessing [--subjects sub-001 ...]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from lyra_mlm.annotations import AnnotationKind

from . import config as C
from . import data_loader as D

FEATURE_NAMES = (
    # sEMG, Welch PSD from 1 s Hann frames (1 Hz bins); "HF" = 20-45 Hz + 55-128 Hz (mains band out)
    "emg_hf_logpow", "emg_frac_20_45", "emg_frac_55_90", "emg_low_ratio", "emg_mains_ratio",
    "emg_median_freq", "emg_spectral_entropy", "emg_frame_logpow_std",
    # sEMG envelope: RMS of the first difference in 1/16 s bins
    "env_cv", "env_peak", "env_rhythm_1_5", "env_dom_freq",
    # accelerometer (magnitude / variance trace / jerk / spectrum of the magnitude)
    "acc_mag_std", "acc_var_trace", "acc_jerk", "acc_frac_1_6", "acc_dom_freq", "acc_peak",
    # gyroscope, scale invariant only (unit unknown in the dataset)
    "gyr_cv", "gyr_frac_1_6", "gyr_dom_freq", "gyr_peak",
    # posture and cross-modal
    "orient_change_deg", "xmod_freq_diff",
)
N_FEATURES = len(FEATURE_NAMES)
ENV_BIN = 16  # EMG samples per envelope bin -> 16 Hz envelope
ORIENT_S = 2  # seconds averaged at each end of the window for the gravity direction


def _nfft(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _spectrum(x: np.ndarray) -> np.ndarray:
    """|rfft|^2 of mean-removed, Hann-windowed rows, zero-padded to the next power of two."""
    n = x.shape[-1]
    x = (x - x.mean(axis=-1, keepdims=True)) * np.hanning(n)
    return np.abs(np.fft.rfft(x, _nfft(n), axis=-1)) ** 2


def _band_stats(p: np.ndarray, freqs: np.ndarray, lo: float, hi: float, band: tuple[float, float]):
    """(fraction of [lo,hi) power inside `band`, dominant frequency in [lo,hi), log peak/mean)."""
    m = (freqs >= lo) & (freqs < hi)
    pb, fb = p[..., m], freqs[m]
    total = pb.sum(-1)
    inside = pb[..., (fb >= band[0]) & (fb < band[1])].sum(-1)
    peak = pb.max(-1)
    return (inside / (total + 1e-12), fb[pb.argmax(-1)],
            np.log10((peak + 1e-12) / (pb.mean(-1) + 1e-12)))


def window_features(emg: np.ndarray, acc: np.ndarray, gyr: np.ndarray, window_s: int) -> np.ndarray:
    """Features for a batch of windows.

    emg: (B, EMG_FS*W) µV; acc: (B, IMU_FS*W, 3) in g; gyr: same shape, any fixed unit.
    Returns (B, N_FEATURES) float64.
    """
    emg = np.asarray(emg, np.float64)
    acc = np.asarray(acc, np.float64)
    gyr = np.asarray(gyr, np.float64)
    b = emg.shape[0]
    imu_shape = (b, C.IMU_FS * window_s, 3)
    if emg.shape != (b, C.EMG_FS * window_s) or acc.shape != imu_shape or gyr.shape != imu_shape:
        raise ValueError(f"bad window shapes {emg.shape} {acc.shape} {gyr.shape} for W={window_s}")
    out = np.empty((b, N_FEATURES))

    # --- sEMG PSD: mean of one-sided periodograms of 1 s frames, in µV²/Hz (1 Hz bins) ---
    w = np.hanning(C.EMG_FS)
    frames = emg.reshape(b, window_s, C.EMG_FS)
    frames = (frames - frames.mean(-1, keepdims=True)) * w
    per = np.abs(np.fft.rfft(frames, axis=-1)) ** 2 / (C.EMG_FS * (w ** 2).sum())
    per[..., 1:-1] *= 2
    psd = per.mean(1)  # (B, 129), bin k = k Hz
    f = np.arange(psd.shape[-1])
    hf = ((f >= 20) & (f < 45)) | ((f >= 55) & (f < 128))
    p_hf = psd[:, hf].sum(-1)
    out[:, 0] = np.log10(p_hf + 1e-6)
    out[:, 1] = psd[:, 20:45].sum(-1) / (p_hf + 1e-12)
    out[:, 2] = psd[:, 55:90].sum(-1) / (p_hf + 1e-12)
    out[:, 3] = np.log10(psd[:, 1:20].sum(-1) + 1e-6) - out[:, 0]
    out[:, 4] = np.log10(psd[:, 45:55].sum(-1) + 1e-6) - out[:, 0]
    cum = np.cumsum(psd[:, hf], -1)
    out[:, 5] = f[hf][np.argmax(cum >= 0.5 * cum[:, -1:], -1)]
    p = psd[:, hf] / (p_hf[:, None] + 1e-12)
    out[:, 6] = -(p * np.log(p + 1e-12)).sum(-1) / np.log(hf.sum())
    out[:, 7] = np.log10(per[..., hf].sum(-1) + 1e-6).std(-1)

    # --- sEMG envelope ---
    d = np.diff(emg, axis=-1, prepend=emg[:, :1])
    env = np.sqrt((d.reshape(b, -1, ENV_BIN) ** 2).mean(-1))  # (B, 16W)
    em = env.mean(-1)
    out[:, 8] = env.std(-1) / (em + 1e-9)
    out[:, 9] = np.log10((env.max(-1) + 1e-9) / (np.median(env, -1) + 1e-9))
    fs_env = C.EMG_FS / ENV_BIN
    pe = _spectrum(env)
    fe = np.arange(pe.shape[-1]) * fs_env / _nfft(env.shape[-1])
    out[:, 10], out[:, 11], _ = _band_stats(pe, fe, 0.5, fs_env / 2, (1, 5))

    # --- IMU ---
    fi = np.arange(_nfft(acc.shape[1]) // 2 + 1) * C.IMU_FS / _nfft(acc.shape[1])
    amag = np.linalg.norm(acc, axis=-1)
    out[:, 12] = np.log10(amag.std(-1) + 1e-4)
    out[:, 13] = np.log10(acc.var(1).sum(-1) + 1e-8)
    out[:, 14] = np.log10(np.abs(np.diff(amag, axis=-1)).mean(-1) * C.IMU_FS + 1e-4)
    out[:, 15], out[:, 16], out[:, 17] = _band_stats(_spectrum(amag), fi, 0.5, C.IMU_FS / 2, (1, 6))
    gmag = np.linalg.norm(gyr, axis=-1)
    out[:, 18] = gmag.std(-1) / (gmag.mean(-1) + 1e-9)
    out[:, 19], out[:, 20], out[:, 21] = _band_stats(_spectrum(gmag), fi, 0.5, C.IMU_FS / 2, (1, 6))

    k = C.IMU_FS * ORIENT_S
    u, v = acc[:, :k].mean(1), acc[:, -k:].mean(1)
    cos = (u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-12)
    out[:, 22] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    out[:, 23] = np.abs(out[:, 11] - out[:, 20])
    return out


# ---- Windows, quality and labels ---------------------------------------------------------------


def window_ends(duration_s: int, window_s: int) -> np.ndarray:
    return np.arange(window_s, duration_s + 1, C.HOP_S, dtype=np.int32)


def _overlap(t_end: np.ndarray, window_s: int, iv: list[tuple[float, float]]) -> np.ndarray:
    start = t_end - window_s
    ov = np.zeros(t_end.shape)
    for a, b in iv:
        ov = np.maximum(ov, np.clip(np.minimum(t_end, b) - np.maximum(start, a), 0, None))
    return ov


def label_windows(run: D.Run, t_end: np.ndarray, window_s: int) -> np.ndarray:
    """1 = target seizure, 0 = background, -1 = excluded from training (guards, other seizures, impd)."""
    tgt = run.intervals(D.TARGET_KINDS)
    oos = run.intervals((AnnotationKind.SEIZURE_OUT_OF_SCOPE,))
    art = run.intervals((AnnotationKind.ARTIFACT,))
    pre, post = C.TARGET_GUARD_S
    opre, opost = C.OOS_GUARD_S
    y = np.zeros(t_end.shape, np.int8)
    y[_overlap(t_end, window_s, [(a - pre, b + post) for a, b in tgt]) > 0] = -1
    y[_overlap(t_end, window_s, [(a - opre, b + opost) for a, b in oos]) > 0] = -1
    y[_overlap(t_end, window_s, art) > 0] = -1
    y[_overlap(t_end, window_s, tgt) >= C.POS_MIN_OVERLAP * window_s] = 1
    return y


def extract_run(run: D.Run, tmp) -> dict:
    """Fetch one run and compute features for every window length. Missing modality -> no windows."""
    meta = {"subject": run.subject, "run": run.run, "duration_s": run.duration_s,
            "has_emg": run.emg_path is not None, "has_imu": run.mov_path is not None, "windows": {}}
    if run.emg_path is None or run.mov_path is None:
        return meta  # the model needs both modalities; don't download half a run
    emg, acc, gyr = D.load_run_signals(run, tmp)
    meta["has_emg"], meta["has_imu"] = emg is not None, acc is not None
    if emg is None or acc is None:
        return meta
    scale = float(np.median(np.linalg.norm(acc[::C.IMU_FS], axis=1)))  # gravity in file units
    meta["acc_norm_median"] = scale
    meta["gyr_norm_median"] = float(np.median(np.linalg.norm(gyr[::C.IMU_FS], axis=1)))
    if not C.ACC_SCALE_VALID[0] <= scale <= C.ACC_SCALE_VALID[1]:
        meta["acc_scale_invalid"] = True  # pinned at the rail, zero, or unknown scale
        return meta
    acc = acc / scale  # -> g
    dur = min(run.duration_s, emg.values.size // C.EMG_FS, acc.shape[0] // C.IMU_FS)
    meta["usable_duration_s"] = int(dur)
    sw = np.lib.stride_tricks.sliding_window_view
    out = {}
    for W in C.WINDOWS_S:
        t_end = window_ends(dur, W)
        if t_end.size == 0:
            continue
        # strided views; only one chunk of windows is ever copied
        e_view, s_view = sw(emg.values, C.EMG_FS * W), sw(emg.saturated, C.EMG_FS * W)
        a_view, g_view = sw(acc, C.IMU_FS * W, axis=0), sw(gyr, C.IMU_FS * W, axis=0)
        X = np.empty((t_end.size, N_FEATURES), np.float32)
        ok = np.empty(t_end.size, bool)
        for i in range(0, t_end.size, 1024):
            st = t_end[i:i + 1024] - W
            e, sat = e_view[st * C.EMG_FS], s_view[st * C.EMG_FS]
            a = a_view[st * C.IMU_FS].transpose(0, 2, 1)
            g = g_view[st * C.IMU_FS].transpose(0, 2, 1)
            X[i:i + 1024] = window_features(e, a, g, W)
            ok[i:i + 1024] = ((e.std(-1) > C.FLAT_STD) & (sat.mean(-1) <= C.EMG_SAT_MAX_FRAC)
                              & (a.std(1).max(-1) > C.FLAT_STD) & (g.std(1).max(-1) > C.FLAT_STD))
        out[W] = {"X": X, "t_end": t_end, "y": label_windows(run, t_end, W), "ok": ok}
        meta["windows"][str(W)] = {"n": int(t_end.size), "ok": int(ok.sum()),
                                   "pos": int((out[W]["y"] == 1).sum())}
    meta["arrays"] = out
    return meta


def feature_path(W: int, run: str):
    return C.CACHE / "features" / f"W{W}" / f"{run}.npz"


def _work(run: D.Run) -> dict:
    tmp = C.CACHE / "tmp" / run.run
    try:
        meta = extract_run(run, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    for W, arr in meta.pop("arrays", {}).items():
        p = feature_path(W, run.run)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **arr)
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subjects", nargs="*", help="limit to these subjects (default: all)")
    ap.add_argument("--workers", type=int, default=C.DOWNLOAD_WORKERS)
    args = ap.parse_args(argv)
    runs = [r for r in D.list_runs() if not args.subjects or r.subject in args.subjects]
    index_path = C.CACHE / "features" / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {}
    todo = [r for r in runs if r.run not in index]  # resumable
    print(f"{len(runs)} runs, {len(todo)} to process", flush=True)
    t0, done, failed = time.time(), 0, []
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(_work, r): r for r in todo}
        for fut in as_completed(futs):
            r = futs[fut]
            try:
                index[r.run] = fut.result()
            except Exception as e:  # recorded and reported; never silently skipped
                failed.append((r.run, repr(e)))
                print(f"FAILED {r.run}: {e!r}", flush=True)
            done += 1
            if done % 25 == 0 or done == len(todo):
                index_path.parent.mkdir(parents=True, exist_ok=True)
                index_path.write_text(json.dumps(index, indent=1))
                print(f"{done}/{len(todo)} runs, {time.time() - t0:.0f}s", flush=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=1))
    if failed:
        (C.CACHE / "features" / "failed.json").write_text(json.dumps(failed, indent=1))
        print(f"{len(failed)} runs failed; see failed.json", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
