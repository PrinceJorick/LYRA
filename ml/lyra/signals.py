"""EDF reading (pyedflib) and the training-side signal chain.

ENV path (locked decision D12): the pod samples the MyoWare 2.0 ENV pin, so training simulates
that analog chain from SeizeIT2 RAW sEMG, causally (an analog filter cannot see the future):
1st-order Butterworth HPF 20.8 Hz -> full-wave rectification -> 1st-order LPF 3.6 Hz (Kyle's
SimulatedEnvConfig) -> 50 Hz by averaging each 20 ms bin (what an oversampling ADC does on the
pod). Filter state starts at zero at the start of each recording; WARMUP_S covers the transient.
No digital band-pass or notch is applied: the MyoWare ENV output has none.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pyedflib
import scipy.signal as sps
from lyra_mlm.preprocess import simulated_env_filters
from lyra_mlm.seizeit2 import calibrate_imu

from . import config as C


def read_edf(path: Path, labels: tuple[str, ...]) -> tuple[dict, np.ndarray, np.ndarray]:
    """(header summary, digital int32 (n, k), physical float64 (n, k)) for the requested channels."""
    with pyedflib.EdfReader(str(path)) as f:
        names = f.getSignalLabels()
        missing = [lb for lb in labels if lb not in names]
        if missing:
            raise KeyError(f"{path.name}: channels missing {missing}")
        idx = [names.index(lb) for lb in labels]
        hdr = {"filetype": int(f.filetype), "start": f.getStartdatetime().isoformat(), "duration_s": f.getFileDuration(),
               "records": int(f.datarecords_in_file), "record_s": float(f.datarecord_duration), "channels": []}
        n = {int(f.getNSamples()[i]) for i in idx}
        if len(n) != 1:
            raise ValueError(f"{path.name}: channels have different lengths {sorted(n)}")
        dig = np.column_stack([f.readSignal(i, digital=True) for i in idx]).astype(np.int32)
        phys = np.empty(dig.shape)
        for c, i in enumerate(idx):
            h = f.getSignalHeader(i)
            hdr["channels"].append({k: h[k] for k in ("label", "dimension", "physical_min", "physical_max",
                                                      "digital_min", "digital_max", "prefilter")}
                                   | {"rate_hz": float(f.getSampleFrequency(i))})
            bit = (h["physical_max"] - h["physical_min"]) / (h["digital_max"] - h["digital_min"])
            phys[:, c] = bit * (dig[:, c] - h["digital_min"]) + h["physical_min"]  # EDF scaling
    return hdr, dig, phys


def at_rails(dig: np.ndarray, hdr: dict) -> np.ndarray:
    """(n, k) True where a sample sits on its channel's digital min/max (ADC saturation/clipping)."""
    lo = np.array([c["digital_min"] for c in hdr["channels"]])
    hi = np.array([c["digital_max"] for c in hdr["channels"]])
    return (dig <= lo) | (dig >= hi)


def simulate_env(raw: np.ndarray, fs: int) -> np.ndarray:
    """Causal MyoWare ENV simulation at 50 Hz; sample k is the mean of [k/50, (k+1)/50)."""
    if fs != int(fs) or fs < 2 * C.ENV_RATE:
        raise ValueError(f"EMG rate {fs} Hz not supported")
    hpf, lpf = simulated_env_filters(float(fs), C.ENV)
    y = sps.sosfilt(lpf, np.abs(sps.sosfilt(hpf, raw)))
    n_out = len(raw) * C.ENV_RATE // int(fs)
    k = np.arange(len(raw)) * C.ENV_RATE // int(fs)
    keep = k < n_out
    return np.bincount(k[keep], weights=y[keep], minlength=n_out) / np.bincount(k[keep], minlength=n_out)


def imu_physical(phys: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, str]:
    """(acc m/s^2, gyro rad/s, calibratable, note) using lyra_mlm.seizeit2.calibrate_imu
    (per-recording gravity scale -> g; gyro -> deg/s), then SI units."""
    vals, flags, note = calibrate_imu(phys)
    ok = flags is None
    return vals[:, :3] * C.GRAVITY, np.radians(vals[:, 3:]), ok, note


def start_time(hdr: dict) -> datetime:
    return datetime.fromisoformat(hdr["start"])
