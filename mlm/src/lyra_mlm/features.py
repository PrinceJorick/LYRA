"""Deterministic feature extraction and multimodal feature fusion.

Every feature documents its modality, signal type, sampling and minimum-data requirements
(FeatureSpec.requires). A feature that cannot be computed reliably is marked unavailable:
its value is NaN, its availability flag False, and the reason is recorded. A genuine 0.0 stays
available. Feature order is fixed per FeatureConfig (feature_names) for training and embedded
parity. Fusion is the concatenation of the EMG, IMU and cross-modal blocks in that order,
keeping the window's quality metadata.

Default parameters are initial research values, not optimised or validated for seizure
detection. EMG feature names carry the signal type (emg_env_rms vs emg_raw_rms), so a model
trained on one MyoWare output can never silently receive another.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal as sps
from scipy.integrate import trapezoid

from .quality import USABLE, WindowQuality
from .signals import EmgSignalType, Modality, check_timestamps
from .windows import Window

FeatureMode = Literal["emg", "imu", "multimodal"]
MODE_MODALITIES: dict[str, tuple[Modality, ...]] = {
    "emg": (Modality.EMG,), "imu": (Modality.IMU,), "multimodal": (Modality.EMG, Modality.IMU),
}


@dataclass(frozen=True)
class FeatureConfig:
    mode: FeatureMode
    emg_signal_type: EmgSignalType | None = None  # required for emg/multimodal, None for imu
    movement_band_hz: tuple[float, float] = (1.0, 8.0)
    rhythm_lag_s: tuple[float, float] = (0.125, 1.0)  # envelope autocorrelation lags (8 Hz .. 1 Hz)
    max_xcorr_lag_s: float = 0.5
    min_paired_samples: int = 20
    gap_factor: float = 3.0

    def __post_init__(self) -> None:
        if self.mode not in MODE_MODALITIES:
            raise ValueError(f"unknown mode {self.mode!r}")
        if (self.mode != "imu") != (self.emg_signal_type is not None):
            raise ValueError("emg_signal_type is required for emg/multimodal modes and must be None for imu")
        if self.mode == "multimodal" and self.emg_signal_type is not EmgSignalType.ENV:
            raise ValueError("multimodal features currently use the ENV path only; RAW fusion is deferred research")
        lo, hi = self.movement_band_hz
        lag_lo, lag_hi = self.rhythm_lag_s
        if not (0 < lo < hi and 0 < lag_lo < lag_hi and self.max_xcorr_lag_s > 0 and self.min_paired_samples >= 3):
            raise ValueError("invalid band, lag or pairing parameters")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    unit: str
    requires: str  # modality, signal type, sampling and minimum valid data


class FeatureUnavailable(Exception):
    pass


def _need(cond: bool, reason: str) -> None:
    if not cond:
        raise FeatureUnavailable(reason)


class _Ctx:
    """Usable samples of one window, prepared once for all features."""

    def __init__(self, window: Window, cfg: FeatureConfig):
        self.cfg = cfg
        e, i = window.emg, window.imu
        if e is not None and cfg.mode != "imu":
            ok = e.usable
            self.emg_t, self.emg_x = e.timestamps[ok], e.values[ok, 0]
            self.emg_runs = [e.values[a:b, 0] for a, b in e.contiguous_runs(cfg.gap_factor)]
            self.emg_rate, self.emg_bw = e.nominal_rate_hz, e.effective_bandwidth_hz
        if i is not None and cfg.mode != "emg":
            ok = i.usable
            self.imu_t, self.acc, self.gyr = i.timestamps[ok], i.values[ok, :3], i.values[ok, 3:]
            self.acc_mag, self.gyr_mag = np.linalg.norm(self.acc, axis=1), np.linalg.norm(self.gyr, axis=1)
            self.acc_runs = [i.values[a:b, :3] for a, b in i.contiguous_runs(cfg.gap_factor)]
            self.imu_rate, self.imu_bw = i.nominal_rate_hz, i.effective_bandwidth_hz
            self.imu_regular = check_timestamps(self.imu_t, self.imu_rate, gap_factor=cfg.gap_factor).is_clean


# --- sEMG ------------------------------------------------------------------------------------

def _emg_rms(c: _Ctx) -> float:
    _need(c.emg_x.size >= 1, "no usable EMG samples")
    return float(np.sqrt(np.mean(c.emg_x**2)))


def _emg_mav(c: _Ctx) -> float:
    _need(c.emg_x.size >= 1, "no usable EMG samples")
    return float(np.mean(np.abs(c.emg_x)))


def _emg_var(c: _Ctx) -> float:
    _need(c.emg_x.size >= 2, "fewer than 2 usable EMG samples")
    return float(np.var(c.emg_x))


def _emg_std(c: _Ctx) -> float:
    return math.sqrt(_emg_var(c))


def _emg_wl(c: _Ctx) -> float:
    runs = [r for r in c.emg_runs if r.size >= 2]
    _need(bool(runs), "no contiguous EMG run of 2+ samples")
    total = sum(float(np.abs(np.diff(r)).sum()) for r in runs)
    return total / sum((r.size - 1) / c.emg_rate for r in runs)


def _emg_autocorr(c: _Ctx) -> float:
    lo, hi = c.cfg.rhythm_lag_s
    _need(c.emg_bw is not None, "ENV bandwidth not declared; sensor-side smoothing unknown")
    _need(c.emg_bw >= 1 / lo, f"ENV bandwidth {c.emg_bw:g} Hz is below the {1 / lo:g} Hz rhythm range")
    r = max(c.emg_runs, key=len, default=np.empty(0))
    _need(r.size / c.emg_rate >= 2 * hi, f"longest contiguous EMG run shorter than {2 * hi:g} s")
    x = r - r.mean()
    energy = float(x @ x)
    _need(energy > 0, "constant envelope")
    ks = range(max(1, round(lo * c.emg_rate)), round(hi * c.emg_rate) + 1)
    return max(float(x[:-k] @ x[k:]) / energy for k in ks)


# --- IMU -------------------------------------------------------------------------------------

def _imu_stat(attr: str, op: Callable[[np.ndarray], float]) -> Callable[[_Ctx], float]:
    def f(c: _Ctx) -> float:
        v = getattr(c, attr)
        _need(v.size >= 2, "fewer than 2 usable IMU samples")
        return float(op(v))
    return f


def _acc_jerk_rms(c: _Ctx) -> float:
    jerks = [np.linalg.norm(np.diff(r, axis=0), axis=1) * c.imu_rate for r in c.acc_runs if len(r) >= 2]
    _need(bool(jerks), "no contiguous IMU run of 2+ samples")
    j = np.concatenate(jerks)
    return float(np.sqrt(np.mean(j**2)))


def _acc_spectrum(c: _Ctx) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = c.cfg.movement_band_hz
    _need(c.imu_bw is not None and c.imu_bw >= hi, f"IMU bandwidth does not reach {hi:g} Hz")
    r = np.linalg.norm(max(c.acc_runs, key=len, default=np.empty((0, 3))), axis=1)
    _need(r.size / c.imu_rate >= 2 / lo, f"longest contiguous IMU run shorter than 2 cycles of {lo:g} Hz")
    f, p = sps.periodogram(r - r.mean(), fs=c.imu_rate, window="hann")
    band = (f >= lo) & (f <= hi)
    _need(band.sum() >= 2, "too few frequency bins in the movement band")
    return f[band], p[band]


def _acc_band_power(c: _Ctx) -> float:
    f, p = _acc_spectrum(c)
    return float(trapezoid(p, f))


def _acc_dominant_freq(c: _Ctx) -> float:
    f, p = _acc_spectrum(c)
    _need(p.max() > 0, "no movement energy in the band")
    return float(f[np.argmax(p)])


def _orientation_change(c: _Ctx) -> float:
    q = len(c.acc) // 4
    _need(q >= 1, "fewer than 4 usable IMU samples")
    g1, g2 = c.acc[:q].mean(axis=0), c.acc[-q:].mean(axis=0)
    n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
    _need(n1 > 0 and n2 > 0, "zero mean acceleration vector")
    return float(np.degrees(np.arccos(np.clip(g1 @ g2 / (n1 * n2), -1.0, 1.0))))


# --- cross-modal (ENV + IMU) -----------------------------------------------------------------

def _env_on_imu_grid(c: _Ctx) -> np.ndarray:
    """Mean ENV value within +-half an IMU period of each IMU sample (boxcar average, then
    decimation onto the IMU timeline). Local to these features; the streams are not resampled."""
    half = 0.5 / c.imu_rate
    lo = np.searchsorted(c.emg_t, c.imu_t - half, side="left")
    hi = np.searchsorted(c.emg_t, c.imu_t + half, side="left")
    cs = np.concatenate(([0.0], np.cumsum(c.emg_x)))
    n = hi - lo
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)


def _paired_regular(c: _Ctx) -> tuple[np.ndarray, np.ndarray]:
    env = _env_on_imu_grid(c)
    _need(c.imu_regular and bool(np.isfinite(env).all()), "needs a gap-free IMU timeline fully covered by ENV")
    _need(env.size >= c.cfg.min_paired_samples, "too few paired samples")
    a, b = env - env.mean(), c.acc_mag - c.acc_mag.mean()
    _need(bool(a @ a > 0 and b @ b > 0), "a constant series has no cross-modal relationship")
    return a, b


def _xmod_corr(c: _Ctx) -> float:
    env = _env_on_imu_grid(c)
    ok = np.isfinite(env)
    _need(ok.sum() >= c.cfg.min_paired_samples, "too few paired samples")
    a, b = env[ok], c.acc_mag[ok]
    _need(a.std() > 0 and b.std() > 0, "a constant series has no correlation")
    return float(np.corrcoef(a, b)[0, 1])


def _xmod_energy_ratio(c: _Ctx) -> float:
    emg, acc = _emg_rms(c), _imu_stat("acc_mag", np.std)(c)
    _need(emg > 0 and acc > 0, "zero energy in one modality")
    return math.log10(emg / acc)


def _xmod_lag(c: _Ctx) -> float:
    a, b = _paired_regular(c)
    max_k = min(round(c.cfg.max_xcorr_lag_s * c.imu_rate), a.size - 1)
    full = sps.correlate(b, a, mode="full")  # index a.size-1 is lag 0; positive lag: IMU follows EMG
    mid = a.size - 1
    k = int(np.argmax(full[mid - max_k: mid + max_k + 1])) - max_k
    return k / c.imu_rate


def _xmod_coherence(c: _Ctx) -> float:
    lo, hi = c.cfg.movement_band_hz
    _need(c.emg_bw is not None and c.emg_bw >= hi, f"ENV bandwidth not declared or below {hi:g} Hz")
    _need(c.imu_bw is not None and c.imu_bw >= hi, f"IMU bandwidth does not reach {hi:g} Hz")
    a, b = _paired_regular(c)
    nperseg = round(2 * c.imu_rate / lo)
    _need(a.size >= 2 * nperseg, f"needs {2 * nperseg} paired samples for coherence segments")
    f, coh = sps.coherence(a, b, fs=c.imu_rate, nperseg=nperseg)
    band = (f >= lo) & (f <= hi)
    _need(bool(band.any()), "no coherence bins in the movement band")
    return float(np.mean(coh[band]))


# --- registry ----------------------------------------------------------------------------------

_Entry = tuple[str, str, str, Callable[[_Ctx], float]]
_EMG: list[_Entry] = [
    ("rms", "sensor units", ">=1 usable sample", _emg_rms),
    ("mav", "sensor units", ">=1 usable sample", _emg_mav),
    ("var", "sensor units^2", ">=2 usable samples", _emg_var),
    ("std", "sensor units", ">=2 usable samples", _emg_std),
    ("wl_per_s", "sensor units/s", "contiguous runs of >=2 samples; waveform length per second", _emg_wl),
]
_EMG_ENV: list[_Entry] = [
    ("autocorr_peak", "unitless", "ENV only; declared bandwidth >= 1/min lag; contiguous run >= 2x max lag; "
     "peak normalised autocorrelation over rhythm_lag_s", _emg_autocorr),
]
_IMU: list[_Entry] = [
    ("acc_mag_mean", "g", ">=2 usable samples; includes gravity", _imu_stat("acc_mag", np.mean)),
    ("acc_mag_std", "g", ">=2 usable samples", _imu_stat("acc_mag", np.std)),
    ("gyr_mag_mean", "deg/s", ">=2 usable samples", _imu_stat("gyr_mag", np.mean)),
    ("gyr_mag_std", "deg/s", ">=2 usable samples", _imu_stat("gyr_mag", np.std)),
    ("gyr_mag_rms", "deg/s", ">=2 usable samples", _imu_stat("gyr_mag", lambda v: np.sqrt(np.mean(v**2)))),
    ("acc_jerk_rms", "g/s", "contiguous runs of >=2 samples", _acc_jerk_rms),
    ("acc_dominant_freq_hz", "Hz", "bandwidth >= band top; run >= 2 cycles of band bottom; energy in band",
     _acc_dominant_freq),
    ("acc_band_power", "g^2", "bandwidth >= band top; run >= 2 cycles of band bottom", _acc_band_power),
    ("orientation_change_deg", "deg", ">=4 usable samples; angle between mean acceleration (gravity estimate) "
     "of first and last quarter; dynamic acceleration biases it", _orientation_change),
]
_XMOD: list[_Entry] = [
    ("env_accmag_corr", "unitless", "ENV + IMU usable; >=min_paired_samples ENV bins on the IMU timeline",
     _xmod_corr),
    ("energy_log_ratio", "log10(units/g)", "ENV + IMU usable; non-zero energy in both; same-setup comparisons only",
     _xmod_energy_ratio),
    ("lag_s", "s", "gap-free IMU timeline fully covered by ENV; timing agreement only, not proof of physical sync",
     _xmod_lag),
    ("coherence_band", "unitless", "both bandwidths >= band top; >= 2 coherence segments; few segments bias it high",
     _xmod_coherence),
]


def _groups(cfg: FeatureConfig) -> list[tuple[tuple[Modality, ...], str, list[_Entry]]]:
    out = []
    if cfg.mode != "imu":
        t = cfg.emg_signal_type
        out.append(((Modality.EMG,), f"emg_{t.lower()}_", _EMG + (_EMG_ENV if t is EmgSignalType.ENV else [])))
    if cfg.mode != "emg":
        out.append(((Modality.IMU,), "imu_", _IMU))
    if cfg.mode == "multimodal":
        out.append(((Modality.EMG, Modality.IMU), "xmod_", _XMOD))
    return out


def feature_specs(cfg: FeatureConfig) -> tuple[FeatureSpec, ...]:
    return tuple(FeatureSpec(prefix + n, unit, req) for _, prefix, es in _groups(cfg) for n, unit, req, _ in es)


def feature_names(cfg: FeatureConfig) -> tuple[str, ...]:
    return tuple(s.name for s in feature_specs(cfg))


@dataclass(frozen=True, eq=False)
class FeatureVector:
    names: tuple[str, ...]
    values: np.ndarray  # NaN where unavailable
    available: np.ndarray  # bool; False means unavailable, never zero
    unavailable_reasons: Mapping[str, str]
    quality: WindowQuality
    participant_id: str
    recording_id: str
    start_s: float
    end_s: float

    @property
    def available_modalities(self) -> tuple[Modality, ...]:
        return self.quality.available_modalities


def extract_features(window: Window, cfg: FeatureConfig) -> FeatureVector:
    """Compute and fuse the feature blocks for cfg.mode. Blocks whose modalities are not usable
    in this window are unavailable as a whole, with the quality state as the reason."""
    if cfg.mode != "imu" and window.emg is not None and window.emg.signal_type is not cfg.emg_signal_type:
        raise ValueError(f"config expects EMG {cfg.emg_signal_type}, window has {window.emg.signal_type}")
    ctx = _Ctx(window, cfg)
    names, values, reasons = [], [], {}
    for needs, prefix, entries in _groups(cfg):
        bad = [f"{m} {window.quality.per_modality[m].state}" for m in needs
               if window.quality.per_modality[m].state not in USABLE]
        for n, _, _, fn in entries:
            name = prefix + n
            names.append(name)
            try:
                if bad:
                    raise FeatureUnavailable("quality: " + ", ".join(bad))
                v = fn(ctx)
                _need(math.isfinite(v), "non-finite result")
                values.append(v)
            except FeatureUnavailable as e:
                values.append(math.nan)
                reasons[name] = str(e)
    vals = np.array(values, dtype=np.float64)
    return FeatureVector(tuple(names), vals, ~np.isnan(vals), reasons, window.quality,
                         window.participant_id, window.recording_id, window.start_s, window.end_s)
