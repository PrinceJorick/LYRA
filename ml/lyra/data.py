"""Load cached windows for one window configuration and one partition.

Development code calls load(..., "dev") only; load(..., "test") refuses to run unless the final
evaluation lock file exists (written by final_eval.py once the configuration is frozen).
Invalid windows (no usable modality, impedance check, filter warm-up) are dropped here: the
event state machine holds on them, so they change nothing except the gap rule, which uses the
times of the valid windows that remain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import config as C
from . import windows as Wn

LOCK = C.RUNS / "FINAL_EVALUATION_LOCK.json"


@dataclass
class Windows:
    name: str
    window_s: float
    hop_s: float
    X: np.ndarray  # (n, 13) float32
    rec: np.ndarray  # (n,) recording index
    t: np.ndarray  # window end, s
    label: np.ndarray  # 1 / 0 / -1 (partial overlap, excluded from fitting)
    nontarget: np.ndarray
    emg_ok: np.ndarray
    imu_ok: np.ndarray
    recordings: list[str]
    subjects: np.ndarray  # (n_recordings,) subject number
    target_iv: list[np.ndarray]
    nontarget_iv: list[np.ndarray]

    def valid(self, modality: str) -> np.ndarray:
        return {"emg": self.emg_ok, "imu": self.imu_ok, "fusion": self.emg_ok & self.imu_ok}[modality]

    @property
    def subject(self) -> np.ndarray:
        return self.subjects[self.rec]


def manifests(partition: str | None = None) -> list[dict]:
    out = [json.loads(p.read_text()) for p in sorted((C.FEATURE_DIR / "manifest").glob("*.json"))]
    return [m for m in out if partition is None or m["partition"] == partition]


def load(name: str, partition: str) -> Windows:
    if partition == "test" and not LOCK.exists():
        raise PermissionError("the final evaluation cohort is locked until final_eval.py freezes the configuration")
    w, hop = C.WINDOWS[name]
    cols: dict[str, list] = {k: [] for k in ("X", "rec", "t", "label", "nontarget", "emg_ok", "imu_ok")}
    recs, subs, tiv, ntiv = [], [], [], []
    for m in manifests(partition):
        if m.get("excluded_reason"):
            continue
        rid = m["recording_id"]
        assert (partition == "dev") == C.is_dev(rid), rid
        with np.load(C.FEATURE_DIR / m["subject"] / f"{rid}.npz") as z:
            keep = (z[f"{name}_emg_ok"] | z[f"{name}_imu_ok"]) & ~z[f"{name}_artifact"] & ~z[f"{name}_warmup"]
            i = len(recs)
            recs.append(rid)
            subs.append(C.subject_number(rid))
            tiv.append(z["iv_target"])
            ntiv.append(z["iv_nontarget"])
            cols["X"].append(z[f"{name}_X"][keep])
            cols["t"].append(z[f"{name}_t_end"][keep])
            cols["label"].append(Wn.labels(z[f"{name}_tfrac"][keep]))
            cols["nontarget"].append(z[f"{name}_nontarget"][keep])
            cols["emg_ok"].append(z[f"{name}_emg_ok"][keep])
            cols["imu_ok"].append(z[f"{name}_imu_ok"][keep])
            cols["rec"].append(np.full(int(keep.sum()), i, np.int32))
    cat = {k: np.concatenate(v) for k, v in cols.items()}
    return Windows(name, w, hop, cat["X"], cat["rec"], cat["t"], cat["label"], cat["nontarget"], cat["emg_ok"],
                   cat["imu_ok"], recs, np.asarray(subs), tiv, ntiv)
