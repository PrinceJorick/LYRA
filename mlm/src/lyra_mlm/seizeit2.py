"""SeizeIT2 (OpenNeuro ds005873) BIDS adapter.

Layout read (verified on the v1.1.0 files): sub-XXX/ses-YY/{emg,mov,eeg}/<stem>_{emg.edf,mov.edf,
events.tsv} plus JSON sidecars. There is no channels.tsv, so channels, units and sample rates
come from each EDF header (the JSON SamplingFrequency is cross-checked and differences noted).

- EMG: 'EMG SD', raw sEMG on the left deltoid -> EmgSignalType.RAW, native rate kept.
- IMU: the 'ECGEMG SD' module (left chest, same device as the EMG) ACC X/Y/Z + GYR A/B/C.
  Stage 3.5 scale rule (Step 1 investigation, Train/CV only, 2026-11-22): v1.1.0 stores every
  ACC and GYR value as sensor value / D, with D = 6 or 2 per recording, whatever the EDF label
  ('g') says. Both sensors clip at exactly 8/D. assess_imu_scale() picks D from the recording's
  own resting gravity, or rejects the recording's IMU (it then loads as an untrusted stream, so
  every window rates it INVALID). Accepted IMU is converted to g (ACC, +-8 g full scale) and deg/s
  (GYR, 125*D deg/s per stored unit, +-1000 deg/s; the 125 is inferred, medium confidence).
  v1.0.0 exported ACC Z as a copy of Y; the v1.1.0 Z re-export is garbage in some recordings.
- Timing: EMG and MOV EDF start times are anonymised differently, so both are assumed to start
  at the recording onset (t = 0) that events.tsv onsets refer to. Not proof of physical sync.
- A recording missing an EMG/MOV file or the expected channels loads that modality as None and
  lists the reason in RecordingInfo.missing.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .annotations import SEIZURE_LIKE, Annotation, AnnotationKind
from .datasets import SOURCES, RecordingInfo
from .edf import EdfError, read_edf, read_edf_header
from .signals import IMU_CHANNELS, DataOrigin, EmgSignalType, Modality, Recording, SignalStream

EMG_LABEL = "EMG SD"
IMU_LABELS = ("ECGEMG SD ACC X", "ECGEMG SD ACC Y", "ECGEMG SD ACC Z",
              "ECGEMG SD GYR A", "ECGEMG SD GYR B", "ECGEMG SD GYR C")
EMG_PLACEMENT = "left_deltoid"
IMU_PLACEMENT = "chest_upper_back"  # dataset tag; the left-chest ECGEMG module is the one used
SOURCE_IMU_UNITS = ("g [EDF label; stored as value / D]",) * 3 + ("g [EDF label; not a gyro unit]",) * 3
IMU_UNITS = ("g",) * 3 + ("deg/s",) * 3  # after conversion

# --- Stage 3.5 IMU scale rule (approved 2026-11-22) -------------------------------------------
ACC_FULL_SCALE_G, GYR_FULL_SCALE_DPS = 8.0, 1000.0
GYR_DPS_PER_UNIT = 125.0  # per stored unit before /D; inferred (medium confidence), not documented
SATURATION_LIMITS = {**{c: (-ACC_FULL_SCALE_G, ACC_FULL_SCALE_G) for c in IMU_CHANNELS[:3]},
                     **{c: (-GYR_FULL_SCALE_DPS, GYR_FULL_SCALE_DPS) for c in IMU_CHANNELS[3:]}}
D_REST_WINDOWS = {6: (0.155, 0.180), 2: (0.45, 0.60)}  # stored resting |g| for each divisor
MAX_REST_SPREAD = {6: 1.15, 2: 1.20}  # p95/p5 of quiet-block |g| (Step 1 "borderline" cut)
REST_G_RANGE = (0.93, 1.08)  # resting |g| after conversion, in g
MAX_AXIS_DISAGREEMENT = 1.10  # resting |g| by gravity-carrying axis, max/min
STORED_CEILINGS = (1.333, 4.0)  # 8/6 and 8/2 at the stored 0.001 resolution


@dataclass(frozen=True)
class ImuScale:
    d: int | None  # None: the recording's IMU is rejected
    category: str  # "ACCEPTED" or the failed check
    reason: str
    evidence: Mapping[str, object]

    @property
    def accepted(self) -> bool:
        return self.d is not None


def rest_gravity(acc: np.ndarray, rate_hz: float) -> dict | None:
    """Resting |g| from the quietest 10% of 1 s blocks (all-zero, i.e. zero-filled, blocks excluded):
    median, p5/p95, and the median per gravity-carrying axis (axes with >= 10 blocks)."""
    b = int(round(rate_hz))
    nb = len(acc) // b
    blocks = acc[: nb * b].reshape(nb, b, 3)
    nonzero = ~(blocks == 0).all(axis=(1, 2))
    if nonzero.sum() < 20:
        return None
    mean, activity = blocks.mean(axis=1), blocks.std(axis=1).sum(axis=1)
    quiet = nonzero & (activity <= np.percentile(activity[nonzero], 10))
    g = np.linalg.norm(mean[quiet], axis=1)
    dom = np.abs(mean[quiet]).argmax(axis=1)
    by_axis = {ax: float(np.median(g[dom == k])) for k, ax in enumerate("XYZ") if (dom == k).sum() >= 10}
    p5, p50, p95 = (float(v) for v in np.percentile(g, [5, 50, 95]))
    return {"rest": p50, "p5": p5, "p95": p95, "by_axis": by_axis,
            "zero_filled_fraction": float(1 - nonzero.mean()), "quiet_blocks": int(quiet.sum())}


def assess_imu_scale(acc: np.ndarray, gyr: np.ndarray, acc_units: tuple[str, ...],
                     z_header_bounds: tuple[float, float], rate_hz: float) -> ImuScale:
    """The approved per-recording rule. Uses only this recording's samples and header: no labels,
    nothing fitted across recordings. acc/gyr are the EDF physical values (n, 3)."""
    acc, gyr = np.round(acc, 3), np.round(gyr, 3)  # undo the header affine map: stored at 0.001
    ev: dict = {}

    def reject(category: str, reason: str) -> ImuScale:
        return ImuScale(None, category, reason, ev)

    if any(u != "g" for u in acc_units):
        return reject("UNIT", f"ACC unit {acc_units} is not 'g'")
    if max(abs(b) for b in z_header_bounds) >= 30:
        return reject("Z_CORRUPT", f"Z header bound {z_header_bounds} (failed v1.1.0 Z re-export)")
    ev["z_rail_fraction"] = float(np.mean(np.abs(acc[:, 2]) >= 32.767))
    if ev["z_rail_fraction"] > 0.01:
        return reject("Z_CORRUPT", f"{ev['z_rail_fraction']:.1%} of Z samples at +-32.767 (failed v1.1.0 Z re-export)")
    both = np.column_stack([acc, gyr])
    at_ceiling = np.isin(np.abs(both), STORED_CEILINGS).mean(axis=0)
    ev["max_fraction_at_a_ceiling"] = float(at_ceiling.max())
    if at_ceiling.max() > 0.5:
        return reject("STUCK_AXIS", f"{IMU_CHANNELS[int(at_ceiling.argmax())]} stuck at the ceiling "
                                    f"in {at_ceiling.max():.0%} of samples")
    r = rest_gravity(acc, rate_hz)
    if r is None:
        return reject("TOO_LITTLE_DATA", "fewer than 20 non-zero 1 s blocks")
    ev.update(r)
    d = next((d for d, (lo, hi) in D_REST_WINDOWS.items() if lo <= r["rest"] <= hi), None)
    if d is None:
        return reject("D_TEST", f"resting |g| {r['rest']:.4f} fits neither D=6 {D_REST_WINDOWS[6]} "
                                f"nor D=2 {D_REST_WINDOWS[2]}")
    ev["d_candidate"] = d
    ceiling = round(ACC_FULL_SCALE_G / d, 3)
    if np.abs(both).max() > ceiling + 5e-4:
        return reject("OVER_CEILING", f"values above the 8/D ceiling {ceiling} for D={d}")
    axes = list(r["by_axis"].values())
    if len(axes) >= 2 and max(axes) / min(axes) > MAX_AXIS_DISAGREEMENT:
        return reject("AXIS_DISAGREEMENT", f"resting |g| differs by gravity axis {r['by_axis']}")
    if r["p95"] / r["p5"] >= MAX_REST_SPREAD[d]:
        return reject("REST_SPREAD", f"quiet-block |g| p95/p5 {r['p95'] / r['p5']:.3f} >= {MAX_REST_SPREAD[d]}")
    ev["rest_g_converted"] = r["rest"] * d
    if not REST_G_RANGE[0] <= ev["rest_g_converted"] <= REST_G_RANGE[1]:
        return reject("POST_CONVERSION", f"converted resting |g| {ev['rest_g_converted']:.3f} outside {REST_G_RANGE}")
    return ImuScale(d, "ACCEPTED", "", ev)


def convert_imu(acc: np.ndarray, gyr: np.ndarray, d: int) -> np.ndarray:
    """Stored values -> g and deg/s. Samples at the stored 8/D ceiling become exactly full scale, so
    the window saturation check (limits +-8 g / +-1000 deg/s) sees them."""
    stored = np.round(np.column_stack([acc, gyr]), 3)
    out = stored * np.array([d] * 3 + [GYR_DPS_PER_UNIT * d] * 3)
    full = np.array([ACC_FULL_SCALE_G] * 3 + [GYR_FULL_SCALE_DPS] * 3)
    at = np.isclose(np.abs(stored), round(ACC_FULL_SCALE_G / d, 3), rtol=0, atol=1e-9)
    out[at] = (np.sign(stored) * full)[at]
    return out


class MissingRecordingData(ValueError):
    pass


class MalformedAnnotation(ValueError):
    """A seizure-like annotation row cannot be parsed, so the recording cannot be labelled safely."""


class SeizeIT2Mapper:
    """eventType -> AnnotationKind. Only sz_foc_f2b is FBTCS. Unknown labels are UNMAPPED."""

    def map_label(self, raw_label: str) -> AnnotationKind:
        if raw_label == "sz_foc_f2b":
            return AnnotationKind.SEIZURE_FBTCS
        if raw_label == "sz_gen_m_tonicClonic":
            return AnnotationKind.SEIZURE_GTCS
        if raw_label == "sz" or raw_label.startswith("sz_"):
            return AnnotationKind.SEIZURE_OUT_OF_SCOPE
        if raw_label == "bckg":
            return AnnotationKind.BACKGROUND
        if raw_label == "impd":
            return AnnotationKind.ARTIFACT  # impedance check, likely artifactual
        return AnnotationKind.UNMAPPED


class SeizeIT2Adapter:
    source = SOURCES["seizeit2"]

    def __init__(self, root: Path, mapper: SeizeIT2Mapper | None = None):
        self.root = Path(root)
        self.mapper = mapper or SeizeIT2Mapper()

    def _paths(self, recording_id: str) -> dict[str, Path]:
        sub, ses = recording_id.split("_")[:2]
        base = self.root / sub / ses
        return {"emg": base / "emg" / f"{recording_id}_emg.edf", "mov": base / "mov" / f"{recording_id}_mov.edf",
                "events": base / "eeg" / f"{recording_id}_events.tsv"}

    def recording_ids(self) -> list[str]:
        return sorted(p.name.removesuffix("_events.tsv") for p in self.root.glob("sub-*/ses-*/eeg/*_events.tsv"))

    def _rate(self, path: Path, labels: tuple[str, ...], notes: list[str], missing: list[str],
              name: str) -> float | None:
        if not path.exists():
            missing.append(f"{name}: no file")
            return None
        try:
            hdr = read_edf_header(path)
        except EdfError as e:
            missing.append(f"{name}: unreadable EDF ({e})")
            return None
        absent = [lb for lb in labels if lb not in hdr.labels]
        if absent:
            missing.append(f"{name}: channels missing {absent}")
            return None
        rates = {hdr.rate_hz(lb) for lb in labels}
        if len(rates) != 1:
            missing.append(f"{name}: channels have different rates {sorted(rates)}")
            return None
        rate = rates.pop()
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            declared = json.loads(sidecar.read_text()).get("SamplingFrequency")
            if declared is not None and float(declared) != rate:
                notes.append(f"{name}: JSON says {declared} Hz, EDF header gives {rate:g} Hz (header used)")
        return rate

    def info(self, recording_id: str) -> RecordingInfo:
        p, notes, missing = self._paths(recording_id), [], []
        emg_rate = self._rate(p["emg"], (EMG_LABEL,), notes, missing, "emg")
        imu_rate = self._rate(p["mov"], IMU_LABELS, notes, missing, "imu")
        rows = self._rows(recording_id)
        duration = float(rows[0]["recordingDuration"]) if rows else None
        return RecordingInfo(
            dataset_id="seizeit2", participant_id=recording_id.split("_")[0], recording_id=recording_id,
            origin=DataOrigin.DATASET, placement=f"emg={EMG_PLACEMENT}; imu={IMU_PLACEMENT}",
            emg_signal_type=EmgSignalType.RAW if emg_rate else None, emg_rate_hz=emg_rate, imu_rate_hz=imu_rate,
            duration_s=duration, notes="; ".join(notes), emg_placement=EMG_PLACEMENT,
            imu_placement=IMU_PLACEMENT, missing=tuple(missing),
        )

    def list_recordings(self) -> list[RecordingInfo]:
        return [self.info(r) for r in self.recording_ids()]

    def _rows(self, recording_id: str) -> list[dict[str, str]]:
        with open(self._paths(recording_id)["events"], newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))

    def parse_annotations(self, recording_id: str) -> tuple[tuple[Annotation, ...], list[str]]:
        """(annotations, skipped-row notes). Malformed non-seizure rows (e.g. negative-duration 'impd'
        rows in the v1.1.0 files) are skipped and reported; a malformed seizure-like row raises,
        because the recording can then not be labelled safely."""
        good, skipped = [], []
        for i, r in enumerate(self._rows(recording_id)):
            kind = self.mapper.map_label(r["eventType"])
            try:
                good.append(Annotation(f"{recording_id}#{i}", float(r["onset"]), float(r["duration"]), kind,
                                       r["eventType"], source="SeizeIT2 events.tsv"))
            except ValueError as e:
                if kind in SEIZURE_LIKE:
                    raise MalformedAnnotation(f"{recording_id} row {i} ({r['eventType']}): {e}") from e
                skipped.append(f"{recording_id} row {i}: {r['eventType']} onset {r['onset']} "
                               f"duration {r['duration']} skipped (malformed)")
        return tuple(good), skipped

    def annotations(self, recording_id: str) -> tuple[Annotation, ...]:
        return self.parse_annotations(recording_id)[0]

    def _imu(self, path: Path, rate_hz: float) -> tuple[SignalStream, ImuScale]:
        hdr, sig = read_edf(path, IMU_LABELS)
        ts = sig[IMU_LABELS[0]][0]
        acc = np.column_stack([sig[lb][1] for lb in IMU_LABELS[:3]])
        gyr = np.column_stack([sig[lb][1] for lb in IMU_LABELS[3:]])
        z = hdr.index(IMU_LABELS[2])
        scale = assess_imu_scale(acc, gyr, tuple(hdr.units[hdr.index(lb)] for lb in IMU_LABELS[:3]),
                                 (hdr.phys_min[z], hdr.phys_max[z]), rate_hz)
        prov = (f"SeizeIT2 {path.name} ECGEMG SD module",)
        if not scale.accepted:  # raw values kept as delivered; the stream is never used
            return SignalStream(Modality.IMU, ts, np.column_stack([acc, gyr]), rate_hz, IMU_CHANNELS,
                                SOURCE_IMU_UNITS, DataOrigin.DATASET, provenance=prov,
                                untrusted=f"IMU rejected for the recording ({scale.category}): {scale.reason}"), scale
        note = (f"Stage 3.5 IMU scale: stored value x D, D={scale.d} (resting |g| {scale.evidence['rest']:.4f} -> "
                f"{scale.evidence['rest_g_converted']:.3f} g); ACC in g (+-8 g full scale); GYR x {GYR_DPS_PER_UNIT:g}*D "
                f"deg/s (+-1000 deg/s; {GYR_DPS_PER_UNIT:g} deg/s per unit inferred, medium confidence); values "
                f"rounded to the stored 0.001 resolution; samples at the 8/D ceiling set to exactly full scale")
        return SignalStream(Modality.IMU, ts, convert_imu(acc, gyr, scale.d), rate_hz, IMU_CHANNELS, IMU_UNITS,
                            DataOrigin.DATASET, provenance=prov + (note,)), scale

    def imu_scale(self, recording_id: str) -> ImuScale | None:
        """The Stage 3.5 decision for a recording's IMU, or None when it has no usable mov file."""
        info = self.info(recording_id)
        return self._imu(self._paths(recording_id)["mov"], info.imu_rate_hz)[1] if info.imu_rate_hz else None

    def load_recording(self, recording_id: str) -> Recording:
        info = self.info(recording_id)
        p = self._paths(recording_id)
        emg = imu = None
        if info.emg_rate_hz:
            _, sig = read_edf(p["emg"], (EMG_LABEL,))
            ts, v = sig[EMG_LABEL]
            emg = SignalStream(Modality.EMG, ts, v, info.emg_rate_hz, ("raw",), ("uV",), DataOrigin.DATASET,
                               EmgSignalType.RAW, provenance=(f"SeizeIT2 {p['emg'].name} '{EMG_LABEL}'",))
        if info.imu_rate_hz:
            imu = self._imu(p["mov"], info.imu_rate_hz)[0]
        if emg is None and imu is None:
            raise MissingRecordingData(f"{recording_id}: {'; '.join(info.missing)}")
        return Recording(info.participant_id, recording_id, DataOrigin.DATASET, emg, imu,
                         placement=info.placement, annotations=self.annotations(recording_id))
