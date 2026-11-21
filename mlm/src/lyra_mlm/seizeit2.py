"""SeizeIT2 (OpenNeuro ds005873) BIDS adapter.

Layout read (verified on the v1.1.0 files): sub-XXX/ses-YY/{emg,mov,eeg}/<stem>_{emg.edf,mov.edf,
events.tsv} plus JSON sidecars. There is no channels.tsv, so channels, units and sample rates
come from each EDF header (the JSON SamplingFrequency is cross-checked and differences noted).

- EMG: 'EMG SD', raw sEMG on the left deltoid -> EmgSignalType.RAW, native rate kept.
- IMU: the 'ECGEMG SD' module (left chest, same device as the EMG) ACC X/Y/Z + GYR A/B/C.
  The EDF labels both 'g', but each recording uses its own storage scale: median |acc| (gravity)
  is ~1/2 or ~1/6 in valid recordings, and pinned at 32.768 (or zero, or ~1.34) in broken ones.
  GYR carries the same per-recording factor (verified by rigid-body kinematics, see
  ml/seizeit2/gyro_calibration.py). calibrate_imu() divides ACC and GYR by the recording's
  gravity estimate (ACC -> g) and converts GYR with GYRO_DPS_PER_UNIT (-> deg/s, estimated);
  recordings outside ACC_SCALE_VALID get every IMU sample flagged INVALID.
- Timing: EMG and MOV EDF start times are anonymised differently, so both are assumed to start
  at the recording onset (t = 0) that events.tsv onsets refer to. Not proof of physical sync.
- A recording missing an EMG/MOV file or the expected channels loads that modality as None and
  lists the reason in RecordingInfo.missing.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .annotations import SEIZURE_LIKE, Annotation, AnnotationKind
from .datasets import SOURCES, RecordingInfo
from .edf import EdfError, read_edf, read_edf_header
from .signals import IMU_CHANNELS, DataOrigin, EmgSignalType, Modality, Recording, SampleFlag, SignalStream

EMG_LABEL = "EMG SD"
IMU_LABELS = ("ECGEMG SD ACC X", "ECGEMG SD ACC Y", "ECGEMG SD ACC Z",
              "ECGEMG SD GYR A", "ECGEMG SD GYR B", "ECGEMG SD GYR C")
EMG_PLACEMENT = "left_deltoid"
IMU_PLACEMENT = "chest_upper_back"  # dataset tag; the left-chest ECGEMG module is the one used
IMU_UNITS = ("g [per-recording gravity calibration]",) * 3 + ("deg/s [estimated, GYRO_DPS_PER_UNIT]",) * 3
# Median |acc| per recording, in the file's unit. Valid recordings sit at ~1/2 or ~1/6 (1 = already g).
ACC_SCALE_VALID = (0.12, 1.2)
# Calibrated gyro unit -> deg/s: median 125.4 (IQR 109-136) over 24 recordings, fitted from
# rigid-body kinematics with the identity axis map in all 24 (ml/seizeit2/gyro_calibration.py,
# models/seizeit2/evaluation/gyro_calibration.json). An estimate, not a datasheet value.
GYRO_DPS_PER_UNIT = 125.0


def calibrate_imu(values: np.ndarray, gyro_to_dps: float = GYRO_DPS_PER_UNIT
                  ) -> tuple[np.ndarray, np.ndarray | None, str]:
    """(values, source flags or None, provenance note) for the 6 IMU columns (ACC x3, GYR x3).

    ACC and GYR share the recording's storage scale, so both are divided by the median |acc|
    (gravity); GYR is then converted to deg/s. A recording whose gravity estimate is outside
    ACC_SCALE_VALID cannot be calibrated: values are returned unchanged and every sample is INVALID.
    """
    acc = values[:, :3]
    mag = np.linalg.norm(acc[np.isfinite(acc).all(axis=1)], axis=1)
    scale = float(np.median(mag)) if mag.size else float("nan")
    if not ACC_SCALE_VALID[0] <= scale <= ACC_SCALE_VALID[1]:
        return values, np.full(len(values), SampleFlag.INVALID, np.int8), (
            f"IMU rejected: median |acc| {scale:.4g} outside {ACC_SCALE_VALID} (pinned, zero or unknown scale)")
    out = values / scale
    out[:, 3:] *= gyro_to_dps
    return out, None, (f"IMU calibrated: ACC and GYR divided by median |acc| {scale:.4f} (gravity -> g); "
                       f"GYR x {gyro_to_dps:g} deg/s per unit")


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
            _, sig = read_edf(p["mov"], IMU_LABELS)
            ts = sig[IMU_LABELS[0]][0]
            vals, flags, note = calibrate_imu(np.column_stack([sig[lb][1] for lb in IMU_LABELS]))
            imu = SignalStream(Modality.IMU, ts, vals, info.imu_rate_hz, IMU_CHANNELS, IMU_UNITS, DataOrigin.DATASET,
                               flags=flags, provenance=(f"SeizeIT2 {p['mov'].name} ECGEMG SD module", note))
        if emg is None and imu is None:
            raise MissingRecordingData(f"{recording_id}: {'; '.join(info.missing)}")
        return Recording(info.participant_id, recording_id, DataOrigin.DATASET, emg, imu,
                         placement=info.placement, annotations=self.annotations(recording_id))
