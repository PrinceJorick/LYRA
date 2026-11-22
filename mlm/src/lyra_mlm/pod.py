"""LYRA pod recordings: on-disk session format and adapter (format spec: docs/POD_DATA_PROTOCOL.md).

A session directory holds:
- session.json: metadata;
- env.csv: t_s, mV[, flag] (MyoWare ENV, ADC-calibrated millivolts, 50 Hz);
- imu.csv: t_s, ax, ay, az (g), gx, gy, gz (deg/s)[, flag] (BMI270, 25 Hz);
- raw.csv (optional, data-collection firmware only): t_s, mV[, flag] (MyoWare RAW, e.g. 1000 Hz);
- events.tsv: onset, duration, eventType[, source] (protocol vocabulary).

Timestamps are seconds on the pod clock from session start, shared by every stream. Nothing is
filtered, resampled or normalized here. Sessions without a consent-protocol reference are refused.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

from .annotations import HARD_NEGATIVE_CATEGORIES, Annotation, AnnotationKind
from .datasets import SOURCES, RecordingInfo
from .signals import IMU_CHANNELS, DataOrigin, EmgSignalType, Modality, Recording, SignalStream, check_timestamps

SCHEMA_VERSION = 1
ENV_UNIT, IMU_UNITS = "mV", ("g",) * 3 + ("deg/s",) * 3
PSEUDONYM = re.compile(r"^[a-z0-9][a-z0-9-]{1,31}$")  # e.g. "pod-p001": never a name
BACKGROUND_LABELS = frozenset({"rest_seated", "rest_supine", "rest_standing", "sleep", "free_living"})
ARTIFACT_ACTIVITIES = frozenset(HARD_NEGATIVE_CATEGORIES["sensor_artifact"])
ACTIVITY_LABELS = frozenset(a for acts in HARD_NEGATIVE_CATEGORIES.values() for a in acts)
REQUIRED = ("schema_version", "participant_id", "session_id", "consent_protocol", "pod_id", "firmware",
            "placement", "streams")


class PodFormatError(ValueError):
    pass


def map_event(label: str) -> tuple[AnnotationKind, str | None]:
    """Protocol vocabulary -> (kind, activity tag). Unknown labels are UNMAPPED, never negative."""
    if label == "sim_convulsion":
        return AnnotationKind.SIMULATED_CONVULSION, None
    if label in BACKGROUND_LABELS:
        return AnnotationKind.BACKGROUND, None
    if label in ARTIFACT_ACTIVITIES:
        return AnnotationKind.ARTIFACT, label
    if label in ACTIVITY_LABELS:
        return AnnotationKind.HARD_NEGATIVE, label
    return AnnotationKind.UNMAPPED, None


def _table(path: Path, columns: int) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """(timestamps, values (n, columns), flags or None) from a CSV with a header row."""
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if data.shape[1] not in (1 + columns, 2 + columns):
        raise PodFormatError(f"{path.name}: expected {1 + columns} or {2 + columns} columns, got {data.shape[1]}")
    flags = data[:, 1 + columns].astype(np.int8) if data.shape[1] == 2 + columns else None
    return data[:, 0], data[:, 1:1 + columns], flags


class PodAdapter:
    source = SOURCES["lyra_pod"]

    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, recording_id: str) -> Path:
        participant, session = recording_id.rsplit("_", 1)
        return self.root / participant / session

    def recording_ids(self) -> list[str]:
        return sorted(f"{p.parent.parent.name}_{p.parent.name}" for p in self.root.glob("*/*/session.json"))

    def meta(self, recording_id: str) -> dict:
        m = json.loads((self._dir(recording_id) / "session.json").read_text())
        missing = [k for k in REQUIRED if not m.get(k)]
        if missing:
            raise PodFormatError(f"{recording_id}: session.json missing {missing}")
        if m["schema_version"] != SCHEMA_VERSION:
            raise PodFormatError(f"{recording_id}: schema_version {m['schema_version']} != {SCHEMA_VERSION}")
        if not PSEUDONYM.match(m["participant_id"]):
            raise PodFormatError(f"{recording_id}: participant_id must be a pseudonym like 'pod-p001'")
        if not PSEUDONYM.match(m["session_id"]) or recording_id != f"{m['participant_id']}_{m['session_id']}":
            raise PodFormatError(f"{recording_id}: session_id must be like 's01' and both ids must match the folders")
        for name, unit in (("env", ENV_UNIT), ("raw", ENV_UNIT)):
            s = m["streams"].get(name)
            if s is not None and s.get("unit") != unit:
                raise PodFormatError(f"{recording_id}: {name} unit must be {unit!r} (ADC-calibrated), got {s.get('unit')!r}")
        imu = m["streams"].get("imu")
        if imu is not None and tuple(imu.get("units", ())) != IMU_UNITS:
            raise PodFormatError(f"{recording_id}: imu units must be {IMU_UNITS}")
        return m

    def info(self, recording_id: str) -> RecordingInfo:
        m, d = self.meta(recording_id), self._dir(recording_id)
        s = m["streams"]
        missing = [f"{k}: no file" for k in ("env", "imu") if not (d / f"{k}.csv").exists()]
        place = f"{m['placement']['side']} {m['placement']['muscle']}"
        return RecordingInfo(
            dataset_id="lyra_pod", participant_id=m["participant_id"], recording_id=recording_id,
            origin=DataOrigin.HARDWARE, placement=f"emg={place}; imu=pod on {place}",
            emg_signal_type=EmgSignalType.ENV if "env" in s else None,
            emg_rate_hz=s["env"]["rate_hz"] if "env" in s else None,
            imu_rate_hz=s["imu"]["rate_hz"] if "imu" in s else None, duration_s=m.get("duration_s"),
            notes=f"pod {m['pod_id']}, firmware {m['firmware']}, consent {m['consent_protocol']}",
            emg_placement=place, imu_placement=place, missing=tuple(missing))

    def _stream(self, recording_id: str, name: str) -> SignalStream | None:
        m, d = self.meta(recording_id), self._dir(recording_id)
        if name not in m["streams"] or not (d / f"{name}.csv").exists():
            return None
        spec = m["streams"][name]
        prov = (f"LYRA pod {m['pod_id']} firmware {m['firmware']}: {name}.csv",)
        if name == "imu":
            t, v, f = _table(d / "imu.csv", 6)
            return SignalStream(Modality.IMU, t, v, spec["rate_hz"], IMU_CHANNELS, IMU_UNITS, DataOrigin.HARDWARE,
                                flags=f, provenance=prov)
        t, v, f = _table(d / f"{name}.csv", 1)
        kind = EmgSignalType.ENV if name == "env" else EmgSignalType.RAW
        return SignalStream(Modality.EMG, t, v, spec["rate_hz"], (name,), (ENV_UNIT,), DataOrigin.HARDWARE, kind,
                            spec.get("bandwidth_hz"), f, prov)

    def annotations(self, recording_id: str) -> tuple[Annotation, ...]:
        path = self._dir(recording_id) / "events.tsv"
        if not path.exists():
            return ()
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        out = []
        for i, r in enumerate(rows):
            kind, activity = map_event(r["eventType"])
            out.append(Annotation(f"{recording_id}#{i}", float(r["onset"]), float(r["duration"]), kind, r["eventType"],
                                  activity, source=r.get("source") or "pod events.tsv"))
        return tuple(out)

    def load_recording(self, recording_id: str) -> Recording:
        """MyoWare ENV + IMU, as deployed. RAW (data-collection mode) comes from load_raw()."""
        info = self.info(recording_id)
        return Recording(info.participant_id, recording_id, DataOrigin.HARDWARE, self._stream(recording_id, "env"),
                         self._stream(recording_id, "imu"), placement=info.placement,
                         annotations=self.annotations(recording_id))

    def load_raw(self, recording_id: str) -> SignalStream | None:
        return self._stream(recording_id, "raw")


def check_session(adapter: PodAdapter, recording_id: str) -> list[str]:
    """Format and integrity problems (empty list = fine). Reports; never repairs."""
    try:
        m = adapter.meta(recording_id)
        streams = {n: adapter._stream(recording_id, n) for n in ("env", "imu", "raw")}
        anns = adapter.annotations(recording_id)
    except (PodFormatError, ValueError, KeyError, OSError) as e:
        return [str(e)]
    problems = [f"{n}: declared in session.json but no {n}.csv" for n in m["streams"] if streams.get(n) is None]
    for n, s in streams.items():
        if s is None:
            continue
        tr = check_timestamps(s.timestamps, s.nominal_rate_hz)
        if tr.duplicates or tr.out_of_order:
            problems.append(f"{n}: {tr.duplicates} duplicate and {tr.out_of_order} out-of-order timestamps")
        if tr.gaps:
            problems.append(f"{n}: {len(tr.gaps)} gaps, first at {tr.gaps[0][0]:.2f} s")
        if tr.measured_rate_hz and abs(tr.measured_rate_hz / s.nominal_rate_hz - 1) > 0.005:
            problems.append(f"{n}: measured {tr.measured_rate_hz:.3f} Hz vs declared {s.nominal_rate_hz:g} Hz")
        if not s.usable.any():
            problems.append(f"{n}: no usable samples")
    problems += [f"events.tsv row {a.annotation_id.split('#')[1]}: unknown eventType {a.raw_label!r} (UNMAPPED)"
                 for a in anns if a.kind is AnnotationKind.UNMAPPED]
    return problems
