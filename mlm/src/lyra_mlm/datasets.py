"""Dataset source registry, metadata schema, adapter contract and integrity checks.

No dataset has been downloaded, opened or processed. Facts are tagged [VERIFIED: ref] (checked
in context.md against primary sources) or [NOT VERIFIED]. Exact counts must be re-checked
against the actual release files before anyone relies on them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .signals import DataOrigin, EmgSignalType, Recording, check_timestamps


@dataclass(frozen=True)
class DatasetSource:
    source_id: str
    name: str
    url: str
    purpose: str
    access: str
    license: str
    facts: Mapping[str, str]
    differences_from_lyra: tuple[str, ...]


SOURCES: Mapping[str, DatasetSource] = {
    "seizeit2": DatasetSource(
        source_id="seizeit2",
        name="SeizeIT2 (OpenNeuro ds005873)",
        url="https://openneuro.org/datasets/ds005873",
        purpose="Annotated FBTCS examples with wearable EMG, accelerometer and gyroscope. Primary seizure dataset.",
        access="Open download, no registration [VERIFIED: R2]",
        license="CC0 [VERIFIED: R2, dataset_description.json]",
        facts={
            "version": "1.1.0, DOI 10.18112/openneuro.ds005873.v1.1.0 [VERIFIED: R2]",
            "participants": "125 [VERIFIED: R1]",
            "recording_hours": "~11,640 h in the v1.1.0 README [VERIFIED: R1, R2]",
            "fbtcs_events": "55 [VERIFIED: R1, R2]; must be re-counted from the events files",
            "population": "Focal epilepsy only, in-hospital EMU recordings [VERIFIED: R1]",
            "emg": "1 channel, left deltoid, released at 256 Hz (device sampled 250 Hz) [VERIFIED: R1, R2]",
            "movement": "ACC + GYRO, 25 Hz, 12 channels over two modules: back of neck and left chest "
                        "[VERIFIED: R2]",
            "format": "BIDS 1.8.0, EDF + JSON sidecars + events TSV [VERIFIED: R1, R2]",
            "suggested_split": "sub-001..sub-096 train, sub-097..sub-125 held out [VERIFIED: R1]; "
                               "inventory not yet checked",
        },
        differences_from_lyra=(
            "EMG is a sampled electrode signal (RAW path), not a MyoWare ENV output: "
            "ENV-trained features cannot use it without an explicit, documented conversion.",
            "256 Hz EMG covers only up to 128 Hz, below the usual sEMG band.",
            "IMU is on the neck and chest, not the upper arm: IMU features may not transfer.",
            "FBTCS only; it does not validate GTCS detection on its own.",
            "No out-of-hospital activities, so real-life false alarms cannot be estimated from it.",
        ),
    ),
    "ninapro_db2": DatasetSource(
        source_id="ninapro_db2",
        name="NinaPro DB2",
        url="https://ninapro.hevs.ch/",
        purpose="Supplemental non-seizure muscle activity and movement-feature exploration.",
        access="[NOT VERIFIED] check the site's registration and terms before download",
        license="[NOT VERIFIED]",
        facts={"all": "[NOT VERIFIED] channel count, placement, sampling rates and hardware must be read "
                      "from the official documentation"},
        differences_from_lyra=(
            "[NOT VERIFIED] Expected multi-electrode forearm sEMG for hand/wrist movements, unlike LYRA's "
            "single deltoid channel; confirm before use.",
        ),
    ),
    "ninapro_db7": DatasetSource(
        source_id="ninapro_db7",
        name="NinaPro DB7",
        url="https://ninapro.hevs.ch/instructions/DB7.html",
        purpose="Supplemental non-seizure muscle activity and movement-feature exploration.",
        access="[NOT VERIFIED] check the site's registration and terms before download",
        license="[NOT VERIFIED]",
        facts={"all": "[NOT VERIFIED] channel count, placement, sampling rates, IMU content and hardware "
                      "must be read from the DB7 instructions page"},
        differences_from_lyra=(
            "[NOT VERIFIED] Expected forearm placement and different acquisition hardware; confirm before use.",
        ),
    ),
    "lyra_pod": DatasetSource(
        source_id="lyra_pod",
        name="LYRA pod recordings (future)",
        url="",
        purpose="Consent-based recordings from LYRA hardware, deltoid placement, including scripted "
                "hard-negative activities.",
        access="Internal; requires an approved consent protocol before any collection",
        license="Not applicable (participant data; governed by consent terms)",
        facts={"status": "No recordings exist yet", "format": "docs/POD_DATA_PROTOCOL.md, schema v1 (lyra_mlm.pod)"},
        differences_from_lyra=(),
    ),
}


@dataclass(frozen=True)
class RecordingInfo:
    dataset_id: str
    participant_id: str
    recording_id: str
    origin: DataOrigin
    placement: str
    emg_signal_type: EmgSignalType | None
    emg_rate_hz: float | None  # native rate as released
    imu_rate_hz: float | None
    duration_s: float | None = None
    sha256: str | None = None  # of the source file(s), when published
    notes: str = ""
    emg_placement: str = ""
    imu_placement: str = ""
    missing: tuple[str, ...] = ()  # e.g. ("emg: no EMG file",); such modalities load as None


class DatasetAdapter(Protocol):
    """One per dataset. Converts source files to Recording without filtering, resampling or
    normalizing, and keeps native rates, timestamps and verbatim annotation labels."""

    source: DatasetSource

    def list_recordings(self) -> Sequence[RecordingInfo]: ...

    def load_recording(self, recording_id: str) -> Recording: ...


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def validate_recording(rec: Recording) -> list[str]:
    """Human-readable integrity issues; an empty list means none were found."""
    issues = []
    spans = []
    for s in (rec.emg, rec.imu):
        if s is None:
            continue
        tr = check_timestamps(s.timestamps, s.nominal_rate_hz)
        if not tr.is_clean:
            issues.append(f"{s.modality}: {tr.missing} missing, {tr.duplicates} duplicate, "
                          f"{tr.out_of_order} out-of-order, {tr.irregular} irregular timestamps, {len(tr.gaps)} gaps")
        if tr.measured_rate_hz is not None and abs(tr.measured_rate_hz / s.nominal_rate_hz - 1) > 0.01:
            issues.append(f"{s.modality}: measured {tr.measured_rate_hz:.4g} Hz vs nominal {s.nominal_rate_hz:g} Hz")
        ok = s.usable
        if ok.any():
            spans.append((s.timestamps[ok].min(), s.timestamps[ok].max()))
    if rec.placement == "NOT SPECIFIED":
        issues.append("sensor placement not specified")
    if spans:
        lo, hi = min(a for a, _ in spans), max(b for _, b in spans)
        issues += [f"annotation {a.annotation_id} lies outside the recorded span"
                   for a in rec.annotations if a.end_s < lo or a.onset_s > hi]
    return issues
