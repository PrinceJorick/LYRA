"""BIDS recording matching and annotation parsing for SeizeIT2 (read-only).

EMG, MOV and events files are matched on their parsed (subject, session, task, run) entities,
never on directory order or approximate names.
"""

from __future__ import annotations

import collections
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from lyra_mlm.seizeit2 import SeizeIT2Adapter

from . import config as C

_FILE = re.compile(r"^(sub-\d{3})_(ses-\d{2})_(task-[A-Za-z0-9]+)_(run-\d+)_(emg|mov|events)\.(edf|tsv)$")
_FOLDER = {"emg": "emg", "mov": "mov", "events": "eeg"}


@dataclass
class RecordingFiles:
    recording_id: str
    subject: str
    emg: Path | None = None
    mov: Path | None = None
    events: Path | None = None
    problems: list[str] = field(default_factory=list)


def match_recordings(root: Path = C.DATASET) -> tuple[list[RecordingFiles], list[str]]:
    """All recordings keyed by BIDS entities, plus dataset-level problems (duplicates, misplaced files)."""
    found: dict[str, RecordingFiles] = {}
    seen: collections.Counter = collections.Counter()
    problems = []
    for kind, folder in _FOLDER.items():
        for p in sorted(root.glob(f"sub-*/ses-*/{folder}/*_{kind}.*")):
            m = _FILE.match(p.name)
            if not m or m.group(5) != kind:
                continue
            sub, ses, task, run = m.group(1, 2, 3, 4)
            if p.parent.parent.name != ses or p.parent.parent.parent.name != sub:
                problems.append(f"{p.relative_to(root)}: entities do not match its folder")
                continue
            rid = f"{sub}_{ses}_{task}_{run}"
            seen[(rid, kind)] += 1
            rec = found.setdefault(rid, RecordingFiles(rid, sub))
            setattr(rec, kind, p)
    problems += [f"{rid}: {n} {kind} files" for (rid, kind), n in seen.items() if n > 1]
    for rec in found.values():
        for kind in ("emg", "mov", "events"):
            if getattr(rec, kind) is None:
                rec.problems.append(f"no {kind} file")
    return [found[k] for k in sorted(found)], problems


@dataclass
class Intervals:
    target: list[tuple[float, float, str]]
    nontarget: list[tuple[float, float, str]]  # every other seizure code: kept as hard negatives
    artifact: list[tuple[float, float, str]]
    other: list[tuple[float, float, str]]  # bckg and anything else (never a seizure)
    skipped: list[str]
    unmapped_seizure_codes: list[str]
    recording_duration_s: float | None


def parse_events(recording_id: str, root: Path = C.DATASET) -> Intervals:
    """Uses lyra_mlm's row parser (it skips the 5 malformed negative-duration impd rows of v1.1.0 and
    raises on malformed seizure rows); codes are mapped here, explicitly."""
    ad = SeizeIT2Adapter(root)
    anns, skipped = ad.parse_annotations(recording_id)
    rows = ad._rows(recording_id)
    out = Intervals([], [], [], [], skipped, [], float(rows[0]["recordingDuration"]) if rows else None)
    for a in anns:
        iv = (a.onset_s, a.onset_s + a.duration_s, a.raw_label)
        if a.raw_label in C.TARGET_CODES:
            out.target.append(iv)
        elif a.raw_label.startswith("sz"):
            out.nontarget.append(iv)
        elif a.raw_label in C.ARTIFACT_CODES:
            out.artifact.append(iv)
        else:
            out.other.append(iv)
            if a.raw_label not in C.BACKGROUND_CODES:
                out.unmapped_seizure_codes.append(a.raw_label)
    return out


def sidecar(path: Path | None) -> dict:
    if path is None:
        return {}
    j = path.with_suffix(".json")
    return json.loads(j.read_text()) if j.exists() else {}
