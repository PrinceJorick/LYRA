"""Verified access to the git-annexed SeizeIT2 EDFs and to the prepared (Stage 3) windows.

pj-training-data/ds005873 is a DataLad clone whose EDF files are git-annex symlinks without
content. `fetch` downloads one file from the dataset's public bucket and checks it against the
SHA256 in the annex key. The clone itself is never modified. Reading EDFs, mapping labels and
parsing annotations is done by lyra_mlm (edf.py, seizeit2.py), not duplicated here.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lyra_mlm.annotations import AnnotationKind
from lyra_mlm.seizeit2 import SeizeIT2Adapter

from . import config as C

_KEY = re.compile(r"SHA256E-s(\d+)--([0-9a-f]{64})")


def annex_key(link: Path) -> tuple[int, str]:
    m = _KEY.search(os.readlink(link))
    if not m:
        raise ValueError(f"{link} is not a SHA256E git-annex link")
    return int(m.group(1)), m.group(2)


def fetch(link: Path, dest: Path, retries: int = 4) -> Path:
    """Download the annexed content of `link` to `dest`, verifying size and SHA256."""
    if link.exists():  # content already present locally (e.g. after `datalad get`)
        return link.resolve()
    size, sha = annex_key(link)
    url = f"{C.S3_BASE}/{link.relative_to(C.DATASET).as_posix()}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            h = hashlib.sha256()
            with urllib.request.urlopen(url, timeout=60) as r, dest.open("wb") as out:
                while chunk := r.read(1 << 20):
                    h.update(chunk)
                    out.write(chunk)
            if dest.stat().st_size != size or h.hexdigest() != sha:
                raise OSError(f"checksum mismatch for {url}")
            return dest
        except OSError:
            dest.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


# ---- Prepared windows ---------------------------------------------------------------------------

USABLE_CODES = (0, 1)  # lyra_mlm.prep.QUALITY_CODE: VALID=0, DEGRADED=1


@dataclass
class RecordingWindows:
    """One recording's Stage 3 windows, reduced to what training and evaluation need."""

    recording_id: str
    participant_id: str
    t_end: np.ndarray  # window end (s) = prediction timestamp
    label: np.ndarray  # 1 positive, 0 negative, -1 ignore (D14)
    evidence: np.ndarray  # lyra_mlm inference routing: both modalities usable, every primary feature available
    x: np.ndarray  # primary features (float32), NaN where unavailable
    fbtcs: list[tuple[float, float]] = field(default_factory=list)
    out_of_scope: list[tuple[float, float]] = field(default_factory=list)
    artifacts: list[tuple[float, float]] = field(default_factory=list)


def window_file(recording_id: str) -> Path:
    return C.PROCESSED / "windows" / recording_id.split("_")[0] / f"{recording_id}.npz"


def load_windows(recording_id: str, adapter: SeizeIT2Adapter) -> RecordingWindows | None:
    p = window_file(recording_id)
    if not p.exists():
        return None
    with np.load(p) as z:
        evidence = np.isin(z["overall"], USABLE_CODES) & z["primary_available"].all(axis=1)
        out = RecordingWindows(recording_id, recording_id.split("_")[0], z["end_s"].astype(np.float64),
                               z["label"].astype(np.int8), evidence, z["primary"])
    for a in adapter.annotations(recording_id):
        iv = (a.onset_s, a.end_s)
        if a.kind in (AnnotationKind.SEIZURE_FBTCS, AnnotationKind.SEIZURE_GTCS):
            out.fbtcs.append(iv)
        elif a.kind in (AnnotationKind.SEIZURE_OUT_OF_SCOPE, AnnotationKind.UNMAPPED):
            out.out_of_scope.append(iv)
        elif a.kind is AnnotationKind.ARTIFACT:
            out.artifacts.append(iv)
    return out
