"""SeizeIT2 (BIDS, OpenNeuro ds005873) access: run index, annotations, verified fetch, EDF+C reading.

The local clone in pj-training-data/ds005873 is a DataLad dataset whose EDF files are git-annex
symlinks. Their content is fetched from the dataset's public bucket into a temporary file,
checked against the SHA256 embedded in the annex key, read, and deleted. The clone itself is
never modified.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lyra_mlm.annotations import Annotation, AnnotationKind

from . import config as C

# ---- Annotations -------------------------------------------------------------------------------


class SeizeIT2Mapper:
    """Raw SeizeIT2 eventType -> lyra_mlm AnnotationKind (implements lyra_mlm AnnotationMapper)."""

    def map_label(self, raw_label: str) -> AnnotationKind:
        if raw_label in C.TARGET_LABELS:
            return AnnotationKind.SEIZURE_FBTCS
        if raw_label.startswith("sz_gen") and "tonicClonic" in raw_label:
            return AnnotationKind.SEIZURE_GTCS  # none in this release; mapped for completeness
        if raw_label.startswith("sz"):
            return AnnotationKind.SEIZURE_OUT_OF_SCOPE
        if raw_label == "impd":
            return AnnotationKind.ARTIFACT  # impedance check: recorded but artefactual
        return AnnotationKind.UNMAPPED


TARGET_KINDS = (AnnotationKind.SEIZURE_FBTCS, AnnotationKind.SEIZURE_GTCS)
_MAPPER = SeizeIT2Mapper()


@dataclass(frozen=True)
class Run:
    subject: str
    run: str  # e.g. "sub-001_ses-01_task-szMonitoring_run-01"
    duration_s: int
    emg_path: Path | None  # annex symlink in the clone (content usually absent)
    mov_path: Path | None
    annotations: tuple[Annotation, ...]
    issues: tuple[str, ...] = ()  # annotation rows dropped as corrupt, kept for the dataset report

    def intervals(self, kinds: tuple[AnnotationKind, ...]) -> list[tuple[float, float]]:
        return [(a.onset_s, a.end_s) for a in self.annotations if a.kind in kinds]


def list_runs(root: Path = C.DATASET) -> list[Run]:
    runs = []
    for ev in sorted(root.glob("sub-*/ses-*/eeg/*_events.tsv")):
        sess = ev.parent.parent
        tag = ev.name.removesuffix("_events.tsv")
        with ev.open() as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        if not rows:
            raise ValueError(f"{ev}: no rows (every SeizeIT2 run has at least a bckg row)")
        anns, issues = [], []
        for i, r in enumerate(rows):
            if r["eventType"] == "bckg":
                continue  # "bckg" marks seizure-free epochs; background is the absence of events
            kind = _MAPPER.map_label(r["eventType"])
            if kind is AnnotationKind.UNMAPPED:
                raise ValueError(f"{ev}: unmapped eventType {r['eventType']!r}")
            if float(r["duration"]) < 0:
                if kind is not AnnotationKind.ARTIFACT:
                    raise ValueError(f"{ev}: negative duration on a seizure row")
                issues.append(f"{tag} row {i}: {r['eventType']} with negative duration {r['duration']} dropped")
                continue
            anns.append(Annotation(f"{tag}#{i}", float(r["onset"]), float(r["duration"]), kind,
                                   raw_label=r["eventType"], source="SeizeIT2 vEEG review"))
        emg = sess / "emg" / f"{tag}_emg.edf"
        mov = sess / "mov" / f"{tag}_mov.edf"
        runs.append(Run(sess.parent.name, tag, round(float(rows[0]["recordingDuration"])),
                        emg if os.path.lexists(emg) else None,
                        mov if os.path.lexists(mov) else None, tuple(anns), tuple(issues)))
    return runs


# ---- Verified fetch ----------------------------------------------------------------------------

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


# ---- EDF+C reader ------------------------------------------------------------------------------


@dataclass(frozen=True)
class EdfSignal:
    label: str
    unit: str
    fs: float
    values: np.ndarray  # float64 physical units
    saturated: np.ndarray  # bool, sample at the digital min/max rail


def read_edf(path: Path, labels: tuple[str, ...]) -> dict[str, EdfSignal]:
    """Read the requested signals of an EDF/EDF+C file. Missing labels are simply absent.

    Raises on EDF+D (discontinuous) files or unequal record layout, which this pipeline does not
    handle; all SeizeIT2 files inspected are EDF+C with 1 s records.
    """
    with path.open("rb") as fh:
        hdr = fh.read(256).decode("latin1")
        ns = int(hdr[252:256])
        if hdr[192:197] == "EDF+D":
            raise ValueError(f"{path}: EDF+D is not supported")
        n_rec, rec_dur = int(hdr[236:244]), float(hdr[244:252])
        sh = fh.read(256 * ns).decode("latin1")
        data = np.fromfile(fh, dtype="<i2")

    def field(off: int, width: int) -> list[str]:
        return [sh[off + i * width: off + (i + 1) * width].strip() for i in range(ns)]

    widths = [("label", 16), ("transducer", 80), ("unit", 8), ("pmin", 8), ("pmax", 8),
              ("dmin", 8), ("dmax", 8), ("prefilter", 80), ("nsamp", 8), ("reserved", 32)]
    f, off = {}, 0
    for name, w in widths:
        f[name] = field(off, w)
        off += w * ns
    nsamp = np.array([int(x) for x in f["nsamp"]])
    rec_len = int(nsamp.sum())
    if data.size < n_rec * rec_len:
        raise ValueError(f"{path}: truncated ({data.size} < {n_rec * rec_len} samples)")
    recs = data[: n_rec * rec_len].reshape(n_rec, rec_len)
    starts = np.concatenate([[0], np.cumsum(nsamp)])
    out = {}
    for i, lab in enumerate(f["label"]):
        if lab not in labels:
            continue
        dig = recs[:, starts[i]:starts[i + 1]].reshape(-1)
        pmin, pmax, dmin, dmax = (float(f[k][i]) for k in ("pmin", "pmax", "dmin", "dmax"))
        gain = (pmax - pmin) / (dmax - dmin)
        out[lab] = EdfSignal(lab, f["unit"][i], nsamp[i] / rec_dur,
                             (dig.astype(np.float64) - dmin) * gain + pmin,
                             (dig <= dmin) | (dig >= dmax))
    return out


def load_run_signals(run: Run, tmp: Path) -> tuple[EdfSignal | None, np.ndarray | None, np.ndarray | None]:
    """Fetch and read one run: (EMG, ACC (n,3), GYR (n,3)); a missing modality is None."""
    emg = acc = gyr = None
    if run.emg_path is not None:
        p = fetch(run.emg_path, tmp / f"{run.run}_emg.edf")
        try:
            emg = read_edf(p, (C.EMG_LABEL,)).get(C.EMG_LABEL)
        finally:
            if p.parent == tmp:
                p.unlink(missing_ok=True)
        if emg is not None and emg.fs != C.EMG_FS:
            raise ValueError(f"{run.run}: EMG at {emg.fs} Hz, expected {C.EMG_FS}")
    if run.mov_path is not None:
        p = fetch(run.mov_path, tmp / f"{run.run}_mov.edf")
        try:
            sig = read_edf(p, C.ACC_LABELS + C.GYR_LABELS)
        finally:
            if p.parent == tmp:
                p.unlink(missing_ok=True)
        if all(lab in sig for lab in C.ACC_LABELS + C.GYR_LABELS):
            if any(sig[lab].fs != C.IMU_FS for lab in sig):
                raise ValueError(f"{run.run}: IMU not at {C.IMU_FS} Hz")
            acc = np.stack([sig[lab].values for lab in C.ACC_LABELS], axis=1)
            gyr = np.stack([sig[lab].values for lab in C.GYR_LABELS], axis=1)
    return emg, acc, gyr
