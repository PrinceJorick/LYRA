"""Minimal 16-bit EDF / EDF+ reader.

Everything comes from the file header: labels, units, per-signal sample rate, scaling, and
record onsets from the 'EDF Annotations' channel (so EDF+D gaps are kept as timestamp gaps).
Nothing is hard-coded per dataset.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ANNOTATIONS = "EDF Annotations"
_FIELDS = (("label", 16), ("transducer", 80), ("unit", 8), ("phys_min", 8), ("phys_max", 8),
           ("dig_min", 8), ("dig_max", 8), ("prefilter", 80), ("samples", 8), ("reserved", 32))
_ONSET = re.compile(rb"([+-]\d+(?:\.\d*)?)\x14")


class EdfError(ValueError):
    pass


@dataclass(frozen=True)
class EdfHeader:
    labels: tuple[str, ...]
    units: tuple[str, ...]
    samples_per_record: tuple[int, ...]
    phys_min: tuple[float, ...]
    phys_max: tuple[float, ...]
    dig_min: tuple[int, ...]
    dig_max: tuple[int, ...]
    record_duration_s: float
    n_records: int  # records actually present in the file
    header_bytes: int
    variant: str  # "EDF+C", "EDF+D" or "" (plain EDF)
    start: str  # as written; may be anonymised

    def index(self, label: str) -> int:
        return self.labels.index(label)

    def rate_hz(self, label: str) -> float:
        return self.samples_per_record[self.index(label)] / self.record_duration_s

    @property
    def duration_s(self) -> float:
        return self.n_records * self.record_duration_s


def read_edf_header(path: Path) -> EdfHeader:
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(256)
        if len(head) < 256:
            raise EdfError(f"{path.name}: truncated header")
        ns = int(head[252:256])
        sig = f.read(256 * ns)
    if len(sig) < 256 * ns:
        raise EdfError(f"{path.name}: truncated signal header")
    cols, off = {}, 0
    for name, width in _FIELDS:
        cols[name] = [sig[off + i * width: off + (i + 1) * width].decode("latin-1").strip() for i in range(ns)]
        off += width * ns
    spr = tuple(int(x) for x in cols["samples"])
    header_bytes = int(head[184:192])
    record_bytes = 2 * sum(spr)
    declared = int(head[236:244])
    available = (path.stat().st_size - header_bytes) // record_bytes
    duration = float(head[244:252])
    if duration <= 0 or record_bytes == 0:
        raise EdfError(f"{path.name}: invalid record layout")
    variant = head[192:236].decode("latin-1").strip()
    return EdfHeader(
        labels=tuple(cols["label"]), units=tuple(cols["unit"]), samples_per_record=spr,
        phys_min=tuple(float(x) for x in cols["phys_min"]), phys_max=tuple(float(x) for x in cols["phys_max"]),
        dig_min=tuple(int(x) for x in cols["dig_min"]), dig_max=tuple(int(x) for x in cols["dig_max"]),
        record_duration_s=duration, n_records=available if declared < 0 else min(declared, available),
        header_bytes=header_bytes, variant=variant if variant.startswith("EDF+") else "",
        start=head[168:184].decode("latin-1"),
    )


def read_edf(path: Path, labels: tuple[str, ...]) -> tuple[EdfHeader, dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return the header and {label: (timestamps_s, physical_values)} for the labels present.
    Missing labels are simply absent from the dict; the caller decides what that means."""
    hdr = read_edf_header(path)
    spr = np.array(hdr.samples_per_record)
    offsets = np.concatenate(([0], np.cumsum(spr)))
    data = np.fromfile(path, dtype="<i2", count=hdr.n_records * int(spr.sum()), offset=hdr.header_bytes)
    data = data.reshape(hdr.n_records, int(spr.sum()))
    onsets = np.arange(hdr.n_records) * hdr.record_duration_s
    if hdr.variant and ANNOTATIONS in hdr.labels:
        i = hdr.index(ANNOTATIONS)
        blocks = data[:, offsets[i]:offsets[i + 1]]
        parsed = [_ONSET.match(row.tobytes()) for row in blocks]
        if all(parsed):
            onsets = np.array([float(m.group(1)) for m in parsed])
        elif hdr.variant == "EDF+D":
            raise EdfError(f"{Path(path).name}: EDF+D without readable record onsets")
    out = {}
    for label in labels:
        if label not in hdr.labels or label == ANNOTATIONS:
            continue
        i = hdr.index(label)
        dig = data[:, offsets[i]:offsets[i + 1]].astype(np.float64)
        scale = (hdr.phys_max[i] - hdr.phys_min[i]) / (hdr.dig_max[i] - hdr.dig_min[i])
        phys = (dig - hdr.dig_min[i]) * scale + hdr.phys_min[i]
        ts = onsets[:, None] + np.arange(spr[i]) * (hdr.record_duration_s / spr[i])
        out[label] = (ts.ravel(), phys.ravel())
    return hdr, out
