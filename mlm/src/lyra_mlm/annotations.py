"""Annotation contracts and the hard-negative movement taxonomy.

Kinds and activity tags let future evaluation group false alarms by activity. Nothing here
has been learned, and no raw dataset label has been mapped yet (that requires inspecting the
real annotation files in the dataset-integration session).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class AnnotationKind(StrEnum):
    SEIZURE_FBTCS = "seizure_fbtcs"
    SEIZURE_GTCS = "seizure_gtcs"
    SEIZURE_OUT_OF_SCOPE = "seizure_out_of_scope"  # non-convulsive, absence, focal non-motor: outside v1
    HARD_NEGATIVE = "hard_negative"  # labelled non-seizure activity
    ARTIFACT = "artifact"
    UNMAPPED = "unmapped"  # raw label not mapped yet; never silently treated as negative


HARD_NEGATIVE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "daily_activity": (
        "brushing_teeth", "eating_drinking", "walking", "writing", "dressing", "posture_change",
    ),
    "repetitive_movement": ("repeated_arm_movement", "rhythmic_household_task", "repetitive_gesture"),
    "physical_activity": ("running", "exercise", "stair_climbing", "lifting"),
    "other_movement": ("shivering", "tremor", "muscle_spasm", "sudden_jerk", "sleep_movement"),
    "sensor_artifact": (
        "loose_electrode", "electrode_displacement", "device_displacement", "external_impact",
        "signal_saturation", "missing_or_corrupted",
    ),
}
_GROUP_OF = {a: g for g, acts in HARD_NEGATIVE_CATEGORIES.items() for a in acts}


def activity_group(activity: str) -> str:
    """Evaluation grouping for an activity tag, e.g. 'walking' -> 'daily_activity'."""
    return _GROUP_OF[activity]


@dataclass(frozen=True)
class Annotation:
    annotation_id: str
    onset_s: float
    duration_s: float
    kind: AnnotationKind
    raw_label: str  # verbatim source label, kept for audit
    activity: str | None = None  # a HARD_NEGATIVE_CATEGORIES tag (HARD_NEGATIVE / ARTIFACT only)
    source: str = ""  # annotator or dataset

    def __post_init__(self) -> None:
        if not (math.isfinite(self.onset_s) and math.isfinite(self.duration_s) and self.duration_s >= 0):
            raise ValueError(f"annotation {self.annotation_id}: onset/duration must be finite, duration >= 0")
        if self.activity is not None:
            if self.activity not in _GROUP_OF:
                raise ValueError(f"unknown activity tag {self.activity!r}")
            if self.kind not in (AnnotationKind.HARD_NEGATIVE, AnnotationKind.ARTIFACT):
                raise ValueError("activity tags apply only to HARD_NEGATIVE or ARTIFACT annotations")

    @property
    def end_s(self) -> float:
        return self.onset_s + self.duration_s

    def overlaps(self, start_s: float, end_s: float) -> bool:
        if self.duration_s == 0:
            return start_s <= self.onset_s < end_s
        return self.onset_s < end_s and self.end_s > start_s


class AnnotationMapper(Protocol):
    """Maps a dataset's raw labels to AnnotationKind.

    Implement per dataset only after inspecting the real annotation files. Unknown labels must
    map to UNMAPPED, never to HARD_NEGATIVE.
    """

    def map_label(self, raw_label: str) -> AnnotationKind: ...
