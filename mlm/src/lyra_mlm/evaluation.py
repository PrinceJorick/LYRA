"""Evaluation contracts: partitions, participant-independent splits and metric definitions.

Nothing here computes a metric. No evaluation has been run and no performance is claimed.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, Sequence


class Partition(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"  # cross-validation folds drawn from training participants
    TEST = "test"  # held-out participants


class LeakageError(ValueError):
    """Raised when data from one partition would influence another."""


def check_participant_independence(participants_by_partition: Mapping[Partition, Iterable[str]]) -> None:
    """Raise LeakageError if any participant appears in more than one partition."""
    seen: dict[str, Partition] = {}
    for partition, ids in participants_by_partition.items():
        for pid in set(ids):
            if pid in seen and seen[pid] is not partition:
                raise LeakageError(f"participant {pid} appears in both {seen[pid]} and {partition}")
            seen[pid] = partition


@dataclass(frozen=True)
class ParticipantSplit:
    train: frozenset[str]
    held_out: frozenset[str]
    source: str
    verified: bool  # True only after checking against the real participant inventory

    def __post_init__(self) -> None:
        check_participant_independence({Partition.TRAIN: self.train, Partition.TEST: self.held_out})

    def partition_of(self, participant_id: str) -> Partition | None:
        if participant_id in self.train:
            return Partition.TRAIN
        if participant_id in self.held_out:
            return Partition.TEST
        return None  # unknown participants are never assigned silently


def _subs(lo: int, hi: int) -> frozenset[str]:
    return frozenset(f"sub-{i:03d}" for i in range(lo, hi + 1))


SEIZEIT2_PLANNED_SPLIT = ParticipantSplit(
    train=_subs(1, 96),
    held_out=_subs(97, 125),
    source="SeizeIT2 paper's suggested split (context.md §7, R1); verified 2026-11-21 against the v1.1.0 "
           "participants.tsv and subject folders (125 participants, sub-001..sub-125)",
    verified=True,
)


def split_manifest(participants: Iterable[str], fbtcs_events: Mapping[str, int], split: ParticipantSplit) -> dict:
    """Check a split against the real participant inventory and describe each partition.
    Every inventory participant must be assigned; partitions must be disjoint."""
    inventory = set(participants)
    unassigned = sorted(p for p in inventory if split.partition_of(p) is None)
    if unassigned:
        raise LeakageError(f"participants not covered by the split: {unassigned}")
    parts = {Partition.TRAIN: sorted(inventory & split.train), Partition.TEST: sorted(inventory & split.held_out)}
    check_participant_independence(parts)
    assert not set(parts[Partition.TRAIN]) & set(parts[Partition.TEST])

    def describe(ids: list[str]) -> dict:
        with_fbtcs = [p for p in ids if fbtcs_events.get(p, 0) > 0]
        return {"participants": ids, "n_participants": len(ids), "with_fbtcs": with_fbtcs,
                "interictal_only": [p for p in ids if p not in with_fbtcs],
                "fbtcs_events": sum(fbtcs_events.get(p, 0) for p in ids)}

    return {"split_source": split.source,
            "in_split_but_not_in_inventory": sorted((split.train | split.held_out) - inventory),
            "train_cv": describe(parts[Partition.TRAIN]), "held_out_test": describe(parts[Partition.TEST]),
            "disjoint": True}


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    definition: str


METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("event_sensitivity", "fraction", "Annotated in-scope events with >=1 possible-event alert matched "
               "to them / all in-scope events. The matching convention is unresolved (context.md U11)."),
    MetricSpec("false_alarms_per_24h", "1/24h", "Alerts not matched to an in-scope event / valid monitored hours "
               "x 24. Suppressed time is reported separately, never counted as monitored."),
    MetricSpec("detection_latency_s", "s", "Alert time minus annotated onset, for matched events only."),
    MetricSpec("precision", "fraction", "Matched alerts / all alerts."),
    MetricSpec("auprc", "unitless", "Area under the window-level precision-recall curve on held-out participants."),
    MetricSpec("specificity", "fraction", "Valid negative windows / all valid non-event windows."),
    MetricSpec("per_participant", "mixed", "Every metric above, per held-out participant."),
)


@dataclass(frozen=True)
class EvaluationReport:
    """Filled only by a future evaluation run. None means 'not computed'."""

    partition: Partition
    split_source: str
    event_sensitivity: float | None = None
    false_alarms_per_24h: float | None = None
    detection_latency_s: float | None = None
    precision: float | None = None
    auprc: float | None = None
    specificity: float | None = None
    per_participant: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)
    false_alarms_by_activity_group: Mapping[str, int] = field(default_factory=dict)  # hard negatives
    valid_monitored_hours: float | None = None
    suppressed_hours: float | None = None  # so suppression cannot hide misses

    def __post_init__(self) -> None:
        if self.partition is Partition.TRAIN:
            raise LeakageError("evaluation reports must not be computed on the training partition")


class Evaluator(Protocol):
    def evaluate(self, predictions: Sequence[object], annotations: Sequence[object],
                 split: ParticipantSplit) -> EvaluationReport: ...
