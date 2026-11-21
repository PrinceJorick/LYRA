"""Stage 3: SeizeIT2 inventory, checksums, window preparation and split manifest.

Reads mlm/data/seizeit2/ and writes only under mlm/data/ (gitignored). Fits nothing: features are
stored raw with availability masks for Stage 4 training-fold normalization. Resumable: recordings
with an existing output file are summarised from it instead of recomputed.
Usage: python scripts/prepare_seizeit2.py [--workers 5] [--skip-checksums] [--data mlm/data]
"""

import argparse
import csv
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from lyra_mlm.annotations import AnnotationKind, WindowLabel
from lyra_mlm.evaluation import SEIZEIT2_PLANNED_SPLIT, split_manifest
from lyra_mlm.features import feature_specs
from lyra_mlm.prep import QUALITY_CODE, PrepConfig, prepare_recording
from lyra_mlm.seizeit2 import MissingRecordingData, SeizeIT2Adapter

DATA = Path(__file__).resolve().parents[1] / "data"
CFG = PrepConfig()
STATE = {v: k.value for k, v in QUALITY_CODE.items()}


def counts(a: np.ndarray, names: dict) -> dict:
    vals, n = np.unique(a, return_counts=True)
    return {names[int(v)]: int(c) for v, c in zip(vals, n)}


def summarise(rid: str, z) -> dict:
    labels = {int(WindowLabel.POSITIVE): "positive", int(WindowLabel.NEGATIVE): "negative",
              int(WindowLabel.IGNORE): "ignore"}
    return {"recording_id": rid, "participant_id": rid.split("_")[0], "n_windows": int(z["label"].size),
            "labels": counts(z["label"], labels), "overall_quality": counts(z["overall"], STATE),
            "emg_quality": counts(z["emg_state"], STATE), "imu_quality": counts(z["imu_state"], STATE),
            "raw_emg_quality": counts(z["raw_emg_state"], STATE),
            "primary_all_available": int(z["primary_available"].all(axis=1).sum()),
            "secondary_all_available": int(z["secondary_available"].all(axis=1).sum())}


def work(rid: str, RAW: Path, OUT: Path) -> dict:
    out = OUT / "windows" / rid.split("_")[0] / f"{rid}.npz"
    if out.exists():
        with np.load(out) as z:
            return summarise(rid, z)
    try:
        rec = SeizeIT2Adapter(RAW).load_recording(rid)
    except MissingRecordingData as e:
        return {"recording_id": rid, "participant_id": rid.split("_")[0], "excluded": str(e)}
    arrays = prepare_recording(rec, CFG)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, participant_id=rec.participant_id, recording_id=rid, placement=rec.placement,
                        emg_provenance="\n".join(rec.emg.provenance) if rec.emg else "",
                        imu_rate_hz=rec.imu.nominal_rate_hz if rec.imu else np.nan,
                        emg_rate_hz=rec.emg.nominal_rate_hz if rec.emg else np.nan, **arrays)
    tmp.replace(out)
    return summarise(rid, arrays)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--skip-checksums", action="store_true")
    ap.add_argument("--data", type=Path, default=DATA, help="folder holding seizeit2/; outputs go to processed/")
    args = ap.parse_args()
    RAW, OUT = args.data / "seizeit2", args.data / "processed"
    OUT.mkdir(parents=True, exist_ok=True)
    adapter = SeizeIT2Adapter(RAW)
    rids = adapter.recording_ids()

    # 1. inventory from headers and events (no signal processing)
    infos = [adapter.info(r) for r in rids]
    with open(RAW / "participants.tsv", newline="") as f:
        participants = [r["participant_id"] for r in csv.DictReader(f, delimiter="\t")]
    raw_labels, fbtcs, gtcs = Counter(), Counter(), Counter()
    seizures = 0
    for r in rids:
        for a in adapter.annotations(r):
            raw_labels[a.raw_label] += 1
            seizures += a.kind in (AnnotationKind.SEIZURE_FBTCS, AnnotationKind.SEIZURE_GTCS,
                                   AnnotationKind.SEIZURE_OUT_OF_SCOPE)
            fbtcs[r.split("_")[0]] += a.kind is AnnotationKind.SEIZURE_FBTCS
            gtcs[r.split("_")[0]] += a.kind is AnnotationKind.SEIZURE_GTCS
    inventory = {
        "participants_tsv": len(participants), "participants_with_recordings": len({i.participant_id for i in infos}),
        "recordings": len(infos), "recording_hours": sum(i.duration_s or 0 for i in infos) / 3600,
        "with_emg": sum(i.emg_rate_hz is not None for i in infos),
        "with_imu": sum(i.imu_rate_hz is not None for i in infos),
        "with_both": sum(i.emg_rate_hz is not None and i.imu_rate_hz is not None for i in infos),
        "with_neither": sum(i.emg_rate_hz is None and i.imu_rate_hz is None for i in infos),
        "emg_rates_hz": dict(Counter(str(i.emg_rate_hz) for i in infos)),
        "imu_rates_hz": dict(Counter(str(i.imu_rate_hz) for i in infos)),
        "missing_reasons": dict(Counter(m for i in infos for m in i.missing)),
        "sidecar_rate_notes": [i.notes for i in infos if i.notes],
        "annotated_seizures": seizures, "fbtcs_events": sum(fbtcs.values()), "gtcs_events": sum(gtcs.values()),
        "fbtcs_by_participant": {p: n for p, n in sorted(fbtcs.items()) if n},
        "event_labels": dict(raw_labels.most_common()),
    }
    (OUT / "inventory.json").write_text(json.dumps(inventory, indent=2))
    print(json.dumps({k: v for k, v in inventory.items() if not isinstance(v, (dict, list))}, indent=1), flush=True)

    # 2. checksums of every downloaded file
    manifest = RAW / "sha256_manifest.json"
    if not args.skip_checksums and not manifest.exists():
        files = sorted(p for p in RAW.rglob("*") if p.is_file() and p.suffix in (".edf", ".json", ".tsv")
                       and p.name != "sha256_manifest.json")
        with ThreadPoolExecutor(8) as pool:
            digests = list(pool.map(sha256, files))
        manifest.write_text(json.dumps({str(p.relative_to(RAW)): d for p, d in zip(files, digests)}, indent=0))
        print(f"checksummed {len(files)} files", flush=True)

    # 3. feature schema and configuration
    (OUT / "feature_schema.json").write_text(json.dumps({
        "primary": [asdict(s) for s in feature_specs(CFG.primary)],
        "secondary": [asdict(s) for s in feature_specs(CFG.secondary)],
        "quality_codes": STATE, "label_codes": {"1": "positive", "0": "negative", "-1": "ignore"},
        "prep_config": json.loads(json.dumps(asdict(CFG), default=str)),
        "normalization": "none fitted; Stage 4 fits on training folds only",
    }, indent=2))

    # 4. windows, one process per recording
    t0, summaries = time.time(), []
    with Pool(args.workers, maxtasksperchild=20) as pool:
        for i, s in enumerate(pool.imap_unordered(partial(work, RAW=RAW, OUT=OUT), rids), 1):
            summaries.append(s)
            if i % 50 == 0:
                print(f"{i}/{len(rids)} recordings, {time.time() - t0:.0f} s", flush=True)
    summaries.sort(key=lambda s: s["recording_id"])
    (OUT / "windows_manifest.json").write_text(json.dumps(summaries, indent=1))

    # 5. participant-independent split
    split = split_manifest(participants, fbtcs, SEIZEIT2_PLANNED_SPLIT)
    by_part = defaultdict(Counter)
    for s in summaries:
        by_part[s["participant_id"]].update(s.get("labels", {}))
    for key in ("train_cv", "held_out_test"):
        split[key]["window_labels"] = dict(sum((by_part[p] for p in split[key]["participants"]), Counter()))
    split["verified_against_inventory"] = sorted(participants) == sorted(SEIZEIT2_PLANNED_SPLIT.train
                                                                         | SEIZEIT2_PLANNED_SPLIT.held_out)
    (OUT / "splits_manifest.json").write_text(json.dumps(split, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
