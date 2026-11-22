"""Regenerate lyra_mlm Stage 3 prepared data for every SeizeIT2 recording, streaming the raw files.

    python -m ml.seizeit2.prepare [--workers 6] [--recordings <id> ...]

Runs Kyle's own per-recording step (mlm/scripts/prepare_seizeit2.py `work`, i.e. locked
Decisions D12-D16: simulated MyoWare ENV at 50 Hz, 4 s / 2 s windows, D14 labels, quality states)
with the IMU calibration fix in lyra_mlm.seizeit2. Instead of a 53 GB local copy, each recording's
EMG/MOV EDFs are fetched from the public bucket (SHA256-verified against the git-annex key) into
a temporary one-recording BIDS folder, prepared, and deleted. Output format is identical to
mlm/data/processed/: windows/<sub>/<recording>.npz, feature_schema.json, windows_manifest.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

from lyra_mlm.seizeit2 import SeizeIT2Adapter

from . import config as C
from . import data_loader as D

OUT = C.PROCESSED
_spec = importlib.util.spec_from_file_location("prepare_seizeit2", C.REPO / "mlm" / "scripts" / "prepare_seizeit2.py")
STAGE3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(STAGE3)


def stage(rid: str) -> Path:
    """A temporary BIDS root holding only this recording (EDFs fetched, small text files copied)."""
    src = SeizeIT2Adapter(C.DATASET)._paths(rid)
    root = C.CACHE / "tmp" / rid
    for kind, path in src.items():
        rel = path.relative_to(C.DATASET)
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if kind == "events":
            shutil.copy(path, dest)
            continue
        if path.is_symlink():
            D.fetch(path, dest)
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            shutil.copy(sidecar, dest.with_suffix(".json"))
    return root


def job(rid: str) -> dict:
    root = stage(rid)
    try:
        return STAGE3.work(rid, root, OUT)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--recordings", nargs="*")
    args = ap.parse_args(argv)
    rids = args.recordings or SeizeIT2Adapter(C.DATASET).recording_ids()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = STAGE3.CFG
    (OUT / "feature_schema.json").write_text(json.dumps({
        "primary": [asdict(s) for s in STAGE3.feature_specs(cfg.primary)],
        "secondary": [asdict(s) for s in STAGE3.feature_specs(cfg.secondary)],
        "quality_codes": STAGE3.STATE, "label_codes": {"1": "positive", "0": "negative", "-1": "ignore"},
        "prep_config": json.loads(json.dumps(asdict(cfg), default=str)),
        "normalization": "none fitted; Stage 4 fits on training folds only",
        "imu_calibration": "lyra_mlm.seizeit2.calibrate_imu (ACC -> g, GYR -> deg/s, invalid scale -> INVALID)",
    }, indent=2))
    t0, summaries, failed = time.time(), [], []
    with Pool(args.workers, maxtasksperchild=1) as pool:  # one recording per process: frees its memory
        for i, (rid, res) in enumerate(pool.imap_unordered(_safe, rids), 1):
            (failed.append if "error" in res else summaries.append)(res)
            if i % 50 == 0 or i == len(rids):
                print(f"{i}/{len(rids)} recordings, {time.time() - t0:.0f} s, {len(failed)} failed", flush=True)
    summaries.sort(key=lambda s: s["recording_id"])
    if not args.recordings:
        (OUT / "windows_manifest.json").write_text(json.dumps(summaries, indent=1))
    if failed:
        (OUT / "failed.json").write_text(json.dumps(failed, indent=1))
        print(f"{len(failed)} recordings failed; see {OUT / 'failed.json'}", file=sys.stderr)
        sys.exit(1)


def _safe(rid: str) -> tuple[str, dict]:
    try:
        return rid, job(rid)
    except Exception as e:  # reported and fails the run; never silently skipped
        return rid, {"recording_id": rid, "error": repr(e)}


if __name__ == "__main__":
    main()
