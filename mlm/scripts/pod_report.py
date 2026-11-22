"""Integrity check and domain-shift report for LYRA pod sessions (docs/POD_DATA_PROTOCOL.md).

Reference: SeizeIT2 Train/CV windows only (sub-001..sub-096), multimodal-usable. Negatives are kept
by a deterministic 2% per-window key, and all positives are kept. Cached in
data/processed/reference_traincv.npz and rebuilt when windows_manifest.json changes.
Writes data/pod_reports/<recording>.json. Descriptive only: no model is fitted or changed.
Usage: python scripts/pod_report.py [--pod data/pod] [--recording <participant>_<session>]
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from lyra_mlm.domain import report
from lyra_mlm.evaluation import SEIZEIT2_PLANNED_SPLIT, Partition
from lyra_mlm.pod import PodAdapter, check_session
from lyra_mlm.training import PRIMARY_NAMES, SUPPRESSED, TRACKS, require_train_cv, track_status, window_keys

DATA = Path(__file__).resolve().parents[1] / "data"
REF_FRACTION = 0.02


def build_reference(proc: Path) -> tuple[np.ndarray, np.ndarray]:
    cache, manifest = proc / "reference_traincv.npz", proc / "windows_manifest.json"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if cache.exists():
        with np.load(cache) as z:
            if str(z["manifest_sha256"]) == digest and list(z["names"]) == list(PRIMARY_NAMES):
                return z["negative"], z["positive"]
    neg, pos, parts = [], [], set()
    for m in json.loads(manifest.read_text()):
        if "excluded" in m or SEIZEIT2_PLANNED_SPLIT.partition_of(m["participant_id"]) is not Partition.TRAIN:
            continue
        rid = m["recording_id"]
        with np.load(proc / "windows" / m["participant_id"] / f"{rid}.npz") as z:
            ok = (track_status(z, TRACKS["multimodal"]) != SUPPRESSED) & ~z["artifact_overlap"]
            neg.append(z["primary"][ok & (z["label"] == 0) & (window_keys(rid, z["label"].size) < REF_FRACTION)])
            pos.append(z["primary"][ok & (z["label"] == 1)])
        parts.add(m["participant_id"])
    require_train_cv(parts)
    negative, positive = np.concatenate(neg), np.concatenate(pos)
    np.savez_compressed(cache, negative=negative, positive=positive, names=np.array(PRIMARY_NAMES),
                        manifest_sha256=digest, participants=np.array(sorted(parts)))
    return negative, positive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pod", type=Path, default=DATA / "pod")
    ap.add_argument("--recording", default=None)
    args = ap.parse_args()
    adapter = PodAdapter(args.pod)
    rids = [args.recording] if args.recording else adapter.recording_ids()
    if not rids:
        raise SystemExit(f"no pod sessions under {args.pod}")
    negative, positive = build_reference(DATA / "processed")
    out_dir = DATA / "pod_reports"
    out_dir.mkdir(exist_ok=True)
    for rid in rids:
        problems = check_session(adapter, rid)
        print(f"{rid}: {len(problems)} integrity problems", *problems[:10], sep="\n  ")
        try:
            r = report(adapter.load_recording(rid), adapter.load_raw(rid), adapter.meta(rid), negative, positive)
        except ValueError as e:
            print(f"  report failed: {e}")
            continue
        r["integrity_problems"] = problems
        (out_dir / f"{rid}.json").write_text(json.dumps(r, indent=1, default=float))
        for path in ("simenv", "env"):
            if path in r:
                cmp_ = r[path]["vs_seizeit2_negatives"]
                worst = sorted((v.get("ks", 0), k) for k, v in cmp_.items())[::-1][:5]
                print(f"  {path}: {r[path]['usable_windows']}/{r[path]['windows']} usable windows; largest KS vs "
                      f"SeizeIT2 negatives: " + ", ".join(f"{k} {ks:.2f}" for ks, k in worst))
        if "env_vs_simenv_signal" in r:
            print(f"  MyoWare ENV vs simulated ENV: {r['env_vs_simenv_signal']}")
        print(f"  written {out_dir / f'{rid}.json'}")


if __name__ == "__main__":
    main()
