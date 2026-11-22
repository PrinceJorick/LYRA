"""One streamed pass over SeizeIT2: manifest + cached windows/features for all window configurations.

    python -m ml.lyra.extract [--workers 10] [--recordings <id> ...]

Each recording's EMG/MOV EDFs are fetched from the dataset's public bucket into ml/lyra/cache/tmp,
verified against the SHA256 in the git-annex key, read with pyedflib, processed and deleted.
Nothing is written inside pj-training-data/. Output: ml/lyra/cache/features/<config hash>/
<sub>/<recording>.npz + manifest/<recording>.json. A cached recording is reused only when its stored
config hash and source-file keys match.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ml.seizeit2.data_loader import annex_key, fetch

from . import bids
from . import config as C
from . import signals as S
from . import windows as Wn

TMP = C.CACHE / "tmp"


def _source_key(path: Path | None) -> str | None:
    if path is None:
        return None
    return annex_key(path)[1] if path.is_symlink() else f"local:{path.stat().st_size}"


def out_paths(rid: str) -> tuple[Path, Path]:
    return C.FEATURE_DIR / rid.split("_")[0] / f"{rid}.npz", C.FEATURE_DIR / "manifest" / f"{rid}.json"


def cached(rec: bids.RecordingFiles) -> bool:
    npz, man = out_paths(rec.recording_id)
    if not man.exists():
        return False
    m = json.loads(man.read_text())
    keys = {k: _source_key(getattr(rec, k)) for k in ("emg", "mov")}
    return m.get("config_hash") == C.FEATURE_HASH and m.get("source_keys") == keys and (
        npz.exists() or m.get("excluded_reason"))


def _local(path: Path, work: Path) -> Path:
    if not path.is_symlink():
        return path
    return fetch(path, work / path.name)


def process(rec: bids.RecordingFiles) -> dict:
    rid = rec.recording_id
    work = TMP / rid
    work.mkdir(parents=True, exist_ok=True)
    m: dict = {"recording_id": rid, "subject": rec.subject, "partition": "dev" if C.is_dev(rid) else "test",
               "config_hash": C.FEATURE_HASH, "source_keys": {k: _source_key(getattr(rec, k)) for k in ("emg", "mov")},
               "problems": list(rec.problems), "notes": []}
    try:
        ev = bids.parse_events(rid)
        m["events"] = {"target": len(ev.target), "nontarget": len(ev.nontarget), "artifact": len(ev.artifact),
                       "codes": sorted({c for *_, c in ev.target + ev.nontarget + ev.artifact + ev.other}),
                       "skipped_rows": ev.skipped, "unmapped_codes": ev.unmapped_seizure_codes,
                       "recording_duration_s": ev.recording_duration_s}
        env = env_src = acc = gyr = imu_src = None
        imu_cal = False
        if rec.emg is not None:
            try:
                hdr, dig, phys = S.read_edf(_local(rec.emg, work), (C.EMG_CHANNEL,))
                fs = hdr["channels"][0]["rate_hz"]
                m["emg"] = hdr | {"sidecar_rate_hz": bids.sidecar(rec.emg).get("SamplingFrequency")}
                if m["emg"]["sidecar_rate_hz"] not in (None, fs):
                    m["notes"].append(f"EMG sidecar {m['emg']['sidecar_rate_hz']} Hz != EDF {fs} Hz (EDF used)")
                raw = phys[:, 0]
                env = S.simulate_env(raw, int(fs))
                env_src = (S.at_rails(dig, hdr)[:, 0] | ~np.isfinite(raw), np.diff(raw) != 0, int(fs))
            except (KeyError, ValueError, OSError) as e:
                m["problems"].append(f"EMG unusable: {e}")
        if rec.mov is not None:
            try:
                hdr, dig, phys = S.read_edf(_local(rec.mov, work), C.IMU_CHANNELS)
                rates = {c["rate_hz"] for c in hdr["channels"]}
                if rates != {float(C.IMU_RATE)}:
                    raise ValueError(f"IMU rates {sorted(rates)} != {C.IMU_RATE} Hz")
                m["mov"] = hdr | {"sidecar_rate_hz": bids.sidecar(rec.mov).get("SamplingFrequency")}
                acc, gyr, imu_cal, note = S.imu_physical(phys)
                m["notes"].append(note)
                imu_src = (S.at_rails(dig, hdr).any(axis=1) | ~np.isfinite(phys).all(axis=1),
                           (np.diff(phys, axis=0) != 0).any(axis=1))
            except (KeyError, ValueError, OSError) as e:
                m["problems"].append(f"IMU unusable: {e}")
        if env is None and acc is None:
            m["excluded_reason"] = "no usable EMG and no usable IMU"
            return m
        if env is not None and acc is not None:
            d_emg, d_mov = m["emg"]["duration_s"], m["mov"]["duration_s"]
            m["sync"] = {"emg_start": m["emg"]["start"], "mov_start": m["mov"]["start"],
                         "duration_mismatch_s": d_emg - d_mov,
                         "assumed_offset_s": 0.0,
                         "basis": "same ECGEMG Sensor Dot module; EDF start times are de-identified differently "
                                  "per file, so the absolute offset is not verifiable from the headers"}
            if abs(d_emg - d_mov) > 1.0:
                m["notes"].append(f"EMG/MOV durations differ by {d_emg - d_mov:g} s: windows truncated to the shorter")
        out = Wn.build(env, env_src, acc, gyr, imu_src, imu_cal, ev)
        m["windows"] = {}
        for name in C.WINDOWS:
            lab = Wn.labels(out[f"{name}_tfrac"])
            excl = out[f"{name}_artifact"] | out[f"{name}_warmup"]
            fus = out[f"{name}_emg_ok"] & out[f"{name}_imu_ok"] & ~excl
            m["windows"][name] = {
                "n": int(lab.size), "emg_ok": int(out[f"{name}_emg_ok"].sum()), "imu_ok": int(out[f"{name}_imu_ok"].sum()),
                "artifact": int(out[f"{name}_artifact"].sum()), "warmup": int(out[f"{name}_warmup"].sum()),
                "fusion_valid": int(fus.sum()), "fusion_valid_positive": int((fus & (lab == 1)).sum()),
                "fusion_valid_negative": int((fus & (lab == 0)).sum()),
                "fusion_valid_partial_excluded": int((fus & (lab == -1)).sum()),
                "fusion_valid_nontarget_seizure": int((fus & (lab == 0) & out[f"{name}_nontarget"]).sum())}
        npz, _ = out_paths(rid)
        npz.parent.mkdir(parents=True, exist_ok=True)
        tmp = npz.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **out)
        tmp.replace(npz)
        return m
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _safe(rec: bids.RecordingFiles) -> dict:
    try:
        m = process(rec)
    except Exception as e:  # reported; the run exits non-zero
        return {"recording_id": rec.recording_id, "error": repr(e)}
    _, man = out_paths(rec.recording_id)
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps(m, indent=1, default=str))
    return m


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--recordings", nargs="*")
    args = ap.parse_args(argv)
    recs, problems = bids.match_recordings()
    if args.recordings:
        recs = [r for r in recs if r.recording_id in set(args.recordings)]
    todo = [r for r in recs if not cached(r)]
    C.FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    (C.FEATURE_DIR / "config.json").write_text(json.dumps(C.feature_config() | {"hash": C.FEATURE_HASH}, indent=1,
                                                          default=str))
    print(f"{len(recs)} recordings, {len(recs) - len(todo)} cached, {len(todo)} to process; "
          f"dataset problems: {problems or 'none'}", flush=True)
    t0, failed = time.time(), []
    with Pool(args.workers, maxtasksperchild=1) as pool:
        for i, m in enumerate(pool.imap_unordered(_safe, todo), 1):
            if "error" in m:
                failed.append(m)
            if i % 50 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} recordings, {time.time() - t0:.0f} s, {len(failed)} failed", flush=True)
    if failed:
        (C.FEATURE_DIR / "failed.json").write_text(json.dumps(failed, indent=1))
        print(f"{len(failed)} failed: {C.FEATURE_DIR / 'failed.json'}", file=sys.stderr)
        sys.exit(1)
    (C.FEATURE_DIR / "failed.json").unlink(missing_ok=True)
    (C.FEATURE_DIR / "dataset_problems.json").write_text(json.dumps(problems, indent=1))


if __name__ == "__main__":
    main()
