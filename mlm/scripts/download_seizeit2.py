"""Download the wearable subset of SeizeIT2 (OpenNeuro ds005873) into mlm/data/seizeit2/.

Fetches EMG and MOV EDF files with their JSON sidecars, every events.tsv and the top-level
metadata from OpenNeuro's public S3 bucket. Resumable: files already on disk with the expected
size are skipped. Each file's size is checked against the bucket listing. EEG and ECG are not
downloaded. Usage: python scripts/download_seizeit2.py [--workers 8]
"""

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "https://s3.amazonaws.com/openneuro.org"
PREFIX = "ds005873/"
WANTED = re.compile(r"(_emg\.edf|_emg\.json|_mov\.edf|_mov\.json|_events\.tsv)$|^ds005873/[^/]+$")
OUT = Path(__file__).resolve().parents[1] / "data" / "seizeit2"


def listing() -> list[tuple[str, int]]:
    keys, token = [], None
    while True:
        q = {"list-type": "2", "prefix": PREFIX, **({"continuation-token": token} if token else {})}
        xml = urllib.request.urlopen(f"{BUCKET}?{urllib.parse.urlencode(q)}", timeout=60).read().decode()
        for blk in re.findall(r"<Contents>(.*?)</Contents>", xml, re.S):
            key = re.search(r"<Key>(.*?)</Key>", blk).group(1)
            keys.append((key, int(re.search(r"<Size>(\d+)</Size>", blk).group(1))))
        m = re.search(r"<NextContinuationToken>(.*?)</NextContinuationToken>", xml)
        if not m:
            return keys
        token = m.group(1)


def fetch(key: str, size: int) -> str:
    dest = OUT / key[len(PREFIX):]
    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for _ in range(3):
        try:
            with urllib.request.urlopen(f"{BUCKET}/{urllib.parse.quote(key)}", timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            if tmp.stat().st_size != size:
                raise OSError(f"size {tmp.stat().st_size} != {size}")
            tmp.replace(dest)
            return "ok"
        except OSError as e:
            err = e
    raise RuntimeError(f"{key}: {err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    keys = [(k, s) for k, s in listing() if WANTED.search(k)]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "download_listing.json").write_text(json.dumps(keys))
    total = sum(s for _, s in keys)
    print(f"{len(keys)} files, {total / 1e9:.1f} GB", flush=True)
    done = failed = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fetch, k, s): (k, s) for k, s in keys}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                fut.result()
                done += futs[fut][1]
            except RuntimeError as e:
                failed += 1
                print("FAILED", e, flush=True)
            if i % 200 == 0:
                print(f"{i}/{len(keys)} files, {done / 1e9:.1f} GB", flush=True)
    print(f"finished: {len(keys) - failed} ok, {failed} failed", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
