"""Turn a captured pod data-collection stream into a session folder (docs/POD_DATA_PROTOCOL.md, schema v1).

The pod writes one line per record (firmware/components/lyra_pod/lyra_logline.h): S (header), E (ENV),
R (RAW), I (IMU), M (marker). Capture the serial stream to a file first, then run this. Participant,
consent and placement are entered here, never on the pod. Malformed lines are dropped and counted,
never repaired. Refuses to overwrite an existing session. Button markers go to markers.tsv; the
operator's annotations (onset, duration, eventType) come from --events.
Usage: python scripts/pod_logger.py capture.txt --participant pod-p001 --session s01 --consent <id>
       --side left [--muscle deltoid] [--events operator_events.tsv] [--out data/pod]
"""

import argparse
import csv
import json
from pathlib import Path

from lyra_mlm.pod import IMU_UNITS, PSEUDONYM, PodAdapter, check_session

DATA = Path(__file__).resolve().parents[1] / "data"
FIELDS = {"E": 4, "R": 4, "I": 9, "M": 3}


def parse(lines) -> tuple[dict, dict[str, list[list[str]]], int]:
    header, rows, bad = {}, {k: [] for k in FIELDS}, 0
    for line in lines:
        parts = line.strip().split(",")
        kind = parts[0]
        if kind == "S" and not header:
            header = dict(p.split("=", 1) for p in parts[1:] if "=" in p)
            continue
        try:
            if kind not in FIELDS or len(parts) != FIELDS[kind]:
                raise ValueError
            if kind == "M":
                float(parts[1])
            else:
                [float(x) for x in parts[1:-1]]
                if parts[-1] not in ("0", "1", "2"):
                    raise ValueError
        except ValueError:
            bad += 1
            continue
        rows[kind].append(parts[1:])
    return header, rows, bad


def write_csv(path: Path, head: str, rows: list[list[str]]) -> None:
    path.write_text(head + "\n" + "".join(",".join(r) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--participant", required=True)
    ap.add_argument("--session", required=True)
    ap.add_argument("--consent", required=True, help="approved consent protocol id and version")
    ap.add_argument("--side", required=True, choices=("left", "right"))
    ap.add_argument("--muscle", default="deltoid")
    ap.add_argument("--events", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DATA / "pod")
    args = ap.parse_args()
    if not (PSEUDONYM.match(args.participant) and PSEUDONYM.match(args.session)):
        raise SystemExit("participant and session must be pseudonyms like pod-p001 / s01 (no names, no '_')")
    d = args.out / args.participant / args.session
    if d.exists():
        raise SystemExit(f"{d} exists: refusing to overwrite a session")
    with open(args.capture, errors="replace") as fh:
        header, rows, bad = parse(fh)
    if header.get("schema") != "1":
        raise SystemExit("capture has no 'S,schema=1,...' header line")
    if not rows["E"] or not rows["I"]:
        raise SystemExit("capture has no ENV or no IMU samples")
    d.mkdir(parents=True)
    write_csv(d / "env.csv", "t_s,mV,flag", rows["E"])
    write_csv(d / "imu.csv", "t_s,ax,ay,az,gx,gy,gz,flag", rows["I"])
    if rows["R"]:
        write_csv(d / "raw.csv", "t_s,mV,flag", rows["R"])
    (d / "markers.tsv").write_text("onset\tlabel\n" + "".join(f"{t}\t{lab}\n" for t, lab in rows["M"]))
    events = "onset\tduration\teventType\tsource\n"
    if args.events is not None:
        with open(args.events, newline="") as fh:
            events += "".join(f"{r['onset']}\t{r['duration']}\t{r['eventType']}\t{r.get('source') or 'operator'}\n"
                              for r in csv.DictReader(fh, delimiter="\t"))
    (d / "events.tsv").write_text(events)
    fs = float(header["full_scale_mv"])
    streams = {"env": {"rate_hz": float(header["env_rate_hz"]), "unit": "mV", "full_scale_mv": fs},
               "imu": {"rate_hz": float(header["imu_rate_hz"]), "units": list(IMU_UNITS),
                       "acc_range_g": int(header["acc_range_g"]), "gyr_range_dps": int(header["gyr_range_dps"])}}
    if rows["R"]:
        streams["raw"] = {"rate_hz": float(header["raw_rate_hz"]), "unit": "mV", "full_scale_mv": fs}
    last = max(float(r[0]) for k in ("E", "I", "R") for r in rows[k][-1:])
    (d / "session.json").write_text(json.dumps({
        "schema_version": 1, "participant_id": args.participant, "session_id": args.session,
        "consent_protocol": args.consent, "pod_id": header["pod_id"], "firmware": header["firmware"],
        "placement": {"side": args.side, "muscle": args.muscle}, "duration_s": last, "streams": streams,
        "notes": f"logged by pod_logger.py; {bad} malformed lines dropped"}, indent=1))
    problems = check_session(PodAdapter(args.out), f"{args.participant}_{args.session}")
    print(f"written {d}: ENV {len(rows['E'])}, IMU {len(rows['I'])}, RAW {len(rows['R'])}, markers {len(rows['M'])}, "
          f"malformed lines dropped {bad}; {len(problems)} integrity problems", *problems[:10], sep="\n  ")


if __name__ == "__main__":
    main()
