"""The app's pod payload decoder (src/services/pod/protocol.ts) against lyra_mlm.ble, byte for byte.
Node runs the TypeScript file directly (type stripping, Node >= 23.6). SYNTHETIC payloads only."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from lyra_mlm.ble import DeviceStatusPayload, EventStatusPayload

PROTOCOL_TS = Path(__file__).resolve().parents[2] / "src" / "services" / "pod" / "protocol.ts"
RUNNER = """
import { readFileSync } from 'node:fs';
const p = await import(process.argv[2]);
const out = JSON.parse(readFileSync(process.argv[3], 'utf8')).map(([kind, hex]) => {
  const bytes = Uint8Array.from(Buffer.from(hex, 'hex'));
  try {
    const v = kind === 'status' ? p.decodeDeviceStatus(bytes) : p.decodeEventStatus(bytes);
    const again = kind === 'status' ? p.encodeDeviceStatus(v) : p.encodeEventStatus(v);
    return { ok: true, value: v, reencoded: Buffer.from(again).toString('hex') };
  } catch (e) {
    return { ok: false, error: e instanceof p.PayloadError ? 'PayloadError' : String(e) };
  }
});
console.log(JSON.stringify(out));
"""
QUALITY = ["VALID", "DEGRADED", "INVALID", "MISSING", "UNSYNCHRONIZED"]
STATES = ["idle", "accumulating", "possible_event", "refractory"]
CANCELS = ["none", "cancelled_on_pod", "cancelled_on_phone"]
ORIGIN = {1: "hardware", 2: "simulated"}


def node_version() -> tuple[int, ...]:
    if shutil.which("node") is None:
        return (0,)
    out = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip().lstrip("v")
    return tuple(int(x) for x in out.split(".")[:2])


pytestmark = pytest.mark.skipif(node_version() < (23, 6), reason="needs Node >= 23.6 (runs TypeScript directly)")


def run(vectors, tmp_path):
    (tmp_path / "runner.mjs").write_text(RUNNER)
    (tmp_path / "vectors.json").write_text(json.dumps(vectors))
    r = subprocess.run(["node", str(tmp_path / "runner.mjs"), PROTOCOL_TS.as_uri(), str(tmp_path / "vectors.json")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_typescript_decoder_matches_python_payloads(tmp_path):
    rng = np.random.default_rng(0)
    statuses, events = [], []
    for _ in range(150):
        statuses.append(DeviceStatusPayload(
            origin=int(rng.integers(1, 3)), battery_percent=int(rng.integers(0, 101)), battery_mv=int(rng.integers(0, 65536)),
            emg_quality=int(rng.integers(0, 5)), imu_quality=int(rng.integers(0, 5)),
            heartbeat_seq=int(rng.integers(0, 65536)), uptime_s=int(rng.integers(0, 2**32))))
        events.append(EventStatusPayload(
            origin=int(rng.integers(1, 3)), event_id=int(rng.integers(0, 65536)), state=int(rng.integers(0, 4)),
            cancellation=int(rng.integers(0, 3)), timestamp_s=int(rng.integers(0, 2**32)),
            available_modalities=int(rng.integers(0, 4))))
    vectors = [["status", s.pack().hex()] for s in statuses] + [["event", e.pack().hex()] for e in events]
    out = run(vectors, tmp_path)
    for (kind, hexs), r, src in zip(vectors, out, statuses + events):
        assert r["ok"] and r["reencoded"] == hexs  # decode -> encode reproduces the exact bytes
        v = r["value"]
        assert v["origin"] == ORIGIN[int(src.origin)]
        if kind == "status":
            assert (v["batteryPercent"], v["batteryMv"], v["heartbeatSeq"], v["uptimeS"]) == \
                (src.battery_percent, src.battery_mv, src.heartbeat_seq, src.uptime_s)
            assert (v["emgQuality"], v["imuQuality"]) == (QUALITY[src.emg_quality], QUALITY[src.imu_quality])
        else:
            assert (v["eventId"], v["timestampS"], v["state"], v["cancellation"]) == \
                (src.event_id, src.timestamp_s, STATES[src.state], CANCELS[src.cancellation])
            assert (v["emgAvailable"], v["imuAvailable"]) == (bool(src.available_modalities & 1), bool(src.available_modalities & 2))


def test_typescript_decoder_rejects_what_python_rejects(tmp_path):
    good_s = DeviceStatusPayload(1, 80, 3900, 0, 1, 5, 100).pack()
    good_e = EventStatusPayload(2, 7, 2, 0, 1234, 3).pack()
    bad = [
        ("status", good_s[:12]),                      # wrong length
        ("status", b"\x02" + good_s[1:]),              # unknown protocol version
        ("status", good_s[:1] + b"\x03" + good_s[2:]),  # unknown origin
        ("status", good_s[:2] + b"\x65" + good_s[3:]),  # battery 101 %
        ("status", good_s[:5] + b"\x05" + good_s[6:]),  # unknown quality code
        ("event", good_e[:4] + b"\x04" + good_e[5:]),   # unknown event state
        ("event", good_e[:5] + b"\x03" + good_e[6:]),   # unknown cancellation code
        ("event", good_e[:10] + b"\x04"),              # modality bits outside 0..3
    ]
    out = run([[k, b.hex()] for k, b in bad], tmp_path)
    for (kind, b), r in zip(bad, out):
        assert r == {"ok": False, "error": "PayloadError"}, (kind, b.hex(), r)
        with pytest.raises(ValueError):
            (DeviceStatusPayload if kind == "status" else EventStatusPayload).unpack(b)
