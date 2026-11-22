# Draft BLE protocol: pod ↔ phone

Draft for alignment only. The GATT layout below is a **draft proposal awaiting owner approval**, and no
real BLE transport exists yet. There are three byte-identical implementations:
- `lyra_mlm/ble.py`: the reference;
- `firmware/components/lyra_pod/lyra_ble.c`: pod side;
- `src/services/pod/protocol.ts`: app decoder.

Parity is tested in `mlm/tests/test_firmware_core.py` and `test_app_pod_protocol.py`, including the
UUID strings. The app receives payloads through `src/services/pod` (`PodLink`), which today is a
simulated pod emitting real payload bytes.

## GATT layout (DRAFT, pending owner approval)

There is one custom service. The base is a random 128-bit UUID (uuid4), never the Bluetooth SIG base.
The low 16 bits of the first group number the service and its characteristics.

| Part | UUID | Properties | Payload |
|---|---|---|---|
| LYRA Pod service | `28b70001-8eb8-490a-aaa2-b981ca09340c` | primary service | |
| Device status | `28b70002-8eb8-490a-aaa2-b981ca09340c` | read, notify | §1 (13 B) |
| Event status | `28b70003-8eb8-490a-aaa2-b981ca09340c` | read, **indicate** | §2 (11 B) |
| Device configuration | `28b70004-8eb8-490a-aaa2-b981ca09340c` | read, write with response | §3 (10 B) |
| Phone command | `28b70005-8eb8-490a-aaa2-b981ca09340c` | write with response | §5 (5 B) |

- Event status uses **indicate**, which the phone acknowledges at link level, so a possible event or
  a cancellation is not silently lost. Status uses notify, because a missed heartbeat is replaced by
  the next one.
- A malformed write (wrong length, version, origin or code) is rejected with an ATT application
  error. It is never applied partly.
- Still open: pairing and bonding/security mode, connection parameters, MTU (all payloads fit the
  default 20 bytes), and whether to add the standard Battery and Device Information services.

- **Roles:** ESP32-S3 pod = BLE peripheral / GATT server. Phone = central / GATT client.
- **Normal operation** sends small status and event notifications only, never continuous raw waveforms.
- **Every payload starts with** `version` (u8, currently 1) and `origin` (u8: 1 = HARDWARE,
  2 = SIMULATED). The phone must show simulated payloads as simulated and never as hardware readings.
- Little-endian. Unknown versions, origins, enum codes and wrong lengths are rejected.

## Domains

### 1. Device status (pod → phone, notify): `DeviceStatusPayload`, 13 bytes, `<BBBHBBHI`

| Field | Type | Notes |
|---|---|---|
| version, origin | u8, u8 | |
| battery_percent | u8 | 0–100, MAX17048 |
| battery_mv | u16 | |
| emg_quality, imu_quality | u8, u8 | `QualityCode`: 0 VALID, 1 DEGRADED, 2 INVALID, 3 MISSING, 4 UNSYNCHRONIZED |
| heartbeat_seq | u16 | increments each status; gaps reveal dropped notifications |
| uptime_s | u32 | pod clock |

### 2. Event status (pod → phone, notify/indicate): `EventStatusPayload`, 11 bytes, `<BBHBBIB`

| Field | Type | Notes |
|---|---|---|
| version, origin | u8, u8 | |
| event_id | u16 | per possible event |
| state | u8 | `EventStateCode`: 0 IDLE, 1 ACCUMULATING, 2 POSSIBLE_EVENT, 3 REFRACTORY |
| cancellation | u8 | 0 NONE, 1 CANCELLED_ON_POD (tactile button), 2 CANCELLED_ON_PHONE |
| timestamp_s | u32 | pod uptime at the state change |
| available_modalities | u8 | bit 0 EMG, bit 1 IMU |

On `POSSIBLE_EVENT` the phone shows **"Possible convulsive seizure detected. This is not a diagnosis."**
and starts the cancellation countdown.

### 3. Device configuration (phone → pod, write): `DeviceConfigPayload`, 10 bytes, `<BBHBHHB`

| Field | Type | Notes |
|---|---|---|
| version, origin | u8, u8 | |
| emg_rate_hz | u16 | MyoWare ENV ADC rate; **50 Hz** (locked Decision 3) |
| imu_odr_hz | u8 | 25 (initial) or 50 (experimental) only |
| window_ms, step_ms | u16, u16 | 0 < step ≤ window |
| countdown_s | u8 | 7–10. App default 8 s vs pod spec 10 s is unresolved |

### 5. Phone command (phone → pod, write with response): `PhoneCommandPayload`, 5 bytes, `<BBBH` (DRAFT)

| Field | Type | Notes |
|---|---|---|
| version, origin | u8, u8 | origin SIMULATED when the phone is talking to a simulated pod |
| command | u8 | 1 CANCEL_EVENT: the user cancelled the alert on the phone. Any other code is rejected |
| event_id | u16 | the `event_id` of the event status being cancelled |

On CANCEL_EVENT for the pod's current `event_id`, the pod cancels as if its own button had been
pressed (`lyra_pod_on_phone_command` → `lyra_pod_cancel`). Event confirmation goes back to IDLE with
no cooldown, and the pod indicates an event status `(IDLE, CANCELLED_ON_PHONE)` as confirmation. A
stale or unknown `event_id` is ignored, so a late cancel never resets a newer event.

### 4. Research data transfer (deferred)

For consented research sessions only: chunked transfer of buffered windows with sequence numbers, CRC,
per-chunk origin and the `SignalStream` metadata (rates, signal type, flags). To be designed with the
firmware. It is not part of normal operation.
