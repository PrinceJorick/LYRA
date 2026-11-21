# Draft BLE protocol: pod ↔ phone

Draft for alignment only. No GATT service or characteristic UUIDs are assigned, and no firmware or phone
transport exists. Reference implementation: `lyra_mlm/ble.py` (pack/unpack with validation).

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
| emg_rate_hz | u16 | MyoWare ENV acquisition rate: **not yet specified** |
| imu_odr_hz | u8 | 25 (initial) or 50 (experimental) only |
| window_ms, step_ms | u16, u16 | 0 < step ≤ window |
| countdown_s | u8 | 7–10. App default 8 s vs pod spec 10 s is unresolved |

### 4. Research data transfer (deferred)

For consented research sessions only: chunked transfer of buffered windows with sequence numbers, CRC,
per-chunk origin and the `SignalStream` metadata (rates, signal type, flags). To be designed with the
firmware. It is not part of normal operation.
