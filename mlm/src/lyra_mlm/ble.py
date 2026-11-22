"""Draft BLE payload schemas: pod = GATT peripheral/server, phone = GATT central/client.

Byte layouts are drafts for firmware/app alignment. No service UUIDs, firmware or app
transport exist yet. Every payload carries an origin byte so a simulated payload can never pass
as a hardware reading. Ordinary operation sends small status/event payloads, not raw waveforms.
Research data transfer is documented in docs/BLE_PROTOCOL.md and deferred.
"""

from __future__ import annotations

import struct
from dataclasses import astuple, dataclass, fields
from enum import IntEnum
from typing import ClassVar

from .confirmation import ConfirmationState
from .quality import QualityState

PROTOCOL_VERSION = 1

# GATT layout: DRAFT, pending owner approval (docs/BLE_PROTOCOL.md). A random 128-bit base (uuid4),
# never the Bluetooth SIG base; the low 16 bits of the first group number the service and its parts.
# The firmware (lyra_ble.h) and the app (src/services/pod/protocol.ts) must use exactly these strings.
GATT_BASE = "28b7{:04x}-8eb8-490a-aaa2-b981ca09340c"
GATT_UUIDS = {
    "service": GATT_BASE.format(0x0001),
    "device_status": GATT_BASE.format(0x0002),  # read + notify
    "event_status": GATT_BASE.format(0x0003),  # read + indicate (acknowledged: safety-relevant)
    "device_config": GATT_BASE.format(0x0004),  # read + write with response
    "phone_command": GATT_BASE.format(0x0005),  # write with response
}


class PayloadOrigin(IntEnum):
    HARDWARE = 1
    SIMULATED = 2


class QualityCode(IntEnum):
    VALID = 0
    DEGRADED = 1
    INVALID = 2
    MISSING = 3
    UNSYNCHRONIZED = 4

    @classmethod
    def of(cls, state: QualityState) -> QualityCode:
        return cls[state.value]


class EventStateCode(IntEnum):
    IDLE = 0
    ACCUMULATING = 1
    POSSIBLE_EVENT = 2
    REFRACTORY = 3

    @classmethod
    def of(cls, state: ConfirmationState) -> EventStateCode:
        return cls[state.name]


class CancellationCode(IntEnum):
    NONE = 0
    CANCELLED_ON_POD = 1  # tactile cancellation button
    CANCELLED_ON_PHONE = 2


class CommandCode(IntEnum):
    CANCEL_EVENT = 1  # the user cancelled the alert on the phone


class _Payload:
    FORMAT: ClassVar[str]  # little-endian; first byte is always PROTOCOL_VERSION
    ENUMS: ClassVar[dict[str, type[IntEnum]]] = {}
    RANGES: ClassVar[dict[str, tuple[int, int]]] = {}

    def __post_init__(self) -> None:
        for name, enum in self.ENUMS.items():
            object.__setattr__(self, name, enum(getattr(self, name)))  # unknown codes raise
        for name, (lo, hi) in self.RANGES.items():
            if not lo <= getattr(self, name) <= hi:
                raise ValueError(f"{name}={getattr(self, name)} outside {lo}..{hi}")

    def pack(self) -> bytes:
        return struct.pack(self.FORMAT, PROTOCOL_VERSION, *astuple(self))

    @classmethod
    def unpack(cls, data: bytes):
        if len(data) != struct.calcsize(cls.FORMAT):
            raise ValueError(f"{cls.__name__}: expected {struct.calcsize(cls.FORMAT)} bytes, got {len(data)}")
        version, *values = struct.unpack(cls.FORMAT, data)
        if version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version {version}")
        return cls(**dict(zip((f.name for f in fields(cls)), values)))


@dataclass(frozen=True)
class DeviceStatusPayload(_Payload):
    """Battery (MAX17048), per-modality sensor health and heartbeat."""

    FORMAT = "<BBBHBBHI"
    ENUMS = {"origin": PayloadOrigin, "emg_quality": QualityCode, "imu_quality": QualityCode}
    RANGES = {"battery_percent": (0, 100)}
    origin: PayloadOrigin
    battery_percent: int
    battery_mv: int
    emg_quality: QualityCode
    imu_quality: QualityCode
    heartbeat_seq: int
    uptime_s: int


@dataclass(frozen=True)
class EventStatusPayload(_Payload):
    """Possible-event state from EventConfirmation plus user cancellation status."""

    FORMAT = "<BBHBBIB"
    ENUMS = {"origin": PayloadOrigin, "state": EventStateCode, "cancellation": CancellationCode}
    RANGES = {"available_modalities": (0, 3)}
    origin: PayloadOrigin
    event_id: int
    state: EventStateCode
    cancellation: CancellationCode
    timestamp_s: int  # pod uptime at the state change
    available_modalities: int  # bit 0 = EMG, bit 1 = IMU


@dataclass(frozen=True)
class DeviceConfigPayload(_Payload):
    """Phone -> pod configuration. EMG rate is not yet specified for the hardware."""

    FORMAT = "<BBHBHHB"
    ENUMS = {"origin": PayloadOrigin}
    RANGES = {"emg_rate_hz": (1, 65535), "window_ms": (1, 65535), "step_ms": (1, 65535),
              "countdown_s": (7, 10)}
    origin: PayloadOrigin
    emg_rate_hz: int
    imu_odr_hz: int  # 25 initial, 50 experimental
    window_ms: int
    step_ms: int
    countdown_s: int  # phone cancellation countdown; 8 s app default vs 10 s pod spec is unresolved

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.imu_odr_hz not in (25, 50):
            raise ValueError("imu_odr_hz must be 25 (initial) or 50 (experimental)")
        if not 0 < self.step_ms <= self.window_ms:
            raise ValueError("need 0 < step_ms <= window_ms")


@dataclass(frozen=True)
class PhoneCommandPayload(_Payload):
    """Phone -> pod command (DRAFT). CANCEL_EVENT names the event_id of the EventStatus being cancelled.
    The pod ignores a stale id and confirms a valid cancel with EventStatus(IDLE, CANCELLED_ON_PHONE)."""

    FORMAT = "<BBBH"
    ENUMS = {"origin": PayloadOrigin, "command": CommandCode}
    origin: PayloadOrigin
    command: CommandCode
    event_id: int
