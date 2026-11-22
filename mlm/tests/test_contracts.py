import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from synthetic import emg, model_config

from lyra_mlm.annotations import Annotation, AnnotationKind
from lyra_mlm.ble import (
    GATT_UUIDS, CancellationCode, CommandCode, DeviceConfigPayload, DeviceStatusPayload, EventStateCode,
    EventStatusPayload, PayloadOrigin, PhoneCommandPayload, QualityCode,
)
from lyra_mlm.confirmation import ConfirmationState
from lyra_mlm.datasets import SOURCES, sha256_file, validate_recording
from lyra_mlm.evaluation import (
    METRICS, SEIZEIT2_PLANNED_SPLIT, EvaluationReport, LeakageError, ParticipantSplit, Partition,
    check_participant_independence,
)
from lyra_mlm.model import FeatureDataset, ModelConfig, ensure_exportable, require_training_partition
from lyra_mlm.preprocess import NormalizationParams
from lyra_mlm.quality import QualityState
from lyra_mlm.signals import DataOrigin, Recording

SRC = Path(__file__).parents[1] / "src" / "lyra_mlm"


def test_model_config_rejects_wrong_feature_order():
    cfg = model_config()
    with pytest.raises(ValueError, match="feature schema"):
        ModelConfig(cfg.model_id, cfg.family, cfg.feature_config, tuple(reversed(cfg.feature_names)),
                    cfg.window, 0.5)


def test_model_config_json_round_trip_keeps_normalization():
    names = model_config().feature_names
    params = NormalizationParams("robust", names, (0.0,) * len(names), (1.0,) * len(names), 3)
    cfg = model_config(normalization=params)
    assert ModelConfig.from_json(cfg.to_json()) == cfg


def test_untrained_models_cannot_be_exported():
    with pytest.raises(ValueError, match="untrained"):
        ensure_exportable(model_config())


def test_training_only_accepts_the_training_partition():
    data = FeatureDataset(np.zeros((2, 1)), np.zeros(2), ("f",), ("sub-097", "sub-098"), Partition.TEST)
    with pytest.raises(LeakageError):
        require_training_partition(data)


def test_planned_seizeit2_split_is_participant_independent_and_verified():
    s = SEIZEIT2_PLANNED_SPLIT
    assert (len(s.train), len(s.held_out), s.verified) == (96, 29, True)
    assert s.partition_of("sub-096") is Partition.TRAIN and s.partition_of("sub-097") is Partition.TEST
    assert s.partition_of("sub-200") is None
    with pytest.raises(LeakageError):
        ParticipantSplit(frozenset({"a"}), frozenset({"a"}), "synthetic", False)


def test_participant_leakage_is_detected():
    with pytest.raises(LeakageError, match="p1"):
        check_participant_independence({Partition.TRAIN: ["p1", "p2"], Partition.TEST: ["p1"]})
    check_participant_independence({Partition.TRAIN: ["p1", "p1"], Partition.TEST: ["p2"]})


def test_evaluation_contract():
    assert {m.name for m in METRICS} >= {"event_sensitivity", "false_alarms_per_24h", "detection_latency_s",
                                         "precision", "auprc", "specificity", "per_participant"}
    with pytest.raises(LeakageError):
        EvaluationReport(Partition.TRAIN, "synthetic")
    assert EvaluationReport(Partition.TEST, "synthetic").event_sensitivity is None  # nothing computed


def test_dataset_registry_marks_verification_status():
    assert set(SOURCES) == {"seizeit2", "ninapro_db2", "ninapro_db7", "lyra_pod"}
    assert "CC0" in SOURCES["seizeit2"].license
    assert all("NOT VERIFIED" in SOURCES[k].license for k in ("ninapro_db2", "ninapro_db7"))


def test_validate_recording_reports_integrity_issues():
    t = emg().timestamps.copy()
    t[5] = t[4]
    rec = Recording("p", "r", DataOrigin.SYNTHETIC, emg=emg(timestamps=t),
                    annotations=(Annotation("a", 100.0, 1.0, AnnotationKind.UNMAPPED, "x"),))
    issues = validate_recording(rec)
    assert any("duplicate" in i for i in issues)
    assert "sensor placement not specified" in issues
    assert any("outside the recorded span" in i for i in issues)


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"lyra")
    assert sha256_file(p, chunk=2) == hashlib.sha256(b"lyra").hexdigest()


@pytest.mark.parametrize("payload", [
    DeviceStatusPayload(PayloadOrigin.HARDWARE, 87, 3912, QualityCode.VALID, QualityCode.DEGRADED, 42, 3600),
    EventStatusPayload(PayloadOrigin.SIMULATED, 7, EventStateCode.POSSIBLE_EVENT, CancellationCode.NONE, 1234, 3),
    DeviceConfigPayload(PayloadOrigin.SIMULATED, 100, 25, 4000, 2000, 8),
    PhoneCommandPayload(PayloadOrigin.HARDWARE, CommandCode.CANCEL_EVENT, 7),
])
def test_ble_payloads_round_trip_with_origin(payload):
    data = payload.pack()
    assert type(payload).unpack(data) == payload
    assert data[1] == payload.origin  # origin travels in every payload


def test_ble_rejects_unknown_origin_version_and_length():
    data = bytearray(EventStatusPayload(PayloadOrigin.HARDWARE, 1, EventStateCode.IDLE,
                                        CancellationCode.NONE, 0, 0).pack())
    with pytest.raises(ValueError):
        EventStatusPayload.unpack(bytes(data[:-1]))
    bad_origin = bytearray(data)
    bad_origin[1] = 9
    with pytest.raises(ValueError):
        EventStatusPayload.unpack(bytes(bad_origin))
    bad_version = bytearray(data)
    bad_version[0] = 2
    with pytest.raises(ValueError, match="version"):
        EventStatusPayload.unpack(bytes(bad_version))


def test_ble_config_ranges():
    with pytest.raises(ValueError):
        DeviceConfigPayload(PayloadOrigin.SIMULATED, 100, 25, 4000, 2000, 11)
    with pytest.raises(ValueError, match="imu_odr_hz"):
        DeviceConfigPayload(PayloadOrigin.SIMULATED, 100, 100, 4000, 2000, 8)


def test_ble_codes_cover_pipeline_states():
    assert {QualityCode.of(s).name for s in QualityState} == {s.value for s in QualityState}
    assert {EventStateCode.of(s).name for s in ConfirmationState} == {s.name for s in ConfirmationState}


def test_stage4_leakage_and_artifact_integrity_guard():
    """Replaces the pre-training guard (Stage 4 authorized 2026-11-22): fitting and ML libraries only in
    the Stage 4 modules, model files only in the gitignored mlm/models/, frozen artifacts unchanged and
    trained on Train/CV participants only."""
    import json
    import subprocess
    from lyra_mlm.evaluation import SEIZEIT2_PLANNED_SPLIT
    root = SRC.parents[1]
    files = {p: p.read_text() for d in (SRC, root / "scripts") for p in d.rglob("*.py")}
    assert {p.name for p, t in files.items() if re.search(r"\.fit\(", t)} <= {"training.py", "train_stage4.py"}
    ml = {p.name for p, t in files.items() if re.search(r"^\s*(import|from)\s+(sklearn|xgboost|joblib|m2cgen|pickle)\b", t, re.M)}
    assert ml <= {"training.py", "train_stage4.py", "evaluate_heldout.py", "export_c99.py"}
    artifacts = [p for ext in ("*.pkl", "*.joblib", "*.onnx", "*.h5", "*.ubj") for p in root.rglob(ext) if ".venv" not in p.parts]
    assert all(p.parent == root / "models" for p in artifacts)
    assert subprocess.run(["git", "check-ignore", "-q", str(root / "models" / "x.joblib")], cwd=root).returncode == 0
    pipe = root / "models" / "frozen_pipeline.json"
    if pipe.exists():
        meta = json.loads(pipe.read_text())
        assert set(meta["training"]["participants"]) <= SEIZEIT2_PLANNED_SPLIT.train
        for m in meta["models"].values():
            assert hashlib.sha256((root / "models" / m["file"]).read_bytes()).hexdigest() == m["sha256"]

def test_phone_command_is_five_bytes_and_rejects_unknown_commands():
    data = PhoneCommandPayload(PayloadOrigin.SIMULATED, CommandCode.CANCEL_EVENT, 513).pack()
    assert data == bytes([1, 2, 1, 1, 2])  # version, origin, CANCEL_EVENT, event_id 513 little-endian
    for bad in (data[:4], data[:2] + b"\x00" + data[3:], data[:2] + b"\x02" + data[3:]):
        with pytest.raises(ValueError):
            PhoneCommandPayload.unpack(bad)


def test_gatt_uuids_are_distinct_custom_128_bit_uuids():
    import uuid
    ids = [uuid.UUID(u) for u in GATT_UUIDS.values()]
    assert len(set(ids)) == len(ids) == 5
    assert not any(str(u).endswith("-0000-1000-8000-00805f9b34fb") for u in ids)  # not the Bluetooth SIG base
