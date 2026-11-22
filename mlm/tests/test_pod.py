"""Pod session format, adapter, integrity checks and domain report, on SYNTHETIC sessions only."""

import json

import numpy as np
import pytest

from lyra_mlm.annotations import AnnotationKind as K
from lyra_mlm.domain import report
from lyra_mlm.pod import PodAdapter, PodFormatError, check_session, map_event
from lyra_mlm.signals import DataOrigin, EmgSignalType
from lyra_mlm.training import PRIMARY_NAMES

EVENTS = [(0, 30, "rest_seated"), (30, 30, "brushing_teeth"), (70, 30, "sim_convulsion"), (105, 5, "made_up_label")]


def write_session(root, participant="pod-p001", session="s01", seconds=120, raw=True, events=EVENTS, **meta_over):
    """A SYNTHETIC pod session in the protocol format (generated signals, not physiological data)."""
    d = root / participant / session
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    te, tr, ti = np.arange(seconds * 50) / 50, np.arange(seconds * 1000) / 1000, np.arange(seconds * 25) / 25
    act = lambda t: 1 + 4 * ((t >= 70) & (t < 100)) * (1 + np.sin(2 * np.pi * 3 * t))  # noqa: E731
    np.savetxt(d / "env.csv", np.column_stack([te, 40 * act(te) + rng.gamma(2, 2, te.size)]), delimiter=",",
               header="t_s,mV", comments="", fmt="%.6f")
    if raw:
        np.savetxt(d / "raw.csv", np.column_stack([tr, 1650 + 30 * act(tr) * rng.standard_normal(tr.size)]),
                   delimiter=",", header="t_s,mV", comments="", fmt="%.6f")
    shake = ((ti >= 70) & (ti < 100))[:, None] * 0.7 * np.sin(2 * np.pi * 3 * ti)[:, None]
    imu = np.column_stack([np.full((ti.size, 3), [0.1, -0.2, 0.97]) + shake + 0.02 * rng.standard_normal((ti.size, 3)),
                           3 * rng.standard_normal((ti.size, 3)) + 80 * shake])
    np.savetxt(d / "imu.csv", np.column_stack([ti, imu]), delimiter=",", header="t_s,ax,ay,az,gx,gy,gz",
               comments="", fmt="%.6f")
    (d / "events.tsv").write_text("onset\tduration\teventType\tsource\n" +
                                  "".join(f"{o}\t{du}\t{e}\tsynthetic\n" for o, du, e in events))
    meta = {"schema_version": 1, "participant_id": participant, "session_id": session,
            "consent_protocol": "SYNTHETIC-TEST", "pod_id": "pod-dev-0", "firmware": "0.0.0-test",
            "placement": {"side": "left", "muscle": "deltoid"}, "duration_s": seconds,
            "streams": {"env": {"rate_hz": 50.0, "unit": "mV", "full_scale_mv": 3300},
                        "imu": {"rate_hz": 25.0, "units": ["g", "g", "g", "deg/s", "deg/s", "deg/s"]},
                        **({"raw": {"rate_hz": 1000.0, "unit": "mV", "full_scale_mv": 3300}} if raw else {})}}
    meta.update(meta_over)
    (d / "session.json").write_text(json.dumps(meta))
    return f"{participant}_{session}"


def test_adapter_loads_a_session_as_hardware_recording(tmp_path):
    rid = write_session(tmp_path)
    ad = PodAdapter(tmp_path)
    assert ad.recording_ids() == [rid]
    info = ad.info(rid)
    assert (info.origin, info.emg_signal_type, info.emg_rate_hz, info.imu_rate_hz) == (DataOrigin.HARDWARE, EmgSignalType.ENV, 50.0, 25.0)
    rec = ad.load_recording(rid)
    assert rec.origin is DataOrigin.HARDWARE and rec.emg.signal_type is EmgSignalType.ENV and rec.emg.units == ("mV",)
    assert rec.imu.units == ("g",) * 3 + ("deg/s",) * 3 and rec.emg.n == 6000 and rec.imu.n == 3000
    assert [a.kind for a in rec.annotations] == [K.BACKGROUND, K.HARD_NEGATIVE, K.SIMULATED_CONVULSION, K.UNMAPPED]
    assert rec.annotations[1].activity == "brushing_teeth"
    raw = ad.load_raw(rid)
    assert raw.signal_type is EmgSignalType.RAW and raw.nominal_rate_hz == 1000.0


@pytest.mark.parametrize("change, message", [
    ({"consent_protocol": ""}, "consent_protocol"),
    ({"participant_id": "John Doe"}, "pseudonym"),
    ({"schema_version": 2}, "schema_version"),
])
def test_adapter_refuses_sessions_that_break_the_protocol(tmp_path, change, message):
    rid = write_session(tmp_path, **change) if "participant_id" not in change else None
    if rid is None:  # a name as participant id: write under a valid folder, then break session.json
        rid = write_session(tmp_path)
        path = tmp_path / "pod-p001" / "s01" / "session.json"
        path.write_text(json.dumps(json.loads(path.read_text()) | change))
    with pytest.raises(PodFormatError, match=message):
        PodAdapter(tmp_path).load_recording(rid)


def test_wrong_units_are_refused(tmp_path):
    rid = write_session(tmp_path, streams={"env": {"rate_hz": 50.0, "unit": "adc_counts"}})
    with pytest.raises(PodFormatError, match="unit"):
        PodAdapter(tmp_path).meta(rid)


def test_check_session_reports_timing_problems_and_unknown_labels(tmp_path):
    rid = write_session(tmp_path)
    ad = PodAdapter(tmp_path)
    assert check_session(ad, rid) == ["events.tsv row 3: unknown eventType 'made_up_label' (UNMAPPED)"]
    env = tmp_path / "pod-p001" / "s01" / "env.csv"
    lines = env.read_text().splitlines()
    lines[101] = lines[100]  # duplicate timestamp
    env.write_text("\n".join(lines[:3000] + lines[3200:]) + "\n")  # and a 4 s gap
    problems = check_session(ad, rid)
    assert any("duplicate" in p for p in problems) and any("gaps" in p for p in problems)


def test_event_vocabulary():
    assert map_event("sim_convulsion") == (K.SIMULATED_CONVULSION, None)
    assert map_event("walking") == (K.HARD_NEGATIVE, "walking")
    assert map_event("loose_electrode") == (K.ARTIFACT, "loose_electrode")
    assert map_event("sleep") == (K.BACKGROUND, None)
    assert map_event("seizure?") == (K.UNMAPPED, None)


def test_domain_report_compares_both_paths_and_keeps_sim_convulsions_out_of_negatives(tmp_path):
    rid = write_session(tmp_path)
    ad = PodAdapter(tmp_path)
    rng = np.random.default_rng(1)
    ref_neg, ref_pos = rng.normal(5, 2, (500, 22)), rng.normal(9, 3, (50, 22))
    r = report(ad.load_recording(rid), ad.load_raw(rid), ad.meta(rid), ref_neg, ref_pos)
    assert set(r) >= {"env", "simenv", "env_vs_simenv_signal", "warning"}
    for path in ("env", "simenv"):
        cmp_ = r[path]["vs_seizeit2_negatives"]
        assert list(cmp_) == list(PRIMARY_NAMES)
        assert all(0 <= v["outside_ref_1_99"] <= 1 and 0 <= v["ks"] <= 1 for v in cmp_.values() if "ks" in v)
        assert r[path]["sim_convulsion_vs_seizeit2_fbtcs"]
        assert r[path]["events"]["sim_convulsion"] > 0
    assert r["simenv"]["sim_convulsion_windows_never_negative"] is True
    assert r["env_vs_simenv_signal"]["paired_samples"] > 5000 and "pearson_r" in r["env_vs_simenv_signal"]
    json.dumps(r, default=float)  # serializable


def test_env_only_session_reports_the_env_path(tmp_path):
    rid = write_session(tmp_path, raw=False)
    ad = PodAdapter(tmp_path)
    r = report(ad.load_recording(rid), ad.load_raw(rid), ad.meta(rid), np.ones((10, 22)), np.ones((2, 22)))
    assert "env" in r and "simenv" not in r and "env_vs_simenv_signal" not in r
