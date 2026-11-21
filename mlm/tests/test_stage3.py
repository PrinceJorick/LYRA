"""Stage 3 tests on SYNTHETIC signals and synthetic BIDS-like fixtures only."""

import json

import numpy as np
import pytest
from scipy import signal as sps
from synthetic import emg, imu, write_edf

from lyra_mlm.annotations import Annotation, AnnotationKind, LabelConfig, WindowLabel, label_window
from lyra_mlm.edf import read_edf, read_edf_header
from lyra_mlm.evaluation import SEIZEIT2_PLANNED_SPLIT, LeakageError, ParticipantSplit, split_manifest
from lyra_mlm.features import feature_names
from lyra_mlm.prep import PrepConfig, prepare_recording
from lyra_mlm.preprocess import FilterConfigError, SimulatedEnvConfig, raw_to_simulated_env, simulated_env_filters
from lyra_mlm.seizeit2 import EMG_LABEL, IMU_LABELS, MissingRecordingData, SeizeIT2Adapter, SeizeIT2Mapper
from lyra_mlm.signals import DataOrigin, EmgSignalType, Recording

RAW, SIM = EmgSignalType.RAW, EmgSignalType.SIMULATED_ENV_FROM_RAW
K = AnnotationKind


def raw(seconds=10.0, rate=256.0, values=None, timestamps=None):
    return emg(seconds, rate, RAW, None, values, timestamps)


# --- raw_to_simulated_env (Decision 2) ---------------------------------------------------------

def test_filters_are_first_order_butterworth_at_the_myoware_cutoffs():
    hpf, lpf = simulated_env_filters(256.0, SimulatedEnvConfig())
    for sos, cutoff in ((hpf, 20.8), (lpf, 3.6)):
        assert sos.shape == (1, 6) and sos[0, 2] == 0 and sos[0, 5] == 0  # a single first-order section
        _, h = sps.sosfreqz(sos, worN=[cutoff], fs=256.0)
        assert abs(h[0]) == pytest.approx(1 / np.sqrt(2), abs=1e-3)  # -3 dB at the cutoff


def test_output_rate_type_and_provenance():
    s = raw()
    out = raw_to_simulated_env(s)
    assert (out.nominal_rate_hz, out.n, out.signal_type) == (50.0, 500, SIM)
    assert out.provenance[-1].startswith("SIMULATED_ENV_FROM_RAW: butterworth highpass order 1 at 20.8 Hz")
    assert "resample_poly 256 Hz -> 50 Hz (25/128)" in out.provenance[-1]
    assert "HARDWARE" not in out.provenance[-1].upper().replace("HARDWARE ENV", "")
    assert (s.nominal_rate_hz, s.signal_type) == (256.0, RAW)  # input untouched
    assert raw_to_simulated_env(raw(rate=250.0)).n == 500  # 250 Hz source: native rate read, 1/5 ratio


def test_highpass_removes_dc_and_rectified_carrier_gives_an_envelope():
    t = np.arange(256 * 10) / 256.0
    assert np.abs(raw_to_simulated_env(raw(values=np.full(t.size, 500.0))).values[50:]).max() < 1.0
    out = raw_to_simulated_env(raw(values=100 * np.sin(2 * np.pi * 80 * t)))
    hpf, _ = simulated_env_filters(256.0, SimulatedEnvConfig())
    hpf_gain = abs(sps.sosfreqz(hpf, worN=[80.0], fs=256.0)[1][0])  # digital (bilinear) gain
    assert out.values[100:].mean() == pytest.approx(100 * 2 / np.pi * hpf_gain, rel=0.01)  # mean of |sin|


def test_lowpass_keeps_slow_modulation_and_attenuates_fast_modulation():
    t = np.arange(256 * 20) / 256.0
    carrier = np.sin(2 * np.pi * 80 * t)

    def depth(mod_hz):
        v = raw_to_simulated_env(raw(20.0, values=100 * (1 + 0.5 * np.sin(2 * np.pi * mod_hz * t)) * carrier)).values
        return np.ptp(v[200:, 0])

    assert depth(20.0) < 0.25 * depth(1.0)  # 1st-order LPF at 3.6 Hz


def test_decimation_is_anti_aliased():
    """With an 80 Hz carrier every rectifier harmonic folds above 25 Hz at 256 Hz (96, 64, 32 Hz...).
    The LPF leaves some of it; naive decimation to 50 Hz would alias it below 25 Hz, resample_poly removes it."""
    t = np.arange(256 * 20) / 256.0
    v = 100 * np.sin(2 * np.pi * 80 * t)
    full = raw_to_simulated_env(raw(20.0, values=v)).values[:, 0]
    out = full[100:]
    assert abs(full[-1] / full[500] - 1) < 0.01  # no zero-padding transient at the run end
    hpf, lpf = simulated_env_filters(256.0, SimulatedEnvConfig())
    env_256 = sps.sosfilt(lpf, np.abs(sps.sosfilt(hpf, v)))
    naive = env_256[np.round(np.arange(1000) * 256 / 50).astype(int)][100:]  # pick samples, no anti-alias filter
    assert out.std() / out.mean() < 0.002
    assert naive.std() / naive.mean() > 5 * out.std() / out.mean()


def test_nyquist_and_type_violations_are_rejected():
    with pytest.raises(FilterConfigError, match="Nyquist"):
        raw_to_simulated_env(raw(rate=40.0))  # 20.8 Hz HPF >= 20 Hz Nyquist
    with pytest.raises(FilterConfigError, match="must not exceed"):
        raw_to_simulated_env(raw(), SimulatedEnvConfig(target_rate_hz=500.0))
    with pytest.raises(FilterConfigError, match="RAW EMG"):
        raw_to_simulated_env(emg())  # ENV input


def test_gaps_are_not_bridged():
    t = np.concatenate([np.arange(2560) / 256.0, 20.0 + np.arange(2560) / 256.0])
    out = raw_to_simulated_env(raw(values=np.sin(2 * np.pi * 80 * t), timestamps=t))
    assert np.diff(out.timestamps).max() > 9.0 and out.n == 1000


# --- window labelling (Decision 1) ----------------------------------------------------------

def sz(onset, duration, kind=K.SEIZURE_FBTCS):
    return Annotation(f"a{onset}", onset, duration, kind, "synthetic")


@pytest.mark.parametrize("onset, expected", [(2.01, WindowLabel.IGNORE), (2.0, WindowLabel.POSITIVE)])
def test_positive_overlap_boundary(onset, expected):
    assert label_window(0.0, 4.0, (sz(onset, 60.0),), LabelConfig()) is expected  # 1.99 s vs 2.00 s


def test_peri_ictal_buffer_and_clean_negative():
    a = (sz(100.0, 60.0),)
    cfg = LabelConfig()
    assert label_window(80.0, 84.0, a, cfg) is WindowLabel.IGNORE  # 16 s before onset: inside 30 s buffer
    assert label_window(186.0, 190.0, a, cfg) is WindowLabel.IGNORE  # 26 s after offset
    assert label_window(66.0, 70.0, a, cfg) is WindowLabel.NEGATIVE  # ends exactly at onset - 30 s
    assert label_window(1000.0, 1004.0, a, cfg) is WindowLabel.NEGATIVE


def test_non_fbtcs_seizures_and_unmapped_labels_block_negatives():
    for kind in (K.SEIZURE_OUT_OF_SCOPE, K.SEIZURE_GTCS, K.UNMAPPED):
        assert label_window(100.0, 104.0, (sz(100.0, 60.0, kind),), LabelConfig()) is WindowLabel.IGNORE
    for kind in (K.BACKGROUND, K.ARTIFACT):
        assert label_window(100.0, 104.0, (sz(100.0, 60.0, kind),), LabelConfig()) is WindowLabel.NEGATIVE


def test_label_parameters_are_configurable():
    cfg = LabelConfig(min_positive_overlap_fraction=0.25, peri_ictal_buffer_s=5.0)
    assert label_window(0.0, 4.0, (sz(3.0, 60.0),), cfg) is WindowLabel.POSITIVE  # 1 s = 25%
    assert label_window(68.0, 72.0, (sz(76.0, 10.0),), cfg) is WindowLabel.IGNORE  # buffer starts at 71 s
    assert label_window(60.0, 64.0, (sz(76.0, 10.0),), cfg) is WindowLabel.NEGATIVE


def test_seizeit2_mapper():
    m = SeizeIT2Mapper()
    assert m.map_label("sz_foc_f2b") is K.SEIZURE_FBTCS
    assert m.map_label("sz_gen_m_tonicClonic") is K.SEIZURE_GTCS
    assert m.map_label("sz_foc_ia_m_clonic") is K.SEIZURE_OUT_OF_SCOPE
    assert m.map_label("bckg") is K.BACKGROUND and m.map_label("impd") is K.ARTIFACT
    assert m.map_label("something_new") is K.UNMAPPED


# --- EDF and the BIDS adapter -----------------------------------------------------------------

STEM = "{sub}_ses-01_task-szMonitoring_run-01"


def bids(root, sub, emg_rate=256.0, emg_label=EMG_LABEL, mov=True, json_rate=None, events=()):
    stem = STEM.format(sub=sub)
    base = root / sub / "ses-01"
    for d in ("emg", "mov", "eeg"):
        (base / d).mkdir(parents=True, exist_ok=True)
    secs, rng = 60, np.random.default_rng(3)
    if emg_label is not None:
        emg_values = rng.standard_normal(int(secs * emg_rate))
        write_edf(base / "emg" / f"{stem}_emg.edf", [(emg_label, "uV", emg_rate, emg_values)])
        (base / "emg" / f"{stem}_emg.json").write_text(json.dumps({"SamplingFrequency": json_rate or emg_rate}))
    if mov:
        write_edf(base / "mov" / f"{stem}_mov.edf",
                  [(lb, "g", 25.0, rng.standard_normal(secs * 25)) for lb in IMU_LABELS])
    rows = ["onset\tduration\teventType\trecordingDuration"] + [f"{o}\t{d}\t{e}\t{secs}" for o, d, e in events]
    (base / "eeg" / f"{stem}_events.tsv").write_text("\n".join(rows) + "\n")
    return stem


def test_edf_reader_takes_rates_and_scaling_from_the_header(tmp_path):
    v = np.sin(np.arange(250 * 4) / 10.0) * 100
    write_edf(tmp_path / "x.edf", [("EMG SD", "uV", 250.0, v)])
    hdr, sig = read_edf(tmp_path / "x.edf", ("EMG SD", "absent"))
    assert hdr.rate_hz("EMG SD") == 250.0 and hdr.variant == "EDF+C" and hdr.n_records == 4
    ts, phys = sig["EMG SD"]
    assert "absent" not in sig and ts[1] == pytest.approx(1 / 250)
    np.testing.assert_allclose(phys, v, atol=0.01)


def test_edf_plus_d_record_onsets_become_timestamp_gaps(tmp_path):
    write_edf(tmp_path / "d.edf", [("EMG SD", "uV", 10.0, np.arange(40.0))], variant="EDF+D", onsets=[0, 1, 10, 11])
    _, sig = read_edf(tmp_path / "d.edf", ("EMG SD",))
    assert sig["EMG SD"][0][20] == pytest.approx(10.0)
    assert read_edf_header(tmp_path / "d.edf").variant == "EDF+D"


def test_adapter_reads_rates_dynamically_and_handles_missing_channels(tmp_path):
    a = bids(tmp_path, "sub-001", events=[(20, 15, "sz_foc_f2b"), (0, 60, "bckg")])
    b = bids(tmp_path, "sub-002", emg_rate=250.0, mov=False, json_rate=256)
    c = bids(tmp_path, "sub-003", emg_label="ECG SD", mov=False)
    ad = SeizeIT2Adapter(tmp_path)
    assert ad.recording_ids() == [a, b, c]
    i1, i2, i3 = ad.list_recordings()
    assert (i1.emg_rate_hz, i1.imu_rate_hz, i1.missing, i1.duration_s) == (256.0, 25.0, (), 60.0)
    assert (i2.emg_rate_hz, i2.imu_rate_hz, i2.missing) == (250.0, None, ("imu: no file",))
    assert "JSON says 256 Hz, EDF header gives 250 Hz" in i2.notes
    assert i3.emg_rate_hz is None and "emg: channels missing ['EMG SD']" in i3.missing
    assert (i1.emg_placement, i1.imu_placement) == ("left_deltoid", "chest_upper_back")

    rec = ad.load_recording(a)
    assert (rec.emg.nominal_rate_hz, rec.emg.signal_type, rec.imu.nominal_rate_hz) == (256.0, RAW, 25.0)
    assert [x.kind for x in rec.annotations] == [K.SEIZURE_FBTCS, K.BACKGROUND]
    assert ad.load_recording(b).imu is None
    with pytest.raises(MissingRecordingData):
        ad.load_recording(c)


# --- preparation ------------------------------------------------------------------------------

def test_prepare_recording_labels_every_window_and_fits_nothing():
    t = np.arange(256 * 200) / 256.0
    v = 100 * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)) * np.sin(2 * np.pi * 80 * t)
    rec = Recording("SYN-P01", "SYN-R01", DataOrigin.SYNTHETIC, raw(200.0, values=v), imu(200.0),
                    placement="synthetic", annotations=(sz(40.0, 60.0), sz(150.0, 10.0, K.ARTIFACT)))
    out = prepare_recording(rec, PrepConfig())
    labels = out["label"]
    assert labels.size == 99  # windows start every 2 s from 0 to 196
    assert ((labels == 1).sum(), (labels == 0).sum(), (labels == -1).sum()) == (31, 38, 30)
    assert out["primary"].shape == (99, len(feature_names(PrepConfig().primary)))
    assert out["secondary"].shape == (99, len(feature_names(PrepConfig().secondary)))
    np.testing.assert_array_equal(out["primary_available"], ~np.isnan(out["primary"]))
    assert out["artifact_overlap"].sum() == 6 and (out["overall"] == 0).all()
    assert out["primary_available"].all() and out["secondary_available"].all()


def test_prepare_marks_a_missing_modality_instead_of_dropping_the_recording():
    rec = Recording("SYN-P01", "SYN-R01", DataOrigin.SYNTHETIC, None, imu(20.0), placement="synthetic")
    out = prepare_recording(rec, PrepConfig())
    assert out["label"].size == 9 and (out["emg_state"] == 3).all() and (out["imu_state"] == 0).all()  # 3 = MISSING
    assert not out["primary_available"][:, 0].any() and not out["secondary_available"].any()


# --- split integrity --------------------------------------------------------------------------

def test_split_manifest_is_disjoint_and_complete():
    inventory = [f"sub-{i:03d}" for i in range(1, 126)]
    m = split_manifest(inventory, {"sub-005": 2, "sub-100": 1}, SEIZEIT2_PLANNED_SPLIT)
    train, test = set(m["train_cv"]["participants"]), set(m["held_out_test"]["participants"])
    assert not train & test and train | test == set(inventory)
    assert (len(train), len(test)) == (96, 29)
    assert m["train_cv"]["with_fbtcs"] == ["sub-005"] and m["held_out_test"]["fbtcs_events"] == 1
    with pytest.raises(LeakageError, match="not covered"):
        split_manifest(inventory + ["sub-200"], {}, SEIZEIT2_PLANNED_SPLIT)
    with pytest.raises(LeakageError):
        ParticipantSplit(frozenset({"sub-001"}), frozenset({"sub-001"}), "synthetic", False)
