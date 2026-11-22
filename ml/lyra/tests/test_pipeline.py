"""Unit tests for ml/lyra (synthetic data only; no network, no dataset access)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pyedflib
import pytest
from scipy.special import expit

from ml.lyra import benchmark as Bm
from ml.lyra import bids
from ml.lyra import config as C
from ml.lyra import data as Dt
from ml.lyra import events as Ev
from ml.lyra import export_c as X
from ml.lyra import features as F
from ml.lyra import golden as G
from ml.lyra import signals as S
from ml.lyra import windows as Wn

REPO = Path(__file__).resolve().parents[3]
RNG = np.random.default_rng(7)


# ---- EDF loading, recording matching, annotations ------------------------------------------------------


def _edf(path: Path, labels, rate, data, dim="uV", pmin=-2000.0, pmax=2000.0):
    with pyedflib.EdfWriter(str(path), len(labels), file_type=pyedflib.FILETYPE_EDFPLUS) as w:
        w.setSignalHeaders([{"label": lb, "dimension": dim, "sample_frequency": rate, "physical_min": pmin,
                             "physical_max": pmax, "digital_min": -32768, "digital_max": 32767, "transducer": "",
                             "prefilter": ""} for lb in labels])
        w.writeSamples([d for d in data])


def test_edf_loading_matches_pyedflib_physical_values_and_flags_rails(tmp_path):
    x = RNG.normal(0, 300, 256 * 4)
    x[100:110] = 5000.0  # beyond physical max -> clipped to the rail
    p = tmp_path / "e.edf"
    _edf(p, ["EMG SD"], 256, [x])
    hdr, dig, phys = S.read_edf(p, ("EMG SD",))
    with pyedflib.EdfReader(str(p)) as f:
        ref = f.readSignal(0)
    assert np.allclose(phys[:, 0], ref, atol=1e-9)
    assert hdr["channels"][0]["rate_hz"] == 256.0
    rails = S.at_rails(dig, hdr)[:, 0]
    assert rails[100:110].all() and rails.sum() >= 10
    with pytest.raises(KeyError):
        S.read_edf(p, ("missing",))


def _bids(root: Path, rid: str, kinds=("emg", "mov", "events")):
    sub, ses = rid.split("_")[:2]
    for k in kinds:
        d = root / sub / ses / {"emg": "emg", "mov": "mov", "events": "eeg"}[k]
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{rid}_{k}.{'tsv' if k == 'events' else 'edf'}").write_text("x")


def test_recordings_are_matched_on_bids_entities_including_three_digit_runs(tmp_path):
    _bids(tmp_path, "sub-001_ses-01_task-szMonitoring_run-01")
    _bids(tmp_path, "sub-002_ses-01_task-szMonitoring_run-100", ("emg", "events"))
    recs, problems = bids.match_recordings(tmp_path)
    assert [r.recording_id for r in recs] == ["sub-001_ses-01_task-szMonitoring_run-01",
                                               "sub-002_ses-01_task-szMonitoring_run-100"]
    assert recs[0].problems == [] and recs[1].problems == ["no mov file"] and problems == []


def test_annotations_map_target_nontarget_artifact_and_skip_malformed_impd(tmp_path):
    rid = "sub-003_ses-01_task-szMonitoring_run-02"
    d = tmp_path / "sub-003" / "ses-01" / "eeg"
    d.mkdir(parents=True)
    rows = [("0", "3600", "bckg"), ("100", "60", "sz_foc_f2b"), ("500", "30", "sz_foc_ia_nm"),
            ("800", "20", "impd"), ("900", "-5", "impd"), ("1000", "40", "sz_gen_m_tonicClonic")]
    (d / f"{rid}_events.tsv").write_text(
        "onset\tduration\teventType\tlateralization\tlocalization\tvigilance\tconfidence\tchannels\tdateTime\trecordingDuration\n"
        + "".join(f"{o}\t{du}\t{e}\tn/a\tn/a\tn/a\tn/a\tn/a\t2000-01-01 00:00:00\t3600\n" for o, du, e in rows))
    ev = bids.parse_events(rid, tmp_path)
    assert [c for *_, c in ev.target] == ["sz_foc_f2b", "sz_gen_m_tonicClonic"]
    assert [c for *_, c in ev.nontarget] == ["sz_foc_ia_nm"]
    assert len(ev.artifact) == 1 and len(ev.skipped) == 1 and ev.recording_duration_s == 3600


# ---- split integrity ---------------------------------------------------------------------------------


def test_partition_boundaries_and_locked_test_cohort():
    assert C.is_dev("sub-096_ses-01_task-szMonitoring_run-01") and not C.is_dev("sub-097")
    assert set(C.DEV).isdisjoint(C.TEST) and len(C.DEV) + len(C.TEST) == 125
    if not Dt.LOCK.exists():
        with pytest.raises(PermissionError):
            Dt.load("w4", "test")


# ---- signals, windows, artifacts, synchronisation --------------------------------------------------------


def test_env_simulation_is_causal_and_50hz():
    raw = RNG.normal(0, 50, 256 * 20)
    env = S.simulate_env(raw, 256)
    assert env.size == 20 * 50 and (env >= 0).all()
    raw2 = raw.copy()
    raw2[256 * 10:] = RNG.normal(0, 500, raw2.size - 256 * 10)  # change the future only
    assert np.array_equal(S.simulate_env(raw2, 256)[:10 * 50 - 1], env[:10 * 50 - 1])


def _events(**kw):
    base = dict(target=[], nontarget=[], artifact=[], other=[], skipped=[], unmapped_seizure_codes=[],
                recording_duration_s=None)
    return bids.Intervals(**(base | kw))


def _recording(seconds=60, env_seconds=None):
    n = seconds * 25
    acc = np.tile([0.0, 0.0, 9.81], (n, 1)) + RNG.normal(0, 0.05, (n, 3))
    gyr = RNG.normal(0, 0.01, (n, 3))
    raw = RNG.normal(0, 50, (env_seconds or seconds) * 256)
    env = S.simulate_env(raw, 256)
    raw_bad = np.zeros(raw.size, bool)
    return env, (raw_bad, np.diff(raw) != 0, 256), acc, gyr, raw


def test_windows_labels_artifacts_saturation_flatline_and_sync_truncation():
    env, src, acc, gyr, raw = _recording(60, env_seconds=58)  # EMG shorter: windows stop at 58 s
    bad = src[0].copy()
    bad[256 * 20:256 * 23] = True  # 3 s saturated
    changes = src[1].copy()
    changes[256 * 40:256 * 44] = False  # 4 s flat EMG
    ev = _events(target=[(30.5, 33.2, "sz_foc_f2b")], nontarget=[(50.0, 52.0, "sz_foc_ia_nm")],
                 artifact=[(10.0, 12.0, "impd")])
    out = Wn.build(env, (bad, changes, 256), acc, gyr, (np.zeros(len(acc), bool), np.ones(len(acc) - 1, bool)), True, ev)
    t = out["w2_t_end"]
    assert t[-1] <= 58 and np.allclose(np.diff(t), 1.0) and t[0] == 2.0
    lab = Wn.labels(out["w2_tfrac"])
    # [29,31): 25% -> excluded; [30,32) 75%, [31,33) 100%, [32,34) 60% -> positive; [33,35) 10% -> excluded
    assert lab[t == 31.0] == -1 and (lab[(t >= 32) & (t <= 34)] == 1).all() and lab[t == 35.0] == -1
    assert lab[t == 36.0] == 0 and (lab == -1).sum() == 2 and (lab == 1).sum() == 3
    assert out["w2_artifact"][(t > 10) & (t < 14)].all() and not out["w2_artifact"][t == 15].any()
    assert not out["w2_emg_ok"][(t > 21) & (t <= 23)].any()  # > 5% saturated
    assert not out["w2_emg_ok"][(t >= 42) & (t <= 44)].any()  # flat window
    assert out["w2_imu_ok"].all() and out["w2_warmup"][t < 3].all()
    assert out["w2_nontarget"][t == 51.0]
    assert np.isfinite(out["w4_X"]).all() and out["w5_t_end"][1] - out["w5_t_end"][0] == pytest.approx(2.48)


def test_uncalibratable_imu_marks_every_imu_window_invalid():
    env, src, acc, gyr, _ = _recording(30)
    out = Wn.build(env, src, acc, gyr, (np.zeros(len(acc), bool), np.ones(len(acc) - 1, bool)), False, _events())
    assert not out["w4_imu_ok"].any() and out["w4_emg_ok"].all()


# ---- features ------------------------------------------------------------------------------------------


def test_feature_edge_cases_and_known_signals():
    n = 100  # 4 s
    t = np.arange(n) / 25
    acc = np.column_stack([np.zeros(n), np.zeros(n), 9.81 + 3 * np.sin(2 * np.pi * 3 * t)])[None]  # along gravity
    gyr = np.zeros((1, n, 3))
    env = (10 + 5 * np.sin(2 * np.pi * 2 * np.arange(2 * n) / 50))[None]
    x = F.compute(env, acc, gyr, 1)[0]
    f = dict(zip(C.FEATURES, x))
    assert f["imu_acc_dom_freq_hz"] == pytest.approx(3.0, abs=0.25)
    assert f["imu_acc_band_ratio_1_5hz"] > 0.5
    assert f["emg_acf_peak"] == pytest.approx((200 - 25) / 200)  # biased ACF of a 2 Hz envelope at lag 25
    assert f["imu_gyr_mag_mean"] == 0 and f["emg_mav"] == pytest.approx(10, rel=1e-3)
    c = F.compute(np.full((1, 2 * n), 5.0), np.tile([0, 0, 9.81], (1, n, 1)), np.zeros((1, n, 3)), 1)[0]
    d = dict(zip(C.FEATURES, c))
    assert d["emg_acf_peak"] == 0 and d["imu_acc_band_ratio_1_5hz"] == 0 and d["imu_acc_dom_freq_hz"] == 0
    assert d["xmod_corr_env_accmag"] == 0 and d["xmod_coh_1_5hz"] == 0 and d["imu_tilt_deg"] == 0
    ang = np.linspace(0, np.pi / 2, n)[:, None]
    tilt = F.compute(env, (9.81 * np.hstack([np.sin(ang), np.zeros((n, 1)), np.cos(ang)]))[None], gyr, 1)[0]
    assert tilt[C.FEATURES.index("imu_tilt_deg")] == pytest.approx(90 - 2 * (24 / 99 * 90) / 2, abs=0.5)  # 1 s means: 10.9 and 79.1 deg
    same = F.xmod_features(np.repeat(np.linspace(1, 20, n), 2)[None] + 0, np.column_stack([np.linspace(1, 20, n), np.zeros(n), np.zeros(n)])[None])
    assert same[0, 0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        F.xmod_features(np.ones((1, 40)), np.ones((1, 20, 3)))  # too few coherence segments


def test_missing_modality_gives_nan_features():
    x = F.compute(None, np.ones((2, 50, 3)), np.ones((2, 50, 3)), 2)
    assert np.isnan(x[:, list(C.MODALITIES["emg"])]).all() and np.isfinite(x[:, list(C.MODALITIES["imu"])]).all()


# ---- state machine and event metrics ----------------------------------------------------------------


def test_detector_rules():
    d = Ev.Detector(0.5, 2, 2)
    seq = [(0.6, 1000), (0.6, 2000), (0.6, 3000), (0.1, 4000), (0.1, 5000), (0.6, 6000), (0.6, 7000)]
    flags = [d.update(p, t) for p, t in seq]
    assert flags[1] == Ev.OPENED | Ev.ALERT and flags[2] == 0 and flags[4] == Ev.CLOSED
    assert flags[6] == Ev.OPENED | Ev.SUPPRESSED  # 5 s after the alert: refractory
    assert d.update(float("nan"), 8000) == Ev.INVALID and d.update(1.5, 8000) == Ev.INVALID
    assert d.update(0.1, 8000 + 7000) & Ev.CLOSED  # gap > 6 s closes
    d.reset()
    assert d.update(0.9, 2**32 - 500) == 0 and d.update(0.9, 500) == Ev.OPENED | Ev.ALERT  # wrap-around


@pytest.mark.parametrize("seed", range(200))
def test_fast_openings_equal_sequential_detector(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 300))
    t = np.cumsum(rng.choice([1.0, 2.0, 2.48, 7.0, 61.0], n, p=[0.6, 0.2, 0.1, 0.07, 0.03]))
    p = rng.uniform(0, 1, n).astype(np.float32)
    valid = rng.uniform(size=n) > 0.15
    thr, no, mc = float(rng.choice([0.3, 0.5, 0.7])), int(rng.choice([2, 3])), int(rng.choice([2, 3]))
    d = Ev.Detector(thr, no, mc)
    seq_open, seq_alert = [], []
    for ti, pi, vi in zip(t, p, valid):
        f = d.update(float(pi) if vi else float("nan"), int(round(ti * 1000)))
        if f & Ev.OPENED:
            seq_open.append(ti)
        if f & Ev.ALERT:
            seq_alert.append(ti)
    fast = Ev.openings(t[valid], p[valid] >= thr, no, mc)
    emitted, suppressed = Ev.alerts(fast)
    assert np.allclose(fast, seq_open) and np.allclose(emitted, seq_alert)
    assert suppressed == len(seq_open) - len(seq_alert)


def test_event_metrics():
    t = np.arange(1.0, 401.0)
    p = np.zeros(t.size)
    p[(t >= 102) & (t <= 106)] = 0.9  # seizure 100-130 detected at 103 (N=2)
    p[(t >= 300) & (t <= 302)] = 0.9  # false alarm, inside a non-target seizure
    valid = np.ones(t.size, bool)
    neg = valid & ~((t > 100) & (t - 1 < 130))
    ec = Ev.score_recording(t, p, valid, neg, 1.0, 1.0, np.array([[100.0, 130.0], [350.0, 351.0]]),
                            np.array([[299.0, 310.0]]), 0.5, 2, 2)
    assert (ec.events, ec.evaluable, ec.detected, ec.false_alerts, ec.true_alerts) == (2, 2, 1, 1, 1)
    assert ec.latencies == [3.0] and ec.false_alerts_during_nontarget_seizures == 1
    s = ec.summary()
    assert s["event_precision"] == 0.5 and s["event_sensitivity"] == 0.5


# ---- training, OOF, calibration (synthetic Windows) ----------------------------------------------------


def _synthetic_windows(n_rec=24, per=400):
    X, rec, t, label = [], [], [], []
    tiv = []
    for r in range(n_rec):
        x = RNG.normal(size=(per, 13)).astype(np.float32)
        lab = np.zeros(per, np.int8)
        if r % 3 == 0:
            lab[200:212] = 1
            x[200:212, :] += 3
        X.append(x), rec.append(np.full(per, r, np.int32)), t.append(np.arange(1, per + 1) * 2.0), label.append(lab)
        tiv.append(np.array([[400.0, 424.0]]) if r % 3 == 0 else np.empty((0, 2)))
    n = n_rec * per
    return Dt.Windows("w4", 4.0, 2.0, np.concatenate(X), np.concatenate(rec), np.concatenate(t), np.concatenate(label),
                      np.zeros(n, bool), np.ones(n, bool), np.ones(n, bool), [f"sub-{r + 1:03d}_r" for r in range(n_rec)],
                      np.arange(1, n_rec + 1), tiv, [np.empty((0, 2))] * n_rec)


@pytest.mark.parametrize("name", ["LR", "RF", "XGB"])
def test_oof_is_patient_wise_calibrated_and_complete(name):
    D = _synthetic_windows()
    folds = {s: (s - 1) % 5 for s in D.subjects}
    res = Bm.oof(name, D, "fusion", folds)
    assert np.isfinite(res["p"]).all() and ((res["p"] >= 0) & (res["p"] <= 1)).all()
    assert all(m["train_subjects"] + m["val_subjects"] == 24 for m in res["folds"])
    assert res["p"][D.label == 1].mean() > res["p"][D.label == 0].mean()
    rows = Bm.sweep(D, res["p"], D.valid("fusion"))
    assert len(rows) == len(C.THRESHOLDS) * 4 and all(r["target_events"] == 8 for r in rows)


def test_pool_keeps_all_positives_and_reweights_negatives():
    D = _synthetic_windows()
    idx, w = Bm.pool(D, np.ones(D.t.size, bool))
    assert (D.label[idx] == 1).sum() == (D.label == 1).sum()
    assert w[D.label[idx] == 0].sum() == pytest.approx((D.label == 0).sum())


# ---- C99 export and parity -------------------------------------------------------------------------------


def _compile_and_run(tmp: Path, name: str, model, cols, A=0.7, B=-0.5) -> str:
    for f in ("lyra_features.c", "lyra_features.h", "lyra_detector.c", "lyra_detector.h", "lyra_model_test.c"):
        (tmp / f).write_text((REPO / "firmware" / "model" / f).read_text())
    hdr, replica, _ = X.model_header(name, model, cols, A, B, "test")
    (tmp / "lyra_model.h").write_text(hdr)
    (tmp / "lyra_feature_spec.h").write_text(X.spec_header(125))
    sel = {"window": "w4", "model": name, "modality": "fusion", "threshold": 0.4, "n_open": 2, "m_close": 3}
    (tmp / "lyra_detector_config.h").write_text(X.detector_config(sel, "test"))
    G.write(tmp / "golden.bin", replica, 0.4, 2, 3)
    subprocess.run(["gcc", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I", str(tmp),
                    str(tmp / "lyra_model_test.c"), str(tmp / "lyra_features.c"), str(tmp / "lyra_detector.c"), "-lm",
                    "-o", str(tmp / "t")], check=True)
    r = subprocess.run([str(tmp / "t"), str(tmp / "golden.bin")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    return r.stdout


@pytest.mark.parametrize("name", ["RF", "ET", "XGB", "LR", "SVC"])
def test_c99_inference_features_and_detector_match_python(tmp_path, name):
    Xs = RNG.normal(size=(2500, 13)) * 5 + 3
    y = (Xs[:, 7] + Xs[:, 2] > 8).astype(int)
    cols = list(range(13)) if name != "LR" else [0, 2, 7, 9]
    k = 1200 if name == "SVC" else len(y)
    m = Bm.make_model(name, int(y.sum()), int((1 - y).sum())).fit(Xs[:k][:, cols], y[:k])
    out = _compile_and_run(tmp_path, name, m, cols)
    assert "failures 0" in out


def test_replicas_match_original_estimators_where_exact_export_is_claimed():
    Xs = (RNG.normal(size=(3000, 13)) * 5).astype(np.float32)
    y = (Xs[:, 1] > 2).astype(int)
    for name in ("XGB", "LR"):
        m = Bm.make_model(name, int(y.sum()), int((1 - y).sum())).fit(Xs, y)
        _, replica, _ = X.model_header(name, m, list(range(13)), 1.0, 0.0, "t")
        ref = expit(Bm.raw_score(name, m, Xs.astype(np.float64)))
        assert np.abs(replica(Xs) - ref).max() < 1e-5


def test_feature_spec_header_matches_canonical_json():
    hdr = X.spec_header(100)
    for i, n in enumerate(C.FEATURES):
        assert f"LYRA_F_{n.upper()} = {i}," in hdr
    assert json.loads((REPO / "ml" / "lyra" / "feature_spec.json").read_text())["version"] in hdr


# ---- ADL loader ------------------------------------------------------------------------------------------


def test_adl_loader_validates_schema_units_and_rates(tmp_path):
    from ml.lyra import adl
    n_imu = 25 * 30
    t_env = np.arange(2 * n_imu) / 50
    rows = {"subject_id": "v01", "recording_id": "r1", "timestamp": t_env, "emg": np.abs(RNG.normal(10, 2, t_env.size))}
    for c in ("ax", "ay", "az", "gx", "gy", "gz"):
        col = np.full(t_env.size, np.nan)
        col[::2] = RNG.normal(0, 0.1, n_imu) + (1.0 if c == "az" else 0)
        rows[c] = col
    import pandas as pd
    pd.DataFrame(rows).to_csv(tmp_path / "r1.csv", index=False)
    (tmp_path / "metadata.json").write_text(json.dumps({"emg_rate_hz": 50, "imu_rate_hz": 25, "acc_unit": "g",
                                                        "gyr_unit": "deg/s"}))
    rec = adl.load_dir(tmp_path, {})[0]
    assert rec["acc"][:, 2].mean() == pytest.approx(9.80665, rel=0.05) and rec["hours"] == pytest.approx(30 / 3600, rel=0.01)
    t, p, v = adl.score(rec, lambda x: np.full(len(x), 0.1, np.float32), "w4")
    assert v.any() and np.nanmax(p) == pytest.approx(0.1)
    fa = adl.false_alarms([rec], [(t, p, v)], 0.5, 2, 2, 2.0)
    assert fa["false_alerts"] == 0 and fa["volunteers"] == 1
    (tmp_path / "metadata.json").write_text(json.dumps({"emg_rate_hz": 250, "imu_rate_hz": 25, "acc_unit": "g",
                                                        "gyr_unit": "deg/s"}))
    with pytest.raises(adl.AdlSchemaError):
        adl.load_dir(tmp_path, {})
    pd.DataFrame(rows).drop(columns=["gz"]).to_csv(tmp_path / "r1.csv", index=False)
    with pytest.raises(adl.AdlSchemaError):
        adl.load_dir(tmp_path, {"gz": "gyro_z"})


# ---- resource reporting ------------------------------------------------------------------------------------


def test_resource_budget_accounting():
    from ml.lyra import deploy
    objs = {"objects": {"model": {"flash_bytes": 30000, "static_ram_bytes": 0, "stack_bytes_per_function": {"lyra_model_entry": 96}},
                        "features": {"flash_bytes": 6000, "static_ram_bytes": 1600,
                                     "stack_bytes_per_function": {"lyra_features_compute": 672}},
                        "detector": {"flash_bytes": 500, "static_ram_bytes": 48, "stack_bytes_per_function": {}}}}
    b = deploy.resource_budget({"window": "w4"}, objs)
    assert b["flash_bytes"] == 36500 and b["sram_input_window_bytes"] == 4 * (200 + 600)
    assert b["sram_total_bytes"] == 1648 + 768 + 3200 and b["meets_flash_target"] and b["meets_sram_target"]
