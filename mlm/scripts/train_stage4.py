"""Stage 4 (authorized 2026-11-22): 5-fold participant-grouped CV on Train/CV (sub-001..sub-096),
model selection, final training on all Train/CV, frozen artifacts in mlm/models/ (gitignored).

Held-out windows are never opened here: recordings are filtered by the locked split before any
file is read, and training_rows() refuses non-Train/CV participants.
Selection rule (fixed before looking at results): per track, the random forest config with the
highest pooled CV window-level AUPRC; ties go to fewer trees, then shallower trees.
Operating points (owner decision 2026-11-22, after CV showed ~48 FA/24h at the locked 0.65 entry
threshold, before any held-out run): each configuration keeps the locked Decision 4 point and gets a
CV-selected entry threshold = the lowest value in ENTRY_GRID whose pooled CV interictal false-alarm
rate is <= FA_CAP_PER_24H. Every other confirmation parameter stays at Decision 4.
Usage: python scripts/train_stage4.py [--ratio 50]
"""

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import sklearn
import xgboost
from sklearn.model_selection import StratifiedGroupKFold

from lyra_mlm.confirmation import ConfirmationConfig
from lyra_mlm.evaluation import SEIZEIT2_PLANNED_SPLIT, Partition
from lyra_mlm.seizeit2 import SeizeIT2Adapter
from lyra_mlm.training import (PRIMARY_NAMES, SUPPRESSED, TRACKS, Tally, fallback_status, make_model,
                               require_train_cv, score_recording, summarise, track_status, train_mask,
                               training_rows, window_keys)

ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "data", ROOT / "models"
PROC = DATA / "processed"
POOL_P = 0.05  # candidate negatives kept per window; must exceed the per-fold sampling fraction
RF_GRID = [{"n_estimators": n, "max_depth": d, "min_samples_leaf": leaf}
           for n in (25, 50) for d in (6, 8) for leaf in (5, 10)]
XGB = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}
CANDIDATES = [("lr", {})] + [("rf", g) for g in RF_GRID] + [("xgb", XGB)]
ENTRY_GRID = (0.65, 0.75, 0.85, 0.90, 0.95, 0.97, 0.99)
FA_CAP_PER_24H = 1.0
CONFIGS = ("multimodal", "multimodal_fallback_unimodal", "simenv", "imu")
KEYS = ("label", "overall", "emg_state", "imu_state", "artifact_overlap", "primary_available", "primary", "end_s")


def cand_key(family: str, params: dict) -> str:
    return family + "".join(f"_{k[0]}{v}" for k, v in params.items() if family == "rf")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(rid: str) -> dict:
    with np.load(PROC / "windows" / rid.split("_")[0] / f"{rid}.npz") as z:
        return {k: z[k] for k in KEYS}


def train_cv_recordings() -> dict[str, list[str]]:
    by_part = defaultdict(list)
    for m in json.loads((PROC / "windows_manifest.json").read_text()):
        if "excluded" not in m and SEIZEIT2_PLANNED_SPLIT.partition_of(m["participant_id"]) is Partition.TRAIN:
            by_part[m["participant_id"]].append(m["recording_id"])
    require_train_cv(by_part)
    return dict(sorted(by_part.items()))


def build_pool(recs: dict[str, list[str]]):
    """Every training-eligible positive plus candidate negatives (key < POOL_P), all tracks at once."""
    X, lab, key, part, bits = [], [], [], [], []
    neg_total = {t: Counter() for t in TRACKS}
    pos_total = {t: Counter() for t in TRACKS}
    labelled = {}
    for p, rids in recs.items():
        n_pos = n_neg = 0
        for rid in rids:
            z = load(rid)
            m = {t: train_mask(z, tr) for t, tr in TRACKS.items()}
            k = window_keys(rid, z["label"].size)
            sel = np.any(list(m.values()), axis=0) & ((z["label"] == 1) | (k < POOL_P))
            X.append(z["primary"][sel].astype(np.float32))
            lab.append(z["label"][sel])
            key.append(k[sel])
            part.append(np.full(sel.sum(), p))
            bits.append(np.column_stack([m[t][sel] for t in TRACKS]))
            for t in TRACKS:
                neg_total[t][p] += int((m[t] & (z["label"] == 0)).sum())
                pos_total[t][p] += int((m[t] & (z["label"] == 1)).sum())
            n_pos += int((z["label"] == 1).sum())
            n_neg += int((z["label"] == 0).sum())
        labelled[p] = (n_pos, n_neg)
    pool = {"X": np.concatenate(X), "label": np.concatenate(lab), "key": np.concatenate(key),
            "part": np.concatenate(part), "bits": np.concatenate(bits)}
    return pool, neg_total, pos_total, labelled


def folds(labelled: dict[str, tuple[int, int]], n_splits: int) -> list[list[str]]:
    """StratifiedGroupKFold over participants, stratified by each participant's labelled windows."""
    parts = list(labelled)
    y = np.concatenate([np.r_[np.ones(a, np.int8), np.zeros(b, np.int8)] for a, b in labelled.values()])
    g = np.concatenate([np.full(a + b, i, np.int32) for i, (a, b) in enumerate(labelled.values())])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    return [sorted({parts[i] for i in np.unique(g[val])}) for _, val in sgkf.split(np.zeros((y.size, 1)), y, g)]


def fit_track(pool, neg_total, track: str, participants: list[str], ratio: int, family: str, params: dict):
    t = list(TRACKS).index(track)
    cand = np.flatnonzero(pool["bits"][:, t] & np.isin(pool["part"], participants))
    rows = cand[training_rows(pool["part"][cand], pool["label"][cand], pool["key"][cand],
                              {p: neg_total[track][p] for p in participants}, ratio)]
    x, y = pool["X"][rows][:, TRACKS[track].columns], pool["label"][rows].astype(int)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    return make_model(family, params, n_pos, n_neg).fit(x, y), {"positives": n_pos, "negatives": n_neg}


def score_participant(rids, models: dict, adapter) -> dict[str, Tally]:
    """Score every window of a participant's recordings with each model (batched per participant)."""
    zs = [load(r) for r in rids]
    out = defaultdict(Tally)
    for track, tr in TRACKS.items():
        status = [track_status(z, tr) for z in zs]
        x = np.concatenate([z["primary"][s != SUPPRESSED][:, tr.columns] for z, s in zip(zs, status)])
        for key, model in models.get(track, {}).items():
            s_all = model.predict_proba(x)[:, 1] if len(x) else np.empty(0)
            off = 0
            for rid, z, s in zip(rids, zs, status):
                score = np.full(s.size, np.nan)
                n = int((s != SUPPRESSED).sum())
                score[s != SUPPRESSED] = s_all[off:off + n]
                off += n
                out[(track, key)].add(score_recording(z["end_s"], z["label"], s, score, adapter.annotations(rid)))
    return out


def operating_points(rids, models: dict, adapter, entries) -> dict[tuple[str, float], Tally]:
    """Frozen-model configurations x entry thresholds. models: track -> fitted model. The fallback
    configuration scores the multimodal model, and the SIMULATED_ENV-only model where only IMU is unusable."""
    out = defaultdict(Tally)
    for rid in rids:
        z, anns, runs = load(rid), adapter.annotations(rid), {}
        for track, tr in TRACKS.items():
            st = track_status(z, tr)
            sc = np.full(st.size, np.nan)
            if (st != SUPPRESSED).any():
                sc[st != SUPPRESSED] = models[track].predict_proba(z["primary"][st != SUPPRESSED][:, tr.columns])[:, 1]
            runs[track] = (st, sc)
        (s_mm, c_mm), (_, c_sim), s_fb = runs["multimodal"], runs["simenv"], fallback_status(z)
        use = (s_mm == SUPPRESSED) & (s_fb != SUPPRESSED)
        runs["multimodal_fallback_unimodal"] = (np.where(s_mm != SUPPRESSED, s_mm, s_fb), np.where(use, c_sim, c_mm))
        for e in entries:
            cfg = ConfirmationConfig(entry_threshold=e)
            for name in CONFIGS:
                st, sc = runs[name]
                out[(name, e)].add(score_recording(z["end_s"], z["label"], st, sc, anns, cfg))
    return out


def select_entry(results: dict, config: str) -> float:
    ok = [e for e in ENTRY_GRID if (results[f"{config}@{e}"]["interictal_fa_per_24h"] or 0) <= FA_CAP_PER_24H]
    return min(ok) if ok else max(ENTRY_GRID)


def d_groups() -> dict[str, str]:
    """Participant -> D2 / D6 / mixed / no_accepted_imu, from the Stage 3.5 IMU scale manifest."""
    ds = defaultdict(set)
    for rid, e in json.loads((PROC / "imu_scale_manifest.json").read_text())["recordings"].items():
        if e["d"] is not None:
            ds[rid.split("_")[0]].add(e["d"])
    return defaultdict(lambda: "no_accepted_imu", {p: "mixed" if len(d) > 1 else f"D{d.pop()}" for p, d in ds.items()})


def total(tallies) -> Tally:
    out = Tally()
    for t in tallies:
        out.add(t)
    return out


def report(per_part: dict[str, Tally], groups: dict[str, str]) -> dict:
    by_group = defaultdict(dict)
    for p, t in per_part.items():
        by_group[groups[p]][p] = t
    out = summarise(total(per_part.values()), per_part)
    out["by_d_group"] = {g: summarise(total(ts.values()), ts) | {"participants": len(ts)}
                         for g, ts in sorted(by_group.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=int, default=50, help="negatives per positive in training folds")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    t0 = time.time()
    adapter, recs, groups = SeizeIT2Adapter(DATA / "seizeit2"), train_cv_recordings(), d_groups()
    pool, neg_total, pos_total, labelled = build_pool(recs)
    for t in TRACKS:
        f = args.ratio * sum(pos_total[t].values()) / sum(neg_total[t].values())
        assert f < POOL_P / 1.5, f"pool too small for track {t}: sampling fraction {f:.4f}"
    print(f"pool {len(pool['label'])} rows ({int((pool['label'] == 1).sum())} positives), {time.time() - t0:.0f} s", flush=True)
    fold_parts = folds(labelled, args.folds)
    require_train_cv(p for f in fold_parts for p in f)

    cv = defaultdict(lambda: defaultdict(Tally))  # (track, key) -> participant -> tally
    fold_models, fold_info = [], []
    for i, val in enumerate(fold_parts):
        train = [p for p in recs if p not in val]
        models, sizes = defaultdict(dict), {}
        for track in TRACKS:
            for family, params in CANDIDATES:
                models[track][cand_key(family, params)], sizes[f"{track}/{cand_key(family, params)}"] = \
                    fit_track(pool, neg_total, track, train, args.ratio, family, params)
        fold_models.append(models)
        fold_info.append({"validation_participants": val, "training_set_sizes": sizes})
        for p in val:
            for k, t in score_participant(recs[p], models, adapter).items():
                cv[k][p] = t
        print(f"fold {i + 1}/{len(fold_parts)} done, {time.time() - t0:.0f} s", flush=True)

    results = {f"{track}/{key}": report(cv[(track, key)], groups) for track, key in cv}
    winners = {}
    for track in TRACKS:
        rf = [(results[f"{track}/{cand_key('rf', g)}"]["auprc"] or 0, -g["n_estimators"], -g["max_depth"], i)
              for i, g in enumerate(RF_GRID)]
        winners[track] = RF_GRID[max(rf)[3]]
    op = defaultdict(dict)  # selected models: configurations x entry thresholds, CV
    for i, val in enumerate(fold_parts):
        chosen = {t: fold_models[i][t][cand_key("rf", winners[t])] for t in TRACKS}
        for p in val:
            for k, t in operating_points(recs[p], chosen, adapter, ENTRY_GRID).items():
                op[k][p] = t
    op_results = {f"{c}@{e}": report(op[(c, e)], groups) for c, e in op}
    points = {c: {"locked_decision4": ConfirmationConfig().entry_threshold, "cv_selected": select_entry(op_results, c)}
              for c in CONFIGS}
    for c, pt in points.items():
        r = op_results[f"{c}@{pt['cv_selected']}"]
        pt["cv_at_selected"] = {k: r[k] for k in ("detected", "fbtcs_events", "event_sensitivity", "interictal_fa_per_24h",
                                                  "non_fbtcs_seizure_alert_rate", "detection_latency_s_median")}

    frozen = {}
    for track, params in winners.items():  # final models: all Train/CV participants
        model, size = fit_track(pool, neg_total, track, list(recs), args.ratio, "rf", params)
        path = MODELS / f"{track}_rf.joblib"
        joblib.dump(model, path)
        frozen[track] = {"file": path.name, "sha256": sha256(path), "family": "random_forest",
                         "hyperparameters": params | {"class_weight": "balanced", "random_state": 0},
                         "feature_names": [PRIMARY_NAMES[c] for c in TRACKS[track].columns],
                         "gate": TRACKS[track].gate, "training_set": size,
                         "cv_auprc": results[f"{track}/{cand_key('rf', params)}"]["auprc"]}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    pipeline = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_commit": commit,
        "models": frozen, "confirmation": asdict(ConfirmationConfig()),
        "operating_points": {"entry_grid": ENTRY_GRID, "fa_cap_per_24h": FA_CAP_PER_24H,
                             "rule": "lowest grid entry threshold with pooled CV interictal FA/24h <= cap; other "
                                     "confirmation parameters at Decision 4", "by_configuration": points},
        "inference": {"degraded_window_policy": "allow_with_flag",
                      "missing_modality_policy": ["suppress", "fallback_unimodal (simenv model when IMU unusable)"]},
        "training": {"participants": list(recs), "negatives_per_positive": args.ratio,
                     "negative_sampling": "deterministic per-window key (crc32 of recording id), lowest keys, "
                                          "proportional per participant, training folds only",
                     "loss_windows": "POSITIVE/NEGATIVE, track usable (VALID/DEGRADED, all features available), "
                                     "outside impedance-check (impd) annotations"},
        "selection_rule": "per track, RF config with highest pooled CV window AUPRC; ties -> fewer, shallower trees",
        "data": {f: sha256(PROC / f) for f in ("windows_manifest.json", "imu_scale_manifest.json", "feature_schema.json")},
        "versions": {"sklearn": sklearn.__version__, "xgboost": xgboost.__version__, "numpy": np.__version__},
    }
    (MODELS / "frozen_pipeline.json").write_text(json.dumps(pipeline, indent=1))
    (MODELS / "cv_report.json").write_text(json.dumps({"folds": fold_info, "winners": winners, "results": results,
                                                       "operating_points": op_results, "selected": points,
                                                       "d_groups": dict(groups)}, indent=1, default=float))
    for track in TRACKS:
        for family, params in CANDIDATES:
            r = results[f"{track}/{cand_key(family, params)}"]
            print(f"{track:10s} {cand_key(family, params):14s} auprc {r['auprc']:.3f} sens {r['event_sensitivity']} "
                  f"fa24 {r['interictal_fa_per_24h']}", flush=True)
    print("winners", winners, "operating points", json.dumps(points, default=float), f"total {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
