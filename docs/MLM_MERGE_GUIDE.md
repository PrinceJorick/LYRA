# LYRA ML: what branch `seizeit2-model` contains, and how to merge the two MLMs

For: the LYRA team (PrinceJorick, Kyle). Written 2026-11-21.

LYRA has two ML pipelines that were developed in parallel on the same dataset (SeizeIT2):

| | **MLM A: `mlm/` (`lyra_mlm`)** | **MLM B: `ml/seizeit2/` + `models/seizeit2/`** |
|---|---|---|
| Author | Kyle (Sessions 2–3) | PrinceJorick (Session 4, this branch) |
| Stage | Stage 3: dataset preparation. No model trained (by design) | Stage 4: a trained, evaluated and exported model, wired into the app |

The plan is to turn them into **one MLM**. This document has three parts:

1. What this branch adds (§1).
2. How the two pipelines differ, and what this branch found that affects `mlm/` (§2–3).
3. The decisions and step-by-step procedure for the merge (§4–6).

> Experimental research software. Nothing here is clinically validated or a diagnostic device. All
> results are on in-hospital SeizeIT2 recordings, not on the LYRA pod.

---

## 0. TL;DR

- **Branch:** `seizeit2-model` = `origin/main` (`0a50a6e`) + 2 commits. It fast-forwards onto `main`.
  1. `2393be6` repairs `main`. Kyle's merge `0a50a6e` was committed with unresolved conflict markers in
     9 files, so `prep.py`, `quality.py`, `audit_stage3.py` and `test_stage3.py` could not be imported.
     The files are restored verbatim from his own `1c1a9a6`. After this commit the tree equals
     `1c1a9a6` exactly, and all 134 `mlm` tests pass.
  2. `ff1373d` adds MLM B.
     - Contents: the SeizeIT2 training pipeline, the trained model `lyra-seizeit2-fbtcs` v1.0.0, its
       model card and metrics, and the TypeScript runtime in the app.
     - Held-out test result: 11/12 FBTCS detected, 0.22 false alarms/24 h.
- **The two MLMs conflict on locked decisions:**
  - D11: inference runs on the pod, not the phone.
  - D12: EMG is MyoWare ENV at 50 Hz, not RAW at 256 Hz.
  - D13: 4 s windows.
  - D17: confirmation logic.
  - D18: model family.
  - The **participant split** also differs: MLM B was trained on 22 of the 29 participants that MLM A
    keeps held out.
- **One finding matters for `mlm/` right away.** SeizeIT2 accelerometer files use different scales
  per recording (~1/2, ~1/6, or pinned at 32.768). `mlm/` keeps "units as given", so its Stage 3
  features very likely mix scales (§3.1).
- **Recommended merge:**
  1. Land the branch as-is (the pipelines coexist).
  2. Sign off the decisions in §4.
  3. Move MLM B's code into `lyra_mlm` (§5).
  4. Retrain once under the agreed split and EMG path, evaluate once, then export.

---

## 1. What branch `seizeit2-model` contains

### 1.1 Commit `2393be6`: repair of `main` (no content change)

`0a50a6e` ("Initial Commit") merged two identical histories (`1dd383b` and `414a214` have the same
tree) and left `<<<<<<<`/`>>>>>>>` markers in 9 files:

- `PROGRESS.md`, `context.md`
- `mlm/docs/{ARCHITECTURE,DATASETS,TRAINING_PLAN}.md`
- `mlm/scripts/audit_stage3.py`, `mlm/src/lyra_mlm/{prep,quality}.py`, `mlm/tests/test_stage3.py`

In every hunk the HEAD side (`1c1a9a6`, "Prep for training") is the newer Stage 3 audit version. The
commit restores those files from `1c1a9a6` unchanged. To verify:

```bash
git diff 1c1a9a6 2393be6 --stat        # prints nothing: identical trees
cd mlm && .venv/bin/python -m pytest -q  # 134 passed
```

If Kyle prefers to fix his merge differently, drop this commit; nothing else depends on its content.

### 1.2 Commit `ff1373d`: MLM B (SeizeIT2 model + app integration)

| Path | What it is |
|---|---|
| `ml/seizeit2/config.py` | Every parameter that changes a result: paths, signal contract, labels, quality rules, split seed, post-processing, selection rule |
| `ml/seizeit2/data_loader.py` | Run index from `events.tsv`; `SeizeIT2Mapper` (implements `lyra_mlm.annotations.AnnotationMapper`); SHA256-verified fetch of git-annexed EDFs from the dataset's public bucket; minimal EDF+C reader |
| `ml/seizeit2/preprocessing.py` | Reference feature extractor (24 features), window labels, quality flags, per-run accelerometer gravity calibration, feature-cache builder (`python -m ml.seizeit2.preprocessing`) |
| `ml/seizeit2/dataset_analysis.py` | Aggregate dataset summary → `models/seizeit2/dataset_summary.json` |
| `ml/seizeit2/train.py` | Subject-level split, 5-fold subject-grouped CV over 4 candidates + 2 ablations × 2 window lengths, operating-point selection, final fit |
| `ml/seizeit2/evaluate.py` | Event matching, FAR/24 h with CIs, latency, window metrics; one-time test evaluation (refuses to run twice) |
| `ml/seizeit2/export.py` | Export to `model.json` + config/labels/metadata; checks exact agreement with sklearn on 200k dev windows; writes parity vectors |
| `ml/seizeit2/parity_check.mjs`, `parity_vectors.json` | Python ↔ TypeScript parity: features, scores, detector. Synthetic inputs only, no dataset samples |
| `ml/seizeit2/requirements.txt` | numpy, scipy, scikit-learn, matplotlib, `-e ./mlm` |
| `models/seizeit2/model.json` | Trained model: sklearn HistGradientBoosting, 300 trees, ≤15 leaves, 288 kB, format `lyra-seizeit2-model/1` |
| `models/seizeit2/config.json`, `labels.json`, `metadata.json`, `split.json` | Inference contract, class/event semantics, provenance, the subject split |
| `models/seizeit2/evaluation/` | `metrics.json` (test + all CV candidates), confusion matrix, training history, CV operating curves |
| `models/seizeit2/README.md` | Model card: method, results, limitations, reproduction |
| `src/services/ml/seizeit2/features.ts` | Line-by-line TypeScript port of the feature extractor |
| `src/services/ml/seizeit2/model.ts` | Tree-ensemble evaluator, asset check, streaming detector (`createDetector`) |
| `src/services/ml/seizeit2/service.ts` | `MlService` implementation (validation, flat-signal rule, detection) |
| `src/services/ml/index.ts` | Serves the SeizeIT2 service; falls back to the mock if bundled assets are incompatible |
| `src/app/(tabs)/insights.tsx`, `src/app/about.tsx` | Wording only: "experimental, not clinically validated, no live pod data" |
| `.gitignore`, `README.md`, `PROGRESS.md` | Merged with Kyle's versions (`.gitignore` keeps all of his rules; `PROGRESS.md` gains Session 4) |

**Not in git, by design:**

- `pj-training-data/`: the DataLad clone of ds005873 plus a 3.5 GB feature cache.
- `ml/seizeit2/.venv/`.
- `ml/seizeit2/runs/`: CV checkpoints and logs.

### 1.3 How MLM B works

1. **Data.** `pj-training-data/ds005873` is a DataLad clone, so its EDF files are git-annex links with
   no content.
   - `preprocessing.py` downloads each EMG/MOV EDF from `s3://openneuro.org/ds005873` and checks its
     size and SHA256 against the annex key.
   - It computes features, then deletes the EDF. Peak disk use is about 5 GB; full extraction takes
     about 50 min at about 25 MB/s.
2. **Signals.**
   - EMG: `EMG SD` (left deltoid), RAW at 256 Hz in µV.
   - IMU: chest `ECGEMG SD`, ACC + GYR at 25 Hz.
   - Runs that lack either modality are skipped.
3. **Accelerometer calibration.** Each run is divided by its median |acc| (gravity), which puts it in
   g. Runs whose median falls outside `[0.12, 0.7]` are excluded (405 runs; see §3.1).
4. **Windows.** 5 s and 10 s windows every 2 s, right-aligned to whole seconds.
   - Unusable: flat EMG or IMU, or more than 5% of EMG samples at the ADC rail.
5. **Features.** 24, all rotation invariant:
   - sEMG Welch spectrum (20–128 Hz without mains, band fractions, median frequency, entropy).
   - sEMG difference-envelope stats and 1–5 Hz rhythmicity.
   - Accelerometer magnitude stats and spectrum, in g.
   - Gyro features that do not depend on its scale (CV, band fraction, dominant frequency, peak ratio).
   - Orientation change, and an EMG-vs-gyro frequency difference.
6. **Labels.**
   - Positive: ≥ 50% overlap with `sz_foc_f2b`.
   - Ignored: windows within (−30 s, +120 s) of an FBTCS, or (−30 s, +60 s) of any other seizure,
     and `impd` windows.
   - Negative: everything else.
7. **Split.**
   - Random but stratified by "has FBTCS", seed 20261121.
   - 25 test participants (7 with FBTCS, 13 events) and 100 dev participants in 5 grouped folds.
8. **Selection.** Out-of-fold over all dev windows, using a rule fixed in advance: maximise event
   sensitivity with FAR ≤ 1/24 h, then lower FAR, then lower latency.
9. **Post-processing.** The mean of the last k = 5 scores must reach the threshold (0.9867). There is
   a 300 s refractory period, and a stream gap resets the mean.
10. **Event matching.**
    - Detection: an alarm in [onset − 10 s, offset + 30 s].
    - Alarms near other seizures or in `impd` are reported separately, not counted as false alarms.

### 1.4 Results (full detail in `models/seizeit2/README.md`)

Dev, out of fold (32 usable FBTCS, 8,049 h):

| Candidate | Sens | FAR/24 h |
|---|---|---|
| Rule baseline | 0.50 | 0.74 |
| Logistic regression | 0.66 | 0.92 |
| **HGB 300×15, W = 5 s (selected)** | **0.97** | **0.23** |
| Ablation: EMG only | 0.91 | 0.52 |
| Ablation: IMU only | 0.97 | 0.76 |

Held-out test (25 participants, 1,983 h, evaluated once):

| Metric | Value |
|---|---|
| Event sensitivity | 11/12 = 0.92 (95% CI 0.62–1.00); 11/13 if the one event in an unusable run counts as missed |
| False alarms | 0.22/24 h (95% CI 0.13–0.34) |
| Latency | median 43 s from the annotated focal onset |
| Window level | ROC-AUC 0.86, PR-AUC 0.40 |

### 1.5 App behaviour after this commit

- `mlService` now reports `lyra-seizeit2-fbtcs v1.0.0 (integrated)`.
- `detect(window)` accepts 5 s windows:
  - 1,280 EMG samples (µV at 256 Hz)
  - 125 samples per IMU axis (acc in g, gyro in any fixed unit)
- **Nothing calls `detect()` yet,** because there is no BLE stream. The alert, cancel countdown and
  caregiver flow are unchanged and still driven only by the demo trigger.
- No score, probability or "seizure detected" wording is shown anywhere.

### 1.6 Reproduce and verify MLM B

```bash
uv venv --python 3.14 ml/seizeit2/.venv
VIRTUAL_ENV=ml/seizeit2/.venv uv pip install -r ml/seizeit2/requirements.txt
P=ml/seizeit2/.venv/bin/python
$P -m ml.seizeit2.preprocessing      # ~50 min, ~3.5 GB cache in pj-training-data/cache/
$P -m ml.seizeit2.dataset_analysis
$P -m ml.seizeit2.train              # ~14 min on 12 cores (CPU; sklearn HGB has no GPU path)
$P -m ml.seizeit2.evaluate --run ml/seizeit2/runs/<id>   # once
$P -m ml.seizeit2.export   --run ml/seizeit2/runs/<id>
node ml/seizeit2/parity_check.mjs    # Node >= 22.18
npx tsc --noEmit && npx expo lint
```

The DataLad clone is expected at `pj-training-data/ds005873`. Create it with
`datalad clone https://github.com/OpenNeuroDatasets/ds005873.git`, or `git clone` the same URL. Do not
`datalad get` it: the pipeline fetches and verifies content itself.

---

## 2. The two MLMs side by side

| Topic | MLM A `mlm/` (Kyle) | MLM B `ml/seizeit2/` (PJ) | Same? |
|---|---|---|---|
| Package | `lyra_mlm` (pyproject, src layout, 134 tests, docs) | Plain modules run with `python -m ml.seizeit2.*`; imports `lyra_mlm.annotations` | — |
| Python / venv | 3.12, `mlm/.venv` | 3.14, `ml/seizeit2/.venv` | No |
| Data location | `mlm/data/seizeit2/` (full 52.8 GB wearable subset downloaded) | `pj-training-data/ds005873` (DataLad) + streamed fetch; 3.5 GB feature cache | No |
| Label mapping | `SeizeIT2Mapper`: f2b→FBTCS, `sz_gen_m_tonicClonic`→GTCS, `sz*`→out of scope, `bckg`→BACKGROUND, `impd`→ARTIFACT | Same semantics (bckg rows skipped instead of mapped) | Yes |
| Corrupt rows | 5 negative-duration `impd` rows skipped | Same 5 rows dropped | Yes |
| EMG input | Primary `SIMULATED_ENV_FROM_RAW` (D15: HPF 20.8 Hz → rectify → LPF 3.6 Hz → 50 Hz); secondary RAW | RAW 256 Hz only (spectral + difference-envelope features) | **No** |
| IMU units | "Kept as given" (resting \|acc\| ≈ 0.5 assumed) | Per-run gravity → g; invalid-scale runs excluded; gyro scale-invariant only | **No** (§3.1) |
| Windows | 4 s / 2 s (D13) | 5 s / 2 s (10 s tested) | No |
| Labels | D14: pos ≥ 50% FBTCS overlap; neg outside ±30 s of any seizure; else IGNORE | Pos ≥ 50%; FBTCS guard (−30, +120) s; other seizures (−30, +60) s; impd ignored | Close |
| Quality | VALID/DEGRADED/INVALID/MISSING/UNSYNCHRONIZED, availability masks, inheritance | Per-window ok flag (flat, EMG rail fraction) + per-run acc-scale check | Different depth |
| Split | `SEIZEIT2_PLANNED_SPLIT`: train/CV sub-001…096, held-out sub-097…125 (paper's suggestion) | Random stratified, 25 test / 100 dev | **No** (§4 D-A) |
| Models | D18: LR baseline, RF 25–50 trees depth 6–8 (embedded), XGBoost offline, SVM optional | Rule baseline, LR, HGB 100/300 trees; ablations | Partly |
| Event logic | D17 `EventConfirmation`: entry 0.65 / exit 0.45, 3 consecutive, hold policy, refractory 60 s | Mean of last 5 ≥ 0.9867, refractory 300 s, gap reset | **No** |
| Inference location | D11: ESP32-S3 on the pod (C99 port); phone does not run the detector | Phone, TypeScript, in `src/services/ml/` | **No** |
| Export | `EmbeddedExporter` contract (emlearn / micromlgen / m2cgen to investigate) | JSON trees + TS evaluator, parity-tested | No |
| Guard test | `test_package_contains_no_training_or_model_files` forbids `.fit(`, sklearn/pickle imports and model files inside `mlm/` | Training code lives outside `mlm/`, so the guard still passes | — |

---

## 3. Findings from MLM B that affect `mlm/`

### 3.1 Accelerometer scale is not constant across SeizeIT2 recordings (likely bug in Stage 3 features)

Median |acc| per run, which is gravity in the file's unit, over 2,487 runs that have EMG and chest IMU:

| Median \|acc\| | Runs | Meaning |
|---|---|---|
| ~0.50 | 245 | ±2 g-like storage scale (sub-001…016, 037–042, 049, 050, 052, 099–101, 107) |
| ~0.167 | 1,837 | 3× smaller scale |
| 32.768 | 347 | Channel pinned at the rail (sub-066–068, 084, 086, 088, 114, 122, 125 and others) |
| other (≈0, 1.3, 16, 22) | 58 | Unusable |

`mlm/` keeps IMU units as given and verified the 0.5 figure on sub-001 run-01 only. The default
quality configuration never flags these runs:

- `flatline_ptp = 0` only catches exactly constant channels.
- `saturation_limits` is empty and is not set by `prep.py`.

As a result, the Stage 3 amplitude features (|acc| statistics, jerk, band power) very likely mix
three scales, and pinned channels are rated VALID. The gyroscope differs between the two valid
regimes too, by about 4–5× in median magnitude, and its unit is undocumented.

What MLM B does:

- Divides each run by its own median |acc|, so gravity = 1 g, which is also the BMI270's native unit.
- Excludes runs outside `ACC_SCALE_VALID = (0.12, 0.7)`. This cost 3 FBTCS events.
- Uses only scale-invariant gyro features.

**Suggested check for `mlm/`:**

```python
# median |acc| per recording from mlm/data/seizeit2; expect clusters at ~0.5, ~0.167 and 32.768
```

If confirmed, adopt the calibration in `prep.py` (or a `quality.py` rule) before training on
`mlm/data/processed/`.

### 3.2 Fusion helps on this data

The ablations used RAW EMG, so they are evidence for the fusion hypothesis, not for the ENV path:

- IMU only reaches the same sensitivity but with 3.3× the false alarms (0.76 vs 0.23/24 h).
- EMG only is worse on both sensitivity and false alarms.

### 3.3 Things both pipelines found independently (good sign)

- 55 FBTCS in 37 participants, and no GTCS.
- The same 5 corrupt `impd` rows.
- 44 recordings have neither EMG nor chest IMU.
- The 256 Hz and 25 Hz sampling rates.
- The mapper semantics.

---

## 4. Decisions to sign off before merging

Each row needs an owner decision. Several of them change locked decisions D11–D18 in `context.md`.
The recommendation is a proposal only.

| # | Question | Options | Recommendation |
|---|---|---|---|
| **D-A** | Which participant split? | (1) Kyle's `SEIZEIT2_PLANNED_SPLIT` (sub-001…096 / 097…125); (2) PJ's random stratified split | **(1).** It was locked first, is documented, and follows the dataset paper. MLM B's current model **must not** be evaluated on 097…125: it trained on 22 of those 29 participants (15 of their 22 FBTCS). Retrain MLM B on 001…096. Note honestly that MLM B's design choices were informed by results that included some 097…125 participants. |
| **D-B** | Which EMG signal does the deployable model use? | (1) `SIMULATED_ENV_FROM_RAW` (D12/D15, matches MyoWare ENV); (2) RAW 256 Hz (MLM B) | **Train both under the same split.** ENV is the deployable path for the planned hardware. RAW serves as the upper-bound benchmark. If RAW clearly wins, revisit D12 (MyoWare RAW output at ≥ 256 Hz). |
| **D-C** | Window length | 4 s (D13) / 5 s / 10 s | **4 s** for consistency. MLM B saw 5 s ≈ 10 s, so 4 s is a cheap re-check. |
| **D-D** | Label buffers | D14 (±30 s around any seizure) / MLM B's asymmetric guards | **D14** for training labels. Keep MLM B's evaluation rule that alarms near out-of-scope seizures are reported separately, not as false alarms. |
| **D-E** | Event logic | D17 `EventConfirmation` / MLM B moving-average detector | **One implementation:** `EventConfirmation` (it has quality states and suppression). Port it to TS (and later C). Re-tune entry/exit on CV only. |
| **D-F** | Where does inference run? | Pod (D11) / phone (MLM B) / both | **Pod as the target, phone as interim.** Keep the TS runtime as the dev harness and fallback until pod firmware exists, and amend D11 to say so. Otherwise remove `services/ml/seizeit2` from the app. |
| **D-G** | Model family | RF 25–50 trees depth 6–8 (D18) / HGB 300×15 | **Add RF per D18 to the candidate list** and keep HGB as the offline comparison. Choose with the same predefined rule, plus a flash/RAM budget for the ESP32-S3. |
| **D-H** | Data root and environment | `mlm/data/` (full download) / `pj-training-data/` (DataLad + streamed fetch) | **One configurable root** (an env var such as `LYRA_DATA`). Support both "full download" and "stream + verify + delete". One venv, one Python version (3.12+; MLM B also runs on 3.14). |
| **D-I** | Stage 4 guard test | Keep / scope / remove | **Scope it:** forbid training code in the core `lyra_mlm` modules, but allow a `lyra_mlm.training` subpackage. Authorise Stage 4 explicitly in `context.md`. |

---

## 5. Merge procedure

### Phase 0: land this branch (pipelines coexist, nothing breaks)

```bash
git fetch origin
git switch seizeit2-model && git log --oneline origin/main..HEAD   # 2 commits
cd mlm && .venv/bin/python -m pytest -q && cd ..                     # 134 passed
node ml/seizeit2/parity_check.mjs && npx tsc --noEmit && npx expo lint
gh pr create --base main --head seizeit2-model --title "SeizeIT2 model + main repair" \
  --body-file docs/MLM_MERGE_GUIDE.md                                # or merge locally:
git switch main && git merge --ff-only seizeit2-model && git push origin main
```

After Phase 0, `main` has both MLMs and a working `mlm/` package. Sign off §4 before Phase 1.

### Phase 1: one package (`mlm/` becomes the single MLM)

Target layout:

```
mlm/
  pyproject.toml                 + optional-deps: train = ["scikit-learn>=1.9", "matplotlib>=3.11"]
  src/lyra_mlm/
    signals, annotations, sync, preprocess, quality, windows, features, model,
    inference, confirmation, evaluation, datasets, ble, edf, seizeit2, prep     (Kyle, unchanged)
    fetch.py                     ← ml/seizeit2/data_loader.py: fetch() + annex_key()
    training/                    ← new subpackage (allowed by the scoped guard test, D-I)
      config.py                  ← ml/seizeit2/config.py (merged with PrepConfig/ModelConfig)
      train.py                   ← ml/seizeit2/train.py
      events.py                  ← ml/seizeit2/evaluate.py event matching, FAR CIs, latency
      export_json.py             ← ml/seizeit2/export.py
      analysis.py                ← ml/seizeit2/dataset_analysis.py
  scripts/                       download / prepare / audit (Kyle) + train / evaluate / export entry points
  tests/                         + parity/ (parity_check.mjs, parity_vectors.json) + training tests
models/seizeit2/                 deployable artefacts (unchanged location)
src/services/ml/seizeit2/        app runtime (unchanged location)
```

File-by-file mapping. Use `git mv` so history follows the code.

| From (MLM B) | To | Action |
|---|---|---|
| `data_loader.SeizeIT2Mapper`, `list_runs`, `Run` | `lyra_mlm.seizeit2.SeizeIT2Adapter` / `SeizeIT2Mapper` | **Delete MLM B's copy.** Kyle's has identical semantics. Adapt the call sites that use `Run.intervals()` |
| `data_loader.read_edf` | `lyra_mlm.edf.read_edf` | **Delete MLM B's copy.** Keep the saturation mask: add it to Kyle's reader if it's missing |
| `data_loader.fetch`, `annex_key` | `lyra_mlm/fetch.py`, used by `scripts/download_seizeit2.py --stream` | **Move** |
| `preprocessing.window_features` (24 features) | `lyra_mlm.features` as a named feature set, e.g. `seizeit2_raw_v1` | **Move unchanged** while `model.json` v1 is in use. Any change needs a re-export and `parity_check.mjs` |
| Per-run acc gravity calibration + `ACC_SCALE_VALID` | `lyra_mlm.prep` / `quality` | **Adopt in both paths** (§3.1) |
| `preprocessing.label_windows` | `lyra_mlm.prep.label_window` | **Use Kyle's** (D-D), keeping MLM B's out-of-scope reporting in evaluation |
| `preprocessing.main` (feature cache) | `scripts/prepare_seizeit2.py` | **Merge.** One cache format: Kyle's `processed/windows/*.npz`, plus MLM B's feature set as extra columns |
| `train.py` | `lyra_mlm/training/train.py` | **Move.** Use `ParticipantSplit`, `check_participant_independence`, `require_training_partition`, and `ModelConfig.training_record` |
| `evaluate.py` | `lyra_mlm/training/events.py` + `lyra_mlm.evaluation.EvaluationReport` | **Merge.** Keep the one-time test guard and the CIs |
| `export.py` | `lyra_mlm/training/export_json.py` (an `EmbeddedExporter` sibling) | **Move** |
| `config.py` | `lyra_mlm/training/config.py` | **Merge** with `PrepConfig` / `ModelConfig` |
| `requirements.txt` | `mlm/pyproject.toml` `[project.optional-dependencies] train` | **Replace** |
| `parity_check.mjs`, `parity_vectors.json` | `mlm/tests/parity/` | **Move.** Update the relative paths to `src/` and `models/` |

Mechanical changes when moving:

1. **Imports.** `from lyra_mlm.annotations import …` becomes `from ..annotations import …`. Inside
   `training/`, `from . import config as C` stays.
2. **Paths.** `config.REPO = Path(__file__).resolve().parents[N]`: recompute N for the new depth, or
   replace it with the `LYRA_DATA` env var (D-H).
3. **Entry points.** `python -m ml.seizeit2.train` becomes `python -m lyra_mlm.training.train` (or
   `scripts/train_seizeit2.py`). `train.py`'s `__main__` re-import trick (it keeps pickled classes
   importable) must name the new module path.
4. **Old pickles.** `ml/seizeit2/runs/*/final_model.pkl` will not unpickle after the move. This
   doesn't matter: `models/seizeit2/model.json` is self-contained, and Phase 2 retrains anyway.
5. **Guard test.** Change `test_package_contains_no_training_or_model_files` to skip `training/`
   (D-I).
6. **Cleanup.** Delete `ml/seizeit2/` once everything has moved. Keep `.gitignore`'s
   `pj-training-data/` rule, and add the new data root if it differs.

Suggested commit sequence: one commit per row group above, with the tests green after each one.

```bash
git switch -c mlm-merge main
git mv ml/seizeit2/train.py mlm/src/lyra_mlm/training/train.py   # etc., then fix imports
cd mlm && .venv/bin/python -m pytest -q
```

### Phase 2: retrain once under the agreed decisions

1. Freeze the analysis plan (D-A…D-I) in `mlm/docs/TRAINING_PLAN.md` **before** touching held-out
   data.
2. Prepare features with the acc calibration (§3.1), for the ENV primary path and the RAW secondary
   path (D-B).
3. Run grouped CV on sub-001…096 only, with candidates LR, RF (D18), HGB, and the ENV/RAW/IMU
   ablations. Pick one candidate with one predefined rule.
4. Evaluate on sub-097…125 **once**. Report per participant: sub-115 and sub-116 hold 11 of the 22
   held-out events.
5. Export:
   - JSON for the phone runtime (if D-F keeps it) and run `parity_check.mjs`.
   - Embedded export for the ESP32 (D11), checked for C-vs-Python score parity.
6. Bump `models/seizeit2/metadata.json` to v2.0.0 and regenerate the model card, `split.json` and
   `evaluation/`.

### Phase 3: verification checklist

```bash
cd mlm && .venv/bin/python -m pytest -q                  # all green, guard test scoped
node mlm/tests/parity/parity_check.mjs                   # Python <-> TS parity
npx tsc --noEmit && npx expo lint                        # app
npx expo export --platform web --output-dir /tmp/lyra-web   # model JSON bundles
git check-ignore -v pj-training-data/ mlm/data/          # raw data ignored
git check-ignore models/seizeit2/model.json; echo $?     # 1 = trackable (expected)
git ls-files pj-training-data mlm/data | wc -l           # 0
```

---

## 6. Rules both pipelines already share (keep them after the merge)

- Raw recordings never go into git. Only code, configs, metrics and deployable models are tracked.
- The split is made before any windowing result is looked at. Held-out data is evaluated once.
  Normalisation and thresholds are fitted on training folds only.
- Accuracy is never the headline metric. Report event sensitivity, FAR/24 h, latency, PR-AUC,
  per-participant results and CIs.
- No "seizure detected" or "confirmed" wording, and no scores or percentages shown to users. The app
  says "Possible Event". The required alert text in `context.md` ("Possible convulsive seizure
  detected. This is not a diagnosis.") is still unreconciled with the app wording.
- Everything is described as experimental, in-hospital, FBTCS-only evidence. It is not GTCS evidence
  and not clinical validation.

## 7. Open risks after the merge

- **LYRA pod domain shift** (sensor, placement, MyoWare ENV vs SeizeIT2 RAW, chest vs pod IMU) is
  untested. A small consented pod recording set is needed before any real-world claim.
- **In-hospital negatives only.** Expect more false alarms in daily life. Labelled hard negatives
  (the `HARD_NEGATIVE_CATEGORIES` taxonomy) are needed.
- **Few events.** 44–47 usable FBTCS in total, with wide confidence intervals and heavy
  concentration in a few participants.
- **Latency** is measured from the annotated focal onset. There is no bilateral-onset timestamp, so
  time-to-convulsion is unknown.
