# lyra-mlm: LYRA multimodal ML foundation (Stage 4: trained research models)

> **Status: Stage 4 (2026-11-22).** SeizeIT2 windows were prepared (Stage 3) and IMU-scale-remediated
> (Stage 3.5). Random forests were trained on Train/CV, evaluated once on held-out participants, and
> exported to C99 (results: docs/TRAINING_PLAN.md §7). These are research estimates on an in-hospital
> cohort, not clinical validation. LYRA is an experimental, unvalidated research prototype, not a
> diagnostic, predictive or preventive medical device.

Python package for LYRA's sEMG (MyoWare 2.0) + 6-axis IMU (BMI270) pipeline. It defines the data
contracts and deterministic signal operations that the future training session and the ESP32-S3 C99
port will share:

`signals → quality → windows → features (+ fusion) → quality-gated inference → event confirmation → BLE payloads`

## Layout

```
mlm/
  pyproject.toml          Python >= 3.11, numpy, scipy (pytest for dev)
  src/lyra_mlm/
    signals.py            SignalStream / Recording schemas, sample flags, timestamp checks
    annotations.py        Annotation kinds, hard-negative activity taxonomy, AnnotationMapper contract
    sync.py               cross-modality clock check, explicit resampling (never implicit)
    preprocess.py         validated filters, fixed transforms, train-only normalization
    quality.py            VALID / DEGRADED / INVALID / MISSING / UNSYNCHRONIZED assessment
    windows.py            4 s / 2 s segmentation on a common timeline, native rates kept
    features.py           sEMG, IMU and cross-modal features with availability masks, fusion
    model.py              Prediction / ModelConfig schemas, predictor, trainer, exporter contracts
    inference.py          quality-gated routing (suppress / degraded / unimodal fallback)
    confirmation.py       EventConfirmation state machine
    evaluation.py         partitions, participant split, leakage guard, metric definitions
    datasets.py           dataset source registry, metadata schema, adapter contract, integrity checks
    ble.py                draft BLE payload layouts with a hardware/simulated origin byte
    edf.py                minimal EDF/EDF+ reader (rates, units, scaling from each header)
    seizeit2.py           SeizeIT2 BIDS adapter and label mapper
    prep.py               Stage 3 per-recording preparation (simulated ENV, windows, labels, features)
    training.py           Stage 4 tracks, negative subsampling, models, offline confirmation, event metrics
  scripts/
    download_seizeit2.py  resumable, size-checked download of the wearable subset
    prepare_seizeit2.py   inventory, checksums, window files, split manifest -> data/processed/
    audit_stage3.py       re-derives labels/grid, checks quality and the Stage 3.5 IMU rule
    train_stage4.py       participant-grouped CV, selection, frozen models -> models/ (gitignored)
    evaluate_heldout.py   the single held-out evaluation (refuses a second run)
    export_c99.py         frozen random forests -> export/c99/ (m2cgen)
  tests/                  pytest suites on SYNTHETIC fixtures only (+ test-only StubScorePredictor)
  docs/                   ARCHITECTURE, DATASETS, TRAINING_PLAN, BLE_PROTOCOL
```

## Setup and tests

```bash
cd mlm
uv venv --python 3.12 .venv          # or: python3.11 -m venv .venv
uv sync --extra dev --extra train     # xgboost on macOS also needs: brew install libomp
.venv/bin/python -m pytest -q
```

The tests check software behaviour on generated signals. They are not ML evaluation and say nothing
about real-world detection.

## Stage 3 data

```bash
.venv/bin/python scripts/download_seizeit2.py      # ~53 GB into data/seizeit2/
.venv/bin/python scripts/prepare_seizeit2.py      # -> data/processed/
```

Never commit anything under `data/`.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): pipeline, contracts, quality semantics, configuration, limitations
- [docs/DATASETS.md](docs/DATASETS.md): source registry, LYRA vs dataset differences, integration procedure
- [docs/TRAINING_PLAN.md](docs/TRAINING_PLAN.md): how the separate training session should begin
- [docs/BLE_PROTOCOL.md](docs/BLE_PROTOCOL.md): draft pod ↔ phone payloads
