# lyra-mlm: LYRA multimodal ML foundation (pre-training phase)

> **Status: software foundation only.** No model has been trained, no dataset has been downloaded or
> processed, and no detection performance exists or is claimed. LYRA is an experimental, unvalidated
> research prototype, not a diagnostic, predictive or preventive medical device.

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
  tests/                  pytest suites on SYNTHETIC fixtures only (+ test-only StubScorePredictor)
  docs/                   ARCHITECTURE, DATASETS, TRAINING_PLAN, BLE_PROTOCOL
```

## Setup and tests

```bash
cd mlm
uv venv --python 3.12 .venv          # or: python3.11 -m venv .venv
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/python -m pytest -q
```

The tests check software behaviour on generated signals. They are not ML evaluation and say nothing
about real-world detection.

## Docs

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): pipeline, contracts, quality semantics, configuration, limitations
- [docs/DATASETS.md](docs/DATASETS.md): source registry, LYRA vs dataset differences, integration procedure
- [docs/TRAINING_PLAN.md](docs/TRAINING_PLAN.md): how the separate training session should begin
- [docs/BLE_PROTOCOL.md](docs/BLE_PROTOCOL.md): draft pod ↔ phone payloads
