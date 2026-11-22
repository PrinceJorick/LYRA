# ml/lyra: SeizeIT2 → ESP32-S3 pipeline (ENV path, D12)

Experimental research code. It is not clinically validated and makes no diagnosis.

This pipeline trains on SeizeIT2 (`pj-training-data/ds005873`, read-only) and exports the C99 package in
`firmware/model/`. EMG is converted to a simulated MyoWare 2.0 ENV signal at 50 Hz (locked decision D12).
All filtering is causal. The results are in `reports/lyra/lyra_report.md`.

## Reproduce

Run from the repository root with the shared environment `ml/seizeit2/.venv`. Run the steps in this order.

```bash
PY=ml/seizeit2/.venv/bin/python
$PY -m ml.lyra.extract                         # features + labels per recording → ml/lyra/cache/features/<hash>/
$PY -m ml.lyra.benchmark                       # 3 windows × 3 modalities × 5 models, subject-wise OOF → ml/lyra/runs/bench-<ts>/
$PY -m ml.lyra.deploy --run ml/lyra/runs/bench-<ts>      # select, export to firmware/model/, measure, write the lock
$PY -m ml.lyra.parity --run ml/lyra/runs/bench-<ts>      # Python/C parity on real development windows
$PY -m ml.lyra.final_eval --run ml/lyra/runs/bench-<ts>  # sub-097..125, runs once only
$PY -m ml.lyra.report --run ml/lyra/runs/bench-<ts>      # reports/lyra/*
$PY -m pytest -q -p no:cacheprovider ml/lyra/tests
```

- **Evaluation lock.** `data.py` refuses to load sub-097..125 until `deploy.py` writes
  `ml/lyra/runs/FINAL_EVALUATION_LOCK.json`. That file freezes the selection and the SHA-256 of the models
  and firmware files. `final_eval.py` refuses to run a second time.
- **Size measurement.** Measuring flash and SRAM needs ESP-IDF v5.5 for esp32s3. Activate it first with
  `. ~/esp/esp-idf/export.sh`; on this machine, put `~/esp` on `PATH` first for the Python 3.13 shim.
- **Daily-life negatives.** When pod recordings exist, run
  `$PY -m ml.lyra.adl --lyra-negatives-dir PATH --run ml/lyra/runs/bench-<ts>`.
- **Resource use.** A full run takes about 45 min to extract, about 100 min to benchmark, and about 8 GB of
  peak RAM.
