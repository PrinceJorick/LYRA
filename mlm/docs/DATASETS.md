# Datasets: registry and integration procedure

No dataset has been downloaded, opened or processed. The machine-readable registry is
`lyra_mlm.datasets.SOURCES`. `[VERIFIED: R#]` facts were checked in `context.md` §7 against primary
sources. Re-check every count against the actual release files before relying on it.

## Sources

| Source | Role | Access / licence | Status |
|---|---|---|---|
| **SeizeIT2**, OpenNeuro `ds005873` v1.1.0 | Primary: annotated FBTCS, wearable EMG + ACC + GYRO | Open, no registration, CC0 `[VERIFIED: R2]` | Documented, not downloaded |
| **NinaPro DB2** (ninapro.hevs.ch) | Supplemental non-seizure muscle activity | `[NOT VERIFIED]` | Documented, nothing verified |
| **NinaPro DB7** (ninapro.hevs.ch/instructions/DB7.html) | Supplemental non-seizure muscle/movement | `[NOT VERIFIED]` | Documented, nothing verified |
| LYRA pod recordings | Consent-based target-domain data incl. hard negatives | Needs an approved consent protocol | None exist |
| Future | SeizeIT2 interictal segments, exercise/fatigue datasets | n/a | Not selected |

### SeizeIT2 facts (from context.md)

125 participants `[R1]`. ~11,640 recording hours `[R1, R2]`. **55 FBTCS** `[R1, R2]`. Focal-epilepsy
in-hospital population `[R1]`. EMG: 1 channel, left deltoid, released at 256 Hz (device sampled 250 Hz)
`[R1, R2]`. Movement: ACC + GYRO at 25 Hz, 12 channels across two modules (back of neck, left chest)
`[R2]`. BIDS 1.8.0 with EDF + JSON sidecars + events TSV `[R1, R2]`. Suggested split: sub-001…096
train, sub-097…125 held out `[R1]`, encoded as `SEIZEIT2_PLANNED_SPLIT` with `verified=False`.

### How SeizeIT2 differs from LYRA (must be handled, not ignored)

1. **Signal type.** SeizeIT2 EMG is a sampled electrode signal, so its adapter must emit
   `EmgSignalType.RAW`. LYRA's first path is the MyoWare **ENV** analog envelope. ENV-schema features
   (`emg_env_*`) cannot be computed from SeizeIT2 without an explicit, documented RAW→envelope
   conversion, and even then that conversion will not equal MyoWare's analog stage.
2. **Bandwidth.** 256 Hz covers only up to 128 Hz, below the usual sEMG band.
3. **IMU placement.** SeizeIT2's IMU is on the neck and chest, while LYRA's is on the upper arm/deltoid.
   IMU features may not transfer, so the adapter must record placement in `Recording.placement`.
4. **Scope.** FBTCS only. It does not validate GTCS detection on its own.
5. **Setting.** In-hospital. Real-life false-alarm rates cannot be estimated from it.

### NinaPro

Expected (`[NOT VERIFIED]`): multi-electrode forearm sEMG for hand/wrist movements, recorded with
different hardware. Before any use, read the official pages and record the channel count, placement,
sampling rates, IMU content, licence and registration terms. Do not bypass registration.

## Integration procedure (next session, after explicit authorization)

1. Confirm the SeizeIT2 version and snapshot. Record the DOI, file list and per-file SHA-256
   (`sha256_file`) in a manifest committed to git. The data itself never goes in git; `mlm/data/` is
   ignored.
2. Inspect the real BIDS sidecars and `events.tsv` vocabulary before writing any mapping. Implement an
   `AnnotationMapper` in which unknown labels map to `UNMAPPED`, never to a negative.
3. Implement `DatasetAdapter` for SeizeIT2. It must produce `Recording` objects with native rates,
   original timestamps, verbatim `raw_label`s and correct placement strings, with no filtering,
   resampling or normalization in the adapter.
4. Run `validate_recording` on every recording and publish an integrity report: timestamp issues,
   rate mismatches, recordings with missing modalities.
5. Re-count participants, recordings, hours and FBTCS events from the files, and reconcile them with
   R1/R2. Check that the participant inventory matches `SEIZEIT2_PLANNED_SPLIT`, then set
   `verified=True` only if it matches.
6. Decide and document the ENV-vs-RAW strategy for SeizeIT2 (decision record) before generating
   any windows.
