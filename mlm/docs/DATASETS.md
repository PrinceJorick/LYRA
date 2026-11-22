# Datasets: registry, verified SeizeIT2 inventory and Stage 3 outputs

The machine-readable registry is `lyra_mlm.datasets.SOURCES`. `[VERIFIED: R#]` facts were checked in
`context.md` §7 against primary sources. `[VERIFIED: files]` numbers were counted from the downloaded
v1.1.0 files on 2026-11-21 (Stage 3). Data lives only in the gitignored `mlm/data/`.

## Sources

| Source | Role | Access / licence | Status |
|---|---|---|---|
| **SeizeIT2**, OpenNeuro `ds005873` v1.1.0 | Primary: annotated FBTCS, wearable EMG + ACC + GYRO | Open, no registration, CC0 `[VERIFIED: R2]` | **Wearable subset downloaded and processed** |
| **NinaPro DB2 / DB7** | Supplemental non-seizure muscle activity | `[NOT VERIFIED]` | Not downloaded. Not needed for Stage 4. Pooling it with SeizeIT2 would invite a dataset-identity shortcut. |
| LYRA pod recordings | Target domain incl. scripted hard negatives | Needs an approved consent protocol | None exist; the main future data priority |

## SeizeIT2: what was downloaded

From OpenNeuro's public S3 bucket over HTTPS, with no credentials (`scripts/download_seizeit2.py`):
**13,568 files, 52.81 GB**, every file size-checked against the bucket listing, and SHA-256 for all
13,568 in `mlm/data/seizeit2/sha256_manifest.json`.

| Files | Count | Size |
|---|---|---|
| `*_emg.edf` + `*_emg.json` | 2,804 + 2,804 | 25.40 GB |
| `*_mov.edf` + `*_mov.json` | 2,551 + 2,551 | 27.41 GB |
| `*_events.tsv` (in `eeg/`) | 2,850 | < 0.01 GB |
| Top-level metadata | 8 | < 0.01 GB |

EEG (47.6 GB) and ECG (25.4 GB) were not downloaded, since LYRA uses neither.

## Verified inventory `[VERIFIED: files]`

| Item | Value | Published |
|---|---|---|
| Participants | 125 (`sub-001`..`sub-125`; `participants.tsv` = subject folders) | 125 `[R1]` |
| Recordings (runs with `events.tsv`) | 2,850 | 2,850 `[R1]` |
| Total duration (sum of `recordingDuration`) | 11,626.2 h | ~11,600 h `[R1]` / 11,640 h `[R2 README]` |
| Seizure annotations (`sz*` rows) | **883** | 883 `[R1]` / 886 `[R2 README]` |
| **FBTCS** (`sz_foc_f2b`) | **55**, in 37 participants | 55 `[R1, R2]` |
| **GTCS** (`sz_gen_m_tonicClonic`) | **0**; no generalized-onset seizures of any kind | focal-epilepsy cohort `[R1]` |
| EMG sampling rate (EDF header, all files) | 256 Hz, channel `EMG SD`, µV | 256 Hz `[R2]` |
| MOV sampling rate (EDF header, all files) | 25 Hz, 12 channels | 25 Hz `[R2]` |
| JSON vs header rate mismatches | 0 | |

Every eventType maps to a known kind (`SeizeIT2Mapper`): `bckg` 2,313 background, `impd` 459 artifact
(after skipping 5), 24 non-FBTCS `sz*` labels covering 828 out-of-scope focal or unknown-onset
seizures, and `sz_foc_f2b` with 55 FBTCS. No labels are `UNMAPPED`. Other categories are **never** reinterpreted as GTCS.

### Channel availability and exclusions

| Recordings | Count |
|---|---|
| With EMG (`EMG SD`) | 2,804 (46 have no EMG file) |
| With a usable 6-axis IMU (`ECGEMG SD` ACC X/Y/Z + GYR A/B/C, left chest) | 2,489 |
| MOV problems | 299 no MOV file; 51 MOV files without any ECGEMG channel; 10 without ECGEMG gyro; 1 without ECGEMG acc |
| With both | 2,487 |
| **With neither → excluded** | **44** (logged in `windows_manifest.json`) |
| **Processed** | **2,806** |

- **sub-097** (held-out) has zero usable recordings: all 11 lack an EMG file and the ECGEMG IMU
  module. It has no FBTCS events.
- Recordings missing one modality are processed, with that modality `MISSING` in every window.
  They are not dropped, so the continuous timeline is kept.

### Annotation anomalies

Five `impd` rows have negative durations in the v1.1.0 files. They are skipped and listed in
`inventory.json` (`skipped_annotation_rows`):

| Recording | Row | onset (s) | duration (s) |
|---|---|---|---|
| sub-031 run-07 | 2 | 51968 | −40258 |
| sub-031 run-07 | 3 | 52025 | −35020 |
| sub-043 run-02 | 1 | 37799 | −28796 |
| sub-043 run-02 | 2 | 37895 | −82 |
| sub-043 run-02 | 3 | 38306 | −11 |

Policy: malformed non-seizure rows are skipped and logged. A malformed seizure or unmapped row would
exclude the whole recording (`MalformedAnnotation`), never become a negative label. There were 0
such recordings.

### Signal facts that matter for modelling

1. **Signal type:** SeizeIT2 EMG is raw sEMG → `RAW`, and the primary path converts it with
   `raw_to_simulated_env` (256 Hz → 50 Hz, ratio 25/128). This is simulated, never hardware ENV.
2. **Bandwidth:** 256 Hz covers only up to 128 Hz, so RAW spectral features use 20–128 Hz.
3. **IMU placement:** the left chest (ECGEMG module) is used, tagged `imu_placement=chest_upper_back`.
   LYRA's IMU is on the upper arm.
4. **IMU units (Step 1 investigation 2026-11-22, Train/CV only; Stage 3.5 remediation):** the EDF
   label "g" is wrong for both sensors. v1.1.0 stores every ACC and GYR value as *sensor value / D*,
   with D = 6 or D = 2 per recording. Both sensors clip at exactly 8/D (1.333 or 4.000), both modules
   of a recording share the same D, and the three ACC axes share one scale. ACC × D is g (±8 g full
   scale). GYR × 125·D is deg/s (±1000 deg/s). The 125 factor was inferred from posture changes
   (ACC tilt vs integrated gyro), medium confidence, and is not documented by the dataset.
   - The authors' loader (`biomedepi/seizeit2`, which redirects to `mbhagubai/seizeit2`, commit
     `fcde7a7`) only calls `pyedflib.readSignal()` (EDF physical units). It applies no movement
     scaling, and their model uses EEG only. EDF physical min/max are taken from the data, so they
     carry no sensor full-scale information.
   - v1.0.0 exported ACC Z as an exact copy of Y. The v1.1.0 "Corrected accelerometer data" release
     re-exported Z only. In 221 Train/CV recordings (sub-066/067/068/084/086/088) the new Z is
     garbage: 77% of samples sit at ±32.767, and the rest are integers.
   - Stage 3.5 rule (`seizeit2.assess_imu_scale`, per recording, no labels, nothing fitted across
     recordings). The IMU is rejected (INVALID) if any of these holds:
     - the unit isn't "g";
     - a Z header bound is ≥ 30, or more than 1% of Z samples are at ±32.767;
     - any channel is more than 50% at a ceiling;
     - the resting |g| doesn't fit D=6 (0.155–0.180) or D=2 (0.45–0.60);
     - a value exceeds 8/D;
     - per-axis resting |g| differs by more than 10%;
     - the quiet-block p95/p5 is 1.15 or more (D=6) / 1.20 or more (D=2);
     - the converted resting |g| is outside 0.93–1.08 g.
   - Accepted IMU is converted. Samples at the stored 8/D ceiling become exactly ±8 g / ±1000 deg/s, so
     the window saturation limits catch them (more than 5% saturated → INVALID, any → DEGRADED).
     Decisions are in `processed/imu_scale_manifest.json`; held-out entries hold only D and the
     category.
5. **Timing:** EMG and MOV EDF start times are anonymised differently, so both are assumed to start
   at t = 0, the reference for `events.tsv` onsets.
6. **Scope:** an in-hospital focal-epilepsy cohort. FBTCS windows are not GTCS evidence and not
   clinical validation.

## Stage 3 outputs (`mlm/data/processed/`, gitignored)

| File | Content |
|---|---|
| `windows/<sub>/<recording>.npz` | per window: `start_s`, `end_s`, `label`, `fbtcs_overlap_s`, `artifact_overlap`, quality codes (`overall`, `emg_state`, `imu_state`, `raw_emg_state`), coverages, `primary` (22) + `primary_available`, `secondary` (8 RAW) + `secondary_available`, `primary_emg_signal_type`, `primary_emg_provenance` |
| `inventory.json` | counts above, rates, missing reasons, skipped rows, FBTCS by participant |
| `feature_schema.json` | feature order and specs, quality and label codes, prep config; "no normalization fitted" |
| `windows_manifest.json` | per-recording window, label and quality counts, or the exclusion reason |
| `splits_manifest.json` | participant split, FBTCS counts, window label counts |
| `imu_scale_manifest.json` | Stage 3.5 IMU decision per recording (D or rejection category; held-out: D and category only), rule, counts |
| `audit.json` | `scripts/audit_stage3.py` results (0 problems after Stage 3.5) |

Features are raw, with no normalization or other dataset statistic fitted.

### Stage 3.5 IMU remediation (2026-11-22)

IMU decisions per recording (`imu_scale_manifest.json`):

| | Accepted (D=6 / D=2) | Z corrupt | Fails D test | Axis disagreement | Stuck axis | Post-conversion | Rest spread | No ECGEMG IMU |
|---|---|---|---|---|---|---|---|---|
| Train/CV | 1,552 (1,376 / 176) | 221 | 44 | 24 | 7 | 5 | 5 | 214 |
| Held-out (aggregate only) | 474 (434 / 40) | 153 | 2 | 1 | 0 | 0 | 1 | 147 |

The rule reproduces the Step 1 classification for 2,051 of 2,072 Train/CV recordings. The 21
differences are all extra rejections, from the approved axis-disagreement (more than 10%) and
post-conversion (0.93–1.08 g) checks; 18 of them are D=2 recordings, where Z often reads 10–15% above
X/Y. The audit verifies:
- every non-IMU array (grid, labels, overlaps, EMG states, the 9 `emg_simenv_*` features, secondary
  features, provenance) is bit-identical to the pre-remediation files (2,806 of 2,806);
- every accepted IMU (2,026) re-loads in g / deg/s within full scale and rests at 0.93–1.08 g;
- no rejected IMU (780, including recordings without one) is ever rated usable.

### Window accounting (4 s windows, 2 s step, labels per Decision 1; after Stage 3.5)

| | All windows | Usable (overall VALID/DEGRADED) | Before Stage 3.5 |
|---|---|---|---|
| **N_total** | **20,296,484** | | |
| Positive | 3,716 | 2,980 (Train/CV 2,022, held-out 958) | 3,456 |
| Negative | 20,244,889 | 17,176,078 | 19,112,674 |
| Ignore | 47,879 | 39,960 | 42,468 |
| **N_excluded (signal)**: overall INVALID or MISSING | | **3,077,466** | 1,137,886 |

Conservation holds: 20,296,484 = 2,980 + 17,176,078 + 39,960 + 3,077,466. Labels are assigned to
every window independently of quality, so the "all windows" column also sums to N_total. "Usable"
here means usable for the multimodal track. The SIMULATED_ENV-only track keeps every EMG-usable window.

Quality per modality across all windows:

| Modality | VALID | DEGRADED | INVALID | MISSING |
|---|---|---|---|---|
| Overall (EMG + IMU) | 14,585,462 | 2,633,556 | 2,049,468 | 1,027,998 |
| Simulated ENV (inherits RAW) | 20,191,789 | – | 92,382 | 12,313 |
| IMU | 14,592,395 | 2,638,767 | 2,038,856 | 1,026,466 |
| RAW EMG | 20,191,789 | – | 92,382 | 12,313 |

Before Stage 3.5, IMU was VALID 15,524,391 / DEGRADED 3,652,695 / INVALID 92,932: corrupted-Z windows
were then rated usable. No window was `UNSYNCHRONIZED`.

### FBTCS coverage (after Stage 3.5)

| Split | FBTCS events | With positive windows | Usable positive: overall / SIMULATED_ENV / IMU |
|---|---|---|---|
| Train/CV | 33 | 30 | 29 / 30 / 29 |
| Held-out | 22 | 21 | 15 / 21 / 15 |

The 4 events without windows (sub-049 run-02, sub-054 run-04, sub-058 run-02, sub-115 run-12) are
in excluded recordings with no EMG and no ECGEMG IMU.

## Re-running

```bash
.venv/bin/python scripts/download_seizeit2.py   # resumable
.venv/bin/python scripts/prepare_seizeit2.py --workers 3   # resumable; 8 GB RAM fits 3 workers
.venv/bin/python scripts/audit_stage3.py [--baseline data/processed_pre35]   # Stage 3.5: bit-identity check
```
