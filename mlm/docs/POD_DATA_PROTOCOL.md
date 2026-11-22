# Pod data-collection protocol and session format (draft v1)

This is for the people running recording sessions and for the firmware/logger developer. Tooling:
`lyra_mlm.pod` (loader and integrity checks), `lyra_mlm.domain` and `scripts/pod_report.py`
(domain-shift report). **No pod recordings exist yet.**

> **Governance first.** No one may be recorded until a consent / ethics protocol has been
> approved. Phase 1 is for healthy adult volunteers only; patients only after a separate approval.
> Participants are identified by pseudonym only (`pod-p001`); never record names, contact details or
> exact dates of birth. Data live in `mlm/data/pod/`, which is gitignored and never goes into git.
> LYRA is an experimental research prototype, not a medical device. Nothing collected here is clinical
> validation.

## 1. Why: the questions this data must answer

The SeizeIT2 models were trained on a **simulated** envelope computed from raw deltoid EMG
(µV-derived units) and on a **chest** IMU. The pod outputs a **MyoWare hardware ENV** (mV) and has its
IMU on the **upper arm**. Before any pod score means anything, we need to know:

1. **Q1 Envelope:** how does the MyoWare ENV relate to `SIMULATED_ENV_FROM_RAW` computed from the
   same electrodes? This needs the MyoWare RAW channel recorded at the same time.
2. **Q2 Domain gap:** how far are pod feature distributions from the SeizeIT2 training data, per
   feature?
   - The `simenv` path (pod RAW → simulated ENV) isolates the device and placement gap.
   - The `env` path (hardware ENV) adds the signal-type gap.
3. **Q3 Hard negatives:** which everyday activities produce windows that look event-like, and how
   often would the frozen pipeline alarm on free-living data (false alarms per 24 h)?
4. **Q4 Engineering checks:** does convulsion-like movement reach the detector at all?
   - Simulated convulsions are scripted movements by healthy volunteers. **They are never
     seizures and never sensitivity evidence.**
   - They are labelled `SIMULATED_CONVULSION`. Those windows can never become negatives, and they
     are excluded from every seizure metric.

## 2. Setup

- **Pod:** ESP32-S3 + MyoWare 2.0 + BMI270, in data-collection firmware mode. That mode must record
  three streams on one MCU clock, with timestamps in seconds from session start:
  - MyoWare **RAW** at **1000 Hz**;
  - MyoWare **ENV** at **50 Hz**;
  - IMU at **25 Hz**.
- **ADC calibration:** convert ADC readings to millivolts on the pod (ESP-IDF `adc_cali`). Store
  ENV/RAW in **mV**, never in raw counts. Declare the ADC full scale (`full_scale_mv`) so saturation
  at the rail is detected.
- **IMU units:** store in physical units, ACC in g and GYR in deg/s. Record the configured ranges, the
  axis mapping, and a photo of the pod orientation on the arm.
- **Placement:** proximal upper arm over the deltoid. Record the side, electrode positions, the
  reference electrode, the MyoWare gain setting and skin preparation. Use the same placement for
  every session of a participant, or record the change.
- **Annotation:**
  - The operator marks the start and end of every block on a logger synchronised to the pod clock,
    or with the pod button. Target precision is ±1 s.
  - Anything unexpected (cable tug, electrode lifting) is marked as an artifact event.

## 3. Session script

Use these `eventType` labels exactly. Each one maps onto `annotations.HARD_NEGATIVE_CATEGORIES`;
anything else loads as `UNMAPPED` and is flagged.

**Phase 1, lab session (about 75 min per participant):**

| Block | `eventType` | Duration |
|---|---|---|
| Seated rest | `rest_seated` | 5 min |
| Supine rest | `rest_supine` | 5 min |
| Sit ↔ stand ↔ lie, 10 transitions | `posture_change` | ~3 min |
| Walking | `walking` | 3 min |
| Stairs | `stair_climbing` | 2 min |
| Jogging in place | `running` | 2 min |
| Brushing teeth (vigorous) | `brushing_teeth` | 2 min |
| Eating / drinking | `eating_drinking` | 3 min |
| Writing / typing | `writing` | 3 min |
| Dressing (jacket on / off) | `dressing` | 2 min |
| Arm raises, repeated | `repeated_arm_movement` | 2 min |
| Stirring / wiping a table | `rhythmic_household_task` | 2 min |
| Clapping / tapping at 2–5 Hz (clonic-band rhythm) | `repetitive_gesture` | 2 min |
| Lifting a bag | `lifting` | 2 min |
| Simulated shivering | `shivering` | 1 min |
| Sudden jerks, 10 × | `sudden_jerk` | ~1 min |
| **Simulated convulsive movement, 3 × 45 s with rests**, seated or on a mat, supervised, stop at any discomfort | `sim_convulsion` | ~5 min |
| Tap / press the pod | `device_displacement` | 1 min |
| Lift one electrode lead briefly | `loose_electrode` | 30 s |

**Phase 2, free living (after Phase 1 works):**
- 24 h with the deployed configuration (ENV + IMU, no RAW), including sleep (`free_living`, `sleep`).
- The participant logs notable activities.
- This estimates false alarms per 24 h on real daily life.

**Planning numbers, not a powered design:**
- Phase 1: at least 10 volunteers.
- Phase 2: at least 10 × 24 h, about 240 h.

## 4. Session format (schema v1)

One folder per session: `mlm/data/pod/<participant>/<session>/`, for example `pod-p001/s01/`. The
folder names must equal `participant_id` and `session_id`. Neither may contain `_`.

| File | Content |
|---|---|
| `session.json` | metadata (below) |
| `env.csv` | header `t_s,mV[,flag]`: MyoWare ENV |
| `imu.csv` | header `t_s,ax,ay,az,gx,gy,gz[,flag]`: g and deg/s |
| `raw.csv` | header `t_s,mV[,flag]`: MyoWare RAW (data-collection mode only) |
| `events.tsv` | header `onset duration eventType source` (tab-separated, seconds on the pod clock) |

`flag` is optional: 0 = OK, 1 = missing, 2 = invalid (for example a CRC error on the link). Never
repair or interpolate samples; flag them.

```json
{
  "schema_version": 1,
  "participant_id": "pod-p001",
  "session_id": "s01",
  "consent_protocol": "<approved protocol id and version>",
  "pod_id": "pod-dev-0",
  "firmware": "<git describe of the firmware>",
  "placement": {"side": "left", "muscle": "deltoid", "electrodes": "<positions>", "reference": "<site>"},
  "duration_s": 4500,
  "streams": {
    "env": {"rate_hz": 50.0, "unit": "mV", "full_scale_mv": 3300, "myoware_gain": "<setting>"},
    "raw": {"rate_hz": 1000.0, "unit": "mV", "full_scale_mv": 3300},
    "imu": {"rate_hz": 25.0, "units": ["g", "g", "g", "deg/s", "deg/s", "deg/s"],
            "acc_range_g": 8, "gyr_range_dps": 1000, "axis_map": "<pod x/y/z on the arm>"}
  },
  "notes": ""
}
```

`PodAdapter` refuses a session in any of these cases:
- no `consent_protocol`;
- a participant id that isn't a lowercase pseudonym;
- a different schema version;
- ENV/RAW not in mV;
- IMU not in g / deg/s.

`check_session` reports duplicates, out-of-order samples, gaps, rate mismatches (over 0.5%), streams
with no usable samples, and unknown event labels.

## 5. After each session

```bash
.venv/bin/python scripts/pod_report.py --recording pod-p001_s01   # writes data/pod_reports/pod-p001_s01.json
```

Fix every integrity problem at its source before the next session. The report contains:
- **quality** state counts, the usable windows, and window counts per event label;
- **`vs_seizeit2_negatives`**: per trained feature, compared with SeizeIT2 Train/CV negative
  windows (never held-out): the pod and reference percentiles, the share of pod windows outside the
  reference 1st–99th percentile range, and the KS statistic. There is one section for `simenv`
  (trained schema) and one for `env` (hardware ENV, compared by feature-name suffix);
- **`sim_convulsion_vs_seizeit2_fbtcs`**: the same comparison for the simulated-convulsion windows
  against SeizeIT2 FBTCS windows (engineering context only);
- **`env_vs_simenv_signal`** (Q1): the correlation, median ratio and least-squares line between
  MyoWare ENV and simulated ENV. Descriptive only; nothing is applied.

## 6. Decision gates (fixed before data exist)

1. **Q1:** if MyoWare ENV and simulated ENV are well related across participants (high correlation
   and a stable line), a documented per-device mapping may be evaluated. Otherwise the envelope
   models must be retrained on pod data.
2. **Q2:** pod features that fall mostly outside the SeizeIT2 range, or that have large KS values,
   mean the frozen SeizeIT2 model cannot be trusted on the pod. Large gaps in the `simenv` path mean
   placement or device; large gaps only in the `env` path mean signal type.
3. **Q3:** false alarms per 24 h from Phase 2 are reported per activity group, as
   `evaluation.METRICS` defines. They are not compared with the SeizeIT2 held-out numbers as if they
   measured the same thing.

## 7. Rules

- The SeizeIT2 held-out set (sub-097…sub-125) has been evaluated once and is never used again, for
  anything.
- Any calibration or retraining with pod data uses **participant-grouped** cross-validation across
  pod participants. A participant's sessions never appear on both sides of a split.
- Simulated convulsions are never counted as seizures and never used as sensitivity evidence.
- Every number from pod data is a research estimate from a small number of volunteers, not
  validation.
