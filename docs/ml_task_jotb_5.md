# JOTB-5: ML task and target contract

Status: proposed for team review. This document defines the first reproducible
baseline task; it does not claim to detect infrastructure incidents.

## Decision in brief

Use **regression** to forecast the median recorded signal in each fixed spatial
zone of `2.csv` at the next regular measurement, 120–121 seconds after the latest
profile. The target is calculated from a real later measurement, not from an
anomaly score or an invented event label. A user can inspect the expected signal
by zone and, when the next profile arrives, prioritize zones with large forecast
residuals for manual review. A residual is not a verified incident.

## Evidence and limits

- `MGTS_data.zip/2.csv` has `dts_id`, `channel_id`, `timestamp`, and JSON `points`
  pairs of coordinate and value. It has 696 rows for one sensor/channel on
  2026-04-10 00:01:41–23:58:41 UTC. There are 512 intervals of 120 seconds,
  170 of 121 seconds, and 13 longer intervals.
- The first 2,121 coordinates are a common prefix in all 696 profiles. Profile
  lengths range from 2,121 to 2,315 points; 14 are shorter than 2,315.
- `reports/data_audit.md` records six source CSV files, temporal gaps, and no
  duplicate `(dts_id, channel_id, timestamp)` keys. No event-label table or
  externally measured outcome is supplied with these files.
- `docs/decisions.md` leaves coordinate and signal units, file-to-modality
  mapping, and the interpretation of `-300` in `5.csv` unresolved. Do not call
  the value in `2.csv` temperature or vibration until this is confirmed.
- The repository's ingestion, windowing, feature, and baseline modules are
  currently placeholders. This contract does not imply they already run.

The CSV counts and coordinate-prefix statements above were checked by parsing
the attached archive. `2.csv` is chosen because its timestamps have UTC offsets,
it covers nearly one day, and its 120–121 second cadence can be filtered without
assuming the timezone or sentinel semantics of `5.csv`. This is a baseline on
one sensor/channel/day, not evidence of transfer to other assets or dates.

## Checklist and target schema

| Question | Contract |
| --- | --- |
| What is predicted? | The numerical median signal of a zone in the next profile. Task type: regression. |
| Who can use it? | An analyst can compare forecast and next observation by zone to select locations for review. No automated incident decision is justified yet. |
| What is one example? | One `(sensor, channel, zone, decision time t)` pair with five consecutive profiles ending at `t`. |
| How is a zone defined? | Use coordinate indices `[0, 2120)` of `2.csv`, split into 53 disjoint blocks of 40 points: zone `z` contains indices `[40z, 40(z+1))`, `z=0..52`. The one remaining common-prefix point is excluded. Forty points is a baseline design choice, not a confirmed physical distance. |
| Where does target come from? | For zone `z`, `y(z,t) = median(points[i].value at next profile for i in zone z)`. The next profile must have the same sensor and channel. It is never a feature for the prediction made at `t`. |
| Forecast horizon? | One *regular* next observation: each of the five consecutive transitions spanning the four input intervals and target interval must be 120–121 seconds. Drop windows crossing longer gaps; do not relabel a delayed profile as a two-minute target. |
| What is available at prediction time? | Once profile `t` is fully received, the five profiles ending at `t`, their timestamps, the sensor/channel identifier and zone index. Candidate features are zone medians, dispersion, last changes, and short-window trend computed only from these profiles. No full-dataset normalization or future profile is available. |
| Primary metric? | Mean absolute error (MAE) of predicted zone medians, first averaged over zones at each decision time and then over decision times. Report in raw signal units until their meaning is confirmed. |
| Simple baseline? | Persistence: predict the current zone median as the next one. Also report a train-only constant median per zone. An ML model must be compared against persistence, not only against the constant. |

The target is a measured signal summary. There are no supplied incident labels;
therefore classification accuracy, anomaly precision/recall, and incident-detection
claims cannot be estimated from the supplied archive. Unsupervised anomaly scores
or cluster IDs must not be treated as ground-truth labels.

## Train, validation, and test

All times below are UTC on 2026-04-10 and refer to source profile timestamps.
The full set of five input profiles **and** the later target profile must lie
inside the same block. No example may use measurements from another block.

| Block | Inclusive start | Exclusive end | Eligible six-profile windows checked in `2.csv` |
| --- | --- | --- | ---: |
| Train | 00:00 | 14:00 | 376 |
| Validation | 14:12 | 19:00 | 136 |
| Test | 19:12 | 24:00 | 112 |

The 12-minute omitted intervals exceed the approximately ten minutes spanned
by four input intervals and the next-profile horizon. Eligibility also requires all
five adjacent intervals in an example to be 120–121 seconds. The counts are
temporal windows; multiplying by 53 zones gives correlated zone examples, **not**
independent time trials. Events could last longer than 12 minutes, so repeat
the analysis with 30- and 60-minute embargoes as a sensitivity check.

The split is global for this channel: neighboring zones and overlapping windows
are not randomly assigned to different partitions. Do not pool other files or
channels into this evaluation. If transfer is later claimed, reserve a complete
sensor/channel or date for a separate external test after verifying comparable
signal semantics and coordinates.

## Leakage and evaluation rules

1. Generate split membership from timestamps before fitting preprocessing or
   selecting features. Fit scaling, imputation, feature selection, and any
   target transformation on train only.
2. For each example, verify `max(feature_timestamp) <= t < target_timestamp`
   and verify all six timestamps belong to its block. Reject long gaps.
3. Keep test untouched while choosing features, zone width, hyperparameters,
   and the decision to use an ML model. Use validation for those choices.
4. Compare validation and final test MAE with persistence and the train-only
   constant. Also report RMSE, MAE by zone and time, and relative MAE improvement
   `1 - MAE_model / MAE_persistence`. Quantify uncertainty by resampling time
   blocks, not individual neighboring zones/windows.
5. A usable baseline is a reproducible, leakage-free comparison. A learned
   model is useful only if it improves on persistence on validation and retains
   improvement on the untouched test. With one day of one channel, this remains
   a narrow empirical result, not a production accuracy claim.

## Handoff to JOTB-6

1. Implement parsing and schema validation for `2.csv`; record a source-file
   fingerprint and verify the counts, UTC timestamps, common coordinate prefix,
   and interval distribution above.
2. Construct the 53 zone series and eligible six-profile windows; assert the
   split and timestamp rules, then save the target/feature table with provenance.
3. Calculate persistence and train-only constant predictions before training.
4. Calculate only causal features from the five input profiles. Fit one simple
   regressor on train, select using validation MAE, and evaluate once on test.
5. Report metrics, the number of distinct decision times, errors during rapid
   signal changes, and sensitivity to wider embargoes. If the regressor does
   not beat persistence, retain persistence as the honest baseline.
