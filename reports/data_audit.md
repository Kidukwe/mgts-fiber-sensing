# Initial data audit

Read-only audit of the six source files in `MGTS_data/`.

- Files: 6 CSV, approximately 759 MiB.
- Measurements: 6,749 rows.
- Spatial points: 23,327,758 pairs.
- All CSV records and JSON-encoded point arrays parsed successfully.
- No duplicate `(dts_id, channel_id, timestamp)` keys were found.
- Coordinates are strictly increasing inside each measurement.
- No NaN or infinite numeric values were found.
- Timestamp formats and spatial grid sizes differ between files.
- File `5.csv` has no timezone and contains 675 trailing values equal to `-300` in
  every row. Their meaning has not been assumed.
- Files `2.csv` and `4.csv` contain some shorter spatial profiles.
- Temporal gaps exist in all files.

This report records observations, not preprocessing decisions. The meaning of sentinel
values, units, sensor modality and timezone must be confirmed by a domain owner.
