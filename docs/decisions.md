# Decisions

## Accepted

- Raw CSV files are immutable and excluded from Git.
- Experiment seed starts at `42` and is stored in configuration.
- Preprocessing must be fitted after splitting to prevent leakage.
- Scripts are thin entry points; reusable code belongs in the package.
- `main` is the presentation-ready branch; `develop` is the integration branch.
- Feature work is merged into `develop` through reviewed pull requests.
- CI runs lint and tests for pull requests; deployment automation is deferred.

## Pending human decisions

- Team review of the JOTB-5 regression target, its operational meaning, MAE,
  and chronological split proposed in [`ml_task_jotb_5.md`](ml_task_jotb_5.md).
- Acceptable error trade-offs for any operational deployment; the JOTB-5
  baseline only compares MAE with persistence.
- Meaning of `-300` in `5.csv`.
- Timezone of naive timestamps in `5.csv`.
- Coordinate and signal units.
- Mapping between files and DTS/DAS modalities.
- Open-source license.
