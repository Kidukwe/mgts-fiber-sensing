# Data layout

- `MGTS_data/` at the repository root contains the original CSV files. Treat them as
  immutable raw data; the directory is excluded from Git.
- `data/sample/` is reserved for a small, non-sensitive synthetic or explicitly
  redistributable sample used in tests and examples.
- Future derived datasets must be written to `data/interim/` or `data/processed/` and
  must not overwrite raw files.

The current CSV rows contain sensor id, channel, timestamp and a JSON-encoded `points`
array. Some files additionally contain `points_size`. Confirmed temperature units, end-of-fiber values and UTC+3 policy are documented
in docs/decisions.md and docs/solution.md. Raw x units remain separate from
index-based physical distances.
