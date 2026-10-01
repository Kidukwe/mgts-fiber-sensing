# Data layout

- `MGTS_data/` at the repository root contains the original CSV files. Treat them as
  immutable raw data; the directory is excluded from Git.
- `data/sample/` is reserved for a small, non-sensitive synthetic or explicitly
  redistributable sample used in tests and examples.
- Future derived datasets must be written to `data/interim/` or `data/processed/` and
  must not overwrite raw files.

The current CSV rows contain sensor id, channel, timestamp and a JSON-encoded `points`
array. Some files additionally contain `points_size`. Units, sensor modality and special
values must be documented before preprocessing is implemented.
