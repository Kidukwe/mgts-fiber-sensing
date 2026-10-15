# Architecture

The project follows a small batch pipeline:

1. Read immutable source CSV files.
2. Validate schema, timestamps and spatial profiles.
3. Split data using an approved validation strategy.
4. Fit preprocessing only on the training partition.
5. Build features and train a reproducible baseline.
6. Save configuration, metrics, data fingerprint and model artifacts.

Reusable logic lives in `src/mgts_sensing/`. Thin commands live in `scripts/`.
Notebooks are used for exploration and presentation, not as the only implementation of
the pipeline.

## Module boundaries

- `io.py` and `validation.py`: source parsing and data contracts.
- `preprocessing.py` and `windowing.py`: transformations and window construction.
- `dsp.py`: reusable signal transforms; `dsp_features.py`: spectral feature extraction.
- `time_features.py`: time-domain feature extraction.
- `features.py`: shared feature-table contract and combination of feature families.
- `modeling.py` and `evaluation.py`: estimators, validation and metrics.
- `visualization.py`: plots shared by reports, notebooks and the demo.


## Event baseline integration

`event_cli.py` composes `iter_profiles` → `iter_feature_results` → `EventDetector`.
`modeling.py` provides a causal rolling median/MAD baseline, `events.py` tracks
confirmed per-zone candidates with hysteresis, and `evaluation.py` records paired
prequential errors against persistence. Both `scripts/run_baseline.py` and
`demo/demo.py` call this pipeline. This proposal does not establish signal units,
labels, or the previously proposed supervised regression experiment. See
[temperature_events.md](temperature_events.md) for the output contract.
