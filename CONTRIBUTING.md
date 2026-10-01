# Contributing

## Branches

- `main` contains only reviewed versions suitable for a demo or presentation.
- `develop` is the integration branch for completed tasks.
- Work starts from an up-to-date `develop` in a short-lived branch:
  `feature/<task>`, `fix/<task>` or `docs/<task>`.
- One task should produce one pull request into `develop`.
- Direct pushes to `main` are not part of the team workflow.

Example:

```bash
git switch develop
git pull --ff-only origin develop
git switch -c feature/time-features
```

## Pull requests

A pull request is ready for review when:

- its scope matches one task;
- relevant tests are included and pass;
- `ruff check .` passes;
- raw data, models and local secrets are absent from the diff;
- preprocessing is fitted only after the validation split;
- configuration, seed and metrics are saved for an experiment;
- the author documents assumptions and known limitations.

At least one teammate reviews changes before merge. Prefer squash merge for a compact
hackathon history. Delete the feature branch after merge.

## Initial ownership

| Area | Primary owner |
|---|---|
| `io.py`, `validation.py` | coordinator |
| `preprocessing.py`, `windowing.py`, `time_features.py` | coordinator |
| `dsp.py`, `dsp_features.py` | Ilya |
| `features.py` shared contract | team review |
| `modeling.py`, `evaluation.py`, `demo/` | Max Zavalov |
| `README.md`, `docs/` | coordinated edits |

Ownership identifies the first reviewer and does not prevent collaboration.
