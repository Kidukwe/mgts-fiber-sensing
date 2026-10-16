"""Explicit train-only standardization; no fitting occurs in inference."""

import numpy as np


def fit_standardizer(training_features):
    values = np.asarray(training_features, float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty finite training feature matrix')
    mean, scale = values.mean(axis=0), values.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return mean, scale
