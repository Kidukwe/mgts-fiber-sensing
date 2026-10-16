"""Train-only standardized ridge regression with a portable JSON artifact."""

from dataclasses import dataclass

import numpy as np

from .preprocessing import fit_standardizer


@dataclass
class RidgeForecaster:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: np.ndarray
    alpha: float

    @classmethod
    def fit(cls, x, y, alpha=10.0):
        x, y = np.asarray(x, float), np.asarray(y, float)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
            raise ValueError("Expected aligned nonempty 2D arrays")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or alpha <= 0:
            raise ValueError("Finite arrays and positive regularization required")
        mean, scale = fit_standardizer(x)
        z = (x - mean) / scale
        intercept = y.mean(axis=0)
        coefficients = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]),
                                       z.T @ (y - intercept))
        return cls(mean, scale, coefficients, intercept, alpha)

    def predict(self, x):
        return ((np.asarray(x) - self.mean) / self.scale) @ self.coefficients + self.intercept

    def to_dict(self):
        return {name: getattr(self, name).tolist() for name in
                ("mean", "scale", "coefficients", "intercept")} | {"alpha": self.alpha}

    @classmethod
    def from_dict(cls, value):
        return cls(*(np.asarray(value[name], float) for name in
                     ("mean", "scale", "coefficients", "intercept")), value["alpha"])


def predict_artifact(artifact, features, last_values, rolling_values, zone_ids=None):
    """Apply a saved champion to causally generated ordered feature rows."""
    selected = artifact['selected_model']
    if selected == 'persistence':
        prediction = np.asarray(last_values, float).copy()
    elif selected == 'train_median':
        if zone_ids is None:
            raise ValueError('Zone IDs are required for the train-median champion')
        prediction = np.array([artifact['train_medians_c'].get(str(z),
                              artifact['global_train_median_c']) for z in zone_ids], float)
    elif selected == 'rolling_median':
        prediction = np.asarray(rolling_values, float).copy()
    else:
        prediction = np.asarray(last_values, float) + RidgeForecaster.from_dict(
            artifact['ridge']).predict(np.asarray(features)[:, artifact.get(
                'feature_indices', list(range(np.asarray(features).shape[1])))])
    if artifact.get('clip_excursions_to_nonnegative', True):
        prediction[:, 1:] = np.maximum(0, prediction[:, 1:])
    return prediction
