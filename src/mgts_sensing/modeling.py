"""Causal, bounded-memory baseline; observations are added after prediction."""

from collections import deque

import numpy as np


class RobustBaseline:
    """Rolling median and MAD over accepted past observations only."""

    def __init__(self, window: int, min_history: int) -> None:
        if (type(window) is not int or type(min_history) is not int
                or not 2 <= min_history <= window):
            raise ValueError("Require integer window >= min_history >= 2")
        self.values: deque[float] = deque(maxlen=window)
        self.min_history = min_history

    def predict(self) -> tuple[float, float] | None:
        if len(self.values) < self.min_history:
            return None
        values = np.array(self.values)
        with np.errstate(over="ignore", invalid="ignore"):
            center = float(np.median(values))
            scale = float(1.4826 * np.median(np.abs(values - center)))
        if not np.isfinite(center) or not np.isfinite(scale):
            return None
        return center, scale

    def observe(self, value: float) -> None:
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError("Baseline observation must be finite")
        self.values.append(float(value))
