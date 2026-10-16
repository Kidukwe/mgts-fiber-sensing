"""Spatial DSP on valid contiguous samples; no temporal resampling or future values."""

import numpy as np

from .dsp_features import spatial_high_frequency_fraction


def spatial_features(values: np.ndarray) -> dict[str, float]:
    """Median low-pass + high-pass energy and normalized spatial FFT energy.

    The spectrum is spatial cycles/sample, never temporal vibration frequency.
    The unsmoothed extrema preserve narrow hot/cold spots.
    """
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("Expected a nonempty finite vector")
    padded = np.pad(y, (1, 1), mode="edge")
    smooth = np.median(np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1)
    median = float(np.median(y))
    return {
        "median": median, "hot": float(y.max() - median),
        "cold": float(median - y.min()), "std": float(y.std()),
        "robust_hot": float(np.quantile(y, .95) - median),
        "robust_cold": float(median - np.quantile(y, .05)),
        "roughness": float(np.sqrt(np.mean((y - smooth) ** 2))),
        "gradient": float(np.mean(np.abs(np.diff(y)))) if len(y) > 1 else 0.0,
        "high_frequency_fraction": spatial_high_frequency_fraction(y),
    }


def change_features(delta, threshold, point_count):
    """Joint spatial support from coordinate-aligned differences, no hole bridging."""
    delta = np.asarray(delta, float)
    result = {"fraction_hot": 0.0, "fraction_cold": 0.0,
              "longest_hot_fraction": 0.0, "longest_cold_fraction": 0.0,
              "pair_fraction": len(delta) / point_count if point_count else 0.0}
    if not len(delta):
        return result
    for sign, name in ((1, "hot"), (-1, "cold")):
        mask = sign * delta > threshold
        run = longest = 0
        for flag in mask:
            run = run + 1 if flag else 0
            longest = max(longest, run)
        result["fraction_" + name] = float(mask.mean())
        result["longest_" + name + "_fraction"] = longest / point_count
    return result
