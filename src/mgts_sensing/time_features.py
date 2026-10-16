"""Compute per-coordinate features using only available profile history."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from .validation import SENTINEL
from .windowing import ProfileWindow, _block_key, _timestamp, _validate_profile


@dataclass(frozen=True)
class TimeFeatures:
    window: ProfileWindow
    x: NDArray[np.float64]
    lag_values: NDArray[np.float64]
    delta: NDArray[np.float64]
    rate: NDArray[np.float64]
    rolling_mean: NDArray[np.float64]
    rolling_std: NDArray[np.float64]
    valid_count: NDArray[np.int64]
    delta_seconds: float


@dataclass(frozen=True)
class TemporalValues:
    """Read-only values for explicit availability, without raw-sentinel semantics."""

    lag_values: NDArray[np.float64]
    delta: NDArray[np.float64]
    rate: NDArray[np.float64]
    rolling_mean: NDArray[np.float64]
    rolling_std: NDArray[np.float64]
    valid_count: NDArray[np.int64]
    delta_seconds: float


def compute_temporal_values(
    values: NDArray[np.float64],
    valid_mask: NDArray[np.bool_],
    timestamps: tuple[datetime, ...] | list[datetime],
) -> TemporalValues:
    """Compute oldest-first history with explicit mask; -300 has no special meaning."""
    if (
        not isinstance(values, np.ndarray) or values.ndim != 2
        or values.shape[0] < 2 or values.shape[1] == 0
        or values.dtype.kind not in "iuf" or not np.isfinite(values).all()
    ):
        raise ValueError("values must be a finite real (H, coordinates) matrix, H >= 2")
    if (
        not isinstance(valid_mask, np.ndarray) or valid_mask.dtype.kind != "b"
        or valid_mask.shape != values.shape
    ):
        raise ValueError("valid_mask must be boolean and match values")
    if len(timestamps) != values.shape[0] or any(
        not isinstance(value, datetime) for value in timestamps
    ):
        raise ValueError("timestamps must contain H datetime values")
    aware = timestamps[0].utcoffset() is not None
    if any((value.utcoffset() is not None) != aware for value in timestamps):
        raise ValueError("timestamps must share one naive/aware domain")
    times = [value.astimezone(timezone.utc) if aware else value for value in timestamps]
    if any(current <= previous for previous, current in pairwise(times)):
        raise ValueError("timestamps must strictly increase")
    delta_seconds = (times[-1] - times[-2]).total_seconds()
    lag_values = values[::-1].astype(np.float64, copy=True)
    valid = valid_mask[::-1]
    lag_values[~valid] = np.nan
    valid_count = valid.sum(axis=0, dtype=np.int64)
    rolling_mean = np.full(values.shape[1], np.nan, dtype=np.float64)
    variance = np.full(values.shape[1], np.nan, dtype=np.float64)
    # Expected missing/overflow values remain NaN/Inf in this numerical API;
    # callers distinguish availability and report unexpected nonfinite results.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = lag_values[0] - lag_values[1]
        rate = delta / delta_seconds
        np.divide(
            np.where(valid, lag_values, 0.0).sum(axis=0), valid_count,
            out=rolling_mean, where=valid_count > 0,
        )
        centered = np.where(valid, lag_values - rolling_mean, 0.0)
        np.divide(
            (centered * centered).sum(axis=0), valid_count,
            out=variance, where=valid_count > 0,
        )
        rolling_std = np.sqrt(variance)
    for array in (lag_values, delta, rate, rolling_mean, rolling_std, valid_count):
        array.setflags(write=False)
    return TemporalValues(
        lag_values, delta, rate, rolling_mean, rolling_std, valid_count, delta_seconds,
    )


def compute_time_features(window: ProfileWindow) -> TimeFeatures:
    """Mask suspect values in copies; calculate changes and population statistics.

    lag_values[0] is current, lag_values[1] previous. A coordinate without
    usable history has NaN mean/std and zero valid_count. Gap policy is the
    responsibility of iter_windows, not inferred here.
    """
    if not isinstance(window, ProfileWindow) or len(window.profiles) < 2:
        raise ValueError("Expected a ProfileWindow with at least two profiles")
    for profile in window.profiles:
        _validate_profile(profile)
    first = window.profiles[0]
    for previous, current in zip(window.profiles, window.profiles[1:]):
        if _block_key(current) != _block_key(first):
            raise ValueError("A window must contain one source/stream/timezone domain")
        if not np.array_equal(current.x, first.x):
            raise ValueError("A window must contain one coordinate grid")
        if _timestamp(current) <= _timestamp(previous):
            raise ValueError("Window timestamps must strictly increase")

    values = np.stack([profile.y for profile in window.profiles])
    valid = np.stack([
        ~(profile.sentinel_mask | (profile.y == SENTINEL)) for profile in window.profiles
    ])
    temporal = compute_temporal_values(
        values, valid, tuple(profile.timestamp for profile in window.profiles),
    )
    x = first.x.astype(np.float64, copy=True)
    x.setflags(write=False)
    return TimeFeatures(
        window=window,
        x=x,
        lag_values=temporal.lag_values,
        delta=temporal.delta,
        rate=temporal.rate,
        rolling_mean=temporal.rolling_mean,
        rolling_std=temporal.rolling_std,
        valid_count=temporal.valid_count,
        delta_seconds=temporal.delta_seconds,
    )
