"""Numerical and provenance tests for features built only from available history."""

import warnings
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from mgts_sensing.time_features import compute_time_features
from mgts_sensing.validation import Profile
from mgts_sensing.windowing import ProfileWindow, iter_windows


def make_profile(seconds, values, *, mask=None, aware=True):
    y = np.array(values, dtype=float)
    x = np.arange(y.size, dtype=float)
    mask = np.zeros(y.size, dtype=bool) if mask is None else np.array(mask, dtype=bool)
    for array in (x, y, mask):
        array.setflags(write=False)
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc if aware else None)
    timestamp += timedelta(seconds=seconds)
    return Profile(
        source_file=Path("a.csv"), row_number=2, line_number=2,
        dts_id="a", channel_id="1", timestamp=timestamp,
        timestamp_raw=timestamp.isoformat(), timezone_raw="+00:00" if aware else None,
        utc_offset_seconds=0.0 if aware else None, timestamp_format_normalized=False,
        x=x, y=y, sentinel_mask=mask, points_size=y.size,
    )


def features(profiles):
    return compute_time_features(ProfileWindow(tuple(profiles), session_index=0))


def test_numerical_formulas_and_irregular_elapsed_time():
    profiles = [make_profile(0, [1, 4]), make_profile(10, [3, 8]),
                make_profile(30, [9, 2])]
    result = features(profiles)
    np.testing.assert_array_equal(result.lag_values, [[9, 2], [3, 8], [1, 4]])
    np.testing.assert_array_equal(result.delta, [6, -6])
    assert result.delta_seconds == 20
    np.testing.assert_allclose(result.rate, [0.3, -0.3])
    np.testing.assert_allclose(result.rolling_mean, [13 / 3, 14 / 3])
    np.testing.assert_allclose(result.rolling_std, [np.sqrt(104 / 9), np.sqrt(56 / 9)])
    np.testing.assert_array_equal(result.valid_count, [3, 3])
    assert result.valid_count.dtype == np.dtype("int64")
    assert result.window.profiles[0] is profiles[0]


def test_sentinel_or_explicit_mask_excluded_without_runtime_warnings():
    profiles = [make_profile(0, [1, -300, -300, 7]),
                make_profile(10, [3, 5, 2, -300], mask=[False, False, True, False]),
                make_profile(30, [-300, 9, 6, -300], mask=[False, False, True, False])]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = features(profiles)
    np.testing.assert_allclose(result.lag_values,
                               [[np.nan, 9, np.nan, np.nan],
                                [3, 5, np.nan, np.nan],
                                [1, np.nan, np.nan, 7]], equal_nan=True)
    np.testing.assert_allclose(result.delta, [np.nan, 4, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(result.rate, [np.nan, 0.2, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(result.rolling_mean, [2, 7, np.nan, 7], equal_nan=True)
    np.testing.assert_allclose(result.rolling_std, [1, 2, np.nan, 0], equal_nan=True)
    np.testing.assert_array_equal(result.valid_count, [2, 2, 0, 1])


def test_outputs_are_readonly_copies_and_inputs_preserved():
    profiles = [make_profile(0, [1, -300]), make_profile(10, [3, 8])]
    snapshots = [(p.x.copy(), p.y.copy(), p.sentinel_mask.copy()) for p in profiles]
    result = features(profiles)
    assert not np.shares_memory(result.x, profiles[0].x)
    for field in ("x", "lag_values", "delta", "rate", "rolling_mean",
                  "rolling_std", "valid_count"):
        array = getattr(result, field)
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0
    for profile, snapshot in zip(profiles, snapshots):
        for array, before in zip((profile.x, profile.y, profile.sentinel_mask), snapshot):
            np.testing.assert_array_equal(array, before)
            assert not array.flags.writeable


def test_future_perturbation_does_not_change_already_available_features():
    prefix = [make_profile(0, [1]), make_profile(10, [3]), make_profile(20, [5])]
    original = prefix + [make_profile(30, [7])]
    perturbed = prefix + [make_profile(30, [70000])]

    def extract(profiles):
        return [compute_time_features(w) for w in iter_windows(
            profiles, history_size=2, max_gap_seconds=10,
        )]

    first, second = extract(original), extract(perturbed)
    for a, b in zip(first[:2], second[:2]):
        for field in ("x", "lag_values", "delta", "rate", "rolling_mean",
                      "rolling_std", "valid_count"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert first[-1].delta[0] != second[-1].delta[0]
    repeated = extract(original)
    for a, b in zip(first, repeated):
        np.testing.assert_array_equal(a.lag_values, b.lag_values)


def test_known_offsets_use_actual_elapsed_seconds():
    previous = make_profile(0, [1])
    current = make_profile(10, [3])
    current = replace(current, timestamp=current.timestamp.astimezone(
        timezone(timedelta(hours=3)),
    ), utc_offset_seconds=10800)
    result = features([previous, current])
    assert result.delta_seconds == 10
    np.testing.assert_allclose(result.rate, [0.2])


@pytest.mark.parametrize("count", [0, 1])
def test_too_short_manual_window_rejected(count):
    with pytest.raises(ValueError):
        features([make_profile(0, [1])] * count)


@pytest.mark.parametrize("changes", [
    {"source_file": Path("b.csv")}, {"dts_id": "b"}, {"channel_id": "2"},
    {"x": np.array([1.0, 2.0])},
    {"x": np.array([0.0, 0.0])},
    {"x": np.array([0.0, np.inf])},
    {"y": np.array([1.0, np.nan])},
    {"y": np.array(["a", "b"])},
    {"y": np.array([1.0])},
    {"sentinel_mask": np.array([False])},
    {"x": np.array([[0.0, 1.0]])},
])
def test_manual_window_revalidates_profiles_and_compatibility(changes):
    with pytest.raises(ValueError):
        features([make_profile(0, [1, 2]), replace(make_profile(10, [3, 4]), **changes)])


@pytest.mark.parametrize("seconds", [0, -1])
def test_duplicate_and_reverse_time_rejected(seconds):
    with pytest.raises(ValueError):
        features([make_profile(0, [1]), make_profile(seconds, [3])])


def test_naive_aware_manual_window_rejected():
    with pytest.raises(ValueError):
        features([make_profile(0, [1]), make_profile(10, [3], aware=False)])


def test_naive_time_supported_without_assigning_timezone():
    profiles = [make_profile(0, [1], aware=False), make_profile(10, [3], aware=False)]
    result = features(profiles)
    assert result.delta_seconds == 10
    assert all(p.timestamp.tzinfo is None for p in result.window.profiles)
