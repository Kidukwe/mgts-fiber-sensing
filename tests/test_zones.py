"""Independent numerical and causal acceptance tests for the proposal zone API."""

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from mgts_sensing.time_features import compute_temporal_values
from mgts_sensing.validation import Issue, ProfileResult, parse_profile
from mgts_sensing.zones import ZoneConfig, aggregate_profile, iter_feature_results


def config(**changes):
    return ZoneConfig(**{
        "index_start": 0, "index_stop": 4, "zone_width": 4,
        "history_size": 3, "max_gap_seconds": 121, "min_valid_fraction": 1.0,
        **changes,
    })


def outcome(values=(1, 2, 3, 4), *, seconds=0, row=1, source="sample.csv",
            channel="1", sensor="a", x=None, naive=False):
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    timestamp_text = (timestamp.replace(tzinfo=None) if naive else timestamp).isoformat()
    return parse_profile({
        "dts_id": sensor, "channel_id": channel, "timestamp": timestamp_text,
        "points": json.dumps(list(zip(range(len(values)) if x is None else x, values))),
    }, source, row, row + 1)


def features(results, **changes):
    return list(iter_feature_results(results, config(**changes)))


@pytest.mark.parametrize("changes", [
    {"index_start": True}, {"index_start": -1}, {"index_start": 4},
    {"index_stop": 4.0}, {"index_stop": False}, {"zone_width": 0},
    {"zone_width": 2.0}, {"zone_width": True}, {"history_size": 1},
    {"history_size": True}, {"max_gap_seconds": 0}, {"max_gap_seconds": float("inf")},
    {"max_gap_seconds": float("nan")}, {"min_valid_fraction": 0},
    {"min_valid_fraction": 1.1}, {"min_valid_fraction": float("nan")},
    {"max_gap_seconds": True}, {"min_valid_fraction": True},
])
def test_invalid_config_is_rejected(changes):
    with pytest.raises((ValueError, TypeError)):
        config(**changes)


def test_spatial_statistics_and_provenance():
    raw = outcome((1, 2, 3, 10), x=[0.1, 0.3, 0.7, 1.2]).profile
    aggregated = aggregate_profile(raw, config())
    zone, = aggregated.zones
    assert aggregated.profile is raw
    assert (zone.zone_id, zone.index_start, zone.index_stop) == (0, 0, 4)
    assert (zone.x_min, zone.x_max) == (0.1, 1.2)
    assert (zone.nominal_count, zone.point_count, zone.valid_count) == (4, 4, 4)
    assert zone.median == 2.5 and zone.minimum == 1 and zone.maximum == 10
    assert zone.maximum_minus_median == 7.5 and zone.median_minus_minimum == 1.5
    assert zone.std == pytest.approx(np.sqrt(12.5))
    assert zone.valid_fraction == 1 and zone.sentinel_fraction == 0
    np.testing.assert_array_equal(aggregated.selected_x, raw.x)
    assert aggregated.selected_x is not raw.x and not aggregated.selected_x.flags.writeable


@pytest.mark.parametrize("values,low,high", [
    ([20, 20, 20, 20, 100], 20, 100), ([20, 20, 20, 20, -10], -10, 20),
])
def test_narrow_excursion_is_not_lost_when_median_unchanged(values, low, high):
    zone, = aggregate_profile(outcome(values).profile, config(index_stop=5, zone_width=5)).zones
    assert zone.median == 20 and zone.minimum == low and zone.maximum == high
    assert zone.maximum_minus_median == high - 20
    assert zone.median_minus_minimum == 20 - low


def test_partial_zone_denominator_is_nominal_width_and_indices_are_absolute():
    zones = aggregate_profile(outcome([99, 1, 2, 3, 4, 5, 6]).profile,
                              config(index_start=1, index_stop=7)).zones
    first, tail = zones
    assert (first.index_start, first.index_stop) == (1, 5)
    assert (tail.index_start, tail.index_stop, tail.nominal_count, tail.point_count) == (5, 7, 4, 2)
    assert tail.valid_fraction == 0.5 and tail.sentinel_fraction == 0
    assert tail.median is None and "INCOMPLETE_ZONE" in tail.quality_flags
    assert "INSUFFICIENT_VALID_POINTS" in tail.quality_flags
    permissive = aggregate_profile(outcome([99, 1, 2, 3, 4, 5, 6]).profile,
                                   config(index_start=1, index_stop=7,
                                          min_valid_fraction=0.5)).zones[-1]
    assert permissive.median == 5.5 and permissive.std == 0.5


def test_raw_marker_or_explicit_mask_is_excluded_only_in_derived_values():
    raw = outcome([1, -300, 3, 7]).profile
    mask = np.array([False, False, True, False], dtype=bool)
    mask.setflags(write=False)
    raw = replace(raw, sentinel_mask=mask)
    zone, = aggregate_profile(raw, config(min_valid_fraction=0.5)).zones
    assert zone.valid_count == 2 and zone.sentinel_count == 2
    assert zone.valid_fraction == zone.sentinel_fraction == 0.5
    assert (zone.minimum, zone.median, zone.maximum, zone.std) == (1, 4, 7, 3)
    np.testing.assert_array_equal(raw.y, [1, -300, 3, 7])
    strict, = aggregate_profile(raw, config()).zones
    assert all(getattr(strict, key) is None for key in (
        "median", "minimum", "maximum", "std", "maximum_minus_median", "median_minus_minimum"))


def test_fully_masked_zone_retained_as_null_not_zero():
    result, = features([outcome([-300] * 4)])
    row, = result.rows
    assert result.status == "aggregated"
    assert row["valid_count"] == 0 and row["sentinel_count"] == 4
    assert row["median"] is None and row["lag_medians"] == [None, None, None]
    assert row["temporal_valid_count"] == 0


def test_derived_median_equal_to_marker_stays_valid_temporally():
    results = features([outcome([-301, -299], seconds=i * 10, row=i + 1)
                        for i in range(3)], index_stop=2, zone_width=2)
    last, = results[-1].rows
    assert last["median"] == -300 and last["sentinel_count"] == 0
    assert last["lag_medians"] == [-300, -300, -300]
    assert last["rolling_mean"] == -300 and last["rolling_std"] == 0
    assert last["temporal_valid_count"] == 3 and last["delta"] == 0


def test_temporal_values_use_actual_intervals_and_population_std():
    results = features([outcome([value] * 4, seconds=seconds, row=index + 1)
                        for index, (seconds, value) in enumerate([(0, 2), (10, 4), (30, 10)])])
    warmup = results[1].rows[0]
    assert warmup["history_ready"] is False and warmup["lag_medians"] == [4, 2, None]
    assert warmup["temporal_valid_count"] == 2 and warmup["rolling_mean"] is None
    assert warmup["delta_seconds"] is None and warmup["rate"] is None
    last = results[-1].rows[0]
    assert last["history_ready"] is True and last["history_count"] == 3
    assert last["lag_medians"] == [10, 4, 2] and last["delta"] == 6
    assert last["delta_seconds"] == 20 and last["rate"] == 0.3
    assert last["rolling_mean"] == pytest.approx(16 / 3)
    assert last["rolling_std"] == pytest.approx(np.sqrt(104 / 9))
    assert [item["row_number"] for item in last["history_sources"]] == [3, 2, 1]
    assert len(last["history_sources"]) == 3


def test_spatial_median_is_computed_before_temporal_statistics():
    # Each profile has median 0, but median(per-coordinate history mean) is 100/3.
    samples = ([0, 0, 100], [0, 100, 0], [100, 0, 0])
    result = features([outcome(values, seconds=index * 10, row=index + 1)
                       for index, values in enumerate(samples)],
                      index_stop=3, zone_width=3)[-1].rows[0]
    assert result["lag_medians"] == [0, 0, 0]
    assert result["rolling_mean"] == result["rolling_std"] == 0
    assert result["maximum"] == 100 and result["maximum_minus_median"] == 100


def test_missing_zonal_medians_do_not_make_full_history_unready():
    result = features([outcome([2] * 4), outcome([-300] * 4, seconds=10, row=2),
                       outcome([8] * 4, seconds=20, row=3)])[-1].rows[0]
    assert result["history_ready"] and result["temporal_valid_count"] == 2
    assert result["lag_medians"] == [8, None, 2]
    assert result["rolling_mean"] == 5 and result["rolling_std"] == 3
    assert result["delta"] is None and result["rate"] is None


def test_all_missing_and_single_available_full_history_have_explicit_statistics():
    empty = features([outcome([-300] * 4, seconds=i * 10, row=i + 1)
                      for i in range(3)])[-1].rows[0]
    assert empty["history_ready"] and empty["temporal_valid_count"] == 0
    assert empty["rolling_mean"] is None and empty["rolling_std"] is None
    single = features([outcome([-300] * 4), outcome([-300] * 4, seconds=10, row=2),
                       outcome([7] * 4, seconds=20, row=3)])[-1].rows[0]
    assert single["history_ready"] and single["temporal_valid_count"] == 1
    assert single["rolling_mean"] == 7 and single["rolling_std"] == 0
    assert single["delta"] is None


def test_available_extreme_temporal_values_export_null_with_explicit_overflow_issue():
    values = [0, 0, -1e308, -1e308, 1e308, 1e308, 0, 0,
              1e308, -1e308, -1e308, 0, 1e308, -1e308, -1e308, -1e308]
    # Arithmetic failures must not escape as runtime warnings or be mistaken
    # for legitimately missing history. Inputs and spatial statistics are finite.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = features([outcome([value], seconds=index, row=index + 1)
                           for index, value in enumerate(values)],
                          index_stop=1, zone_width=1, history_size=16)[-1]
    row, = result.rows
    assert row["history_ready"] and row["temporal_valid_count"] == 16
    assert row["median"] == -1e308 and row["std"] == 0
    assert row["rolling_mean"] is None and row["rolling_std"] is None
    assert "NONFINITE_DERIVED_STATISTIC" in row["quality_flags"]
    issue = next(issue for issue in result.issues if issue.code == "NONFINITE_DERIVED_STATISTIC")
    assert issue.row_number == 16 and issue.line_number == 17
    assert issue.severity == "warning" and issue.description
    json.dumps(row, allow_nan=False)


def test_partial_temporal_availability_is_explicit_not_numeric_overflow():
    result = features([outcome([2] * 4), outcome([-300] * 4, seconds=10, row=2),
                       outcome([8] * 4, seconds=20, row=3)])[-1]
    row, = result.rows
    assert row["history_ready"] and row["temporal_valid_count"] == 2
    assert row["delta"] is None and row["rate"] is None
    assert row["rolling_mean"] == 5 and row["rolling_std"] == 3
    assert "PARTIAL_TEMPORAL_VALUES" in row["quality_flags"]
    assert "NONFINITE_DERIVED_STATISTIC" not in row["quality_flags"]
    assert "PARTIAL_TEMPORAL_VALUES" in {issue.code for issue in result.issues}
    assert "NONFINITE_DERIVED_STATISTIC" not in {issue.code for issue in result.issues}


def test_no_available_temporal_values_has_quality_flag_and_contextual_issue():
    result = features([outcome([-300] * 4, seconds=index * 10, row=index + 1)
                       for index in range(3)])[-1]
    row, = result.rows
    assert row["history_ready"] and row["temporal_valid_count"] == 0
    assert all(row[key] is None for key in ("delta", "rate", "rolling_mean", "rolling_std"))
    assert "NO_VALID_TEMPORAL_VALUES" in row["quality_flags"]
    assert "NONFINITE_DERIVED_STATISTIC" not in row["quality_flags"]
    issue = next(issue for issue in result.issues if issue.code == "NO_VALID_TEMPORAL_VALUES")
    assert issue.row_number == 3 and issue.line_number == 4 and issue.description


def test_temporal_core_respects_explicit_mask_and_preserves_finite_input():
    values = np.array([[-300, 1], [-300, 3], [-300, 5]], dtype=float)
    mask = np.array([[True, False], [True, True], [True, True]], dtype=bool)
    original_values, original_mask = values.copy(), mask.copy()
    times = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=i * 10)
             for i in range(3)]
    result = compute_temporal_values(values, mask, times)
    np.testing.assert_array_equal(result.valid_count, [3, 2])
    np.testing.assert_array_equal(result.rolling_mean, [-300, 4])
    np.testing.assert_array_equal(result.rolling_std, [0, 1])
    assert np.isnan(result.lag_values[-1, 1])
    assert result.delta_seconds == 10
    for array in (result.lag_values, result.delta, result.rate, result.rolling_mean,
                  result.rolling_std, result.valid_count):
        assert not array.flags.writeable
    np.testing.assert_array_equal(values, original_values)
    np.testing.assert_array_equal(mask, original_mask)


@pytest.mark.parametrize("kind", ["nan", "shape", "mask_type", "duplicate", "reversed", "domain"])
def test_temporal_core_rejects_invalid_inputs(kind):
    values = np.ones((3, 2), dtype=float)
    mask = np.ones((3, 2), dtype=bool)
    times = [datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=i * 10)
             for i in range(3)]
    if kind == "nan":
        values[0, 0] = np.nan
    elif kind == "shape":
        mask = mask[:, :1]
    elif kind == "mask_type":
        mask = mask.astype(float)
    elif kind == "duplicate":
        times[2] = times[1]
    elif kind == "reversed":
        times = times[::-1]
    elif kind == "domain":
        times[1] = times[1].replace(tzinfo=None)
    with pytest.raises(ValueError):
        compute_temporal_values(values, mask, times)


def test_feature_iterator_does_not_read_a_future_profile_to_emit_current():
    def inputs():
        yield outcome()
        raise AssertionError("A future outcome was consumed before current emission")
    result = next(iter_feature_results(inputs(), config()))
    assert result.rows[0]["history_count"] == 1


def test_naive_timestamp_and_quality_are_not_assigned_timezone():
    rows = features([outcome(seconds=i * 10, row=i + 1, naive=True) for i in range(3)])[-1].rows
    row, = rows
    assert row["timezone_known"] is False
    assert row["timezone_raw"] is None and row["utc_offset_seconds"] is None
    assert row["timestamp"] == "2026-01-01T00:00:20"
    assert "UNKNOWN_TIMEZONE" in row["quality_flags"]


@pytest.mark.parametrize("change", ["file", "channel", "sensor", "timezone", "grid"])
def test_identity_or_selected_grid_change_starts_new_block(change):
    second_kwargs = {"seconds": 10, "row": 2}
    second_kwargs.update({
        "file": {"source": "b.csv"}, "channel": {"channel": "2"},
        "sensor": {"sensor": "b"}, "timezone": {"naive": True},
        "grid": {"x": [0, 1.1, 2, 3]},
    }[change])
    first, second = features([outcome(), outcome(**second_kwargs)])
    assert second.block_id != first.block_id
    assert second.rows[0]["history_count"] == 1 and not second.rows[0]["history_ready"]


def test_stream_switch_does_not_resume_old_stream_history():
    results = features([outcome(), outcome(seconds=10, row=2, channel="2"),
                        outcome(seconds=20, row=3)])
    assert [result.block_id for result in results] == [0, 1, 2]
    assert [result.rows[0]["history_count"] for result in results] == [1, 1, 1]


def test_full_tail_changes_do_not_reset_compatible_selected_grid():
    results = features([outcome([1, 2, 3, 4, 99]),
                        outcome([1, 2, 3, 4, 99, 100], seconds=10, row=2),
                        outcome([1, 2, 3, 4], seconds=20, row=3)])
    assert [result.block_id for result in results] == [0, 0, 0]
    assert results[-1].rows[0]["history_ready"]


@pytest.mark.parametrize("gap,expected_block", [(121, 0), (122, 1)])
def test_gap_boundary_is_inclusive(gap, expected_block):
    results = features([outcome(), outcome(seconds=gap, row=2)], history_size=2)
    assert results[-1].block_id == expected_block
    assert results[-1].rows[0]["history_ready"] == (gap == 121)


@pytest.mark.parametrize("bad", ["range", "x", "json"])
def test_unusable_record_reported_and_history_restarts(bad):
    invalid = {"range": outcome([1, 2], seconds=10, row=2),
               "x": outcome(x=[0, 1, 1, 3], seconds=10, row=2),
               "json": parse_profile({"dts_id": "a", "channel_id": "1",
                                      "timestamp": "2026-01-01T00:00:10+00:00",
                                      "points": "invalid"}, "sample.csv", 2, 3)}[bad]
    results = features([outcome(), invalid, outcome(seconds=20, row=3),
                        outcome(seconds=30, row=4), outcome(seconds=40, row=5)])
    assert len(results) == 5 and results[1].status == ("rejected" if bad == "json" else "excluded")
    assert results[1].rows == () and results[1].issues
    assert all(issue.source_file.name == "sample.csv" and issue.row_number == 2
               for issue in results[1].issues)
    assert results[2].rows[0]["history_count"] == 1
    assert not results[3].rows[0]["history_ready"] and results[4].rows[0]["history_ready"]


def test_file_error_is_reported_separately_and_breaks_existing_history():
    first = outcome()
    issue = Issue(first.source_file, 0, 0, "error", "FILE_READ_ERROR", "File unavailable")
    file_event = ProfileResult(first.source_file, 0, 0, None, (issue,))
    results = features([first, file_event, outcome(seconds=20, row=3)])
    assert results[1].status == "file_error" and results[1].rows == ()
    assert results[1].issues == (issue,)
    assert results[2].rows[0]["history_count"] == 1 and results[2].block_id == 1


@pytest.mark.parametrize("mutation", ["nonfinite_x", "nonfinite_y", "invalid_mask"])
def test_aggregate_profile_revalidates_direct_api_input(mutation):
    profile = outcome().profile
    if mutation == "nonfinite_x":
        profile = replace(profile, x=np.array([0, 1, float("inf"), 3]))
    elif mutation == "nonfinite_y":
        profile = replace(profile, y=np.array([0, 1, float("nan"), 3]))
    else:
        profile = replace(profile, sentinel_mask=np.zeros(4, dtype=float))
    with pytest.raises(ValueError):
        aggregate_profile(profile, config())


@pytest.mark.parametrize("seconds,code", [(10, "DUPLICATE_KEY"), (5, "NONINCREASING_TIMESTAMP")])
def test_duplicate_or_reversed_time_is_excluded_and_breaks_history(seconds, code):
    results = features([outcome(), outcome(seconds=10, row=2),
                        outcome(seconds=seconds, row=3), outcome(seconds=20, row=4)])
    assert results[2].status == "excluded" and results[2].rows == ()
    assert code in {issue.code for issue in results[2].issues}
    assert results[3].rows[0]["history_count"] == 1


def test_known_offsets_compare_absolute_time_not_local_wall_clock():
    first = outcome()
    second = outcome(seconds=10, row=2)
    p = second.profile
    local = datetime.fromisoformat("2026-01-01T03:00:10+03:00")
    second = replace(second, profile=replace(p, timestamp=local,
        timestamp_raw=local.isoformat(), timezone_raw="+03:00", utc_offset_seconds=10800))
    results = features([first, second], history_size=2)
    assert results[-1].rows[0]["history_ready"]
    assert results[-1].rows[0]["delta_seconds"] == 10


def test_future_perturbation_does_not_change_past_rows_or_input_arrays():
    past = [outcome([i] * 4, seconds=i * 10, row=i + 1) for i in range(3)]
    before = [(r.profile.x.copy(), r.profile.y.copy(), r.profile.sentinel_mask.copy()) for r in past]
    left = features([*past, outcome([10] * 4, seconds=30, row=4)])
    right = features([*past, outcome([999] * 4, seconds=1000, row=4)])
    assert [r.rows for r in left[:3]] == [r.rows for r in right[:3]]
    for result, original in zip(past, before):
        for current, snapshot in zip((result.profile.x, result.profile.y,
                                      result.profile.sentinel_mask), original):
            np.testing.assert_array_equal(current, snapshot)
            assert not current.flags.writeable
    keys = set(left[0].rows[0])
    for result in left:
        for row in result.rows:
            assert set(row) == keys and isinstance(row["quality_flags"], list)
            assert isinstance(row["history_sources"], list) and isinstance(row["lag_medians"], list)
            json.dumps(row, allow_nan=False)
