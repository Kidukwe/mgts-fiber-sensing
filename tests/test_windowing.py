"""Independent contract tests for causal windows; no CSV fixtures needed."""

import csv
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from itertools import groupby
from pathlib import Path

import numpy as np
import pytest

from mgts_sensing.io import iter_profiles
from mgts_sensing.validation import Profile
from mgts_sensing.windowing import iter_windows


def make_profile(seconds=0, *, file="a.csv", device="a", channel="1", aware=True):
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc if aware else None)
    timestamp += timedelta(seconds=seconds)
    x = np.array([0.0, 1.0])
    y = np.array([seconds, seconds + 1.0])
    mask = np.array([False, False])
    for array in (x, y, mask):
        array.setflags(write=False)
    return Profile(
        source_file=Path(file), row_number=2, line_number=2,
        dts_id=device, channel_id=channel, timestamp=timestamp,
        timestamp_raw=timestamp.isoformat(), timezone_raw="+00:00" if aware else None,
        utc_offset_seconds=0.0 if aware else None, timestamp_format_normalized=False,
        x=x, y=y, sentinel_mask=mask, points_size=2,
    )


def windows(profiles, *, history_size=2, max_gap_seconds=10):
    return list(iter_windows(
        profiles, history_size=history_size, max_gap_seconds=max_gap_seconds,
    ))


def test_warmup_sliding_order_identity_and_determinism():
    profiles = [make_profile(t) for t in (0, 2, 5, 8)]
    result = windows(iter(profiles), history_size=3)
    assert len(result) == 2
    assert [w.session_index for w in result] == [0, 0]
    for index, window in enumerate(result):
        assert isinstance(window.profiles, tuple)
        assert len(window.profiles) == 3
        assert all(a is b for a, b in zip(window.profiles, profiles[index:index + 3]))
    repeated = windows(profiles, history_size=3)
    assert [w.session_index for w in repeated] == [0, 0]
    assert all(a is b for a, b in zip(repeated[1].profiles, result[1].profiles))
    assert windows([]) == []
    assert windows(profiles[:1]) == []


def test_generator_does_not_consume_future_before_yield():
    seen = []

    def source():
        for t in (0, 1, 2, 3):
            seen.append(t)
            yield make_profile(t)

    generator = iter_windows(source(), history_size=2, max_gap_seconds=10)
    first = next(generator)
    assert seen == [0, 1]
    assert len(first.profiles) == 2


def test_gap_equal_threshold_allowed_and_larger_gap_resets():
    result = windows([make_profile(t) for t in (0, 10, 21, 22)])
    assert [w.session_index for w in result] == [0, 1]
    assert [w.profiles[-1].timestamp.second for w in result] == [10, 22]


@pytest.mark.parametrize("change", [
    {"file": "b.csv"}, {"device": "b"}, {"channel": "2"}, {"aware": False},
])
def test_block_boundary_resets_history_even_if_new_time_rewinds(change):
    profiles = [make_profile(10), make_profile(11),
                make_profile(0, **change), make_profile(1, **change)]
    result = windows(profiles)
    assert [w.session_index for w in result] == [0, 1]
    assert all(w.profiles[0].source_file == w.profiles[1].source_file for w in result)


def test_exact_grid_change_resets_history():
    changed_x = np.array([0.0, 1.0 + 1e-12])
    profiles = [make_profile(0), make_profile(1),
                replace(make_profile(2), x=changed_x), replace(make_profile(3), x=changed_x)]
    result = windows(profiles)
    assert [w.session_index for w in result] == [0, 1]
    assert result[1].profiles[0] is profiles[2]


def test_simultaneous_boundaries_increment_session_once():
    profiles = [make_profile(0), make_profile(1)]
    for t in (50, 51):
        profiles.append(replace(make_profile(t, file="b.csv", device="b"),
                                x=np.array([2.0, 3.0])))
    assert [w.session_index for w in windows(profiles)] == [0, 1]


@pytest.mark.parametrize("seconds", [0, -1])
@pytest.mark.parametrize("change_grid", [False, True])
def test_duplicate_or_reverse_time_rejected_even_with_grid_change(seconds, change_grid):
    second = make_profile(seconds)
    if change_grid:
        second = replace(second, x=np.array([2.0, 3.0]))
    with pytest.raises(ValueError):
        windows([make_profile(0), second])


def test_different_known_offsets_use_real_elapsed_time():
    previous = make_profile(0)
    timestamp = (previous.timestamp + timedelta(seconds=5)).astimezone(
        timezone(timedelta(hours=3)),
    )
    current = replace(make_profile(5), timestamp=timestamp, utc_offset_seconds=10800)
    assert len(windows([previous, current])) == 1
    same_instant = replace(current, timestamp=previous.timestamp.astimezone(timestamp.tzinfo))
    with pytest.raises(ValueError):
        windows([previous, same_instant])


@pytest.mark.parametrize("history_size", [True, False, 0, 1, -2, 2.5, "2", None])
def test_invalid_history_size_rejected_even_on_empty_input(history_size):
    with pytest.raises(ValueError):
        windows([], history_size=history_size)


@pytest.mark.parametrize("max_gap_seconds", [True, False, 0, -1, np.nan, np.inf,
                                            -np.inf, "10", None])
def test_invalid_gap_rejected_even_on_empty_input(max_gap_seconds):
    with pytest.raises(ValueError):
        windows([], max_gap_seconds=max_gap_seconds)


@pytest.mark.parametrize("changes", [
    {"x": np.array([]), "y": np.array([]), "sentinel_mask": np.array([], dtype=bool)},
    {"x": np.array([1.0, 0.0])},
    {"x": np.array([1.0, 1.0])},
    {"x": np.array([0.0, np.nan])},
    {"y": np.array([0.0, np.inf])},
    {"y": np.array(["a", "b"])},
    {"x": np.array([[0.0, 1.0]])},
    {"y": np.array([1.0])},
    {"sentinel_mask": np.array([False])},
])
def test_manually_malformed_profile_is_rejected(changes):
    with pytest.raises(ValueError):
        windows([replace(make_profile(0), **changes), make_profile(1)])


def test_windowing_preserves_readonly_input_arrays_and_values():
    profiles = [make_profile(0), make_profile(1)]
    snapshots = [(p.x.copy(), p.y.copy(), p.sentinel_mask.copy()) for p in profiles]
    windows(profiles)
    for profile, snapshot in zip(profiles, snapshots):
        for array, before in zip((profile.x, profile.y, profile.sentinel_mask), snapshot):
            np.testing.assert_array_equal(array, before)
            assert not array.flags.writeable


def test_documented_ingestion_grouping_does_not_bridge_rejected_record(tmp_path):
    source = tmp_path / "profiles.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "dts_id", "channel_id", "timestamp", "points",
        ])
        writer.writeheader()
        for seconds in range(5):
            writer.writerow({
                "dts_id": "a", "channel_id": "1",
                "timestamp": make_profile(seconds).timestamp.isoformat(),
                "points": "broken JSON" if seconds == 2 else "[[0, 1], [1, 2]]",
            })

    def collect(history_size):
        output = []
        rejected = []
        for run_index, (loaded, run) in enumerate(groupby(
            iter_profiles([source]), key=lambda result: result.loaded,
        )):
            if not loaded:
                rejected.extend(run)
                continue
            profiles = (result.profile for result in run)
            output.extend((run_index, window) for window in iter_windows(
                profiles, history_size=history_size, max_gap_seconds=10,
            ))
        assert [result.row_number for result in rejected] == [3]
        assert any(issue.code == "INVALID_JSON" for issue in rejected[0].issues)
        return output

    assert collect(history_size=3) == []
    result = collect(history_size=2)
    assert len(result) == 2
    assert result[0][0] != result[1][0]
    assert [[p.row_number for p in window.profiles] for _, window in result] == [[1, 2], [4, 5]]
    # Each call has its own session numbering; run identity must be retained separately.
    assert [window.session_index for _, window in result] == [0, 0]
