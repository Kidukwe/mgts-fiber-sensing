import csv
import json
from datetime import timedelta

import numpy as np
import pytest

from mgts_sensing.io import iter_profiles
from mgts_sensing.validation import REQUIRED_COLUMNS, parse_profile


def row(**changes):
    result = {
        "dts_id": "sensor-a", "channel_id": "1", "timestamp": "2026-02-12 00:00:00+03",
        "points": "[[0, 21.5], [0.25, 22]]", "points_size": "2",
    }
    result.update(changes)
    return result


def parse(**changes):
    return parse_profile(row(**changes), "sample.csv", row_number=1, line_number=2)


def codes(result):
    return {issue.code for issue in result.issues}


def write_csv(tmp_path, name, rows, fields=None):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(row()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_valid_profile_preserves_values_timestamp_and_source():
    result = parse()
    assert result.loaded and not result.issues
    p = result.profile
    assert p.source_file.name == "sample.csv" and p.row_number == 1 and p.line_number == 2
    assert p.dts_id == "sensor-a" and p.channel_id == "1"
    assert p.timestamp_raw == "2026-02-12 00:00:00+03"
    assert p.timezone_raw == "+03" and p.utc_offset_seconds == 10800
    assert p.timestamp.utcoffset() == timedelta(hours=3)
    assert p.timestamp_format_normalized and p.timezone_known
    np.testing.assert_array_equal(p.x, [0, 0.25])
    np.testing.assert_array_equal(p.y, [21.5, 22])
    assert p.x.dtype == np.float64 and p.sentinel_mask.dtype == np.bool_
    assert not p.x.flags.writeable and not p.y.flags.writeable


@pytest.mark.parametrize("points,code", [
    ("{broken", "INVALID_JSON"),
    ("null", "INVALID_POINTS_STRUCTURE"),
    ("{}", "INVALID_POINTS_STRUCTURE"),
    ("[]", "EMPTY_PROFILE"),
    ("[[0,1],[2]]", "INVALID_POINT_PAIR"),
    ('[[0,"1"]]', "NONNUMERIC_POINT"),
    ('[[0,"bad"]]', "NONNUMERIC_POINT"),
    ("[[0,true]]", "NONNUMERIC_POINT"),
    ("[[null,1]]", "NONNUMERIC_POINT"),
    ("[[0,NaN]]", "NONFINITE_POINT"),
    ("[[Infinity,1]]", "NONFINITE_POINT"),
    ("[[0,-Infinity]]", "NONFINITE_POINT"),
    ("[[0,1e400]]", "NONFINITE_POINT"),
])
def test_invalid_points_reject_whole_profile(points, code):
    result = parse(points=points)
    assert result.profile is None and code in codes(result)
    problem = next(issue for issue in result.issues if issue.code == code)
    assert problem.severity == "error" and problem.line_number == 2
    assert problem.source_file.name == "sample.csv" and problem.description


@pytest.mark.parametrize("field", REQUIRED_COLUMNS)
def test_missing_required_fields(field):
    result = parse(**{field: ""})
    assert not result.loaded and "MISSING_REQUIRED_FIELD" in codes(result)


@pytest.mark.parametrize("timestamp", ["not-a-time", "2026-02-30 00:00:00", "2026-02-12",
                                       "2026-02-12T00:00:00+03:99"])
def test_invalid_timestamp(timestamp):
    result = parse(timestamp=timestamp)
    assert not result.loaded and "INVALID_TIMESTAMP" in codes(result)


def test_unknown_timezone_remains_naive():
    result = parse(timestamp="2025-10-16 12:00:06")
    p = result.profile
    assert result.loaded and codes(result) == {"UNKNOWN_TIMEZONE"}
    assert p.timestamp.tzinfo is None and p.timezone_raw is None
    assert p.utc_offset_seconds is None and not p.timezone_known


@pytest.mark.parametrize("timestamp", ["2026-02-12T00:00:00Z", "2026-02-12 00:00:00+00"])
def test_utc_formats(timestamp):
    assert parse(timestamp=timestamp).profile.utc_offset_seconds == 0


def test_timezone_offset_information_is_preserved():
    result = parse(timestamp="2026-02-12T00:00:00+03:30:15.500000")
    assert result.loaded
    assert result.profile.timezone_raw == "+03:30:15.500000"
    assert result.profile.utc_offset_seconds == 12615.5
    assert result.profile.timestamp.utcoffset().total_seconds() == 12615.5


def test_no_points_size_is_valid(tmp_path):
    data = row()
    del data["points_size"]
    path = write_csv(tmp_path, "a.csv", [data], list(REQUIRED_COLUMNS))
    result = next(iter_profiles([path]))
    assert result.loaded and result.profile.points_size is None and not result.issues


@pytest.mark.parametrize("size,code", [("5", "POINTS_SIZE_MISMATCH"),
                                      ("abc", "INVALID_POINTS_SIZE"),
                                      ("", "INVALID_POINTS_SIZE")])
def test_declared_size_quality_warning(size, code):
    result = parse(points_size=size)
    assert result.loaded and codes(result) == {code}
    np.testing.assert_array_equal(result.profile.y, [21.5, 22])


@pytest.mark.parametrize("points", ["[[1,2],[0,3]]", "[[1,2],[1,3]]"])
def test_coordinate_order_warning_keeps_original_order(points):
    result = parse(points=points)
    assert result.loaded and codes(result) == {"NONINCREASING_X"}
    np.testing.assert_array_equal(result.profile.x, [pair[0] for pair in json.loads(points)])


def test_sentinel_keeps_values_and_exposes_mask():
    result = parse(points="[[0,20],[1,-300]]")
    assert result.loaded and codes(result) == {"SUSPECT_SENTINEL"}
    np.testing.assert_array_equal(result.profile.y, [20, -300])
    np.testing.assert_array_equal(result.profile.sentinel_mask, [False, True])
    assert "SUSPECT_SENTINEL" in result.profile.quality_flags


def test_rejected_record_does_not_hide_following_records(tmp_path):
    path = write_csv(tmp_path, "a.csv", [row(), row(points="bad"),
                                         row(timestamp="2026-02-12 00:01:00+03")])
    results = list(iter_profiles([path]))
    assert [r.loaded for r in results] == [True, False, True]
    assert [r.row_number for r in results] == [1, 2, 3]
    assert [r.line_number for r in results] == [2, 3, 4]


def test_cross_file_duplicate_same_utc_instant_and_unknown_time_domain(tmp_path):
    first = write_csv(tmp_path, "a.csv", [row()])
    second = write_csv(tmp_path, "b.csv", [
        row(timestamp="2026-02-11T21:00:00Z"),
        row(timestamp="2026-02-12 00:00:00"),
        row(timestamp="2026-02-12T00:00:00"),
        row(dts_id="sensor-b"),
        row(channel_id="2"),
    ])
    results = list(iter_profiles([first, second]))
    assert all(r.loaded for r in results)
    assert "DUPLICATE_KEY" in codes(results[1])
    assert "a.csv:2" in next(i.description for i in results[1].issues if i.code == "DUPLICATE_KEY")
    assert "DUPLICATE_KEY" not in codes(results[2])
    assert "DUPLICATE_KEY" in codes(results[3])
    assert "DUPLICATE_KEY" not in codes(results[4]) | codes(results[5])


def test_lengths_and_grids_can_change_without_rejection(tmp_path):
    path = write_csv(tmp_path, "a.csv", [row(), row(points="[[0,20]]", points_size="1"),
                                         row(points="[[0.1,20]]", points_size="1")])
    results = list(iter_profiles([path]))
    assert all(r.loaded for r in results)
    assert {"PROFILE_LENGTH_CHANGED", "GRID_CHANGED"} <= codes(results[1])
    assert "GRID_CHANGED" in codes(results[2])
    assert "PROFILE_LENGTH_CHANGED" not in codes(results[2])


def test_header_error_accounts_for_each_record(tmp_path):
    data = row()
    del data["points"]
    path = write_csv(tmp_path, "bad.csv", [data, data], list(data))
    results = list(iter_profiles([path]))
    assert results[0].row_number == 0  # separate header event
    assert [r.row_number for r in results[1:]] == [1, 2]
    assert all(not r.loaded and "MISSING_COLUMN" in codes(r) for r in results)


def test_multiline_record_reports_start_line(tmp_path):
    path = write_csv(tmp_path, "a.csv", [row(points="[[0,1],\n[1,2]]"), row()])
    results = list(iter_profiles([path]))
    assert [r.line_number for r in results] == [2, 4]


def test_csv_syntax_error_is_explicit_and_next_file_is_read(tmp_path):
    broken = tmp_path / "broken.csv"
    broken.write_text('dts_id,channel_id,timestamp,points\na,1,2026-01-01T00:00:00,"unterminated\n')
    valid = write_csv(tmp_path, "valid.csv", [row()])
    results = list(iter_profiles([broken, valid]))
    assert not results[0].loaded and "CSV_READ_ERROR" in codes(results[0])
    assert "remainder unread" in results[0].issues[0].description
    assert results[1].loaded


def test_rejected_points_still_participate_in_duplicate_check(tmp_path):
    path = write_csv(tmp_path, "a.csv", [row(points="bad"), row()])
    results = list(iter_profiles([path]))
    assert not results[0].loaded and results[1].loaded
    assert "DUPLICATE_KEY" in codes(results[1])


def test_every_bad_pair_gets_a_problem():
    result = parse(points='[[0,"bad"],[1,NaN],[2]]')
    assert not result.loaded
    assert {"NONNUMERIC_POINT", "NONFINITE_POINT", "INVALID_POINT_PAIR"} <= codes(result)


def test_excessively_nested_json_is_explicit_error():
    result = parse(points="[" * 2000 + "0" + "]" * 2000)
    # Python 3.12 can decode deeper JSON than 3.10; either path must reject it.
    assert not result.loaded
    assert codes(result) & {"INVALID_JSON", "INVALID_POINT_PAIR"}


def test_csv_column_count_and_blank_record_are_explicit(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text('dts_id,channel_id,timestamp,points\n\na,1\n', encoding="utf-8")
    results = list(iter_profiles([path]))
    assert len(results) == 2
    assert all("CSV_COLUMN_COUNT" in codes(r) for r in results)


def test_duplicate_header_and_missing_file_are_reported(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text('dts_id,channel_id,timestamp,points,points\na,1,x,[],[]\n')
    results = list(iter_profiles([path, tmp_path / "missing.csv"]))
    assert "DUPLICATE_COLUMN" in codes(results[0])
    assert not results[1].loaded and results[1].row_number == 1
    assert "FILE_READ_ERROR" in codes(results[2])
