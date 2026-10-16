"""Independent regression checks for PR7 semantic/topology fixes, not event labels."""

import csv
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from mgts_sensing.io import iter_profiles
from mgts_sensing.solution import SEMANTICS, build_alerts, extract, make_samples, run_file


def fixture(path, *, x=(0, 1, 2, 3), count=60, offset=None):
    start = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    if offset is None:
        start = start.replace(tzinfo=None)
    else:
        start = start.astimezone(timezone(timedelta(hours=offset)))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dts_id", "channel_id", "timestamp", "points"])
        for index in range(count):
            writer.writerow(["synthetic", "1", (start + timedelta(minutes=index)).isoformat(),
                             json.dumps([[value, 20] for value in x])])
    return path


def injected_alerts(records, *, floors_c=None):
    _, _, values, _, _, meta, split, cadence = make_samples(records)
    values[split == 2, 0] += 10
    return build_alerts(meta, values, values.copy(), np.ones(3), split, cadence, "synthetic.csv",
                        floors_c=floors_c)


def test_only_5_gets_automatic_offset_and_original_ingestion_stays_naive(tmp_path):
    path = fixture(tmp_path / "5.csv", count=2)
    original = path.read_bytes()
    raw = next(iter_profiles([path])).profile
    assert raw.timestamp.tzinfo is None and raw.timestamp_raw == "2026-10-01T12:00:00"
    records, _ = extract(path, 2)
    first = records[0]
    assert first["timestamp"] == "2026-10-01T12:00:00+03:00"
    assert datetime.fromisoformat(first["timestamp"]).astimezone(timezone.utc).isoformat() \
        == "2026-10-01T09:00:00+00:00"
    assert first["seconds"] == datetime(2026, 10, 1, 9, tzinfo=timezone.utc).timestamp()
    assert path.read_bytes() == original


@pytest.mark.parametrize("name", ["1.csv", "4.csv", "50.csv", "unknown.csv"])
def test_other_naive_files_are_explicitly_excluded_not_converted_by_host_timezone(tmp_path, name):
    path = fixture(tmp_path / name, count=3)
    original = path.read_bytes()
    issue_path = tmp_path / "issues.jsonl"
    records, counts = extract(path, 2, issue_path=issue_path)
    assert len(records) == 3 and all(record["invalid"] for record in records)
    assert all("seconds" not in record and "zones" not in record for record in records)
    assert counts["UNKNOWN_TIMEZONE_UNSUPPORTED"] == 3
    issues = [json.loads(line) for line in issue_path.read_text().splitlines()]
    unresolved = [issue for issue in issues if issue["code"] == "UNKNOWN_TIMEZONE_UNSUPPORTED"]
    assert [issue["row_number"] for issue in unresolved] == [1, 2, 3]
    assert [issue["line_number"] for issue in unresolved] == [2, 3, 4]
    assert all(issue["source_file"] == str(path) and issue["description"] for issue in unresolved)
    assert path.read_bytes() == original


@pytest.mark.parametrize("offset", [-4, 0, 3, 5])
@pytest.mark.parametrize("name", ["5.csv", "2.csv"])
def test_aware_time_preserves_original_instant_and_offset(tmp_path, name, offset):
    path = fixture(tmp_path / name, offset=offset, count=2)
    raw = next(iter_profiles([path])).profile
    original = path.read_bytes()
    records, _ = extract(path, 2)
    assert records[0]["timestamp"] == raw.timestamp.isoformat()
    assert records[0]["seconds"] == raw.timestamp.timestamp()
    assert records[0]["seconds"] == datetime(2026, 10, 1, 12, tzinfo=timezone.utc).timestamp()
    assert path.read_bytes() == original


def test_coordinate_jump_between_zones_prevents_one_spatial_event_component(tmp_path):
    path = fixture(tmp_path / "5.csv", x=(0, 1, 1000, 1001))
    original = path.read_bytes()
    records, counts = extract(path, 2)
    assert not counts.get("COORDINATE_GAP_NONUNIFORM")
    assert all(set(record["zones"]) == {0, 1} for record in records)
    assert [(zone["index_start"], zone["index_stop"], zone["raw_x_start"], zone["raw_x_end"])
            for zone in records[0]["zones"].values()] == [(0, 2, 0, 1), (2, 4, 1000, 1001)]
    alerts = [event for event in injected_alerts(records) if event["kind"] == "heating"]
    assert len(alerts) == 2
    assert {(event["index_start"], event["index_stop"], event["raw_x_start"], event["raw_x_end"])
            for event in alerts} == {(0, 2, 0, 1), (2, 4, 1000, 1001)}
    for event in alerts:
        assert event["distance_start_m"] is event["distance_stop_m"] is event["peak_extent_m"] is None
        assert event["peak_extent_points"] == 2
        assert math.isfinite(event["priority_score"]) and event["priority_score"] > 0
        json.dumps(event, allow_nan=False)
    assert path.read_bytes() == original


def test_truly_adjacent_uniform_zones_still_form_one_component(tmp_path):
    path = fixture(tmp_path / "5.csv")
    records, _ = extract(path, 2)
    alerts = [event for event in injected_alerts(records) if event["kind"] == "heating"]
    assert len(alerts) == 1
    assert (alerts[0]["index_start"], alerts[0]["index_stop"], alerts[0]["peak_extent_points"]) == (0, 4, 4)
    assert alerts[0]["peak_extent_m"] is None


def test_nonuniform_grid_inside_one_zone_is_explicitly_skipped(tmp_path):
    path = fixture(tmp_path / "5.csv", x=(0, 1, 1000, 1001), count=3)
    original = path.read_bytes()
    issue_path = tmp_path / "issues.jsonl"
    records, counts = extract(path, 4, issue_path=issue_path)
    assert len(records) == 3 and all(record["zones"] == {} for record in records)
    assert counts["COORDINATE_GAP_NONUNIFORM"] == 3
    issues = [json.loads(line) for line in issue_path.read_text().splitlines()]
    gaps = [issue for issue in issues if issue["code"] == "COORDINATE_GAP_NONUNIFORM"]
    assert [issue["row_number"] for issue in gaps] == [1, 2, 3]
    assert all(issue["source_file"] == str(path) and issue["description"] for issue in gaps)
    assert path.read_bytes() == original


def test_event_floors_are_explicit_research_parameters_not_instrument_accuracy(tmp_path):
    path = fixture(tmp_path / "5.csv")
    records, _ = extract(path, 2)
    ordinary = injected_alerts(records, floors_c=(0.2, 0.4, 0.4))
    suppressed = injected_alerts(records, floors_c=(100, 100, 100))
    assert ordinary and suppressed == []
    assert SEMANTICS["accuracy_c"] == 0.1
    assert SEMANTICS["sample_spacing_m"] is None and SEMANTICS["spatial_resolution_m"] is None
    assert SEMANTICS["coordinate_unit"] is None
    assert "0,25 см" in json.dumps(SEMANTICS, ensure_ascii=False)


def test_small_synthetic_artifact_persists_unknown_metres_and_requested_research_floors(tmp_path):
    pytest.importorskip("matplotlib")
    path = fixture(tmp_path / "5.csv")
    original = path.read_bytes()
    output = tmp_path / "output"
    result, _ = run_file(path, output, width=2, floors_c=(0.3, 0.6, 0.9))
    model = json.loads((output / "model.json").read_text())
    assert model["semantics"]["sample_spacing_m"] is None
    assert model["semantics"]["spatial_resolution_m"] is None
    assert model["semantics"]["coordinate_unit"] is None
    assert model["research_threshold_floors_c"] == [0.3, 0.6, 0.9]
    assert np.all(np.array(model["thresholds_c"]) >= [0.3, 0.6, 0.9])
    assert result["event_precision"] is None and result["event_recall"] is None
    assert result["raw_unchanged"] is True and path.read_bytes() == original
