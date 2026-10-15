"""Behavior tests on synthetic profiles; these are not real event labels."""

import csv
import json
from dataclasses import replace
from pathlib import Path

import pytest

from mgts_sensing.event_cli import main
from mgts_sensing.events import DetectorConfig, EventDetector
from mgts_sensing.io import iter_profiles
from mgts_sensing.modeling import RobustBaseline
from mgts_sensing.zones import ZoneConfig, iter_feature_results


def make_profiles(tmp_path, values, *, name="profiles.csv", timestamps=None):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dts_id", "channel_id", "timestamp", "points"])
        for i, y in enumerate(values):
            time = timestamps[i] if timestamps else f"2026-01-01T00:{i:02}:00Z"
            writer.writerow(["synthetic", "1", time, json.dumps(list(enumerate(y)))])
    return path


def features(path):
    return list(iter_feature_results(iter_profiles([path]), ZoneConfig(
        index_stop=6, zone_width=3, history_size=3, max_gap_seconds=61,
    )))


def run(results, **overrides):
    config = DetectorConfig(min_history=3, baseline_window=5, threshold=3,
                            **overrides)
    detector = EventDetector(config)
    scores, events = [], []
    for result in results:
        batch = detector.process(result)
        scores.extend(batch.scores)
        events.extend(batch.events)
    events.extend(detector.finish())
    return scores, events


def test_baseline_is_causal_and_robust():
    baseline = RobustBaseline(5, 3)
    assert baseline.predict() is None
    for value in [10, 10, 10]:
        baseline.observe(value)
    assert baseline.predict() == (10, 0)
    baseline.observe(1000)
    assert baseline.predict() == (10, 0)
    with pytest.raises(ValueError):
        baseline.observe(float("nan"))


def test_constant_and_isolated_spike_are_not_events(tmp_path):
    path = make_profiles(tmp_path, [[10] * 6] * 4 + [[50] * 6] + [[10] * 6] * 3)
    scores, events = run(features(path))
    assert events == []
    assert any(row.get("candidate") for row in scores)


@pytest.mark.parametrize("value,kind", [(20, "heating"), (0, "cooling")])
def test_sustained_step_freezes_baseline_and_recovers(tmp_path, value, kind):
    path = make_profiles(tmp_path, [[10] * 6] * 3 + [[value] * 6] * 12 + [[10] * 6] * 2)
    scores, events = run(features(path))
    assert len(events) == 2  # separate event for each zone
    assert {event["kind"] for event in events} == {kind}
    for event in events:
        assert event["start"] == "2026-01-01T00:03:00+00:00"
        assert event["detected_at"] == "2026-01-01T00:04:00+00:00"
        assert event["end"] == "2026-01-01T00:14:00+00:00"
        assert event["close_reason"] == "recovered"
        assert event["observations"] == 12
        assert event["baseline_at_peak"] == 10
    active = [row for row in scores if row.get("active") and row["kind"] == kind]
    assert all(row["baseline"] == 10 for row in active)


def test_peak_and_dip_preserved_when_median_does_not_move(tmp_path):
    path = make_profiles(tmp_path, [[10] * 6] * 3 + [[10, 10, 30, -10, 10, 10]] * 2)
    _, events = run(features(path))
    assert {(e["kind"], e["zone_id"]) for e in events} == {("local_peak", 0), ("local_dip", 1)}
    assert all(e["close_reason"] == "end_of_input" for e in events)


def test_permanent_spatial_shape_is_baseline_not_event(tmp_path):
    path = make_profiles(tmp_path, [[10, 10, 30, -10, 10, 10]] * 8)
    _, events = run(features(path))
    assert not events


def test_invalid_zone_closes_event_and_rewarms_only_that_zone(tmp_path):
    path = make_profiles(tmp_path, [[10] * 6] * 3 + [[20] * 6] * 2
                         + [[-300] * 3 + [20] * 3] + [[20] * 6] * 2)
    scores, events = run(features(path))
    assert {(e["zone_id"], e["close_reason"]) for e in events} == {
        (0, "invalid_observation"), (1, "end_of_input"),
    }
    assert any(row["status"] == "invalid_observation" for row in scores)
    after = [r for r in scores if r["timestamp"] == "2026-01-01T00:06:00+00:00"
             and r["zone_id"] == 0]
    assert all(r["status"] == "baseline_warmup" for r in after)


def test_gap_grid_change_and_rejection_do_not_bridge_events(tmp_path):
    path = make_profiles(tmp_path, [[10] * 6] * 3 + [[20] * 6] * 2 + [[20] * 6],
                         timestamps=[f"2026-01-01T00:{i:02}:00Z" for i in range(5)]
                         + ["2026-01-01T01:00:00Z"])
    results = features(path)
    _, events = run(results)
    assert len(events) == 2
    assert all(e["close_reason"] == "block_boundary" for e in events)
    changed = replace(results[-1], block_id=0,
                      rows=tuple({**r, "block_id": 0, "selected_grid_sha256": "changed"}
                                 for r in results[-1].rows))
    _, events = run(results[:-1] + [changed])
    assert all(e["close_reason"] == "block_boundary" for e in events)
    rejected = replace(results[-1], status="rejected", rows=(), block_id=None)
    _, events = run(results[:-1] + [rejected])
    assert all(e["close_reason"] == "upstream_exclusion" for e in events)


def test_prefix_does_not_depend_on_future(tmp_path):
    path = make_profiles(tmp_path, [[10] * 6] * 3 + [[20] * 6] * 3 + [[-100] * 6] * 5)
    results = features(path)
    prefix_scores, _ = run(results[:6])
    all_scores, _ = run(results)
    assert prefix_scores == all_scores[:len(prefix_scores)]


@pytest.mark.parametrize("kwargs", [
    {"threshold": 0}, {"threshold": float("nan")}, {"mad_multiplier": -1},
    {"release_ratio": 1}, {"min_consecutive": True}, {"min_history": 1},
    {"baseline_window": 2}, {"min_valid_fraction": 0},
])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        DetectorConfig(**kwargs)


def test_cli_integration_reproducible_and_immutable(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    path = make_profiles(raw_dir, [[10] * 6] * 3 + [[20] * 6] * 2 + [[10] * 6] * 2)
    original = path.read_bytes()
    output = tmp_path / "output"
    argv = ["--input", str(path), "--output-dir", str(output), "--index-stop", "6",
            "--zone-width", "3", "--min-history", "3"]
    assert main(argv) == 0
    expected = {p.name: p.read_bytes() for p in output.iterdir()}
    assert main(argv) == 0
    assert {p.name: p.read_bytes() for p in output.iterdir()} == expected
    assert path.read_bytes() == original
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert len(events) == 2
    report = json.loads((output / "report.json").read_text())
    assert report["raw_unchanged"]
    assert report["baseline_metrics"]["profiles"] == 4
    assert report["baseline_metrics"]["robust"]["mae"] == pytest.approx(5)
    assert report["baseline_metrics"]["persistence"]["mae"] == pytest.approx(5)
    assert report["baseline_metrics"]["event_precision"] is None
    with pytest.raises(SystemExit) as exc:
        main(["--input", str(path), "--output-dir", str(raw_dir / "generated")])
    assert exc.value.code == 2


def test_cli_rejected_records_reported(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    path = make_profiles(raw, [[10] * 6] * 3)
    with path.open("a") as handle:
        handle.write('synthetic,1,invalid,"[]"\n')
    output = tmp_path / "out"
    assert main(["--input", str(path), "--output-dir", str(output), "--index-stop", "6",
                 "--zone-width", "3"]) == 1
    report = json.loads((output / "report.json").read_text())
    assert report["counts"]["rejected"] == 1
    assert (output / "issues.jsonl").read_text()


def test_cli_refuses_foreign_outputs_and_symlinks(tmp_path):
    path = Path("data/sample/jotb7_profiles.csv")
    output = tmp_path / "out"
    output.mkdir()
    (output / "events.jsonl").write_text("existing data")
    args = ["--input", str(path), "--output-dir", str(output)]
    with pytest.raises(SystemExit):
        main(args)
    assert (output / "events.jsonl").read_text() == "existing data"
    (output / "events.jsonl").unlink()
    (output / "events.jsonl").symlink_to(tmp_path / "target")
    with pytest.raises(SystemExit):
        main(args)


def test_hysteresis_inclusive_threshold_and_short_recovery(tmp_path):
    # Residual 3 confirms; residual 1.5 holds exactly at release boundary.
    levels = [10, 10, 10, 13, 13, 11.5, 10, 11.5, 10, 10]
    path = make_profiles(tmp_path, [[value] * 6 for value in levels])
    _, events = run(features(path))
    assert len(events) == 2
    assert all(e["observations"] == 4 for e in events)
    assert all(e["end"] == "2026-01-01T00:07:00+00:00" for e in events)
    assert all(e["duration_seconds"] == 240 for e in events)
    assert all(e["close_reason"] == "recovered" for e in events)


def test_sample_has_four_recovered_kinds():
    results = list(iter_feature_results(
        iter_profiles(["data/sample/temperature_events.csv"]),
        ZoneConfig(index_stop=6, zone_width=3),
    ))
    detector = EventDetector()
    events = []
    for result in results:
        events.extend(detector.process(result).events)
    events.extend(detector.finish())
    assert {event["kind"] for event in events} == {
        "heating", "cooling", "local_peak", "local_dip",
    }
    assert len(events) == 4
    assert all(event["close_reason"] == "recovered" for event in events)


def test_metrics_weight_profiles_equally():
    from mgts_sensing.evaluation import BaselineMetrics

    metrics = BaselineMetrics()
    assert metrics.report()["robust"]["mae"] is None

    def scored(value):
        return {"status": "scored", "kind": "heating", "value": value,
                "baseline": 0, "persistence": 0}

    metrics.update((scored(2), scored(4)))
    metrics.update((scored(9),))
    report = metrics.report()
    assert report["profiles"] == 2
    assert report["zone_observations"] == 3
    assert report["robust"]["mae"] == 6
    assert report["robust"]["rmse"] == pytest.approx((45.5) ** 0.5)
