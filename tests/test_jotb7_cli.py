"""End-to-end export contract tests using tiny CSVs, not hackathon raw data."""

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FILES = {"features.jsonl", "schema.json", "config.json", "report.json", "issues.jsonl"}


def write_input(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dts_id", "channel_id", "timestamp", "points"])
        for index, values in enumerate(samples):
            points = values if isinstance(values, str) else json.dumps(list(enumerate(values)))
            writer.writerow(["fixture", "1", f"2026-01-01T00:00:{index * 10:02d}", points])
    return path


def run_cli(inputs, output, extra=()):
    command = [sys.executable, str(ROOT / "scripts/build_features.py")]
    for path in inputs:
        command.extend(["--input", str(path)])
    command.extend(["--output-dir", str(output), "--index-start", "0", "--index-stop", "5",
                    "--zone-width", "3", "--history-size", "3", "--max-gap-seconds", "121",
                    "--min-valid-fraction", str(2 / 3), *extra])
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(command, capture_output=True, text=True, env=environment,
                          timeout=30, check=False)


def strict_json(text):
    def invalid_constant(value):
        pytest.fail(f"Non-JSON floating value {value}")
    return json.loads(text, parse_constant=invalid_constant)


def jsonl(path):
    return [strict_json(line) for line in path.read_text().splitlines()]


def test_export_accounts_bad_record_and_is_deterministic_without_raw_writes(tmp_path):
    raw = write_input(tmp_path / "raw/sample.csv", [
        [0, 0, 6, 10, 20], [-300, 2, 4, 10, 20], "broken-json",
        [6, 6, 6, 12, 18], [8, 8, 8, 14, 16], [10, 10, 10, 16, 14],
    ])
    original = raw.read_bytes()
    output = tmp_path / "derived"
    completed = run_cli([raw], output)
    assert completed.returncode in (0, 1), completed.stdout + completed.stderr
    assert OUTPUT_FILES <= {path.name for path in output.iterdir()}
    report = strict_json((output / "report.json").read_text())
    assert report["ingestion_records"] == 6
    assert report["loaded"] == 5 and report["rejected"] == 1
    assert report["feature_excluded"] == 0
    assert report["zone_count"] == report["feature_rows"] == 10
    assert report["warmup_rows"] == 8 and report["full_history_rows"] == 2
    assert report["full_windows"] == 1 and report["temporal_blocks"] == 2
    assert report["accounting_complete"] is True and report["raw_unchanged"] is True
    assert report["issue_counts"]
    rows = jsonl(output / "features.jsonl")
    assert len(rows) == 10
    first = rows[0]
    assert first["median"] == 0 and first["maximum_minus_median"] == 6
    assert first["std"] == pytest.approx(8 ** 0.5)
    assert first["timezone_raw"] is None and first["timezone_known"] is False
    assert first["history_ready"] is False and first["rolling_mean"] is None
    assert first["lag_medians"] == [0, None, None]
    assert rows[1]["valid_fraction"] == pytest.approx(2 / 3)
    assert "INCOMPLETE_ZONE" in rows[1]["quality_flags"]
    assert all(row["row_number"] != 3 for row in rows)
    last_zone_zero = next(row for row in rows if row["row_number"] == 6 and row["zone_id"] == 0)
    assert last_zone_zero["lag_medians"] == [10, 8, 6]
    assert last_zone_zero["rate"] == 0.2 and last_zone_zero["rolling_mean"] == 8
    assert last_zone_zero["history_ready"] is True
    assert [r["row_number"] for r in last_zone_zero["history_sources"]] == [6, 5, 4]
    keys = set(first)
    assert all(set(row) == keys for row in rows)
    assert all(isinstance(row["lag_medians"], list) and isinstance(row["quality_flags"], list)
               for row in rows)
    issues = jsonl(output / "issues.jsonl")
    bad = [issue for issue in issues if issue["code"] == "INVALID_JSON"]
    assert len(bad) == 1 and bad[0]["row_number"] == 3 and bad[0]["line_number"] == 4
    assert bad[0]["severity"] == "error" and bad[0]["source_file"] == str(raw)
    assert raw.read_bytes() == original
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == hashlib.sha256(original).hexdigest()
    schema = strict_json((output / "schema.json").read_text())
    config = strict_json((output / "config.json").read_text())
    assert "proposal" in json.dumps(schema) and "jotb7-zones-v1-proposal" in json.dumps(schema)
    assert set(schema["columns"]) == keys
    assert set(schema["row_schema"]["required"]) == keys
    assert schema["row_schema"]["additionalProperties"] is False
    assert schema["coordinate_unit"] is None and schema["signal_unit"] is None
    parameters = config["parameters"]
    assert parameters["index_stop"] == 5 and parameters["history_size"] == 3
    assert parameters["min_valid_fraction"] == pytest.approx(2 / 3)
    expected_hash = hashlib.sha256(original).hexdigest()
    assert config["inputs"] == [{"source_file": str(raw), "sha256_before": expected_hash}]
    assert report["input_files"][0]["sha256_before"] == expected_hash
    assert report["input_files"][0]["sha256_after"] == expected_hash
    snapshot = {name: (output / name).read_bytes() for name in OUTPUT_FILES}
    rerun = run_cli([raw], output)
    assert rerun.returncode == completed.returncode
    assert {name: (output / name).read_bytes() for name in OUTPUT_FILES} == snapshot


def test_repeated_inputs_preserve_file_order_and_never_share_history(tmp_path):
    first = write_input(tmp_path / "raw/z.csv", [[1] * 5, [2] * 5])
    second = write_input(tmp_path / "raw/a.csv", [[10] * 5, [20] * 5])
    output = tmp_path / "derived"
    completed = run_cli([first, second], output)
    assert completed.returncode in (0, 1), completed.stdout + completed.stderr
    rows = jsonl(output / "features.jsonl")
    # The second file duplicates wall-clock keys: ingestion must flag those,
    # then feature exclusions must be explicit rather than joining sources.
    report = strict_json((output / "report.json").read_text())
    assert report["ingestion_records"] == report["loaded"] == 4
    assert report["rejected"] == 0
    assert rows[0]["source_file"] == str(first)
    assert all(not row["history_ready"] for row in rows)
    assert all(len({source["source_file"] for source in row["history_sources"]}) == 1
               for row in rows)
    assert report["loaded"] == report["feature_excluded"] + len(rows) // 2
    assert report["feature_excluded"] == 2
    assert report["issue_counts"]["DUPLICATE_KEY"] == 2


def test_irrecoverable_csv_explicitly_marks_incomplete_accounting(tmp_path):
    raw = write_input(tmp_path / "raw/broken.csv", [[1] * 5])
    with raw.open("a", encoding="utf-8") as handle:
        handle.write('fixture,1,2026-01-01T00:00:10,"unfinished\n')
    output = tmp_path / "derived"
    completed = run_cli([raw], output)
    assert completed.returncode != 0
    report = strict_json((output / "report.json").read_text())
    assert report["accounting_complete"] is False
    assert report["ingestion_records"] == report["loaded"] + report["rejected"]
    assert report["raw_unchanged"] is True
    assert any(issue["code"] == "CSV_READ_ERROR" for issue in jsonl(output / "issues.jsonl"))


@pytest.mark.parametrize("relative", [".", "nested"])
def test_output_inside_input_directory_is_rejected_before_export(tmp_path, relative):
    raw = write_input(tmp_path / "raw/sample.csv", [[1] * 5])
    original = raw.read_bytes()
    output = raw.parent / relative
    completed = run_cli([raw], output)
    assert completed.returncode != 0
    assert raw.read_bytes() == original
    assert not (output / "features.jsonl").exists()


def test_cli_requires_explicit_input(tmp_path):
    completed = subprocess.run([sys.executable, str(ROOT / "scripts/build_features.py"),
                                "--output-dir", str(tmp_path / "derived")],
                               capture_output=True, text=True, timeout=30, check=False,
                               env={**os.environ, "PYTHONPATH": str(ROOT / "src")})
    assert completed.returncode != 0 and "--input" in completed.stderr


def test_cli_loaded_but_feature_invalid_rows_have_reasons_and_break_history(tmp_path):
    raw = write_input(tmp_path / "raw/sample.csv", [[1] * 5, [2] * 2,
                      [3] * 5, [4] * 5, [5] * 5])
    output = tmp_path / "derived"
    completed = run_cli([raw], output)
    assert completed.returncode == 1
    report = strict_json((output / "report.json").read_text())
    assert report["ingestion_records"] == report["loaded"] == 5
    assert report["rejected"] == 0 and report["feature_excluded"] == 1
    assert report["aggregated_profiles"] == 4 and report["temporal_blocks"] == 2
    assert report["full_windows"] == 1 and report["feature_rows"] == 8
    issues = jsonl(output / "issues.jsonl")
    missing_slice = [issue for issue in issues if issue["code"] == "SELECTED_SLICE_UNAVAILABLE"]
    assert len(missing_slice) == 1 and missing_slice[0]["row_number"] == 2
    recovered = next(row for row in jsonl(output / "features.jsonl") if row["row_number"] == 3)
    assert recovered["history_count"] == 1 and not recovered["history_ready"]


@pytest.mark.parametrize("symlink", [False, True])
def test_cli_does_not_replace_foreign_outputs_or_symbolic_links(tmp_path, symlink):
    raw = write_input(tmp_path / "raw/sample.csv", [[1] * 5])
    output = tmp_path / "derived"
    output.mkdir()
    destination = output / "features.jsonl"
    original = b"User-owned, not a JOTB-7 export\n"
    if symlink:
        foreign = tmp_path / "foreign.txt"
        foreign.write_bytes(original)
        destination.symlink_to(foreign)
    else:
        destination.write_bytes(original)
    completed = run_cli([raw], output)
    assert completed.returncode != 0
    assert destination.read_bytes() == original
    assert destination.is_symlink() == symlink
    assert not (output / "report.json").exists()


def test_checked_in_synthetic_export_matches_api_and_versioned_schema(monkeypatch):
    from mgts_sensing.io import iter_profiles
    from mgts_sensing.zones import ZoneConfig, iter_feature_results

    monkeypatch.chdir(ROOT)
    example = strict_json((ROOT / "data/sample/jotb7_config.json").read_text())
    schema = strict_json((ROOT / "data/sample/jotb7_schema.json").read_text())
    captured = jsonl(ROOT / "data/sample/jotb7_features.jsonl")
    assert example["synthetic"] is True and example["contract_status"] == "proposal"
    assert len(captured) == 8 and sum(row["history_ready"] for row in captured) == 4
    assert "x_start" in schema["columns"] and "x_end" in schema["columns"]
    assert "x_min" not in schema["columns"] and "x_max" not in schema["columns"]
    expected_keys = set(schema["columns"])
    assert set(schema["row_schema"]["required"]) == expected_keys
    assert schema["row_schema"]["additionalProperties"] is False

    def check_schema(value, specification):
        """Check the fixture's used JSON-schema subset without another dependency."""
        datatype = specification["type"]
        allowed = datatype if isinstance(datatype, list) else [datatype]
        types = {
            "null": value is None, "boolean": type(value) is bool,
            "integer": type(value) is int, "number": type(value) in (int, float),
            "string": type(value) is str, "array": type(value) is list,
            "object": type(value) is dict,
        }
        assert any(types[name] for name in allowed)
        if isinstance(value, dict):
            assert set(specification.get("required", [])) <= set(value)
            if specification.get("additionalProperties") is False:
                assert set(value) <= set(specification["properties"])
            for key, item in value.items():
                check_schema(item, specification["properties"][key])
        elif isinstance(value, list):
            assert len(value) >= specification.get("minItems", 0)
            assert len(value) <= specification.get("maxItems", len(value))
            if specification.get("uniqueItems"):
                assert len({json.dumps(item, sort_keys=True) for item in value}) == len(value)
            for item in value:
                check_schema(item, specification["items"])
        elif type(value) in (int, float):
            if "minimum" in specification:
                assert value >= specification["minimum"]

    for row in captured:
        assert set(row) == expected_keys
        check_schema(row, schema["row_schema"])
        assert isinstance(row["lag_medians"], list) and isinstance(row["history_sources"], list)
        assert isinstance(row["quality_flags"], list)
        if not row["history_ready"]:
            assert row["rolling_mean"] is None and row["rate"] is None
        json.dumps(row, allow_nan=False)
    source = Path(example["input"])
    original_bytes = source.read_bytes()
    parameters = {key: example[key] for key in (
        "index_start", "index_stop", "zone_width", "history_size", "max_gap_seconds",
        "min_valid_fraction",
    )}
    generated = [row for result in iter_feature_results(iter_profiles([source]), ZoneConfig(**parameters))
                 for row in result.rows]
    assert len(generated) == len(captured)
    for actual, expected in zip(generated, captured):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            if type(value) is float:
                assert actual[key] == pytest.approx(value)
            else:
                assert actual[key] == value
    assert source.read_bytes() == original_bytes
    assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(original_bytes).digest()
