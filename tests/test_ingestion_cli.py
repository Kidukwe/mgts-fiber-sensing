import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_cli(raw_dir, report_dir):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts/validate_data.py"),
         "--data-dir", str(raw_dir), "--report-dir", str(report_dir)],
        env=env, capture_output=True, text=True, check=False,
    )


@pytest.mark.parametrize("include_invalid", [False, True])
def test_cli_accounting_reports_and_raw_integrity(tmp_path, include_invalid):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    path = raw_dir / "1.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dts_id", "channel_id", "timestamp", "points"])
        writer.writerow(["a", "1", "2025-10-16T12:00:00", "[[0,-300],[1,20]]"])
        if include_invalid:
            writer.writerow(["a", "1", "2025-10-16T12:00:01", "not json"])
    before = path.read_bytes()
    report_dir = tmp_path / "reports"
    command = run_cli(raw_dir, report_dir)
    assert command.returncode == int(include_invalid), command.stderr
    summary = json.loads((report_dir / "summary.json").read_text())
    assert summary["records"] == 1 + int(include_invalid)
    assert summary["loaded"] == 1 and summary["rejected"] == int(include_invalid)
    assert summary["accounting_complete"]
    item = summary["files"][0]
    assert item["raw_unchanged"] and item["sentinel_points"] == 1
    assert item["timezone_unknown"] == 1 and len(item["sha256"]) == 64
    issues = [json.loads(line) for line in (report_dir / "issues.jsonl").read_text().splitlines()]
    assert {"UNKNOWN_TIMEZONE", "SUSPECT_SENTINEL"} <= {i["code"] for i in issues}
    if include_invalid:
        invalid = next(i for i in issues if i["code"] == "INVALID_JSON")
        assert invalid["line_number"] == 3 and invalid["row_number"] == 2
        assert invalid["severity"] == "error" and invalid["source_file"] == str(path)
    assert path.read_bytes() == before


def test_cli_broken_csv_does_not_claim_complete_accounting(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "1.csv").write_text(
        'dts_id,channel_id,timestamp,points\na,1,2025-10-16T12:00:00,"unterminated\n'
    )
    report_dir = tmp_path / "reports"
    command = run_cli(raw_dir, report_dir)
    assert command.returncode == 1
    summary = json.loads((report_dir / "summary.json").read_text())
    assert not summary["accounting_complete"]
    assert summary["files"][0]["errors"]["CSV_READ_ERROR"] == 1


def test_cli_cannot_write_reports_inside_raw_directory(tmp_path):
    (tmp_path / "1.csv").write_text("placeholder")
    command = run_cli(tmp_path, tmp_path / "derived")
    assert command.returncode == 2
    assert "outside the raw data directory" in command.stderr
    assert not (tmp_path / "derived").exists()
