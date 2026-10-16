"""Read-only ingestion audit: python scripts/validate_data.py --data-dir MGTS_data."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import Counter, defaultdict
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from mgts_sensing.io import iter_profiles


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("MGTS_data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/ingestion"))
    parser.add_argument("--naive-utc-offset-hours", type=int, default=None,
                        help="Assign a confirmed fixed offset; generic API otherwise preserves naive time")
    args = parser.parse_args()
    paths = sorted(args.data_dir.glob("*.csv"))
    if not paths:
        parser.error(f"No CSV files found in {args.data_dir}")
    raw_dir = args.data_dir.resolve()
    report_dir = args.report_dir.resolve()
    if report_dir == raw_dir or raw_dir in report_dir.parents:
        parser.error("Report directory must be outside the raw data directory")

    fingerprints = {path: file_hash(path) for path in paths}
    stats = {
        path: {
            "source_file": str(path), "sha256": fingerprints[path], "records": 0,
            "loaded": 0, "rejected": 0, "file_errors": 0, "warnings": Counter(),
            "errors": Counter(), "lengths": Counter(), "grids": set(), "steps": Counter(),
            "dts_ids": set(), "channels": set(), "sentinel_points": 0, "sentinel_profiles": 0,
            "timezone_unknown": 0, "streams": defaultdict(list),
        }
        for path in paths
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / "issues.jsonl").open("w", encoding="utf-8") as report:
        for result in iter_profiles(paths, naive_utc_offset_hours=args.naive_utc_offset_hours):
            item = stats[result.source_file]
            if result.row_number == 0:
                item["file_errors"] += 1
            else:
                item["records"] += 1
                item["loaded" if result.loaded else "rejected"] += 1
            for issue in result.issues:
                data = asdict(issue)
                data["source_file"] = str(issue.source_file)
                report.write(json.dumps(data, ensure_ascii=False) + "\n")
                item["errors" if issue.severity == "error" else "warnings"][issue.code] += 1
            if not result.loaded:
                continue
            profile = result.profile
            item["dts_ids"].add(profile.dts_id)
            item["channels"].add(profile.channel_id)
            item["lengths"][len(profile.x)] += 1
            item["grids"].add(hashlib.sha256(profile.x.tobytes()).hexdigest())
            item["steps"].update(np.round(np.diff(profile.x), 9).tolist())
            item["sentinel_points"] += int(profile.sentinel_mask.sum())
            item["sentinel_profiles"] += int(profile.sentinel_mask.any())
            item["timezone_unknown"] += int(not profile.timezone_known)
            stream = (profile.dts_id, profile.channel_id, profile.timezone_known)
            item["streams"][stream].append(profile.timestamp)

    summary_rows = []
    for path, item in stats.items():
        assert item["records"] == item["loaded"] + item["rejected"]
        intervals = []
        time_ranges = []
        for stream, times in item.pop("streams").items():
            ordered = sorted(times)
            intervals.extend((b - a).total_seconds() for a, b in pairwise(ordered))
            time_ranges.append({
                "dts_id": stream[0], "channel_id": stream[1], "timezone_known": stream[2],
                "start": min(times).isoformat(), "end": max(times).isoformat(),
            })
        # A broken CSV/encoding may make the remaining record boundaries unknowable.
        item["accounting_complete"] = not any(
            item["errors"][code] for code in ("CSV_READ_ERROR", "FILE_READ_ERROR")
        )
        item["time_ranges"] = time_ranges
        median = float(np.median(intervals)) if intervals else None
        item["intervals_seconds"] = {
            "min": min(intervals) if intervals else None, "median": median,
            "max": max(intervals) if intervals else None,
            "diagnostic_gaps": sum(value > 1.5 * median for value in intervals) if median else 0,
        }
        item["grid_variants"] = len(item.pop("grids"))
        item["step_mode"] = item["steps"].most_common(1)[0][0] if item["steps"] else None
        del item["steps"]
        item["dts_ids"] = sorted(item["dts_ids"])
        item["channels"] = sorted(item["channels"])
        item["raw_unchanged"] = file_hash(path) == fingerprints[path]
        summary_rows.append(item)
        print(f"{path.name}: records={item['records']} loaded={item['loaded']} "
              f"rejected={item['rejected']} warnings={sum(item['warnings'].values())}")

    summary = {
        "configuration": {"data_dir": str(args.data_dir), "gap_factor": 1.5,
                          "sentinel": -300, "naive_utc_offset_hours": args.naive_utc_offset_hours,
                          "duplicate_scope": "all files in caller order"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "files": summary_rows,
        "records": sum(item["records"] for item in summary_rows),
        "loaded": sum(item["loaded"] for item in summary_rows),
        "rejected": sum(item["rejected"] for item in summary_rows),
        "accounting_complete": all(item["accounting_complete"] for item in summary_rows),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"Reports: {report_dir / 'summary.json'}, {report_dir / 'issues.jsonl'}")
    return int(any(item["rejected"] or item["file_errors"] or not item["raw_unchanged"]
                   for item in summary_rows))


if __name__ == "__main__":
    raise SystemExit(main())
