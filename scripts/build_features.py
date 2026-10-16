"""Build a proposal feature table; no target, detector, preprocessing or model."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from mgts_sensing.io import iter_profiles
from mgts_sensing.zones import SCHEMA_VERSION, ZoneConfig, iter_feature_results

OUTPUT_NAMES = ("features.jsonl", "issues.jsonl", "schema.json", "report.json", "config.json")


def file_hash(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None  # iter_profiles supplies the contextual file error, not a silent skip.


def build_schema(config: ZoneConfig) -> dict:
    columns = {}

    def column(name, datatype, meaning, nullable=False, **extra):
        columns[name] = {
            "type": [datatype, "null"] if nullable else datatype,
            "description": meaning, **extra,
        }

    for name, meaning in {
        "schema_version": "Versioned proposal contract, not an approved team contract.",
        "source_file": "Source CSV path supplied to ingestion.",
        "dts_id": "Source/device identifier, physical meaning not assumed.",
        "channel_id": "Original channel identifier as string.",
        "timestamp": "Current measurement ISO datetime; naive time remains naive.",
        "timestamp_raw": "Unchanged original timestamp text.",
        "selected_grid_sha256": "SHA256 of selected float64 x bytes, not a physical line ID.",
        "history_start": "Oldest available measurement in current history.",
        "history_end": "Current measurement, not proof of online availability.",
    }.items():
        column(name, "string", meaning)
    column("timezone_raw", "string", "Original timezone offset suffix, if known.", True)
    column("utc_offset_seconds", "number", "Known UTC offset seconds; null is not UTC.", True)
    column("timezone_known", "boolean", "Whether current measurement has a known offset.")
    for name, meaning in {
        "row_number": "1-based data-record number, excluding header.",
        "line_number": "1-based physical start line in source CSV.",
        "block_id": "Increasing local ID of a contiguous compatible history block.",
        "zone_id": "0-based zone ordinal inside configured selected slice.",
        "index_start": "Absolute first original point index, inclusive.",
        "index_stop": "Absolute original point index, exclusive.",
        "nominal_count": "Configured zone width, including a partial final zone.",
        "point_count": "Actual coordinate count in this zone.",
        "valid_count": "Available original y samples after suspect-value masking.",
        "sentinel_count": "Masked point count: sentinel_mask or original y == -300.",
        "history_size": "Configured H, including current profile.",
        "history_count": "Available profiles in this block, at most H.",
        "temporal_valid_count": "Usable zonal medians in available history.",
    }.items():
        column(name, "integer", meaning, minimum=0)
    for name, meaning in {
        "x_start": "Coordinate of first actual point in zone; physical units unknown.",
        "x_end": "Coordinate of last actual point in zone, not a room boundary.",
        "valid_fraction": "valid_count / nominal_count; geometry and mask both affect it.",
        "sentinel_fraction": "sentinel_count / point_count; not probability of failure.",
    }.items():
        column(name, "number", meaning)
    for name, meaning in {
        "median": "Median of current zone's usable original values.",
        "minimum": "Minimum of current zone's usable original values.",
        "maximum": "Maximum of current zone's usable original values.",
        "maximum_minus_median": "Upper spatial excursion; retains narrow peaks.",
        "median_minus_minimum": "Lower spatial excursion; retains narrow dips.",
        "std": "Population spatial std (ddof=0), not a temporal std.",
        "delta_seconds": "Actual elapsed seconds between last two profiles; null on warmup.",
        "delta": "Current minus previous zonal median; null if either is unavailable.",
        "rate": "delta / delta_seconds in unknown y units per second.",
        "rolling_mean": "Equal-weight mean of usable medians in the complete H window.",
        "rolling_std": "Population std of usable medians in the complete H window.",
    }.items():
        column(name, "number", meaning, True)
    column("history_ready", "boolean", "H profiles available; this does not guarantee zone quality.")
    column("lag_medians", "array", "Current-first, fixed H; null for missing/padded medians.",
           items={"type": ["number", "null"]},
           minItems=config.history_size, maxItems=config.history_size)
    lineage = {
        "source_file": {"type": "string"}, "row_number": {"type": "integer"},
        "line_number": {"type": "integer"}, "timestamp": {"type": "string"},
        "timestamp_raw": {"type": "string"},
    }
    column("history_sources", "array", "Actual history current-first, without padded sources.",
           items={"type": "object", "required": list(lineage), "properties": lineage,
                  "additionalProperties": False}, minItems=1, maxItems=config.history_size)
    column("quality_flags", "array", "Unique quality codes, never event labels.",
           items={"type": "string"}, uniqueItems=True)
    return {
        "schema_version": SCHEMA_VERSION, "contract_status": "proposal", "format": "jsonl",
        "coordinate_unit": None, "signal_unit": None,
        "aggregation_order": "spatial zone statistics first, temporal zonal-median features second",
        "missing_policy": "null; raw suspect values masked before spatial aggregation",
        "warmup_policy": "spatial rows retained; complete-window temporal statistics null",
        "block_scope": "local to one invocation; namespace by input hashes and configuration",
        "columns": columns,
        "row_schema": {"type": "object", "properties": columns, "required": list(columns),
                       "additionalProperties": False},
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                               allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True,
                        help="CSV input, repeat to preserve explicit file order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-start", type=int, default=0)
    parser.add_argument("--index-stop", type=int, default=2120)
    parser.add_argument("--zone-width", type=int, default=40)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--max-gap-seconds", type=float, default=121)
    parser.add_argument("--min-valid-fraction", type=float, default=1.0)
    args = parser.parse_args()
    try:
        config = ZoneConfig(**{key: getattr(args, key) for key in (
            "index_start", "index_stop", "zone_width", "history_size",
            "max_gap_seconds", "min_valid_fraction",
        )})
    except ValueError as exc:
        parser.error(str(exc))
    output_dir = args.output_dir.resolve()
    for source in args.input:
        raw_dir = source.resolve().parent
        if output_dir == raw_dir or raw_dir in output_dir.parents:
            parser.error("Output directory must be outside the raw input directory")
    if any((output_dir / name).is_symlink() for name in OUTPUT_NAMES):
        parser.error("Refusing to replace symbolic links in output directory")
    if any((output_dir / name).exists() for name in OUTPUT_NAMES):
        try:
            previous = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            parser.error("Existing outputs are not a recognized JOTB-7 run; use a new directory")
        if previous.get("schema_version") != SCHEMA_VERSION:
            parser.error("Existing outputs have a different schema; use a new directory")
    before = {source: file_hash(source) for source in args.input}
    schema = build_schema(config)
    manifest = {
        "schema_version": SCHEMA_VERSION, "contract_status": "proposal",
        "parameters": asdict(config),
        "inputs": [{"source_file": str(source), "sha256_before": before[source]}
                   for source in args.input],
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
        "random_seed": 42, "randomness_used": False,
        "coordinate_unit": None, "signal_unit": None,
        "not_ml_ready": True,
        "implementation_sha256": {
            name: file_hash(Path(__file__).resolve().parents[1] / name)
            for name in (
                "scripts/build_features.py", "src/mgts_sensing/io.py",
                "src/mgts_sensing/validation.py", "src/mgts_sensing/windowing.py",
                "src/mgts_sensing/time_features.py", "src/mgts_sensing/zones.py",
            )
        },
    }
    report = {name: 0 for name in (
        "ingestion_records", "loaded", "rejected", "file_events", "aggregated_profiles",
        "feature_excluded", "feature_rows", "warmup_rows", "full_history_rows", "full_windows",
    )}
    issues, severities, flags = Counter(), Counter(), Counter()
    blocks = set()
    per_file = {str(source): Counter() for source in args.input}
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".jotb7-", dir=output_dir) as temporary:
        staging = Path(temporary)
        with (staging / "features.jsonl").open("w", encoding="utf-8") as table, (
            staging / "issues.jsonl"
        ).open("w", encoding="utf-8") as issue_log:
            for result in iter_feature_results(iter_profiles(args.input), config):
                raw = result.input_result
                item = per_file[str(raw.source_file)]
                if raw.row_number == 0:
                    report["file_events"] += 1
                    item["file_events"] += 1
                else:
                    report["ingestion_records"] += 1
                    item["records"] += 1
                    report["loaded" if raw.loaded else "rejected"] += 1
                    item["loaded" if raw.loaded else "rejected"] += 1
                if result.status == "excluded":
                    report["feature_excluded"] += 1
                    item["feature_excluded"] += 1
                for issue in result.issues:
                    data = asdict(issue)
                    data["source_file"] = str(issue.source_file)
                    issue_log.write(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                               allow_nan=False) + "\n")
                    issues[issue.code] += 1
                    severities[issue.severity] += 1
                if result.status != "aggregated":
                    continue
                report["aggregated_profiles"] += 1
                item["aggregated_profiles"] += 1
                blocks.add(result.block_id)
                ready = bool(result.rows[0]["history_ready"])
                report["full_windows"] += int(ready)
                for row in result.rows:
                    if set(row) != set(schema["columns"]):
                        raise ValueError("Feature row violates versioned schema keys")
                    table.write(json.dumps(row, ensure_ascii=False, sort_keys=True,
                                           allow_nan=False, separators=(",", ":")) + "\n")
                    report["feature_rows"] += 1
                    item["feature_rows"] += 1
                    report["full_history_rows" if ready else "warmup_rows"] += 1
                    flags.update(row["quality_flags"])
        after = {source: file_hash(source) for source in args.input}
        report.update({
            "schema_version": SCHEMA_VERSION, "contract_status": "proposal",
            "zone_count": report["feature_rows"], "temporal_blocks": len(blocks),
            "issue_counts": dict(issues), "issue_severities": dict(severities),
            "zone_flag_counts": dict(flags),
            "accounting_complete": not any(issues[code] for code in
                                           ("CSV_READ_ERROR", "FILE_READ_ERROR")),
            "raw_unchanged": all(before[source] is not None and before[source] == after[source]
                                 for source in args.input),
            "input_files": [
                {"source_file": str(source), "sha256_before": before[source],
                 "sha256_after": after[source], **dict(per_file[str(source)])}
                for source in args.input
            ],
        })
        assert report["ingestion_records"] == report["loaded"] + report["rejected"]
        assert report["loaded"] == report["aggregated_profiles"] + report["feature_excluded"]
        assert report["feature_rows"] == report["warmup_rows"] + report["full_history_rows"]
        write_json(staging / "schema.json", schema)
        write_json(staging / "report.json", report)
        write_json(staging / "config.json", manifest)
        # Publish only our named derived files; never write to raw or remove user's files.
        for name in OUTPUT_NAMES:
            (staging / name).replace(output_dir / name)
    print(f"Profiles: {report['ingestion_records']}; loaded: {report['loaded']}; "
          f"ingestion rejected: {report['rejected']}; feature excluded: {report['feature_excluded']}")
    print(f"Rows: {report['feature_rows']}; full windows: {report['full_windows']}; "
          f"blocks: {report['temporal_blocks']}; output: {output_dir}")
    print("PROPOSAL: no physical units, events, targets, detector thresholds or model.")
    return int(bool(report["rejected"] or report["file_events"] or report["feature_excluded"]
                    or not report["raw_unchanged"]))


if __name__ == "__main__":
    raise SystemExit(main())
