"""Run raw CSV → zonal features → causal baseline → event candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import tempfile
from collections import Counter
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from .evaluation import BaselineMetrics
from .events import EVENT_SCHEMA_VERSION, DetectorConfig, EventDetector
from .io import iter_profiles
from .zones import SCHEMA_VERSION, ZoneConfig, iter_feature_results

OUTPUT_NAMES = ("scores.jsonl", "events.jsonl", "issues.jsonl", "config.json", "report.json")


def _hash(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for cls in (ZoneConfig, DetectorConfig):
        for item in fields(cls):
            # Shared name intentionally configures both stages.
            option = "--" + item.name.replace("_", "-")
            if option in parser._option_string_actions:
                continue
            parser.add_argument(option, type=type(item.default), default=item.default)
    args = parser.parse_args(argv)
    try:
        zone_config = ZoneConfig(**{item.name: getattr(args, item.name) for item in fields(ZoneConfig)})
        detector_config = DetectorConfig(**{
            item.name: getattr(args, item.name) for item in fields(DetectorConfig)
        })
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output_dir.resolve()
    for source in args.input:
        parent = source.resolve().parent
        if output == parent or parent in output.parents:
            parser.error("Output directory must be outside every raw input directory")
    if any((output / name).is_symlink() for name in OUTPUT_NAMES):
        parser.error("Refusing to replace symbolic links")
    if any((output / name).exists() for name in OUTPUT_NAMES):
        try:
            previous = json.loads((output / "config.json").read_text(encoding="utf-8"))
            if previous["schema_version"] != EVENT_SCHEMA_VERSION:
                raise ValueError("Different schema")
        except (OSError, ValueError, KeyError, TypeError):
            parser.error("Unrecognized existing outputs; choose a new output directory")
    before = {source: _hash(source) for source in args.input}
    detector = EventDetector(detector_config)
    metrics = BaselineMetrics()
    counts: Counter = Counter()
    issue_counts: Counter = Counter()
    kinds: Counter = Counter()
    close_reasons: Counter = Counter()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".events-", dir=output) as temporary:
        staging = Path(temporary)
        with (staging / "scores.jsonl").open("w", encoding="utf-8") as scores, (
            staging / "events.jsonl"
        ).open("w", encoding="utf-8") as events, (
            staging / "issues.jsonl"
        ).open("w", encoding="utf-8") as issues:
            def save_events(items):
                for event in items:
                    events.write(_json(event) + "\n")
                    kinds[event["kind"]] += 1
                    close_reasons[event["close_reason"]] += 1
                    counts["events"] += 1

            for result in iter_feature_results(
                iter_profiles(args.input, naive_utc_offset_by_source={"5.csv": 3}), zone_config,
            ):
                counts[result.status] += 1
                counts["feature_rows"] += len(result.rows)
                for issue in result.issues:
                    item = asdict(issue)
                    item["source_file"] = str(item["source_file"])
                    issues.write(_json(item) + "\n")
                    issue_counts[issue.code] += 1
                batch = detector.process(result)
                for item in batch.scores:
                    scores.write(_json({"schema_version": EVENT_SCHEMA_VERSION, **item}) + "\n")
                metrics.update(batch.scores)
                save_events(batch.events)
            save_events(detector.finish())
        after = {source: _hash(source) for source in args.input}
        unchanged = all(before[source] is not None and before[source] == after[source]
                        for source in args.input)
        manifest = {
            "schema_version": EVENT_SCHEMA_VERSION, "feature_schema_version": SCHEMA_VERSION,
            "detector": asdict(detector_config), "zones": asdict(zone_config),
            "inputs": [{"source_file": str(source), "sha256_before": before[source],
                        "sha256_after": after[source]} for source in args.input],
            "environment": {"python": platform.python_version(), "numpy": np.__version__},
            "implementation_sha256": {
                name: _hash(Path(__file__).with_name(name)) for name in (
                    "events.py", "modeling.py", "evaluation.py", "event_cli.py", "zones.py",
                    "io.py", "validation.py", "time_features.py", "windowing.py",
                )
            },
            "signal_unit": "degC", "coordinate_unit": None,
            "sample_spacing_m": None, "accuracy_c": 0.1,
            "spatial_resolution_m": None,
            "spatial_resolution_statement": "пространственное разрешение 0,25 см",
            "semantic_source": "MGTS specialist 2026-10-15 relayed by user; no direct transcript",
            "accuracy_interpretation": "claimed accuracy; not noise sigma or alarm threshold",
            "naive_utc_offset_hours": {"5.csv": 3},
            "random_seed": 42, "randomness_used": False,
            "status": "proposal; events are candidates, not confirmed incidents",
        }
        report = {
            "schema_version": EVENT_SCHEMA_VERSION, "counts": dict(counts),
            "issue_counts": dict(issue_counts), "event_kinds": dict(kinds),
            "close_reasons": dict(close_reasons), "raw_unchanged": unchanged,
            "accounting_complete": not any(issue_counts[code] for code in (
                "CSV_READ_ERROR", "FILE_READ_ERROR",
            )), "baseline_metrics": metrics.report(),
        }
        for name, value in (("config.json", manifest), ("report.json", report)):
            (staging / name).write_text(_json(value) + "\n", encoding="utf-8")
        for name in OUTPUT_NAMES:
            (staging / name).replace(output / name)
    print(f"Profiles: {counts['aggregated']}; candidate events: {counts['events']}; output: {output}")
    print("Temperature in degC; event precision/recall unavailable without labels.")
    return int(bool(counts["rejected"] or counts["excluded"] or counts["file_error"] or not unchanged))
