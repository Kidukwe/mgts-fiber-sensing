"""Complete temperature-monitoring experiment: DSP, learned forecasts and alerts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np

from .dsp import change_features, spatial_features
from .features import FEATURE_NAMES, TARGET_NAMES
from .forecast import RidgeForecaster
from .io import iter_profiles

SEMANTICS = {
    "signal_unit": "degC", "accuracy_c": .1, "sample_spacing_m": .0025,
    "naive_utc_offset_hours": 3, "fiber_end_value": -300,
    "distance_policy": "index * 0.0025 m; raw CSV x exported separately",
    "spatial_resolution_m": .0025,
    "distance_is_index_based_assumption": True,
    "profile_availability": "recorded timestamp; acquisition latency unknown",
}



def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def extract(path, width, *, issue_path=None, change_thresholds=None, cadence_override=None):
    """Retain only DSP summaries in memory; ignore fiber-end tails, preserve gaps."""
    records, issues = [], Counter()
    previous = {}
    details = []
    for result in iter_profiles([path], naive_utc_offset_hours=3):
        issues.update(item.code for item in result.issues)
        for issue in result.issues:
            details.append({"source_file": str(issue.source_file), "row_number": issue.row_number,
                            "line_number": issue.line_number, "severity": issue.severity,
                            "code": issue.code, "description": issue.description})
        if not result.loaded:
            records.append({"invalid": True, "row": result.row_number})
            previous.clear()
            continue
        p = result.profile
        zones = {}
        current_points = {}
        if "DUPLICATE_KEY" not in p.quality_flags and np.all(np.diff(p.x) > 0):
            for start in range(0, len(p.y), width):
                stop = min(start + width, len(p.y))
                mask = p.sentinel_mask[start:stop]
                valid = np.flatnonzero(~mask)
                if not len(valid):
                    continue
                # Do not run spatial FFT across missing internal samples.
                if np.any(np.diff(valid) != 1):
                    issues["INTERNAL_FIBER_END_HOLE"] += 1
                    details.append({"source_file": str(path), "row_number": result.row_number,
                                    "line_number": result.line_number, "severity": "warning",
                                    "code": "INTERNAL_FIBER_END_HOLE",
                                    "description": f"Zone {start // width}: noncontiguous valid samples"})
                    continue
                a, b = start + int(valid[0]), start + int(valid[-1]) + 1
                stats = spatial_features(p.y[a:b])
                zone_id = start // width
                old = previous.get(zone_id)
                delta = np.array([], dtype=float)
                delta_seconds = None
                if old is not None and old["stream"] == (p.dts_id, p.channel_id) and np.array_equal(
                        old["x"], p.x[a:b]):
                    delta_seconds = (p.timestamp - old["timestamp"]).total_seconds()
                    if delta_seconds > 0:
                        delta = p.y[a:b] - old["y"]
                current_points[zone_id] = {"x": p.x[a:b], "y": p.y[a:b],
                                           "timestamp": p.timestamp,
                                           "stream": (p.dts_id, p.channel_id)}
                zones[zone_id] = {
                    "_delta": delta, "_delta_seconds": delta_seconds,
                    **stats, "index_start": a, "index_stop": b,
                    "raw_x_start": float(p.x[a]), "raw_x_end": float(p.x[b - 1]),
                    "grid": hashlib.sha256(p.x[a:b].tobytes()).hexdigest(),
                }
        previous = current_points
        records.append({"row": p.row_number, "line_number": p.line_number,
                        "quality_flags": list(p.quality_flags),
                        "timestamp": p.timestamp.isoformat(),
                        "seconds": p.timestamp.timestamp(), "stream": (p.dts_id, p.channel_id),
                        "zones": zones})
    # Thresholds for pointwise differences are fitted only to early training rows.
    training = records[:int(len(records) * .6)]
    intervals = [b["seconds"] - a["seconds"] for a, b in pairwise(training)
                 if "seconds" in a and "seconds" in b and a["stream"] == b["stream"]
                 and b["seconds"] > a["seconds"]]
    cadence = cadence_override or (float(np.median(intervals)) if intervals else 120.0)
    if change_thresholds is None:
        deltas = defaultdict(list)
        for record in training:
            for zone_id, zone in record.get("zones", {}).items():
                dt = zone["_delta_seconds"]
                if dt is not None and cadence * .5 <= dt <= cadence * 1.5:
                    deltas[str(zone_id)].append(np.abs(zone["_delta"]))
        change_thresholds = {z: max(.2, float(np.quantile(np.concatenate(parts), .95)))
                             for z, parts in deltas.items() if any(len(v) for v in parts)}
    for record in records:
        for zone_id, zone in record.get("zones", {}).items():
            dt = zone.pop("_delta_seconds")
            delta = zone.pop("_delta")
            if dt is None or not cadence * .5 <= dt <= cadence * 1.5:
                delta = np.array([], dtype=float)
            threshold = change_thresholds.get(str(zone_id), .2)
            zone.update(change_features(delta, threshold, zone["index_stop"] - zone["index_start"]))
            zone["change_threshold_c"] = threshold
            zone["change_threshold_from_train"] = str(zone_id) in change_thresholds
    if issue_path is not None:
        with Path(issue_path).open("w", encoding="utf-8") as handle:
            for issue in details:
                handle.write(json.dumps(issue, ensure_ascii=False, allow_nan=False) + "\n")
    return records, dict(issues)


def make_samples(records, *, cadence_override=None):
    """60/20/20 chronological target rows; causal history can cross split boundaries.

    No target in training reaches validation. Every feature ends before its target.
    Tail/grid changes reset only the affected zone. Rejections and time gaps reset all.
    """
    n = len(records)
    train_end, validation_end = int(n * .6), int(n * .8)
    positive_deltas = [b["seconds"] - a["seconds"] for a, b in
                       pairwise(records[:train_end])
                       if "seconds" in a and "seconds" in b
                       and a["stream"] == b["stream"] and b["seconds"] > a["seconds"]]
    cadence = cadence_override or (float(np.median(positive_deltas)) if positive_deltas else 120.0)
    history, last, block = {}, None, 0
    x, y, current, persistence, rolling, meta, splits = [], [], [], [], [], [], []
    for index, record in enumerate(records):
        if record.get("invalid"):
            history.clear()
            last = None
            block += 1
            continue
        if last is not None and (record["stream"] != last["stream"]
                                 or not cadence * .5 <= record["seconds"] - last["seconds"] <= cadence * 1.5):
            history.clear()
            block += 1
        for zone_id in list(history):
            if zone_id not in record["zones"]:
                del history[zone_id]
        for zone_id, zone in record["zones"].items():
            h = history.setdefault(zone_id, deque(maxlen=5))
            if h and h[-1][1]["grid"] != zone["grid"]:
                h.clear()
            value = np.array([zone["median"], zone["hot"], zone["cold"]])
            if len(h) >= 5:
                past = np.array([[z["median"], z["hot"], z["cold"]] for _, z in h])
                previous_time = datetime.fromisoformat(h[-1][0]["timestamp"])
                # Time-of-day features use only the last available profile, in UTC+3.
                previous_time = previous_time.astimezone(timezone(timedelta(hours=3)))
                phase = 2 * math.pi * (previous_time.hour + previous_time.minute / 60) / 24
                features = np.r_[past[-1] - past[-2], past[-2] - past[-3],
                                 past[-1] - np.median(past, axis=0),
                                 [h[-1][1][key] for key in
                                  ("hot", "cold", "std", "roughness", "gradient",
                                   "high_frequency_fraction", "robust_hot", "robust_cold",
                                   "fraction_hot", "fraction_cold", "longest_hot_fraction",
                                   "longest_cold_fraction", "pair_fraction")],
                                 math.sin(phase), math.cos(phase)]
                x.append(features)
                y.append(value - past[-1])
                current.append(value)
                persistence.append(past[-1])
                rolling.append(np.median(past, axis=0))
                meta.append({"profile_index": index, "zone_id": zone_id, "block": block,
                             "row_number": record["row"], "line_number": record["line_number"],
                             "quality_flags": record["quality_flags"],
                             "timestamp": record["timestamp"],
                             "forecast_origin": h[-1][0]["timestamp"],
                             "previous_temperature_c": float(past[-1, 0]),
                             "dts_id": record["stream"][0], "channel_id": record["stream"][1],
                             **{k: zone[k] for k in ("index_start", "index_stop",
                                                    "raw_x_start", "raw_x_end", "grid", "change_threshold_c",
                                                    "change_threshold_from_train", "fraction_hot",
                                                    "fraction_cold", "longest_hot_fraction",
                                                    "longest_cold_fraction")}})
                splits.append(0 if index < train_end else 1 if index < validation_end else 2)
            h.append((record, zone))
        last = record
    if not x:
        raise ValueError("Insufficient continuous observations for five-profile history")
    return (np.array(x), np.array(y), np.array(current), np.array(persistence),
            np.array(rolling), meta, np.array(splits), cadence)


def metric(truth, prediction, profile_ids):
    errors = truth - prediction
    # Give each timestamp equal weight, despite varying numbers of valid zones.
    _, inverse = np.unique(profile_ids, return_inverse=True)
    counts = np.bincount(inverse)
    mae = np.mean(np.bincount(inverse, weights=np.abs(errors)) / counts)
    mse = np.mean(np.bincount(inverse, weights=errors ** 2) / counts)
    return {"mae_c": float(mae), "rmse_c": float(np.sqrt(mse)),
            "profiles": len(counts), "zone_observations": len(truth)}


def build_alerts(meta, current, predicted, threshold, split, cadence, source):
    """Connected components in space/time on test only, followed by joint ranking.

    Offline event stitching is separate from forecasting. Two distinct timestamps
    are required; extent is instrumented fiber length, not room size or leak area.
    """
    # Two complementary channels: learned innovations and frozen robust level.
    floors = np.array([.2, .4, .4])
    evidence = {}
    history = {}
    last_index = {}
    for i, row in enumerate(meta):
        key = (row["block"], row["dts_id"], row["channel_id"], row["zone_id"], row["grid"])
        h = history.setdefault(key, deque(maxlen=30))
        if key in last_index and row["profile_index"] != last_index[key] + 1:
            h.clear()
        last_index[key] = row["profile_index"]
        innovation = current[i] - predicted[i]
        level = innovation.copy()
        level_threshold = threshold.copy()
        if len(h) >= 5:
            past = np.array(h)
            center = np.median(past, axis=0)
            scale = 1.4826 * np.median(np.abs(past - center), axis=0)
            level = current[i] - center
            level_threshold = np.maximum(floors, 6 * scale)
        # Forecast overshoot at recovery must not turn a return to background
        # into an opposite physical alarm. Require directional level evidence
        # outside the sensor's worst-case difference uncertainty (0.2 C).
        corroborated = (np.abs(level) > floors) & (innovation * level > 0)
        innovation_ratio = np.where(corroborated, innovation / threshold, 0.0)
        level_ratio = level / level_threshold
        choose_level = np.abs(level_ratio) > np.abs(innovation_ratio)
        ratio = np.where(choose_level, level_ratio, innovation_ratio)
        amplitude = np.where(choose_level, level, innovation)
        evidence[i] = (amplitude, ratio, choose_level)
        # Keep the baseline fixed throughout suspicion; a single transient may
        # delay adaptation, but cannot become a confirmed event alone.
        if split[i] != 2 or np.all(np.abs(ratio) < 1):
            h.append(current[i])
    active, complete = [], []
    grouped = defaultdict(list)
    for i in np.flatnonzero(split == 2):
        grouped[(meta[i]["block"], meta[i]["profile_index"])].append(i)
    next_id = 1
    for (block, profile_index), indices in sorted(grouped.items()):
        time = datetime.fromisoformat(meta[indices[0]]["timestamp"])
        candidates = defaultdict(list)
        for i in indices:
            r, ratios, level_channels = evidence[i]
            values = {"heating": (r[0], ratios[0]), "cooling": (-r[0], -ratios[0]),
                      "local_peak": (r[1], ratios[1]), "local_dip": (r[2], ratios[2])}
            for kind, (amplitude, ratio) in values.items():
                ratio = float(ratio)
                if ratio >= 1:
                    channel = 0 if kind in ("heating", "cooling") else (1 if kind == "local_peak" else 2)
                    evidence_row = {**meta[i], "actual_temperature_c": float(current[i, 0]),
                                    "reference_temperature_c": float(current[i, 0] - r[0]),
                                    "evidence_channel": "robust_level" if level_channels[channel]
                                    else "learned_forecast",
                                    "change_rate_c_per_minute": float((current[i, 0] -
                                         meta[i]["previous_temperature_c"]) / (
                                             datetime.fromisoformat(meta[i]["timestamp"]) -
                                             datetime.fromisoformat(meta[i]["forecast_origin"])
                                         ).total_seconds() * 60),
                                    "feature_value_c": float(current[i, channel]),
                                    "reference_value_c": float(current[i, channel] - r[channel])}
                    candidates[kind].append((evidence_row, float(amplitude), ratio))
        components = []
        for kind, points in candidates.items():
            points.sort(key=lambda item: item[0]["zone_id"])
            runs = []
            for point in points:
                if not runs or point[0]["index_start"] != runs[-1][-1][0]["index_stop"]:
                    runs.append([])
                runs[-1].append(point)
            components.extend((kind, run) for run in runs)
        previous, active = active, []
        for kind, run in components:
            first, last_point = run[0][0], run[-1][0]
            matches = [e for e in previous if e["kind"] == kind and e["block"] == block
                       and e["last_profile_index"] == profile_index - 1
                       and e["dts_id"] == first["dts_id"]
                       and e["channel_id"] == first["channel_id"]
                       and e["grid_by_zone"].get(str(first["zone_id"])) == first["grid"]
                       and e["index_start"] < last_point["index_stop"]
                       and first["index_start"] < e["index_stop"]
                       and 0 < (time - datetime.fromisoformat(e["end"])).total_seconds() <= cadence * 1.5]
            if matches:
                e = matches[0]
                previous.remove(e)
            else:
                e = {"event_id": f"{source}:{next_id}", "source_file": source,
                     "kind": kind, "block": block, "start": first["timestamp"],
                     "end": first["timestamp"], "detected_at": None, "observations": 0,
                     "dts_id": first["dts_id"], "channel_id": first["channel_id"],
                     "index_start": first["index_start"], "index_stop": last_point["index_stop"],
                     "peak_amplitude_c": 0.0, "peak_ratio": 0.0, "peak_extent_m": 0.0,
                     "raw_x_start": first["raw_x_start"], "raw_x_end": last_point["raw_x_end"]}
                next_id += 1
            e["grid_by_zone"] = {str(p[0]["zone_id"]): p[0]["grid"] for p in run}
            e["quality_flags"] = sorted(set(e.get("quality_flags", [])) |
                                         {f for p in run for f in p[0]["quality_flags"]})
            if max(p[1] for p in run) >= e["peak_amplitude_c"]:
                peak = max(run, key=lambda p: p[1])[0]
                e["peak_evidence"] = {key: peak[key] for key in
                    ("row_number", "line_number", "timestamp", "actual_temperature_c",
                     "reference_temperature_c", "evidence_channel", "change_rate_c_per_minute",
                     "feature_value_c", "reference_value_c",
                     "fraction_hot", "fraction_cold", "longest_hot_fraction", "longest_cold_fraction")}
            e["observations"] += 1
            e["end"] = first["timestamp"]
            e["last_profile_index"] = profile_index
            if e["observations"] == 2:
                e["detected_at"] = first["timestamp"]
            e["index_start"] = min(e["index_start"], first["index_start"])
            e["index_stop"] = max(e["index_stop"], last_point["index_stop"])
            e["raw_x_start"] = min(e["raw_x_start"], first["raw_x_start"])
            e["raw_x_end"] = max(e["raw_x_end"], last_point["raw_x_end"])
            e["peak_amplitude_c"] = max(e["peak_amplitude_c"], max(p[1] for p in run))
            e["peak_ratio"] = max(e["peak_ratio"], max(p[2] for p in run))
            e["peak_extent_m"] = max(e["peak_extent_m"],
                                     sum(p[0]["index_stop"] - p[0]["index_start"] for p in run) * .0025)
            active.append(e)
        for e in previous:
            e["close_reason"] = "candidate_discontinued"
        complete.extend(previous)
    for e in active:
        e["close_reason"] = "end_of_input"
    complete.extend(active)
    alerts = []
    for e in complete:
        if e["observations"] < 2:
            continue
        e["duration_seconds"] = (datetime.fromisoformat(e["end"]) -
                                 datetime.fromisoformat(e["start"])).total_seconds()
        e["distance_start_m"] = e["index_start"] * .0025
        e["distance_stop_m"] = e["index_stop"] * .0025
        e["priority_score"] = float(np.log1p(e["peak_ratio"]) *
                                    np.log1p(1 + e["duration_seconds"] / cadence) *
                                    np.sqrt(e["peak_extent_m"] / .0025))
        e["status"] = "temperature_anomaly_candidate"
        e["interpretation"] = ("possible overheating or hot-fluid leak" if e["kind"] in
                               ("heating", "local_peak") else
                               "possible cooling, cold-fluid leak or insulation problem")
        alerts.append(e)
    return sorted(alerts, key=lambda e: e["priority_score"], reverse=True)


def run_file(path, output, width=64, quantile=.995):
    before = sha256(path)
    output.mkdir(parents=True, exist_ok=True)
    records, issues = extract(path, width, issue_path=output / "issues.jsonl")
    x, y, values, persistence, rolling, meta, split, cadence = make_samples(records)
    if any(not np.any(split == part) for part in range(3)):
        raise ValueError("Need nonempty chronological train, validation and test")
    train, validation, test = split == 0, split == 1, split == 2
    ids = np.array([r["profile_index"] for r in meta])
    candidates, models = {}, {}
    feature_sets = {"temporal": [0, 3, 6, len(FEATURE_NAMES) - 2, len(FEATURE_NAMES) - 1],
                    "spatial_dsp": list(range(len(FEATURE_NAMES)))}
    model_indices = {}
    for family, columns in feature_sets.items():
        for alpha in (.1, 10., 1000.):
            model = RidgeForecaster.fit(x[train][:, columns], y[train], alpha)
            name = f"ridge_{family}_alpha_{alpha:g}"
            candidates[name] = persistence + model.predict(x[:, columns])
            candidates[name][:, 1:] = np.maximum(0, candidates[name][:, 1:])
            models[name] = model
            model_indices[name] = columns
    candidates["persistence"] = persistence
    candidates["rolling_median"] = rolling
    # Constant per-zone training median is a separate, non-adapting comparator.
    global_train_median = np.median(values[train], axis=0)
    zone_train_medians = {}
    constant = np.tile(global_train_median, (len(values), 1))
    for zone_id in {m["zone_id"] for m in meta}:
        same_zone = np.array([m["zone_id"] == zone_id for m in meta])
        if np.any(same_zone & train):
            center = np.median(values[same_zone & train], axis=0)
            constant[same_zone] = center
            zone_train_medians[str(zone_id)] = center.tolist()
    candidates["train_median"] = constant
    validation_metrics = {name: metric(values[validation, 0], p[validation, 0], ids[validation])
                          for name, p in candidates.items()}
    selected = min(validation_metrics, key=lambda name: validation_metrics[name]["mae_c"])
    best_ridge = min(models, key=lambda name: validation_metrics[name]["mae_c"])
    prediction = candidates[selected]
    residuals = np.abs(values[validation] - prediction[validation])
    thresholds = np.maximum([.2, .4, .4], np.quantile(residuals, quantile, axis=0))
    alerts = build_alerts(meta, values, prediction, thresholds, split, cadence, path.name)
    output.mkdir(parents=True, exist_ok=True)
    artifact = {"schema_version": "building-temperature-v2", "semantics": SEMANTICS,
                "source_sha256": before, "source_file": path.name,
                "selected_model": selected, "best_ridge": best_ridge,
                "ridge": models[best_ridge].to_dict(), "feature_names": FEATURE_NAMES,
                "feature_indices": model_indices[best_ridge],
                "allowed_streams": sorted({(m["dts_id"], m["channel_id"]) for m in meta}),
                "point_change_thresholds_c": {str(m["zone_id"]): m["change_threshold_c"]
                                               for m in meta if m["change_threshold_from_train"]},
                "feature_ablation": feature_sets,
                "train_medians_c": zone_train_medians,
                "global_train_median_c": global_train_median.tolist(),
                "target_names": TARGET_NAMES, "thresholds_c": thresholds.tolist(),
                "calibration_quantile": quantile, "zone_width_points": width,
                "min_gap_seconds": cadence * .5, "cadence_seconds": cadence, "max_gap_seconds": cadence * 1.5,
                "split_target_rows": [int(len(records) * .6), int(len(records) * .8)],
                "history_profiles": 5, "clip_excursions_to_nonnegative": True}
    dump(output / "model.json", artifact)
    dump(output / "events.json", alerts)
    np.savez_compressed(output / "predictions.npz", actual=values[test], predicted=prediction[test],
                        persistence=persistence[test], rolling=rolling[test], train_median=constant[test],
                        features=x[test],
                        ridge_predicted=candidates[best_ridge][test],
                        timestamp=np.array([m["timestamp"] for m in meta])[test],
                        forecast_origin=np.array([m["forecast_origin"] for m in meta])[test],
                        row_number=np.array([m["row_number"] for m in meta])[test],
                        index_start=np.array([m["index_start"] for m in meta])[test],
                        index_stop=np.array([m["index_stop"] for m in meta])[test],
                        profile_index=ids[test], zone_id=np.array([m["zone_id"] for m in meta])[test])
    after = sha256(path)
    if before != after:
        raise RuntimeError(f"Raw input changed: {path}")
    result = {"source_file": path.name, "sha256": before, "raw_unchanged": before == after,
              "profiles": len(records), "issues": issues, "selected_model": selected,
              "loaded_profiles": sum(not r.get("invalid", False) for r in records),
              "invalid_records": sum(r.get("invalid", False) for r in records),
              "accounting_complete": not any(issues.get(c, 0) for c in
                                              ("CSV_READ_ERROR", "FILE_READ_ERROR")),
              "feature_ablation": {family: min(
                    (name for name in models if f"_{family}_" in name),
                    key=lambda name: validation_metrics[name]["mae_c"]) for family in feature_sets},
              "validation_metrics": validation_metrics,
              "test_metrics": {name: metric(values[test, 0], p[test, 0], ids[test])
                               for name, p in candidates.items()},
              "test_metrics_selected": metric(values[test, 0], prediction[test, 0], ids[test]),
              "train_observations": int(train.sum()), "validation_observations": int(validation.sum()),
              "test_observations": int(test.sum()), "cadence_seconds": cadence,
              "thresholds_c": thresholds.tolist(), "candidate_events": len(alerts),
              "event_precision": None, "event_recall": None}
    result["test_metrics_by_target"] = {
        name: metric(values[test, i], prediction[test, i], ids[test])
        for i, name in enumerate(("median", "hot_excursion", "cold_excursion"))}
    result["test_error_by_zone"] = {}
    for zone_id in {m["zone_id"] for m in meta}:
        mask = test & np.array([m["zone_id"] == zone_id for m in meta])
        if np.any(mask):
            result["test_error_by_zone"][str(zone_id)] = metric(
                values[mask, 0], prediction[mask, 0], ids[mask])
    cutoff = max(.2, float(np.quantile(np.abs(y[train, 0]), .95)))
    result["rapid_change_cutoff_from_train_c"] = cutoff
    result["test_error_by_regime"] = {}
    for name, condition in (("rapid_change", np.abs(y[:, 0]) > cutoff),
                            ("small_change", np.abs(y[:, 0]) <= cutoff)):
        mask = test & condition
        result["test_error_by_regime"][name] = metric(
            values[mask, 0], prediction[mask, 0], ids[mask]) if np.any(mask) else None
    from .visualization import plot_solution
    plot_solution(records, meta, values, prediction, split, alerts, output)
    dump(output / "metrics.json", result)
    return result, alerts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("MGTS_data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/final"))
    parser.add_argument("--zone-width", type=int, default=64,
                        help="Tunable spatial support in samples; not a required physical event size")
    parser.add_argument("--quantile", type=float, default=.995)
    args = parser.parse_args(argv)
    if args.zone_width < 2 or not .5 < args.quantile < 1:
        parser.error("Require zone-width >= 2 and .5 < quantile < 1")
    sources = sorted(args.data_dir.glob("*.csv"))
    if not sources:
        parser.error("No input CSV files")
    output = args.output_dir.resolve()
    raw = args.data_dir.resolve()
    if output == raw or raw in output.parents:
        parser.error("Output must be outside the raw data directory")
    if output.exists() and (output.is_symlink() or any(p.is_symlink() for p in output.rglob("*"))):
        parser.error("Output tree must not contain symbolic links")
    if output.exists() and any(output.iterdir()) and not (output / "solution.json").is_file():
        parser.error("Use an empty directory or an existing solution output")
    output.mkdir(parents=True, exist_ok=True)
    # Marker permits a reproducible rerun even if an earlier run was interrupted.
    dump(output / "solution.json", {"status": "running"})
    results, alerts = [], []
    for path in sources:
        print(f"Processing {path.name} ...", flush=True)
        result, events = run_file(path, output / path.stem, args.zone_width, args.quantile)
        results.append(result)
        alerts.extend(events)
        print(f"  {result['profiles']} profiles; selected={result['selected_model']}; "
              f"test MAE={result['test_metrics_selected']['mae_c']:.4f} C; "
              f"{len(events)} candidates", flush=True)
    summary = {"schema_version": "building-temperature-v2", "status":
               "complete" if all(f["accounting_complete"] and not f["invalid_records"] for f in results)
               else "complete_with_data_issues",
               "semantics": SEMANTICS, "files": results,
               "implementation_sha256": {name: sha256(Path(__file__).with_name(name)) for name in
                                         ("solution.py", "forecast.py", "dsp.py", "visualization.py",
                                          "io.py", "validation.py", "inference.py", "features.py",
                                          "preprocessing.py", "dsp_features.py")},
               "environment": {"numpy": np.__version__, "python": platform.python_version()},
               "alert_policy": "test-only; residual quantile on validation; two consecutive profiles; "
                               "joint amplitude-duration-extent priority, not calibrated probability",
               "events": sorted(alerts, key=lambda e: e["priority_score"], reverse=True)}
    dump(output / "solution.json", summary)
    from .visualization import write_solution_report
    write_solution_report(summary, output)
    print(f"Report: {output / 'report.html'}", flush=True)
    return int(summary["status"] != "complete")
