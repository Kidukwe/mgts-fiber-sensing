"""Proposal-only spatial zones followed by causal temporal features.

No events, labels, physical units, interpolation or learned transformations are
inferred. Input profiles remain untouched, and every ingestion outcome survives.
"""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .time_features import TemporalValues, compute_temporal_values
from .validation import SENTINEL, Issue, Profile, ProfileResult
from .windowing import _block_key, _timestamp, _validate_profile, iter_windows

SCHEMA_VERSION = "jotb7-zones-v1-proposal"


@dataclass(frozen=True)
class ZoneConfig:
    index_start: int = 0
    index_stop: int = 2120
    zone_width: int = 40
    history_size: int = 5
    max_gap_seconds: float = 121.0
    min_valid_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name in ("index_start", "index_stop", "zone_width", "history_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer, not bool")  # noqa: TRY004
        if not 0 <= self.index_start < self.index_stop:
            raise ValueError("Require 0 <= index_start < index_stop")
        if self.zone_width < 1 or self.history_size < 2:
            raise ValueError("Require zone_width >= 1 and history_size >= 2")
        for name in ("max_gap_seconds", "min_valid_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite and real")
        if self.max_gap_seconds <= 0 or not 0 < self.min_valid_fraction <= 1:
            raise ValueError("Require max_gap_seconds > 0 and 0 < min_valid_fraction <= 1")


@dataclass(frozen=True)
class ZoneSummary:
    zone_id: int
    index_start: int
    index_stop: int
    x_min: float
    x_max: float
    nominal_count: int
    point_count: int
    valid_count: int
    sentinel_count: int
    valid_fraction: float
    sentinel_fraction: float
    median: float | None
    minimum: float | None
    maximum: float | None
    maximum_minus_median: float | None
    median_minus_minimum: float | None
    std: float | None
    quality_flags: tuple[str, ...]


@dataclass(frozen=True)
class AggregatedProfile:
    profile: Profile
    selected_x: NDArray[np.float64]
    zones: tuple[ZoneSummary, ...]


@dataclass(frozen=True)
class FeatureResult:
    input_result: ProfileResult
    status: Literal["aggregated", "rejected", "excluded", "file_error"]
    rows: tuple[dict, ...]
    issues: tuple[Issue, ...]
    block_id: int | None


def aggregate_profile(profile: Profile, config: ZoneConfig) -> AggregatedProfile:
    """Mask raw suspect values in derived statistics; preserve the raw profile."""
    _validate_profile(profile)
    if profile.x.size < config.index_stop:
        raise ValueError("Profile does not contain the entire selected slice")
    selected_x = profile.x[config.index_start:config.index_stop].astype(np.float64, copy=True)
    selected_x.setflags(write=False)
    zones: list[ZoneSummary] = []
    for zone_id, start in enumerate(range(config.index_start, config.index_stop, config.zone_width)):
        stop = min(start + config.zone_width, config.index_stop)
        y = profile.y[start:stop]
        missing = profile.sentinel_mask[start:stop] | (y == SENTINEL)
        valid = y[~missing]
        count = stop - start
        flags: list[str] = []
        if count < config.zone_width:
            flags.append("INCOMPLETE_ZONE")
        sentinel_count = int(missing.sum())
        if sentinel_count:
            flags.append("SUSPECT_SENTINEL")
        valid_count = int(valid.size)
        fraction = valid_count / config.zone_width
        median = minimum = maximum = std = maximum_minus_median = median_minus_minimum = None
        if valid_count == 0 or fraction < config.min_valid_fraction:
            flags.append("INSUFFICIENT_VALID_POINTS")
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                median = float(np.median(valid))
                minimum = float(np.min(valid))
                maximum = float(np.max(valid))
                std = float(np.std(valid, ddof=0))
                maximum_minus_median = maximum - median
                median_minus_minimum = median - minimum
            if not all(np.isfinite(value) for value in (
                median, minimum, maximum, std, maximum_minus_median, median_minus_minimum,
            )):
                flags.append("NONFINITE_DERIVED_STATISTIC")
                median = minimum = maximum = std = maximum_minus_median = median_minus_minimum = None
        zones.append(ZoneSummary(
            zone_id=zone_id,
            index_start=int(start), index_stop=int(stop),
            x_min=float(profile.x[start]), x_max=float(profile.x[stop - 1]),
            nominal_count=int(config.zone_width), point_count=int(count),
            valid_count=valid_count, sentinel_count=sentinel_count,
            valid_fraction=fraction, sentinel_fraction=sentinel_count / count,
            median=median, minimum=minimum, maximum=maximum,
            maximum_minus_median=maximum_minus_median,
            median_minus_minimum=median_minus_minimum, std=std,
            quality_flags=tuple(flags),
        ))
    return AggregatedProfile(profile, selected_x, tuple(zones))


def _issue(result: ProfileResult, code: str, description: str) -> Issue:
    return Issue(
        result.source_file, result.row_number, result.line_number,
        "warning", code, description,
    )


def _project(aggregated: AggregatedProfile) -> Profile:
    """Internal finite representation only; marker semantics never applied here."""
    values = np.array([
        zone.median if zone.median is not None else 0.0 for zone in aggregated.zones
    ], dtype=np.float64)
    missing = np.array([zone.median is None for zone in aggregated.zones], dtype=np.bool_)
    x = np.arange(len(aggregated.zones), dtype=np.float64)
    for array in (x, values, missing):
        array.setflags(write=False)
    return replace(
        aggregated.profile, x=x, y=values, sentinel_mask=missing, points_size=len(values),
    )


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _rows(
    aggregated: AggregatedProfile,
    history: deque[Profile],
    config: ZoneConfig,
    block_id: int,
    boundary_flags: tuple[str, ...],
    temporal: TemporalValues | None,
) -> tuple[dict, ...]:
    profile = aggregated.profile
    ready = temporal is not None
    history_sources = [
        {"source_file": str(item.source_file), "row_number": item.row_number,
         "line_number": item.line_number, "timestamp": item.timestamp.isoformat(),
         "timestamp_raw": item.timestamp_raw}
        for item in reversed(history)
    ]
    grid_hash = hashlib.sha256(aggregated.selected_x.tobytes()).hexdigest()
    rows: list[dict] = []
    for zone in aggregated.zones:
        index = zone.zone_id
        if ready:
            lags = [_number(value) for value in temporal.lag_values[:, index]]
            temporal_count = int(temporal.valid_count[index])
        else:
            lags = [
                None if item.sentinel_mask[index] else float(item.y[index])
                for item in reversed(history)
            ]
            temporal_count = sum(value is not None for value in lags)
            lags.extend([None] * (config.history_size - len(lags)))
        flags = [*profile.quality_flags, *zone.quality_flags, *boundary_flags]
        if not ready:
            flags.append("HISTORY_WARMUP")
        else:
            if temporal_count < config.history_size:
                flags.append("PARTIAL_TEMPORAL_VALUES")
            if temporal_count == 0:
                flags.append("NO_VALID_TEMPORAL_VALUES")
            # Missing-history NaN is expected. NaN/Inf despite available inputs
            # is arithmetic failure, including overflow cancellation in sums.
            unexpected_nonfinite = temporal_count > 0 and any(
                not np.isfinite(getattr(temporal, name)[index])
                for name in ("rolling_mean", "rolling_std")
            )
            latest_two_valid = all(not item.sentinel_mask[index] for item in tuple(history)[-2:])
            unexpected_nonfinite |= latest_two_valid and any(
                not np.isfinite(getattr(temporal, name)[index]) for name in ("delta", "rate")
            )
            if unexpected_nonfinite:
                flags.append("NONFINITE_DERIVED_STATISTIC")
        rows.append({
            "schema_version": SCHEMA_VERSION,
            "source_file": str(profile.source_file), "row_number": profile.row_number,
            "line_number": profile.line_number, "dts_id": profile.dts_id,
            "channel_id": profile.channel_id, "timestamp": profile.timestamp.isoformat(),
            "timestamp_raw": profile.timestamp_raw, "timezone_raw": profile.timezone_raw,
            "utc_offset_seconds": profile.utc_offset_seconds,
            "timezone_known": profile.timezone_known, "block_id": block_id,
            "zone_id": zone.zone_id, "selected_grid_sha256": grid_hash,
            "index_start": zone.index_start, "index_stop": zone.index_stop,
            "x_start": zone.x_min, "x_end": zone.x_max, "nominal_count": zone.nominal_count,
            "point_count": zone.point_count, "valid_count": zone.valid_count,
            "sentinel_count": zone.sentinel_count, "valid_fraction": zone.valid_fraction,
            "sentinel_fraction": zone.sentinel_fraction, "median": zone.median,
            "minimum": zone.minimum, "maximum": zone.maximum,
            "maximum_minus_median": zone.maximum_minus_median,
            "median_minus_minimum": zone.median_minus_minimum, "std": zone.std,
            "history_size": config.history_size, "history_count": len(history),
            "history_ready": ready, "history_start": history[0].timestamp.isoformat(),
            "history_end": profile.timestamp.isoformat(), "history_sources": history_sources,
            "lag_medians": lags, "temporal_valid_count": temporal_count,
            "delta_seconds": temporal.delta_seconds if ready else None,
            "delta": _number(temporal.delta[index]) if ready else None,
            "rate": _number(temporal.rate[index]) if ready else None,
            "rolling_mean": _number(temporal.rolling_mean[index]) if ready else None,
            "rolling_std": _number(temporal.rolling_std[index]) if ready else None,
            "quality_flags": list(dict.fromkeys(flags)),
        })
    return tuple(rows)


def iter_feature_results(
    results: Iterable[ProfileResult], config: ZoneConfig,
) -> Iterator[FeatureResult]:
    """Yield one accounted outcome per input and causal zone rows, including warmup."""
    history: deque[Profile] = deque(maxlen=int(config.history_size))
    previous: AggregatedProfile | None = None
    block_id = -1
    for result in results:
        issues = list(result.issues)
        if result.row_number == 0 or not result.loaded:
            history.clear()
            previous = None
            yield FeatureResult(
                result, "file_error" if result.row_number == 0 else "rejected",
                (), tuple(issues), None,
            )
            continue
        try:
            aggregated = aggregate_profile(result.profile, config)
        except ValueError as error:
            code = (
                "SELECTED_SLICE_UNAVAILABLE"
                if isinstance(result.profile.x, np.ndarray)
                and result.profile.x.size < config.index_stop else "INVALID_FEATURE_PROFILE"
            )
            issues.append(_issue(result, code, str(error)))
            history.clear()
            previous = None
            yield FeatureResult(result, "excluded", (), tuple(issues), None)
            continue

        current = aggregated.profile
        duplicate = "DUPLICATE_KEY" in current.quality_flags or any(
            item.code == "DUPLICATE_KEY" for item in issues
        )
        same_stream = previous is not None and _block_key(current) == _block_key(previous.profile)
        nonincreasing = same_stream and _timestamp(current) <= _timestamp(previous.profile)
        if duplicate or nonincreasing:
            code = "DUPLICATE_KEY" if duplicate or (
                nonincreasing and _timestamp(current) == _timestamp(previous.profile)
            ) else "NONINCREASING_TIMESTAMP"
            if not any(item.code == code for item in issues):
                issues.append(_issue(result, code, "Profile excluded; timestamp does not increase"))
            history.clear()
            previous = None
            yield FeatureResult(result, "excluded", (), tuple(issues), None)
            continue

        boundary_flags: list[str] = []
        if previous is not None:
            past = previous.profile
            if current.source_file != past.source_file:
                boundary_flags.append("SOURCE_CHANGED")
            if (current.dts_id, current.channel_id) != (past.dts_id, past.channel_id):
                boundary_flags.append("STREAM_CHANGED")
            if current.timezone_known != past.timezone_known:
                boundary_flags.append("TIMEZONE_DOMAIN_CHANGED")
            if not np.array_equal(aggregated.selected_x, previous.selected_x):
                boundary_flags.append("SELECTED_GRID_CHANGED")
            if same_stream and (
                _timestamp(current) - _timestamp(past)
            ).total_seconds() > config.max_gap_seconds:
                boundary_flags.append("TIME_GAP")
        if previous is None or boundary_flags:
            history.clear()
            block_id += 1
        for code in boundary_flags:
            issues.append(_issue(result, code, "New temporal block; history reset"))
        for zone in aggregated.zones:
            for code in zone.quality_flags:
                issues.append(_issue(result, code, f"Zone {zone.zone_id}: {code}"))
        projected = _project(aggregated)
        history.append(projected)
        temporal = None
        if len(history) == config.history_size:
            # Delegate full-window validity to the existing JOTB-3 contract.
            window = next(iter_windows(
                tuple(history), history_size=config.history_size,
                max_gap_seconds=config.max_gap_seconds,
            ))
            temporal = compute_temporal_values(
                np.stack([item.y for item in window.profiles]),
                np.stack([~item.sentinel_mask for item in window.profiles]),
                tuple(item.timestamp for item in window.profiles),
            )
        rows = _rows(aggregated, history, config, block_id, tuple(boundary_flags), temporal)
        if temporal is None:
            issues.append(_issue(result, "HISTORY_WARMUP", "Fewer than H profiles in current block"))
        for zone, row in zip(aggregated.zones, rows):
            for code in ("PARTIAL_TEMPORAL_VALUES", "NO_VALID_TEMPORAL_VALUES"):
                if code in row["quality_flags"]:
                    issues.append(_issue(
                        result, code,
                        f"Zone {zone.zone_id}: {row['temporal_valid_count']} of "
                        f"{config.history_size} history medians are available",
                    ))
            if "NONFINITE_DERIVED_STATISTIC" in row["quality_flags"] and (
                "NONFINITE_DERIVED_STATISTIC" not in zone.quality_flags
            ):
                issues.append(_issue(
                    result, "NONFINITE_DERIVED_STATISTIC",
                    f"Zone {zone.zone_id}: nonfinite temporal statistic exported as null",
                ))
        yield FeatureResult(result, "aggregated", rows, tuple(issues), block_id)
        previous = aggregated
