"""Causal candidates for temperature-like changes in JOTB-7 zonal signals.

Thresholds are in input signal units, not presumed degrees. No learned labels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from .modeling import RobustBaseline
from .zones import FeatureResult

EVENT_SCHEMA_VERSION = "temperature-events-v1-proposal"
KINDS = {
    "heating": ("median", 1),
    "cooling": ("median", -1),
    "local_peak": ("maximum_minus_median", 1),
    "local_dip": ("median_minus_minimum", 1),
}
BAD_FLAGS = {"INSUFFICIENT_VALID_POINTS", "NONFINITE_DERIVED_STATISTIC", "INCOMPLETE_ZONE"}


@dataclass(frozen=True)
class DetectorConfig:
    baseline_window: int = 30
    min_history: int = 5
    threshold: float = 3.0
    mad_multiplier: float = 6.0
    release_ratio: float = 0.5
    min_consecutive: int = 2
    release_consecutive: int = 2
    min_valid_fraction: float = 1.0

    def __post_init__(self) -> None:
        for name in ("baseline_window", "min_history", "min_consecutive", "release_consecutive"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 2 <= self.min_history <= self.baseline_window:
            raise ValueError("Require baseline_window >= min_history >= 2")
        for name in ("threshold", "mad_multiplier", "release_ratio", "min_valid_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite and numeric")
        if self.threshold <= 0 or self.mad_multiplier < 0:
            raise ValueError("Require threshold > 0 and mad_multiplier >= 0")
        if not 0 < self.release_ratio < 1 or not 0 < self.min_valid_fraction <= 1:
            raise ValueError("Require 0 < release_ratio < 1 and 0 < min_valid_fraction <= 1")


@dataclass
class _Episode:
    pending: list[dict] = field(default_factory=list)
    event: dict | None = None
    release_count: int = 0


@dataclass
class _ZoneState:
    baselines: dict[str, RobustBaseline]
    episodes: dict[str, _Episode]
    previous: float | None = None


@dataclass(frozen=True)
class DetectionBatch:
    scores: tuple[dict, ...]
    events: tuple[dict, ...]


def _closed(event: dict, reason: str) -> dict:
    duration = (datetime.fromisoformat(event["end"]) - datetime.fromisoformat(event["start"]))
    return {**event, "close_reason": reason, "duration_seconds": duration.total_seconds()}


def _location(row: dict) -> dict:
    return {key: row[key] for key in (
        "source_file", "dts_id", "channel_id", "block_id", "selected_grid_sha256",
        "zone_id", "index_start", "index_stop", "x_start", "x_end",
    )}


class EventDetector:
    """Process ordered FeatureResults; one instance per run, finish once at EOF.

    Baselines reset on upstream block boundaries and invalid zone observations.
    Suspected excursions freeze the corresponding baseline, including confirmation.
    Episodes remain separate per zone and kind; spatial extents are zone extents.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.config = config or DetectorConfig()
        self._block = None
        self._zones: dict[int, _ZoneState] = {}
        self._next_id = 1

    def _state(self) -> _ZoneState:
        cfg = self.config
        return _ZoneState(
            {name: RobustBaseline(cfg.baseline_window, cfg.min_history)
             for name in {item[0] for item in KINDS.values()}},
            {kind: _Episode() for kind in KINDS},
        )

    def _close(self, state: _ZoneState, reason: str) -> list[dict]:
        events = []
        for episode in state.episodes.values():
            if episode.event is not None:
                events.append(_closed(episode.event, reason))
            episode.event = None
            episode.pending.clear()
            episode.release_count = 0
        return events

    def finish(self, reason: str = "end_of_input") -> tuple[dict, ...]:
        events = tuple(event for state in self._zones.values() for event in self._close(state, reason))
        self._zones.clear()
        self._block = None
        return events

    def _advance(self, episode: _Episode, score: dict, row: dict) -> list[dict]:
        cfg = self.config
        if episode.event is None:
            if score["score"] < 1:
                episode.pending.clear()
                return []
            episode.pending.append(score)
            if len(episode.pending) < cfg.min_consecutive:
                return []
            first = episode.pending[0]
            peak = max(episode.pending, key=lambda item: item["score"])
            episode.event = {
                "schema_version": EVENT_SCHEMA_VERSION, "event_id": self._next_id,
                "status": "candidate", **_location(row), "kind": score["kind"],
                "start": first["timestamp"], "detected_at": score["timestamp"],
                "end": score["timestamp"], "observations": len(episode.pending),
                "peak_score": peak["score"], "peak_timestamp": peak["timestamp"],
                "peak_residual": peak["residual"], "threshold_at_peak": peak["threshold"],
                "baseline_at_peak": peak["baseline"],
                "start_row_number": first["row_number"], "end_row_number": row["row_number"],
                "quality_flags": sorted({flag for item in episode.pending
                                         for flag in item["quality_flags"]}),
            }
            self._next_id += 1
            episode.pending.clear()
            return []
        if score["score"] < cfg.release_ratio:
            episode.release_count += 1
            if episode.release_count >= cfg.release_consecutive:
                event = _closed(episode.event, "recovered")
                episode.event = None
                episode.release_count = 0
                return [event]
            return []
        episode.release_count = 0
        event = episode.event
        event["end"] = score["timestamp"]
        event["end_row_number"] = row["row_number"]
        event["observations"] += 1
        event["quality_flags"] = sorted(set(event["quality_flags"]) | set(row["quality_flags"]))
        if score["score"] > event["peak_score"]:
            event.update(peak_score=score["score"], peak_timestamp=score["timestamp"],
                         peak_residual=score["residual"], threshold_at_peak=score["threshold"],
                         baseline_at_peak=score["baseline"])
        return []

    def process(self, result: FeatureResult) -> DetectionBatch:
        if result.status != "aggregated":
            return DetectionBatch((), self.finish("upstream_exclusion"))
        first = result.rows[0]
        block = tuple(first[name] for name in (
            "source_file", "dts_id", "channel_id", "block_id", "selected_grid_sha256",
        ))
        events = list(self.finish("block_boundary")) if self._block != block else []
        self._block = block
        scores = []
        for row in result.rows:
            zone_id = row["zone_id"]
            if zone_id not in self._zones:
                self._zones[zone_id] = self._state()
            state = self._zones[zone_id]
            invalid = (row["valid_fraction"] < self.config.min_valid_fraction
                       or bool(BAD_FLAGS.intersection(row["quality_flags"]))
                       or any(row[name] is None or not math.isfinite(row[name])
                              for name in state.baselines))
            if invalid:
                events.extend(self._close(state, "invalid_observation"))
                self._zones[zone_id] = self._state()
                scores.append({**_location(row), "timestamp": row["timestamp"],
                               "status": "invalid_observation", "kind": None,
                               "quality_flags": row["quality_flags"]})
                continue
            predictions = {name: baseline.predict() for name, baseline in state.baselines.items()}
            frozen = set()
            for kind, (name, direction) in KINDS.items():
                prediction = predictions[name]
                if prediction is None:
                    scores.append({**_location(row), "timestamp": row["timestamp"],
                                   "kind": kind, "status": "baseline_warmup",
                                   "quality_flags": row["quality_flags"]})
                    continue
                center, scale = prediction
                threshold = max(self.config.threshold, self.config.mad_multiplier * scale)
                residual = direction * (row[name] - center)
                score = residual / threshold
                if not all(math.isfinite(value) for value in (threshold, residual, score)):
                    events.extend(self._close(state, "nonfinite_score"))
                    self._zones[zone_id] = self._state()
                    scores.append({**_location(row), "timestamp": row["timestamp"],
                                   "kind": kind, "status": "nonfinite_score",
                                   "quality_flags": row["quality_flags"]})
                    break
                item = {**_location(row), "timestamp": row["timestamp"],
                        "row_number": row["row_number"], "kind": kind, "status": "scored",
                        "baseline": center, "scale": scale, "threshold": threshold,
                        "residual": residual, "score": score, "value": row[name],
                        "persistence": state.previous if name == "median" else None,
                        "quality_flags": row["quality_flags"]}
                events.extend(self._advance(state.episodes[kind], item, row))
                episode = state.episodes[kind]
                item["active"] = episode.event is not None
                item["candidate"] = score >= 1
                scores.append(item)
                if score >= 1 or episode.event is not None:
                    frozen.add(name)
            else:
                for name, baseline in state.baselines.items():
                    if name not in frozen:
                        baseline.observe(row[name])
                state.previous = row["median"]
        return DetectionBatch(tuple(scores), tuple(events))
