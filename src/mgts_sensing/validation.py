"""Parse a complete profile or return explicit errors; never drop individual points."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

REQUIRED_COLUMNS = ("dts_id", "channel_id", "timestamp", "points")
SENTINEL = -300.0
_OFFSET = re.compile(r"(Z|[+-]\d{2}(?::?\d{2})?(?::?\d{2}(?:\.\d+)?)?)$")
_SHORT_OFFSET = re.compile(r"([+-]\d{2})$")
_TIME_PREFIX = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}")


@dataclass(frozen=True)
class Issue:
    source_file: Path
    row_number: int
    line_number: int
    severity: Literal["error", "warning"]
    code: str
    description: str


@dataclass(frozen=True)
class Profile:
    source_file: Path
    row_number: int
    line_number: int
    dts_id: str
    channel_id: str
    timestamp: datetime
    timestamp_raw: str
    timezone_raw: str | None
    utc_offset_seconds: float | None
    timestamp_format_normalized: bool
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    sentinel_mask: NDArray[np.bool_]
    points_size: int | None
    quality_flags: tuple[str, ...] = ()

    @property
    def timezone_known(self) -> bool:
        return self.utc_offset_seconds is not None


@dataclass(frozen=True)
class ProfileResult:
    """One outcome per CSV record; header/file failures use row_number=0."""

    source_file: Path
    row_number: int
    line_number: int
    profile: Profile | None
    issues: tuple[Issue, ...]

    @property
    def loaded(self) -> bool:
        return self.profile is not None


def parse_timestamp(raw: str, *, naive_utc_offset_hours: int | None = None) -> tuple[datetime, str | None, bool]:
    """Preserve naive time; support PostgreSQL short offsets and ISO Z on Python 3.10."""
    value = raw.strip()
    if not _TIME_PREFIX.match(value):
        raise ValueError("Expected a datetime with date and time, not a date alone")
    match = _OFFSET.search(value)
    timezone_raw = match.group(1) if match else None
    if timezone_raw and timezone_raw != "Z":
        digits = timezone_raw[1:].split(".")[0].replace(":", "")
        parts = [int(digits[i:i + 2]) for i in range(0, len(digits), 2)]
        if parts[0] > 23 or any(part > 59 for part in parts[1:]):
            raise ValueError("Invalid timezone offset")
    normalized = _SHORT_OFFSET.sub(r"\1:00", value)
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    timestamp = datetime.fromisoformat(normalized)
    if timestamp.tzinfo is None and naive_utc_offset_hours is not None:
        timestamp = timestamp.replace(tzinfo=timezone(timedelta(hours=naive_utc_offset_hours)))
    return timestamp, timezone_raw, normalized != raw


def profile_key(profile: Profile) -> tuple[str, str, bool, datetime]:
    """Known offsets compare as UTC instants; naive times remain a separate domain."""
    timestamp = profile.timestamp
    if profile.timezone_known:
        timestamp = timestamp.astimezone(timezone.utc)
    return profile.dts_id, profile.channel_id, profile.timezone_known, timestamp


def record_key(row: Mapping[str, str | None], *, naive_utc_offset_hours: int | None = None) -> tuple[str, str, bool, datetime] | None:
    """Keys can still be checked for a row whose points failed to parse."""
    if not all(str(row.get(name) or "").strip() for name in REQUIRED_COLUMNS[:3]):
        return None
    try:
        timestamp, _, _ = parse_timestamp(row["timestamp"], naive_utc_offset_hours=naive_utc_offset_hours)
    except ValueError:
        return None
    known = timestamp.utcoffset() is not None
    if known:
        timestamp = timestamp.astimezone(timezone.utc)
    return row["dts_id"], row["channel_id"], known, timestamp


def parse_profile(
    row: Mapping[str, str | None],
    source_file: str | Path,
    row_number: int,
    line_number: int,
    *,
    naive_utc_offset_hours: int | None = None,
) -> ProfileResult:
    """Structural errors reject the entire row. Quality warnings preserve all values."""
    source_file = Path(source_file)
    issues: list[Issue] = []

    def add(severity: Literal["error", "warning"], code: str, description: str) -> None:
        issues.append(Issue(source_file, row_number, line_number, severity, code, description))

    missing = [name for name in REQUIRED_COLUMNS if not str(row.get(name) or "").strip()]
    for name in missing:
        add("error", "MISSING_REQUIRED_FIELD", f"Required field {name!r} is absent or empty")

    timestamp = None
    timezone_raw = None
    normalized = False
    if "timestamp" not in missing:
        try:
            timestamp, timezone_raw, normalized = parse_timestamp(row["timestamp"], naive_utc_offset_hours=naive_utc_offset_hours)
            if timezone_raw is None and naive_utc_offset_hours is not None:
                add("warning", "TIMEZONE_ASSIGNED", "Naive time assigned confirmed fixed UTC offset")
            if timestamp.tzinfo is None:
                add("warning", "UNKNOWN_TIMEZONE", "Timestamp has no timezone; none was assigned")
        except ValueError:
            add("error", "INVALID_TIMESTAMP", "Timestamp cannot be parsed as an ISO datetime")

    points = None
    if "points" not in missing:
        try:
            points = json.loads(row["points"])
        except (ValueError, TypeError, RecursionError):
            add("error", "INVALID_JSON", "points is not a JSON value")
    if points is not None and not isinstance(points, list):
        add("error", "INVALID_POINTS_STRUCTURE", "points must be a list of numeric pairs")
        points = None
    elif points is None and "points" not in missing and not any(
        issue.code == "INVALID_JSON" for issue in issues
    ):
        add("error", "INVALID_POINTS_STRUCTURE", "points must be a list, not null")

    arrays = None
    if isinstance(points, list):
        if not points:
            add("error", "EMPTY_PROFILE", "points contains no samples")
        # Report every invalid pair, without coercing numeric strings, bools or nulls.
        for index, pair in enumerate(points):
            if not isinstance(pair, list) or len(pair) != 2:
                add("error", "INVALID_POINT_PAIR", f"points[{index}] must have exactly two items")
                continue
            if any(type(value) not in (int, float) for value in pair):
                add("error", "NONNUMERIC_POINT", f"points[{index}] contains a nonnumeric value")
                continue
            try:
                finite = all(np.isfinite(float(value)) for value in pair)
            except OverflowError:
                finite = False
            if not finite:
                add("error", "NONFINITE_POINT", f"points[{index}] contains NaN, Inf or overflow")
        if points and not any(issue.code in {
            "INVALID_POINT_PAIR", "NONNUMERIC_POINT", "NONFINITE_POINT"
        } for issue in issues):
            arrays = np.asarray(points, dtype=np.float64)

    declared_size = None
    declared_raw = row.get("points_size")
    if declared_raw is not None:
        try:
            declared_size = int(declared_raw)
            if declared_size < 0:
                raise ValueError
        except ValueError:
            add("warning", "INVALID_POINTS_SIZE", "points_size must be a nonnegative integer")
            declared_size = None
        if declared_size is not None and isinstance(points, list) and declared_size != len(points):
            add("warning", "POINTS_SIZE_MISMATCH", f"Declared {declared_size}, actual {len(points)}")

    if arrays is not None:
        x = arrays[:, 0].copy()
        y = arrays[:, 1].copy()
        mask = y == SENTINEL
        if np.any(x[1:] <= x[:-1]):
            add("warning", "NONINCREASING_X", "x is not strictly increasing; original order kept")
        if mask.any():
            add("warning", "SUSPECT_SENTINEL", f"{int(mask.sum())} y values equal -300; confirmed fiber end, excluded from analytics")

    if any(issue.severity == "error" for issue in issues):
        return ProfileResult(source_file, row_number, line_number, None, tuple(issues))

    # Arrays are read-only to make accidental in-place preprocessing explicit.
    for array in (x, y, mask):
        array.setflags(write=False)
    offset = timestamp.utcoffset()
    profile = Profile(
        source_file=source_file,
        row_number=row_number,
        line_number=line_number,
        dts_id=row["dts_id"],
        channel_id=row["channel_id"],
        timestamp=timestamp,
        timestamp_raw=row["timestamp"],
        timezone_raw=timezone_raw,
        utc_offset_seconds=offset.total_seconds() if offset is not None else None,
        timestamp_format_normalized=normalized,
        x=x,
        y=y,
        sentinel_mask=mask,
        points_size=declared_size,
        quality_flags=tuple(issue.code for issue in issues),
    )
    return ProfileResult(source_file, row_number, line_number, profile, tuple(issues))


def add_warning(result: ProfileResult, code: str, description: str) -> ProfileResult:
    """Attach contextual warnings to the outcome and its loaded profile."""
    issue = Issue(
        result.source_file, result.row_number, result.line_number, "warning", code, description
    )
    profile = (
        replace(result.profile, quality_flags=(*result.profile.quality_flags, code))
        if result.profile is not None else None
    )
    return replace(result, profile=profile, issues=(*result.issues, issue))
