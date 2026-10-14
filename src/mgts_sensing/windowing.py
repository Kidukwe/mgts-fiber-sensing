"""Build causal history windows without crossing measurement boundaries."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral, Real

import numpy as np

from .validation import Profile


@dataclass(frozen=True)
class ProfileWindow:
    """Exactly one compatible history, ordered from oldest to current."""

    profiles: tuple[Profile, ...]
    session_index: int


def _block_key(profile: Profile) -> tuple:
    return profile.source_file, profile.dts_id, profile.channel_id, profile.timezone_known


def _timestamp(profile: Profile) -> datetime:
    """Use UTC instants for aware time, keeping naive time in its local domain."""
    return (
        profile.timestamp.astimezone(timezone.utc)
        if profile.timezone_known else profile.timestamp
    )


def _validate_profile(profile: Profile) -> None:
    """Validate manually constructed profiles as well as ingestion output."""
    if not isinstance(profile, Profile):
        raise ValueError("Expected a Profile")  # noqa: TRY004 -- API uses ValueError
    if not isinstance(profile.timestamp, datetime):
        raise ValueError("Profile timestamp must be a datetime")  # noqa: TRY004
    if (profile.timestamp.utcoffset() is not None) != profile.timezone_known:
        raise ValueError("Timestamp awareness and timezone metadata disagree")
    for name, values in (("x", profile.x), ("y", profile.y)):
        if not isinstance(values, np.ndarray) or values.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional numeric array")
        if values.dtype.kind not in "iuf" or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite real numbers")
    if profile.x.size == 0 or profile.x.shape != profile.y.shape:
        raise ValueError("x and y must have the same nonempty shape")
    mask = profile.sentinel_mask
    if (
        not isinstance(mask, np.ndarray)
        or mask.dtype.kind != "b"
        or mask.shape != profile.x.shape
    ):
        raise ValueError("sentinel_mask must be a boolean array matching x and y")
    if np.any(profile.x[1:] <= profile.x[:-1]):
        raise ValueError("x must be strictly increasing")


def iter_windows(
    profiles: Iterable[Profile],
    *,
    history_size: int,
    max_gap_seconds: float,
) -> Iterator[ProfileWindow]:
    """Yield full windows; reset history at source, stream, grid or gap boundaries.

    Input must already be ordered within each source/stream block. Rejected
    ingestion records require a separate call so history cannot bridge them.
    """
    if isinstance(history_size, bool) or not isinstance(history_size, Integral):
        raise ValueError("history_size must be an integer >= 2")  # noqa: TRY004
    if history_size < 2:
        raise ValueError("history_size must be an integer >= 2")
    if (
        isinstance(max_gap_seconds, bool)
        or not isinstance(max_gap_seconds, Real)
        or not np.isfinite(max_gap_seconds)
        or max_gap_seconds <= 0
    ):
        raise ValueError("max_gap_seconds must be finite and positive")

    history: deque[Profile] = deque(maxlen=int(history_size))
    session_index = 0
    previous: Profile | None = None
    for profile in profiles:
        _validate_profile(profile)
        boundary = False
        if previous is not None:
            boundary = _block_key(profile) != _block_key(previous)
            if not boundary:
                seconds = (_timestamp(profile) - _timestamp(previous)).total_seconds()
                if seconds <= 0:
                    raise ValueError("Timestamps must strictly increase within a block")
                boundary = (
                    seconds > max_gap_seconds
                    or not np.array_equal(profile.x, previous.x)
                )
        if boundary:
            session_index += 1
            history.clear()
        history.append(profile)
        previous = profile
        if len(history) == history_size:
            yield ProfileWindow(tuple(history), session_index)
