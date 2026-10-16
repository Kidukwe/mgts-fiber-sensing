"""Streaming CSV ingestion. Yield every row, including explicitly rejected rows."""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable, Iterator
from pathlib import Path

from mgts_sensing.validation import (
    REQUIRED_COLUMNS,
    Issue,
    ProfileResult,
    add_warning,
    parse_profile,
    record_key,
)


def iter_profiles(paths: Iterable[str | Path], *, naive_utc_offset_hours: int | None = None) -> Iterator[ProfileResult]:
    """Read files in caller order; track duplicate keys across this entire invocation.

    Physical line_number is 1-based (header is line 1); row_number is a 1-based
    data-record index. Multiline records keep their first physical line number.
    A file/header failure has row_number=0. Unrecoverable CSV syntax stops that
    file with an explicit CSV_READ_ERROR, then reading continues with the next file.
    Only keys and one grid fingerprint per stream are kept, not all profiles.
    """
    csv.field_size_limit(2**31 - 1)
    seen = {}
    previous_grids = {}
    for source in paths:
        source = Path(source)
        try:
            handle = source.open(newline="", encoding="utf-8-sig")
        except OSError as exc:
            issue = Issue(source, 0, 0, "error", "FILE_READ_ERROR", str(exc))
            yield ProfileResult(source, 0, 0, None, (issue,))
            continue
        with handle:
            reader = csv.reader(handle, strict=True)
            row_number = 0
            line_number = 1
            try:
                header = next(reader, None)
                if header is None:
                    issue = Issue(source, 0, 1, "error", "EMPTY_FILE", "CSV has no header")
                    yield ProfileResult(source, 0, 1, None, (issue,))
                    continue
                schema_errors = []
                if len(header) != len(set(header)):
                    schema_errors.append(("DUPLICATE_COLUMN", "Header contains duplicate column names"))
                for name in REQUIRED_COLUMNS:
                    if name not in header:
                        schema_errors.append(("MISSING_COLUMN", f"Header lacks required column {name!r}"))
                if schema_errors:
                    issues = tuple(
                        Issue(source, 0, 1, "error", code, text) for code, text in schema_errors
                    )
                    yield ProfileResult(source, 0, 1, None, issues)
                while True:
                    line_number = reader.line_num + 1
                    values = next(reader, None)
                    if values is None:
                        break
                    row_number += 1
                    if schema_errors:
                        issues = tuple(
                            Issue(source, row_number, line_number, "error", code, text)
                            for code, text in schema_errors
                        )
                        yield ProfileResult(source, row_number, line_number, None, issues)
                        continue
                    if len(values) != len(header):
                        issue = Issue(
                            source, row_number, line_number, "error", "CSV_COLUMN_COUNT",
                            f"Header has {len(header)} columns, record has {len(values)}",
                        )
                        yield ProfileResult(source, row_number, line_number, None, (issue,))
                        continue
                    row = dict(zip(header, values))
                    result = parse_profile(row, source, row_number, line_number,
                                           naive_utc_offset_hours=naive_utc_offset_hours)
                    key = record_key(row, naive_utc_offset_hours=naive_utc_offset_hours)
                    if key is not None:
                        if key in seen:
                            first_file, first_line = seen[key]
                            result = add_warning(
                                result, "DUPLICATE_KEY",
                                f"Same sensor/channel/instant as {first_file}:{first_line}",
                            )
                        else:
                            seen[key] = (source, line_number)
                    if result.profile is not None:
                        profile = result.profile
                        # Quality diagnostics, not a requirement to share a fixed grid.
                        stream = (profile.dts_id, profile.channel_id, profile.timezone_known)
                        grid = hashlib.sha256(profile.x.tobytes()).hexdigest()
                        previous = previous_grids.get(stream)
                        if previous is not None:
                            old_length, old_grid = previous
                            if len(profile.x) != old_length:
                                result = add_warning(
                                    result, "PROFILE_LENGTH_CHANGED",
                                    f"Previous profile length {old_length}, current {len(profile.x)}",
                                )
                            if grid != old_grid:
                                result = add_warning(
                                    result, "GRID_CHANGED", "x grid differs from previous loaded profile"
                                )
                        previous_grids[stream] = (len(profile.x), grid)
                    yield result
            except (csv.Error, UnicodeError, OSError) as exc:
                issue = Issue(
                    source, row_number + 1, line_number, "error", "CSV_READ_ERROR",
                    f"Cannot safely recover CSV record boundaries; remainder unread: {exc}",
                )
                yield ProfileResult(source, row_number + 1, line_number, None, (issue,))
