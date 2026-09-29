"""Robust loading utilities for Seoul sewer water-level archives.

The yearly ZIP archives use multiple CSV schemas and encodings. Older files
can be headerless, while newer files can contain duplicate measurement-date
columns. This module normalises all supported variants to station_id,
timestamp, and water_level_m.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from typing import Iterable

import pandas as pd


HEADERLESS_COLUMNS = [
    "고유번호",
    "구분코드",
    "구분명",
    "oracle_date",
    "측정일자",
    "측정수위",
    "통신상태",
]
STATION_ID_RE = re.compile(r"^\d{2}-\d{4}$")


def find_year_archive(obs_dir: Path, year: int) -> Path | None:
    """Return the preferred archive for a year, if present."""
    for name in (
        f"하수관로_수위_현황_{year}.zip",
        f"하수관로_수위_현황_{year} (1).zip",
    ):
        candidate = obs_dir / name
        if candidate.exists():
            return candidate
    return None


def _detect_encoding(raw: bytes) -> str:
    """Detect header encoding; headerless data are parsed loss-tolerantly."""
    head = raw[:8192]
    if bytes.fromhex("eab3a0ec9ca0ebb288ed98b8") in head:
        return "utf-8-sig"
    if bytes.fromhex("b0edc0afb9f8c8a3") in head:
        return "cp949"
    return "utf-8-sig"


def _first_row(raw: bytes, encoding: str) -> list[str]:
    text = raw[:8192].decode(encoding, errors="replace")
    first_line = next((line for line in text.splitlines() if line.strip()), "")
    if not first_line:
        return []
    return next(csv.reader([first_line]))


def _clean_column_name(value: object) -> str:
    return str(value).lstrip("\ufeff?").strip().strip('"')


def _find_column(columns: Iterable[object], exact: str, contains: str) -> str | None:
    cleaned = [_clean_column_name(column) for column in columns]
    if exact in cleaned:
        return exact
    return next((column for column in cleaned if contains in column), None)


def _parse_timestamps(values: pd.Series) -> pd.Series:
    """Parse legacy and ISO timestamps without format-inference warnings."""
    return pd.to_datetime(
        values.astype(str).str.strip(),
        format="mixed",
        errors="coerce",
    )


def _select_timestamp_column(frame: pd.DataFrame) -> pd.Series:
    candidates = [
        column
        for column in frame.columns
        if _clean_column_name(column).startswith("측정일자")
        or _clean_column_name(column) in {"oracle_date", "unix_date"}
    ]
    if not candidates:
        raise ValueError("No measurement-date column was found")

    best_values = _parse_timestamps(frame[candidates[0]])
    best_count = int(best_values.notna().sum())
    for column in candidates[1:]:
        values = _parse_timestamps(frame[column])
        count = int(values.notna().sum())
        # Prefer the later duplicate date column when parse coverage is tied.
        if count >= best_count:
            best_values = values
            best_count = count
    return best_values


def parse_observation_csv(
    raw: bytes,
    stations: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Parse one archive member and return normalised long observations."""
    encoding = _detect_encoding(raw)
    first_row = [_clean_column_name(value) for value in _first_row(raw, encoding)]
    has_header = "고유번호" in first_row or "측정수위" in first_row
    if not has_header:
        # IDs, timestamps, and numeric levels are ASCII in legacy rows. Invalid
        # bytes occur only in descriptive Korean fields that are not analysed.
        encoding = "utf-8-sig"

    read_kwargs = {
        "encoding": encoding,
        "encoding_errors": "replace",
        "low_memory": False,
        "on_bad_lines": "skip",
    }
    if has_header:
        frame = pd.read_csv(io.BytesIO(raw), **read_kwargs)
    else:
        frame = pd.read_csv(
            io.BytesIO(raw),
            header=None,
            names=HEADERLESS_COLUMNS,
            **read_kwargs,
        )

    frame.columns = [_clean_column_name(column) for column in frame.columns]
    station_column = _find_column(frame.columns, "고유번호", "고유번호")
    level_column = _find_column(frame.columns, "측정수위", "측정수위")
    if station_column is None or level_column is None:
        raise ValueError("Required station or water-level column was not found")

    station_ids = frame[station_column].astype(str).str.strip().str.strip('"')
    station_set = set(stations) if stations is not None else None
    keep = station_ids.str.match(STATION_ID_RE)
    if station_set is not None:
        keep &= station_ids.isin(station_set)
    if not bool(keep.any()):
        return pd.DataFrame(columns=["station_id", "timestamp", "water_level_m"])

    selected = frame.loc[keep].copy()
    timestamps = _select_timestamp_column(selected)
    levels = pd.to_numeric(selected[level_column], errors="coerce")
    result = pd.DataFrame(
        {
            "station_id": station_ids.loc[keep].to_numpy(),
            "timestamp": timestamps.to_numpy(),
            "water_level_m": levels.to_numpy(),
        }
    )
    result = result.dropna(subset=["timestamp", "water_level_m"])
    return result.sort_values(["timestamp", "station_id"]).reset_index(drop=True)


def load_monthly_observations(
    obs_dir: Path,
    year: int,
    month: int,
    stations: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Load one calendar month from a yearly ZIP archive."""
    archive = find_year_archive(obs_dir, year)
    if archive is None:
        return pd.DataFrame(columns=["station_id", "timestamp", "water_level_m"])

    month_pattern = f"{year}{month:02d}"
    with zipfile.ZipFile(archive) as handle:
        members = [
            member
            for member in handle.infolist()
            if month_pattern in member.filename
            and member.filename.lower().endswith((".csv", ".txt"))
            and "__macosx" not in member.filename.lower()
        ]
        if not members:
            return pd.DataFrame(
                columns=["station_id", "timestamp", "water_level_m"]
            )
        raw = handle.read(sorted(members, key=lambda item: item.filename)[0])
    return parse_observation_csv(raw, stations=stations)


def load_observation_window(
    obs_dir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    stations: Iterable[str] | None = None,
    cache: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Load a raw observation window as timestamp-by-station values."""
    start = pd.Timestamp(start)
    end = pd.Timestamp(end)
    if end < start:
        raise ValueError("Observation window end precedes start")

    cache = cache if cache is not None else {}
    parts: list[pd.DataFrame] = []
    for period in pd.period_range(start.to_period("M"), end.to_period("M"), freq="M"):
        key = f"{period.year}{period.month:02d}"
        if key not in cache:
            cache[key] = load_monthly_observations(
                obs_dir,
                period.year,
                period.month,
                stations=stations,
            )
        monthly = cache[key]
        if not monthly.empty:
            parts.append(monthly)
    if not parts:
        return pd.DataFrame()

    long = pd.concat(parts, ignore_index=True)
    long = long.loc[
        (long["timestamp"] >= start) & (long["timestamp"] <= end)
    ].copy()
    if long.empty:
        return pd.DataFrame()
    long = long.drop_duplicates(["timestamp", "station_id"], keep="last")
    wide = long.pivot(
        index="timestamp",
        columns="station_id",
        values="water_level_m",
    )
    return wide.sort_index()


