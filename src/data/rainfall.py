"""Shared rainfall loading utilities.

KMA hourly rainfall is timestamped at the end of the accumulation interval.
Corrected experiments can therefore request a fixed negative timestamp shift.
The shift is global and is never estimated separately for individual events.

Station selection: the study catchment (Seocho-gu) is served by KMA AWS
station 401 ("서초"/Seocho). A neighbouring station, AWS 400 ("강남"/Gangnam,
a different administrative district), reports overlapping timestamps for
June-September 2022-2024 in the AWS_400_401_*_summer.csv files. Station 401
is explicitly selected below; relying on drop_duplicates() row order to
resolve this overlap previously and silently preferred station 400 for
these months (audited 2026-08-24, see
timestamp_and_gauge_input_audit_20260824.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

STUDY_STATION_ID = 401


def load_hourly_rainfall(
    rain_dir: str | Path,
    timestamp_shift_minutes: int = 0,
    station_id: int = STUDY_STATION_ID,
) -> pd.Series:
    """Load the merged hourly rainfall record for a single explicit station,
    with an optional fixed shift. Defaults to AWS 401 (Seocho), the station
    co-located with the study catchment."""
    frames: list[pd.DataFrame] = []
    for path in sorted(Path(rain_dir).glob("AWS_4*.csv")):
        frame = pd.read_csv(
            path,
            encoding="cp949",
            header=0,
            names=["stn_id", "stn_name", "datetime", "rain_mm"],
        )
        frame = frame[frame["stn_id"] == station_id]
        if frame.empty:
            continue
        frame["rain_mm"] = (
            pd.to_numeric(frame["rain_mm"], errors="coerce")
            .fillna(0)
            .clip(lower=0)
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        frames.append(frame.dropna(subset=["datetime"])[["datetime", "rain_mm"]])

    if not frames:
        raise FileNotFoundError(
            f"No AWS_4*.csv rainfall rows for station {station_id} found in {rain_dir}"
        )

    combined = (
        pd.concat(frames)
        .drop_duplicates("datetime")
        .sort_values("datetime")
    )
    if combined["datetime"].duplicated().any():
        raise RuntimeError(
            f"Duplicate timestamps remain for station {station_id} in {rain_dir} "
            "after filtering; expected exactly one row per timestamp for a single station."
        )
    hourly = (
        combined.set_index("datetime")["rain_mm"]
        .resample("1h")
        .sum()
        .fillna(0)
    )
    if timestamp_shift_minutes:
        hourly.index = hourly.index + pd.Timedelta(
            minutes=int(timestamp_shift_minutes)
        )
        hourly = hourly.sort_index()
    hourly.attrs["timestamp_shift_minutes"] = int(timestamp_shift_minutes)
    hourly.attrs["station_id"] = int(station_id)
    return hourly


def rainfall_shift_from_config(data_config: dict) -> int:
    """Return the fixed rainfall shift declared by a data configuration."""
    return int(data_config.get("rainfall_timestamp_shift_minutes", 0))
