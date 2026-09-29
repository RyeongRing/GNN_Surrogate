"""Create an immutable -60 minute rainfall-input mirror for GNN scripts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT / "data" / "rainfall_raw"
DEFAULT_OUTPUT = PROJECT / "data" / "rainfall_timecorrected_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shift-minutes", type=int, default=-60)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sources = sorted(args.source.glob("AWS_4*.csv"))
    if not sources:
        raise FileNotFoundError(f"No AWS_4*.csv files in {args.source}")
    manifest_path = args.output / "rainfall_transform_manifest.json"
    if args.output.exists() and any(args.output.iterdir()):
        if not manifest_path.exists():
            raise RuntimeError(f"Output directory is not empty: {args.output}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            item["name"]: item["output_sha256"] for item in manifest["files"]
        }
        if (
            manifest.get("timestamp_shift_minutes") == args.shift_minutes
            and all(
                (args.output / name).exists()
                and sha256(args.output / name) == digest
                for name, digest in expected.items()
            )
        ):
            print(f"Verified existing corrected rainfall mirror: {args.output}")
            return 0
        raise RuntimeError(
            "Existing corrected rainfall mirror does not match its manifest"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sources:
        frame = pd.read_csv(
            source,
            encoding="cp949",
            header=0,
            names=["stn_id", "stn_name", "datetime", "rain_mm"],
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
        if frame["datetime"].isna().any():
            raise ValueError(f"Invalid timestamps in {source}")
        frame["datetime"] = frame["datetime"] + pd.Timedelta(
            minutes=args.shift_minutes
        )
        destination = args.output / source.name
        frame.to_csv(
            destination,
            index=False,
            encoding="cp949",
            date_format="%Y-%m-%d %H:%M:%S",
        )
        records.append(
            {
                "name": source.name,
                "rows": int(len(frame)),
                "source_sha256": sha256(source),
                "output_sha256": sha256(destination),
                "first_shifted_timestamp": str(frame["datetime"].min()),
                "last_shifted_timestamp": str(frame["datetime"].max()),
            }
        )

    manifest = {
        "protocol_id": "paper002_rainfall_timecorrected_v2",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_directory": str(args.source.resolve()),
        "output_directory": str(args.output.resolve()),
        "timestamp_shift_minutes": args.shift_minutes,
        "event_specific_lag_optimisation": False,
        "files": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"Created {len(records)} shifted rainfall files in {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
