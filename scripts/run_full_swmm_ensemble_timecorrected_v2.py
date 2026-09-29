"""Run the timestamp-corrected SWMM ensemble without touching legacy outputs.

Protocol changes relative to the legacy ensemble:
1. Apply one fixed -60 minute shift to every KMA hourly rainfall timestamp.
2. Start SWMM reporting one hour before each locked event start.
3. Keep event IDs, event ends, LHS ranges, LHS seed, and SWMM parameters fixed.

The extra reporting hour supports symmetric observed/simulated baseline removal.
GNN entry points must use ``run_timecorrected_entrypoint.py`` so the six
pre-event 10-minute steps are excluded from surrogate targets.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.rainfall import load_hourly_rainfall
from src.data.swmm_ensemble import SWMMEnsembleRunner


DEFAULT_INP = PROJECT / "inp_versions" / "seocho_imperv_landcover_strict.inp"
DEFAULT_CATALOG = PROJECT / "data" / "event_catalog.json"
DEFAULT_RAIN_DIR = PROJECT / "data" / "rainfall_raw"
DEFAULT_OUTPUT = PROJECT / "results" / "swmm_timecorrected_v2_gauge401_fix"


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inp", type=Path, default=DEFAULT_INP)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--rain-dir", type=Path, default=DEFAULT_RAIN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--n-sets", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rain-shift-minutes", type=int, default=-60)
    parser.add_argument("--report-lead-hours", type=float, default=1.0)
    parser.add_argument(
        "--event-ids",
        default="",
        help="Optional comma-separated event IDs for a pilot run.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_catalog(path: Path, lead_hours: float) -> tuple[list[dict], dict[str, str]]:
    source = json.loads(path.read_text(encoding="utf-8"))
    adjusted: list[dict] = []
    locked_starts: dict[str, str] = {}
    for item in source:
        event = copy.deepcopy(item)
        original_start = event["start"]
        locked_starts[event["event_id"]] = original_start
        timestamp = __import__("pandas").Timestamp(original_start)
        event["start"] = (
            timestamp - __import__("pandas").Timedelta(hours=lead_hours)
        ).strftime("%Y-%m-%d %H:%M:%S")
        event["_locked_evaluation_start"] = original_start
        adjusted.append(event)
    return adjusted, locked_starts


def protocol_record(
    args: argparse.Namespace,
    n_events: int,
    locked_starts: dict[str, str],
) -> dict:
    report_lead_steps = int(round(args.report_lead_hours * 6))
    return {
        "protocol_id": "paper002_gauge401_fix_v1",
        "created_at": now_iso(),
        "inp_version": "landcover_strict",
        "inp_path": str(args.inp.resolve()),
        "inp_sha256": sha256(args.inp),
        "catalog_path": str(args.catalog.resolve()),
        "catalog_sha256": sha256(args.catalog),
        "rain_dir": str(args.rain_dir.resolve()),
        "rainfall_timestamp_shift_minutes": args.rain_shift_minutes,
        "rainfall_station_id": 401,
        "event_specific_lag_optimisation": False,
        "report_lead_hours": args.report_lead_hours,
        "report_lead_steps": report_lead_steps,
        "report_interval_minutes": 10,
        "locked_evaluation_start_by_event": locked_starts,
        "symmetric_baseline_window": "[event start - 60 min, event start)",
        "n_events": n_events,
        "n_sets": args.n_sets,
        "n_total_runs": n_events * args.n_sets,
        "workers": args.workers,
        "seed": args.seed,
    }


def compatible(existing: dict, requested: dict) -> bool:
    keys = (
        "protocol_id",
        "inp_sha256",
        "catalog_sha256",
        "rainfall_timestamp_shift_minutes",
        "rainfall_station_id",
        "report_lead_hours",
        "report_lead_steps",
        "n_events",
        "n_sets",
        "seed",
    )
    return all(existing.get(key) == requested.get(key) for key in keys)


def main() -> int:
    args = parse_args()
    if args.workers < 1 or args.n_sets < 1:
        raise ValueError("workers and n_sets must be positive")
    if args.report_lead_hours < 0:
        raise ValueError("report-lead-hours must be non-negative")
    for required in (args.inp, args.catalog, args.rain_dir):
        if not required.exists():
            raise FileNotFoundError(required)

    catalog, locked_starts = load_catalog(args.catalog, args.report_lead_hours)
    selected = [value for value in args.event_ids.split(",") if value]
    if selected:
        selected_set = set(selected)
        catalog = [event for event in catalog if event["event_id"] in selected_set]
        missing = sorted(selected_set - {event["event_id"] for event in catalog})
        if missing:
            raise ValueError(f"Unknown event IDs: {missing}")
        locked_starts = {
            key: value for key, value in locked_starts.items() if key in selected_set
        }

    manifest = protocol_record(args, len(catalog), locked_starts)
    manifest_path = args.out_dir / "run_manifest.json"
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not compatible(prior, manifest):
            raise RuntimeError(
                f"Existing output has an incompatible protocol: {args.out_dir}"
            )
        manifest = prior
        manifest["resumed_at"] = now_iso()
        manifest["workers"] = args.workers
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    rainfall = load_hourly_rainfall(
        args.rain_dir,
        timestamp_shift_minutes=args.rain_shift_minutes,
    )
    print(
        f"Rainfall: {rainfall.index.min()} to {rainfall.index.max()} "
        f"(fixed shift {args.rain_shift_minutes:+d} min)",
        flush=True,
    )
    print(
        f"Ensemble: {len(catalog)} events x {args.n_sets} sets, "
        f"{args.workers} workers -> {args.out_dir}",
        flush=True,
    )

    runner = SWMMEnsembleRunner(
        inp_path=str(args.inp),
        rain_dat_path=str(args.rain_dir),
        output_dir=str(args.out_dir),
    )
    started = now_iso()
    summary = runner.run_ensemble(
        catalog=catalog,
        rain_series=rainfall,
        n=args.n_sets,
        seed=args.seed,
        max_workers=args.workers,
        ensemble_dir=args.out_dir,
        skip_existing=True,
    )
    summary.update(
        {
            "protocol_id": manifest["protocol_id"],
            "started_at": started,
            "completed_at": now_iso(),
            "output_dir": str(args.out_dir.resolve()),
        }
    )
    (args.out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["failed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
