"""Re-evaluate 50 SWMM sets with fixed rainfall timing and symmetric baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.evaluate_calibration_50sets import (
    aggregate_set,
    fisher_mean_correlations,
    finite,
    pair_metrics,
    pearson_r,
)
from src.data.observation_loader import load_observation_window


CATALOG = PROJECT / "data" / "event_catalog.json"
MAPPING = PROJECT / "data" / "station_node_mapping.csv"
OBS_DIR = PROJECT / "data" / "observations"
DEFAULT_ENSEMBLE = PROJECT / "results" / "swmm_timecorrected_v2_gauge401_fix"
DEFAULT_OUTPUT = PROJECT / "results" / "timecorrected_v2" / "calibration"
DEFAULT_DATA_OUTPUT = PROJECT / "data" / "timecorrected_v2"
REPORT_INTERVAL = pd.Timedelta(minutes=10)
BASELINE_HOURS = 1
MIN_BASELINE_SAMPLES = 30


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-dir", type=Path, default=DEFAULT_ENSEMBLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-output-dir", type=Path, default=DEFAULT_DATA_OUTPUT)
    parser.add_argument("--n-sets", type=int, default=50)
    parser.add_argument("--observations", type=Path, default=OBS_DIR)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    return parser.parse_args()


def prepare_pairs(
    event: dict,
    ensemble_dir: Path,
    stations: list[str],
    station_node: dict[str, str],
    observation_cache: dict[str, pd.DataFrame],
) -> tuple[list[dict], dict]:
    event_id = event["event_id"]
    start = pd.Timestamp(event["start"])
    report_start = start - pd.Timedelta(hours=BASELINE_HOURS)
    evaluation_end = pd.Timestamp(event["end"]) + pd.Timedelta(hours=6)
    reference_path = ensemble_dir / event_id / "000.npz"
    if not reference_path.exists():
        return [], {"status": "missing_reference_npz"}

    with np.load(reference_path, allow_pickle=True) as archive:
        names = [str(value) for value in archive["node_names"]]
        n_steps = int(archive["node_depth"].shape[1])
    times = pd.date_range(
        report_start,
        periods=n_steps,
        freq=REPORT_INTERVAL,
    )
    if not len(times) or times[0] != report_start:
        raise RuntimeError(f"{event_id}: invalid SWMM time axis")
    endpoint_delta_minutes = float(
        (evaluation_end - times[-1]).total_seconds() / 60.0
    )
    if endpoint_delta_minutes < -1e-9:
        raise RuntimeError(f"{event_id}: SWMM output exceeds locked evaluation end")

    node_index = {name: index for index, name in enumerate(names)}
    simulation_baseline_indices = np.flatnonzero(
        (times >= report_start) & (times < start)
    )
    if len(simulation_baseline_indices) < 2:
        raise RuntimeError(f"{event_id}: insufficient simulated baseline steps")

    observations = load_observation_window(
        OBS_DIR,
        report_start,
        evaluation_end,
        stations=stations,
        cache=observation_cache,
    )
    if observations.empty:
        return [], {"status": "no_observations"}

    pairs: list[dict] = []
    for station in stations:
        node_id = station_node[station]
        if station not in observations or node_id not in node_index:
            continue
        series = observations[station]
        baseline_samples = series.loc[
            (series.index >= report_start) & (series.index < start)
        ].dropna()
        if len(baseline_samples) < MIN_BASELINE_SAMPLES:
            continue
        observation_baseline = float(
            np.median(baseline_samples.to_numpy(dtype=float))
        )
        evaluation = series.loc[
            (series.index >= start) & (series.index <= evaluation_end)
        ]
        observation_10min = evaluation.resample(REPORT_INTERVAL).mean()
        common = observation_10min.index.intersection(times)
        observation_values = observation_10min.loc[common].to_numpy(dtype=float)
        valid = np.isfinite(observation_values)
        if int(valid.sum()) < 6:
            continue
        common = common[valid]
        observation_delta = observation_values[valid] - observation_baseline
        time_indices = times.get_indexer(common)
        if np.any(time_indices < 0):
            raise RuntimeError(f"{event_id}/{station}: timestamp mismatch")
        pairs.append(
            {
                "event_id": event_id,
                "station_id": station,
                "node_id": node_id,
                "node_index": node_index[node_id],
                "times": common,
                "time_indices": time_indices,
                "simulation_baseline_indices": simulation_baseline_indices,
                "observation_delta": observation_delta,
            }
        )

    qc = {
        "status": "valid" if pairs else "no_eligible_pairs",
        "n_pairs": len(pairs),
        "stations": [pair["station_id"] for pair in pairs],
        "npz_periods": n_steps,
        "npz_first_time": str(times[0]),
        "npz_last_time": str(times[-1]),
        "locked_event_start": str(start),
        "event_end_plus_6h": str(evaluation_end),
        "endpoint_delta_minutes": endpoint_delta_minutes,
    }
    return pairs, qc


def main() -> int:
    global OBS_DIR, CATALOG, MAPPING
    args = parse_args()
    OBS_DIR, CATALOG, MAPPING = args.observations, args.catalog, args.mapping
    manifest_path = args.ensemble_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required = {
        "rainfall_timestamp_shift_minutes": -60,
        "report_lead_steps": 6,
        "event_specific_lag_optimisation": False,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in required.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Corrected protocol mismatch: {mismatches}")

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    events = [item for item in catalog if item.get("split") == "calib"]
    mapping = pd.read_csv(MAPPING)
    mapping["station_id"] = mapping["station_id"].astype(str)
    mapping["swmm_node"] = mapping["swmm_node"].astype(str)
    stations = sorted(mapping["station_id"].tolist())
    station_node = dict(zip(mapping["station_id"], mapping["swmm_node"]))

    observation_cache: dict[str, pd.DataFrame] = {}
    pairs_by_event: dict[str, list[dict]] = {}
    event_qc: dict[str, dict] = {}
    for event in events:
        pairs, qc = prepare_pairs(
            event,
            args.ensemble_dir,
            stations,
            station_node,
            observation_cache,
        )
        if pairs:
            pairs_by_event[event["event_id"]] = pairs
        event_qc[event["event_id"]] = qc
        print(f"prepared {event['event_id']}: {len(pairs)} station pairs")
    if not pairs_by_event:
        raise RuntimeError("No eligible calibration station-event pairs")

    summaries: list[dict] = []
    all_pair_rows: list[dict] = []
    station_set_correlations: dict[str, list[float]] = defaultdict(list)
    for set_id in range(args.n_sets):
        set_rows: list[dict] = []
        station_obs: dict[str, list[np.ndarray]] = defaultdict(list)
        station_sim: dict[str, list[np.ndarray]] = defaultdict(list)
        for event in events:
            pairs = pairs_by_event.get(event["event_id"], [])
            if not pairs:
                continue
            path = args.ensemble_dir / event["event_id"] / f"{set_id:03d}.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=True) as archive:
                depths = np.asarray(archive["node_depth"], dtype=float)
                for pair in pairs:
                    node_series = depths[pair["node_index"]]
                    simulation_baseline = float(
                        np.median(
                            node_series[pair["simulation_baseline_indices"]]
                        )
                    )
                    simulation_delta = (
                        node_series[pair["time_indices"]] - simulation_baseline
                    )
                    metrics = pair_metrics(
                        pair["times"],
                        pair["observation_delta"],
                        simulation_delta,
                    )
                    metrics.update(
                        {
                            "set_id": set_id,
                            "event_id": event["event_id"],
                            "station_id": pair["station_id"],
                        }
                    )
                    set_rows.append(metrics)
                    all_pair_rows.append(metrics)
                    station_obs[pair["station_id"]].append(
                        pair["observation_delta"]
                    )
                    station_sim[pair["station_id"]].append(simulation_delta)

        aggregated = aggregate_set(set_rows, set_id)
        valid_correlations = finite([row["pearson_r"] for row in set_rows])
        aggregated["mean_r"] = (
            float(np.mean(valid_correlations))
            if len(valid_correlations)
            else np.nan
        )
        aggregated["median_r"] = (
            float(np.median(valid_correlations))
            if len(valid_correlations)
            else np.nan
        )
        summaries.append(aggregated)
        for station in stations:
            if station not in station_obs:
                station_set_correlations[station].append(np.nan)
                continue
            station_set_correlations[station].append(
                pearson_r(
                    np.concatenate(station_obs[station]),
                    np.concatenate(station_sim[station]),
                )
            )
        print(
            f"set {set_id:02d}: median NSE={aggregated['nse_median']:.3f}, "
            f"pooled NSE={aggregated['pooled_nse_station_event_centered']:.3f}, "
            f"mean r={aggregated['mean_r']:.3f}"
        )

    ranking = pd.DataFrame(summaries)
    ranking = ranking.sort_values(
        ["mean_r", "nse_median", "pooled_nse_station_event_centered"],
        ascending=False,
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    threshold = float(ranking["mean_r"].quantile(0.50))
    ranking["behavioral"] = ranking["mean_r"] >= threshold
    behavioral = sorted(
        ranking.loc[ranking["behavioral"], "set_id"].astype(int).tolist()
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_output_dir.mkdir(parents=True, exist_ok=True)
    ranking_path = args.output_dir / "calibration_multimetric_50sets.csv"
    pair_path = args.output_dir / "calibration_station_event_metrics.csv"
    ranking.to_csv(ranking_path, index=False)
    pd.DataFrame(all_pair_rows).to_csv(pair_path, index=False)
    ranking[["set_id", "mean_r", "n_station_event_pairs", "rank", "behavioral"]].to_csv(
        args.data_output_dir / "calibration_summary.csv",
        index=False,
    )
    pd.DataFrame(all_pair_rows).to_csv(
        args.data_output_dir / "calibration_scores.csv",
        index=False,
    )

    station_audit = []
    for station in stations:
        correlations = station_set_correlations[station]
        station_audit.append(
            {
                "station_id": station,
                "swmm_node": station_node[station],
                "fisher_mean_r_across_sets": fisher_mean_correlations(
                    correlations
                ),
                "n_sets_with_r": int(len(finite(correlations))),
            }
        )
    pd.DataFrame(station_audit).to_csv(
        args.output_dir / "station_mapping_performance_audit.csv",
        index=False,
    )

    behavioral_record = {
        "protocol_id": "paper002_timecorrected_v2",
        "threshold_method": "50th percentile of corrected per-set mean Pearson r",
        "threshold_r": threshold,
        "n_behavioral": len(behavioral),
        "n_total": int(len(ranking)),
        "behavioral_set_ids": behavioral,
        "non_behavioral_set_ids": sorted(
            set(range(args.n_sets)) - set(behavioral)
        ),
        "calibration_only": True,
    }
    (args.data_output_dir / "behavioral_sets.json").write_text(
        json.dumps(behavioral_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    audit = {
        "protocol_id": "paper002_timecorrected_v2",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "ensemble_manifest_sha256": sha256(manifest_path),
        "rainfall_timestamp_shift_minutes": -60,
        "event_specific_lag_optimisation": False,
        "observation_baseline": "median over [start - 60 min, start)",
        "simulation_baseline": "median over [start - 60 min, start)",
        "evaluation_window": "[locked event start, event end + 6 h]",
        "n_calibration_events": len(events),
        "n_usable_events": len(pairs_by_event),
        "best_set_id": int(ranking.iloc[0]["set_id"]),
        "best_mean_r": float(ranking.iloc[0]["mean_r"]),
        "best_median_nse": float(ranking.iloc[0]["nse_median"]),
        "best_pooled_nse": float(
            ranking.iloc[0]["pooled_nse_station_event_centered"]
        ),
        "behavioral_screening": behavioral_record,
        "event_qc": event_qc,
    }
    (args.output_dir / "calibration_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
