"""Compute raw storm-surge SWMM performance for the timestamp+gauge-corrected
ensemble (results/swmm_timecorrected_v2_gauge401_fix), for fig_swmm_validation
regeneration.

Same protocol as compute_swmm_raw_performance.py, adapted for two differences
in the corrected-v2 ensemble:
  1. Each NPZ's first timestep is (event start - 1h), not event start
     (report_lead_hours=1.0 in run_full_swmm_ensemble_timecorrected_v2.py),
     so the SWMM time axis must be shifted back by 1h to align.
  2. Simulated baseline uses the symmetric pre-event 1h median (matching the
     observed baseline convention), not just the single first timestep.

Output: results/timecorrected_v2/swmm_raw_performance_gauge401fix.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))
from src.data.observation_loader import load_observation_window

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[1]
_PAPER002 = BASE.parent / "paper_002"
MAPPING_CSV = BASE / "data" / "station_node_mapping.csv"
ENS_DIR = BASE / "results" / "swmm_timecorrected_v2_gauge401_fix"
CATALOG = BASE / "data" / "event_catalog.json"
OBS_DIR = _PAPER002 / "서초구 하수관망"
REPORT_INTERVAL = pd.Timedelta(minutes=10)
REPORT_LEAD_HOURS = 1.0
MIN_BASELINE_SAMPLES = 30
_OBS_WINDOW_CACHE: dict[str, pd.DataFrame] = {}

mapping_df = pd.read_csv(MAPPING_CSV)
mapping_df["swmm_node"] = mapping_df["swmm_node"].astype(str)
STATION_NODE = dict(zip(mapping_df["station_id"], mapping_df["swmm_node"]))
STATIONS = sorted(STATION_NODE.keys())

TARGET_SETS = [7, 31, 46]


def load_monthly_obs(year: str, month: str) -> pd.DataFrame:
    start = pd.Timestamp(int(year), int(month), 1)
    end = start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=59)
    return load_observation_window(
        OBS_DIR, start, end, stations=STATIONS, cache=_OBS_WINDOW_CACHE
    )


def compute_station_metrics(event: dict, obs_month: pd.DataFrame | None, set_id: int):
    del obs_month
    event_id = event["event_id"]
    start = pd.Timestamp(event["start"])
    report_start = start - pd.Timedelta(hours=REPORT_LEAD_HOURS)
    end = pd.Timestamp(event["end"]) + pd.Timedelta(hours=6)
    npz_path = ENS_DIR / event_id / f"{set_id:03d}.npz"
    if not npz_path.exists():
        return {}

    with np.load(npz_path, allow_pickle=True) as data:
        node_names = [str(value) for value in data["node_names"]]
        depths = data["node_depth"].astype(float)
    times = pd.date_range(report_start, periods=depths.shape[1], freq=REPORT_INTERVAL)
    if len(times) == 0 or times[-1] > end:
        raise RuntimeError(f"{event_id}: NPZ time axis exceeds event end + 6 h")
    node_index = {node: index for index, node in enumerate(node_names)}

    raw = load_observation_window(
        OBS_DIR, report_start, end, stations=STATIONS, cache=_OBS_WINDOW_CACHE,
    )
    if raw.empty:
        return {}

    result = {}
    for station, node_id in STATION_NODE.items():
        if event.get("split") == "test_temporal" and station == "22-0009":
            continue
        if station not in raw.columns or node_id not in node_index:
            continue
        series = raw[station]
        baseline_samples = series.loc[
            (series.index >= report_start) & (series.index < start)
        ].dropna()
        if len(baseline_samples) < MIN_BASELINE_SAMPLES:
            continue
        obs_baseline = float(np.median(baseline_samples.to_numpy(dtype=float)))
        evaluation = series.loc[(series.index >= start) & (series.index <= end)]
        observation_10min = evaluation.resample(REPORT_INTERVAL).mean()
        common = observation_10min.index.intersection(times)
        obs_values = observation_10min.loc[common].to_numpy(dtype=float)
        valid = np.isfinite(obs_values)
        if int(valid.sum()) < 6:
            continue
        common = common[valid]
        obs_delta = obs_values[valid] - obs_baseline
        time_indices = times.get_indexer(common)
        if np.any(time_indices < 0):
            raise RuntimeError(f"{event_id}/{station}: timestamp mismatch")

        node_series = depths[node_index[node_id]]
        sim_baseline_mask = (times >= report_start) & (times < start)
        sim_baseline = float(np.median(node_series[sim_baseline_mask]))
        sim_delta = node_series[time_indices] - sim_baseline

        residual = sim_delta - obs_delta
        ss_res = float(np.sum(residual ** 2))
        centred = obs_delta - float(np.mean(obs_delta))
        ss_tot = float(np.sum(centred ** 2))
        nse = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        sigma = float(np.std(obs_delta))
        rsr = rmse / sigma if sigma > 1e-12 else float("nan")
        obs_sum = float(np.sum(obs_delta))
        sim_sum = float(np.sum(sim_delta))
        pbias = 100.0 * (sim_sum - obs_sum) / obs_sum if abs(obs_sum) > 1e-9 else float("nan")
        pearson = (
            float(np.corrcoef(obs_delta, sim_delta)[0, 1])
            if np.std(obs_delta) > 1e-8 and np.std(sim_delta) > 1e-8 else float("nan")
        )
        obs_peak = max(0.0, float(np.max(obs_delta)))
        sim_peak = max(0.0, float(np.max(sim_delta)))
        peak_error = 100.0 * (sim_peak - obs_peak) / obs_peak if obs_peak > 1e-9 else float("nan")
        result[station] = {
            "NSE": nse, "RMSE_m": rmse, "RSR": rsr, "PBIAS_pct": pbias,
            "vol_err_pct": pbias, "peak_err_pct": peak_error, "pearson_r": pearson,
            "ss_res": ss_res, "ss_tot": ss_tot, "sum_obs": obs_sum, "sum_sim": sim_sum,
            "sum_obs_sq": float(np.square(obs_delta).sum()),
            "sum_sim_sq": float(np.square(sim_delta).sum()),
            "sum_obs_sim": float((obs_delta * sim_delta).sum()),
            "n_pts": int(len(obs_delta)),
        }
    return result


def pool_metrics(rec_list):
    keys = ["NSE", "RMSE_m", "RSR", "PBIAS_pct", "pearson_r"]
    agg = {k: [] for k in keys}
    ss_res_total = 0.0
    ss_tot_total = 0.0
    n_points_total = 0
    sum_obs = sum_sim = 0.0
    sum_obs_sq = sum_sim_sq = sum_obs_sim = 0.0
    n_sta_events = 0

    for m in rec_list:
        for k in keys:
            v = m.get(k, float("nan"))
            if not np.isnan(v):
                agg[k].append(v)
        if "ss_res" in m and "ss_tot" in m:
            ss_res_total += m["ss_res"]
            ss_tot_total += m["ss_tot"]
            n_points_total += int(m.get("n_pts", 0))
            sum_obs += float(m.get("sum_obs", 0.0))
            sum_sim += float(m.get("sum_sim", 0.0))
            sum_obs_sq += float(m.get("sum_obs_sq", 0.0))
            sum_sim_sq += float(m.get("sum_sim_sq", 0.0))
            sum_obs_sim += float(m.get("sum_obs_sim", 0.0))
        n_sta_events += 1

    pooled_nse = 1.0 - ss_res_total / ss_tot_total if ss_tot_total > 1e-10 else float("nan")
    pooled_rmse = float(np.sqrt(ss_res_total / n_points_total)) if n_points_total > 0 else float("nan")
    pooled_pbias = (sum_sim - sum_obs) / sum_obs * 100.0 if abs(sum_obs) > 1e-12 else float("nan")
    corr_num = n_points_total * sum_obs_sim - sum_obs * sum_sim
    corr_den = np.sqrt(
        max(n_points_total * sum_obs_sq - sum_obs ** 2, 0.0)
        * max(n_points_total * sum_sim_sq - sum_sim ** 2, 0.0)
    )
    pooled_r = float(corr_num / corr_den) if corr_den > 1e-12 else float("nan")

    out = {
        "pooled_NSE": pooled_nse, "pooled_RMSE_m": pooled_rmse,
        "pooled_PBIAS_pct": pooled_pbias, "pooled_pearson_r": pooled_r,
        "n_points": n_points_total,
    }
    for k, vals in agg.items():
        a = np.array(vals)
        if len(a) == 0:
            out[k] = {"mean": None, "median": None, "std": None, "n": 0}
        else:
            out[k] = {
                "mean": float(np.mean(a)), "median": float(np.median(a)),
                "std": float(np.std(a, ddof=1)) if len(a) > 1 else 0.0, "n": len(a),
            }
    out["n_sta_events"] = n_sta_events
    return out


def run_for_splits(splits_to_run, label, set_ids):
    with open(CATALOG, encoding="utf-8") as f:
        cat = json.load(f)
    events = [e for e in cat if e.get("split") in splits_to_run]
    print(f"\n{'='*60}\n  {label}: {len(events)} events, sets={set_ids}\n{'='*60}")

    obs_cache = {}
    records = {sid: [] for sid in set_ids}
    n_ev_with_obs = 0

    for ev in events:
        yr = ev["start"][:4]
        mo = ev["start"][5:7]
        key = f"{yr}{mo}"
        if key not in obs_cache:
            obs_cache[key] = load_monthly_obs(yr, mo)
        obs_month = obs_cache[key]
        if obs_month.empty:
            print(f"  No obs: {ev['event_id']} ({key})")
            continue
        ev_has_data = False
        for sid in set_ids:
            m = compute_station_metrics(ev, obs_month, sid)
            for sta, met in m.items():
                records[sid].append(met)
                ev_has_data = True
        if ev_has_data:
            n_ev_with_obs += 1

    print(f"  Events with obs data: {n_ev_with_obs}/{len(events)}")
    results = {}
    for sid in set_ids:
        stats = pool_metrics(records[sid])
        n = stats.get("NSE", {}).get("n", 0)
        nse_m = stats.get("NSE", {}).get("mean")
        nse_med = stats.get("NSE", {}).get("median")
        rmse_m = stats.get("RMSE_m", {}).get("mean")
        r_m = stats.get("pearson_r", {}).get("mean")
        if n > 0:
            print(f"  Set {sid:>2}: NSE mean={nse_m:.4f} median={nse_med:.4f}  "
                  f"RMSE={rmse_m:.4f}m  r={r_m:.4f}  n={n}")
        else:
            print(f"  Set {sid:>2}: no valid station-event pairs")
        results[str(sid)] = stats
    return results


def main():
    target_sets = TARGET_SETS
    output = {}
    output["val_modelsel"] = run_for_splits(["val_modelsel"], "Model-selection validation (2022, n=6)", target_sets)
    output["val_conformal"] = run_for_splits(["val_conformal"], "Conformal validation (2023, n=7)", target_sets)
    output["test_temporal"] = run_for_splits(["test_temporal"], "Temporal test (2024, n=5)", target_sets)
    output["assessment"] = run_for_splits(["val_modelsel", "val_conformal"], "Assessment (2022-2023, n=13)", target_sets)
    output["holdout"] = output["test_temporal"]

    out_dir = BASE / "results" / "timecorrected_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "swmm_raw_performance_gauge401fix.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
