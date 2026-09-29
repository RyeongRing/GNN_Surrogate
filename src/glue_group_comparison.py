"""
GLUE Behavioral vs Non-Behavioral Group Comparison
===================================================
Validation events (2022-2023) only.
Compares the current behavioral and non-behavioral sets from
data/calibration_summary.csv
on four metrics: NSE, RMSE, peak error (%), volume error (PBIAS %).

Storm-surge convention (same as GLUE calibration):
  obs_delta  = obs - pre-event baseline  (clipped to >= 0)
  sim_delta  = SWMM node_depth - pre-event baseline  (clipped to >= 0)
  threshold  = obs_peak >= 0.02 m AND sim_peak >= 0.002 m

Output: results/glue_group_comparison.json  +  console table
"""
import io, json, os, sys, zipfile, warnings
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

import compute_swmm_raw_performance as raw_protocol

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE          = Path(__file__).resolve().parents[1]
_PAPER002     = Path(os.environ.get("PAPER002_ROOT", str(BASE.parent / "paper_002")))
MAPPING_CSV   = BASE / "data" / "station_node_mapping.csv"
ENS_DIR       = BASE / "results" / "swmm_gis_strict"
OBS_DIR       = _PAPER002 / "서초구 하수관망"
CATALOG       = BASE / "data" / "event_catalog.json"
CALIB_CSV     = BASE / "data" / "calibration_summary.csv"

# ── 관측 ZIP 목록 ──────────────────────────────────────────────────────────────
OBS_ZIPS = {}
for yr in range(2012, 2025):
    for pat in [f"하수관로_수위_현황_{yr}.zip", f"하수관로_수위_현황_{yr} (1).zip"]:
        p = OBS_DIR / pat
        if p.exists():
            OBS_ZIPS[str(yr)] = p
            break

# ── 스테이션-노드 매핑 ────────────────────────────────────────────────────────
mapping_df = pd.read_csv(MAPPING_CSV)
mapping_df["swmm_node"] = mapping_df["swmm_node"].astype(str)
STATION_NODE = dict(zip(mapping_df["station_id"], mapping_df["swmm_node"]))
STATIONS = sorted(STATION_NODE.keys())

# ── Behavioral / Non-behavioral 분류 ─────────────────────────────────────────
calib_df = pd.read_csv(CALIB_CSV)
behavioral_ids    = set(calib_df.loc[calib_df["behavioral"] == True,  "set_id"].astype(int))
nonbehavioral_ids = set(calib_df.loc[calib_df["behavioral"] == False, "set_id"].astype(int))
ALL_SET_IDS = sorted(behavioral_ids | nonbehavioral_ids)
print(f"Behavioral sets   : n={len(behavioral_ids)}  IDs={sorted(behavioral_ids)}")
print(f"Non-behavioral sets: n={len(nonbehavioral_ids)}  IDs={sorted(nonbehavioral_ids)}")


# ── 관측 데이터 로더 ───────────────────────────────────────────────────────────
def load_monthly_obs(year: str, month: str) -> pd.DataFrame:
    zip_path = OBS_ZIPS.get(year)
    if not zip_path:
        return pd.DataFrame()
    month_pattern = f"{year}{month.zfill(2)}"
    with zipfile.ZipFile(zip_path) as zf:
        entry = next((e for e in zf.infolist() if month_pattern in e.filename), None)
        if entry is None:
            return pd.DataFrame()
        with zf.open(entry) as f:
            raw = f.read()
    try:
        obs = pd.read_csv(io.BytesIO(raw), encoding="cp949",
                          dtype={"고유번호": str}, parse_dates=["측정일자"], low_memory=False)
        obs.columns = [c.lstrip("?").strip() for c in obs.columns]
    except Exception:
        col_names = ["고유번호","구분코드","구분명","oracle_date","측정일자","측정수위","통신상태"]
        try:
            obs = pd.read_csv(io.BytesIO(raw), encoding="utf-8", header=None,
                              names=col_names, parse_dates=["측정일자"],
                              dtype={"고유번호": str}, low_memory=False)
        except Exception:
            return pd.DataFrame()
    if "고유번호" not in obs.columns or "측정수위" not in obs.columns:
        return pd.DataFrame()
    obs22 = obs[obs["고유번호"].isin(STATIONS)][["고유번호","측정일자","측정수위"]].copy()
    obs22["측정수위"] = pd.to_numeric(obs22["측정수위"], errors="coerce")
    obs22 = obs22.dropna(subset=["측정일자","측정수위"])
    if obs22.empty:
        return pd.DataFrame()
    wide = obs22.pivot_table(index="측정일자", columns="고유번호",
                              values="측정수위", aggfunc="mean")
    return wide.sort_index()


# ── 이벤트별 지표 산출 ─────────────────────────────────────────────────────────
def compute_event_metrics(event: dict, obs_month: pd.DataFrame, set_id: int):
    """
    Returns dict: station_id -> {NSE, RMSE_m, peak_err_pct, vol_err_pct}
    or empty dict on data-gap.
    """
    eid   = event["event_id"]
    start = event["start"]
    dur_h = float(event.get("duration_h", 24))
    end_ts = pd.Timestamp(start) + pd.Timedelta(hours=dur_h + 2)

    npz_path = ENS_DIR / eid / f"{set_id:03d}.npz"
    if not npz_path.exists():
        return {}

    d = np.load(str(npz_path), allow_pickle=True)
    node_names = [str(n) for n in d["node_names"]]
    depths     = d["node_depth"]          # shape (N_nodes, T)
    d.close()

    T_swmm = depths.shape[1]
    swmm_times = pd.date_range(start, periods=T_swmm, freq="10min")

    sta_node_idx = {sta: node_names.index(nd)
                    for sta, nd in STATION_NODE.items()
                    if nd in node_names}
    if not sta_node_idx:
        return {}

    mask = (obs_month.index >= start) & (obs_month.index <= str(end_ts))
    obs_ev = obs_month.loc[mask]
    if obs_ev.empty:
        return {}
    obs_10min = obs_ev.resample("10min").mean().interpolate(limit=3)
    common = obs_10min.index.intersection(swmm_times)
    if len(common) < 6:
        return {}

    obs_at   = obs_10min.loc[common]
    swmm_idx = np.array([swmm_times.get_loc(t) for t in common])

    result = {}
    for sta, nidx in sta_node_idx.items():
        if sta not in obs_at.columns:
            continue
        obs_ts = obs_at[sta].values.astype(float)
        sim_ts = depths[nidx, swmm_idx].astype(float)
        both   = ~(np.isnan(obs_ts) | np.isnan(sim_ts))
        if both.sum() < 6:
            continue

        # storm-surge delta (symmetric ΔH: baseline removed from both obs and SWMM)
        n_base = min(6, max(1, both.sum() // 8))
        obs_baseline = float(np.nanmedian(obs_ts[both][:n_base]))
        obs_d = np.clip(obs_ts[both] - obs_baseline, 0, None)
        sim_baseline = float(np.nanmedian(sim_ts[both][:n_base]))
        sim_d = np.clip(sim_ts[both] - sim_baseline, 0, None)

        obs_peak = float(obs_d.max())
        sim_peak = float(sim_d.max())
        if obs_peak < 0.02 or sim_peak < 0.002:
            continue

        # NSE
        obs_mean = obs_d.mean()
        ss_res = float(np.sum((obs_d - sim_d) ** 2))
        ss_tot = float(np.sum((obs_d - obs_mean) ** 2))
        nse = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else float("nan")

        # RMSE [m]
        rmse = float(np.sqrt(np.mean((obs_d - sim_d) ** 2)))

        # Peak error [%]  positive = over-prediction
        peak_err = (sim_peak - obs_peak) / obs_peak * 100.0

        # Volume error / PBIAS [%]
        obs_sum = float(obs_d.sum())
        sim_sum = float(sim_d.sum())
        vol_err = (sim_sum - obs_sum) / obs_sum * 100.0 if obs_sum > 1e-6 else float("nan")

        # Pearson r (what GLUE actually optimised)
        if np.std(obs_d) > 1e-8 and np.std(sim_d) > 1e-8:
            r_val = float(np.corrcoef(obs_d, sim_d)[0, 1])
        else:
            r_val = float("nan")

        result[sta] = {"NSE": nse, "RMSE_m": rmse,
                       "peak_err_pct": peak_err, "vol_err_pct": vol_err,
                       "pearson_r": r_val}
    return result


# ── 검증 이벤트 로드 ──────────────────────────────────────────────────────────
# Strict-protocol overrides; the legacy implementations above remain only for
# provenance and are deliberately shadowed here.
def load_monthly_obs(year: str, month: str) -> pd.DataFrame:
    return raw_protocol.load_monthly_obs(year, month)


def compute_event_metrics(event: dict, obs_month: pd.DataFrame, set_id: int):
    """Return Table 3 metrics using the frozen raw-performance protocol."""
    del obs_month
    strict = raw_protocol.compute_station_metrics(event, None, set_id)
    return {
        station: {
            "NSE": metrics["NSE"],
            "RMSE_m": metrics["RMSE_m"],
            "peak_err_pct": metrics["peak_err_pct"],
            "vol_err_pct": metrics["vol_err_pct"],
            "pearson_r": metrics["pearson_r"],
        }
        for station, metrics in strict.items()
    }


with open(CATALOG, encoding="utf-8") as f:
    catalog = json.load(f)

val_events = [e for e in catalog if e.get("split") in ("val_modelsel", "val_conformal")]
print(f"\nValidation events: {len(val_events)}")

obs_cache = {}

# ── 전체 (set_id, event) 루프 ────────────────────────────────────────────────
# records[set_id] = list of per-(sta,event) metric dicts
records = defaultdict(list)

for ev in val_events:
    yr  = ev["start"][:4]
    mo  = ev["start"][5:7]
    key = f"{yr}{mo}"
    if key not in obs_cache:
        obs_cache[key] = load_monthly_obs(yr, mo)
    obs_month = obs_cache[key]
    if obs_month.empty:
        continue

    for sid in ALL_SET_IDS:
        m = compute_event_metrics(ev, obs_month, sid)
        for sta, metrics in m.items():
            records[sid].append(metrics)

print(f"\nTotal set_ids with data: {sum(1 for v in records.values() if v)}/{len(ALL_SET_IDS)}")


# ── 그룹별 집계 ───────────────────────────────────────────────────────────────
def aggregate_group(set_ids):
    all_nse, all_rmse, all_peak, all_vol, all_r = [], [], [], [], []
    for sid in set_ids:
        for r in records.get(sid, []):
            if not np.isnan(r["NSE"]):          all_nse.append(r["NSE"])
            if not np.isnan(r["RMSE_m"]):       all_rmse.append(r["RMSE_m"])
            if not np.isnan(r["peak_err_pct"]): all_peak.append(r["peak_err_pct"])
            if not np.isnan(r["vol_err_pct"]):  all_vol.append(r["vol_err_pct"])
            if not np.isnan(r.get("pearson_r", float("nan"))): all_r.append(r["pearson_r"])
    def stats(arr):
        a = np.array(arr)
        if len(a) == 0:
            return {"mean": float("nan"), "std": float("nan"),
                    "median": float("nan"), "n": 0}
        return {"mean": float(np.mean(a)), "std": float(np.std(a)),
                "median": float(np.median(a)), "n": len(a)}
    return {
        "NSE":           stats(all_nse),
        "RMSE_m":        stats(all_rmse),
        "peak_err_pct":  stats(all_peak),
        "vol_err_pct":   stats(all_vol),
        "pearson_r":     stats(all_r),
    }

beh_stats  = aggregate_group(behavioral_ids)
nbeh_stats = aggregate_group(nonbehavioral_ids)

# ── 콘솔 출력 ─────────────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  GLUE Group Comparison — Validation Events (2022–2023)")
print("="*72)
print(f"  {'Metric':<20} {'Behavioral (n=25)':>22}  {'Non-behavioral (n=25)':>22}")
print(f"  {'':20} {'mean ± std  [median]':>22}  {'mean ± std  [median]':>22}")
print(f"  {'-'*20} {'-'*22}  {'-'*22}")

metrics_labels = [
    ("NSE",          "NSE",           ""),
    ("RMSE_m",       "RMSE",          " m"),
    ("peak_err_pct", "Peak error",    " %"),
    ("vol_err_pct",  "Vol. error",    " %"),
    ("pearson_r",    "Pearson r",     ""),
]
for key, label, unit in metrics_labels:
    b = beh_stats[key]
    n = nbeh_stats[key]
    b_str = f"{b['mean']:+.3f} ± {b['std']:.3f}  [{b['median']:+.3f}]"
    n_str = f"{n['mean']:+.3f} ± {n['std']:.3f}  [{n['median']:+.3f}]"
    print(f"  {label+unit:<20} {b_str:>22}  {n_str:>22}")

print(f"  {'Observations (n)':20} {beh_stats['NSE']['n']:>22}  {nbeh_stats['NSE']['n']:>22}")
print("="*72)

# ── 저장 ─────────────────────────────────────────────────────────────────────
out = {
    "description": (
        f"Behavioral (n={len(behavioral_ids)}) vs non-behavioral "
        f"(n={len(nonbehavioral_ids)}) SWMM parameter sets "
        "evaluated on independent validation events (2022-2023). "
        "Metric: symmetric storm-surge delta with pre-event baselines "
        "removed from both observations and simulations. "
        "NSE>0 indicates better than climatological mean."
    ),
    "n_val_events": len(val_events),
    "behavioral_set_ids":    sorted(behavioral_ids),
    "nonbehavioral_set_ids": sorted(nonbehavioral_ids),
    "behavioral":    beh_stats,
    "nonbehavioral": nbeh_stats,
}
out_path = BASE / "results" / "glue_group_comparison.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nSaved → {out_path}")
