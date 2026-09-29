"""
GLUE-style calibration pipeline
50개 LHS 파라미터 세트를 관측 수위 데이터와 비교해 behavioral 세트를 선택한다.

출력:
  data/calibration_scores.csv  -- (event_id, set_id, station_id, pearson_r)
  data/calibration_summary.csv -- (set_id, mean_r, rank, behavioral)
  data/behavioral_sets.json    -- behavioral set IDs (top 50%)
"""
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parents[2]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from src.data.observation_loader import load_monthly_observations

# paper_002_update: 관측 데이터는 원본 paper_002 절대경로 참조 (3.9GB 복사 불가)
# 환경변수 PAPER002_ROOT로 재정의 가능 (기본값: 형제 디렉토리 paper_002)
_PAPER002 = Path(os.environ.get("PAPER002_ROOT", str(BASE.parent / "paper_002")))

MAPPING_CSV = BASE / "data" / "station_node_mapping.csv"
# 앙상블 디렉토리: 환경변수 SWMM_ENS_DIR로 재정의 가능 (기본값: results/swmm_gis_strict)
ENS_DIR     = Path(sys.argv[sys.argv.index("--ens-dir") + 1]) if "--ens-dir" in sys.argv \
              else BASE / "results" / "swmm_gis_strict"
OBS_DIR     = _PAPER002 / "서초구 하수관망"
CATALOG     = BASE / "data" / "event_catalog.json"

# Load station-node mapping
mapping_df = pd.read_csv(MAPPING_CSV)
mapping_df['swmm_node'] = mapping_df['swmm_node'].astype(str)
STATION_NODE = dict(zip(mapping_df['station_id'], mapping_df['swmm_node']))
STATIONS = sorted(STATION_NODE.keys())

def load_monthly_obs(year: str, month: str) -> pd.DataFrame:
    """Load one month through the audited multi-schema observation parser."""
    long = load_monthly_observations(
        OBS_DIR,
        int(year),
        int(month),
        stations=STATIONS,
    )
    if long.empty:
        return pd.DataFrame()
    wide = long.pivot_table(
        index="timestamp",
        columns="station_id",
        values="water_level_m",
        aggfunc="mean",
    )
    return wide.sort_index()


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 5:
        return np.nan
    a, b = a[mask], b[mask]
    sa, sb = a.std(), b.std()
    if sa < 1e-6 or sb < 1e-6:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def process_event(event: dict, obs_month: pd.DataFrame) -> list:
    """단일 이벤트 × 50 세트 × 14 관측소 Pearson r 계산"""
    eid = event['event_id']
    start = event['start']
    dur_h = float(event.get('duration_h', 24))
    end_ts = pd.Timestamp(start) + pd.Timedelta(hours=dur_h + 2)

    ens_path = ENS_DIR / eid
    npz_files = sorted(ens_path.glob("*.npz"))
    if not npz_files:
        return []

    # Node ordering from first npz
    first = np.load(npz_files[0], allow_pickle=True)
    node_names = [str(n) for n in first['node_names']]
    T_swmm = first['node_depth'].shape[1]
    first.close()

    # Station → node index
    sta_node_idx = {sta: node_names.index(nd)
                    for sta, nd in STATION_NODE.items()
                    if nd in node_names}
    if not sta_node_idx:
        return []

    swmm_times = pd.date_range(start, periods=T_swmm, freq='10min')

    # Extract obs event window
    mask = (obs_month.index >= start) & (obs_month.index <= str(end_ts))
    obs_ev = obs_month.loc[mask]
    if obs_ev.empty:
        return []
    obs_10min = obs_ev.resample('10min').mean().interpolate(limit=3)

    common = obs_10min.index.intersection(swmm_times)
    if len(common) < 6:
        return []

    obs_arr = obs_10min.loc[common]
    swmm_t_idx = np.array([swmm_times.get_loc(t) for t in common])

    rows = []
    for npz_path in npz_files:
        set_id = int(npz_path.stem)
        d = np.load(npz_path, allow_pickle=True)
        depths = d['node_depth']  # (N, T)
        d.close()

        for sta, nidx in sta_node_idx.items():
            if sta not in obs_arr.columns:
                continue
            sim_ts = depths[nidx, swmm_t_idx].astype(float)
            obs_ts = obs_arr[sta].values.astype(float)
            r = pearson_r(obs_ts, sim_ts)
            rows.append({'event_id': eid, 'set_id': set_id,
                         'station_id': sta, 'pearson_r': r})
    return rows


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=== GLUE Calibration Pipeline ===\n")

    with open(CATALOG, encoding='utf-8') as f:
        catalog = json.load(f)

    # Calib events with existing ensembles
    calib_events = []
    for ev in catalog:
        if ev.get('split') != 'calib':
            continue
        eid = ev['event_id']
        ens_path = ENS_DIR / eid
        if ens_path.exists() and len(list(ens_path.glob("*.npz"))) >= 50:
            calib_events.append(ev)
    print(f"Calib events with ensemble: {len(calib_events)}")

    # Group events by (year, month) for efficient obs loading
    event_by_month = defaultdict(list)
    for ev in calib_events:
        ym = ev['start'][:7]  # "2020-06"
        event_by_month[ym].append(ev)

    print(f"Year-months to process: {sorted(event_by_month.keys())}\n")

    all_rows = []
    total_ym = len(event_by_month)
    for i, (ym, events) in enumerate(sorted(event_by_month.items())):
        year, month = ym.split('-')
        print(f"[{i+1}/{total_ym}] Loading obs {ym} ({len(events)} events)...", flush=True)
        obs_month = load_monthly_obs(year, month)
        if obs_month.empty:
            print(f"  No obs data for {ym}, skipping {len(events)} events")
            continue
        print(f"  Obs: {obs_month.shape[0]} timesteps, {obs_month.shape[1]} stations present")

        for ev in events:
            print(f"  {ev['event_id']} ({ev['start'][:16]}, {ev.get('total_mm',0)}mm)...",
                  end=' ', flush=True)
            rows = process_event(ev, obs_month)
            all_rows.extend(rows)
            if rows:
                valid = [r['pearson_r'] for r in rows
                         if r['pearson_r'] is not None and not np.isnan(r['pearson_r'])]
                print(f"{len(rows)} pairs, mean_r={np.mean(valid):.3f}" if valid else "no valid data")
            else:
                print("no data")

    if not all_rows:
        print("\nERROR: No calibration data!")
        sys.exit(1)

    # Save detailed scores
    scores_df = pd.DataFrame(all_rows)
    out_scores = BASE / "data" / "calibration_scores.csv"
    scores_df.to_csv(out_scores, index=False)
    print(f"\nSaved: {out_scores} ({len(scores_df)} rows)")

    # Summarize by parameter set
    valid_scores = scores_df.dropna(subset=['pearson_r'])
    summary = (valid_scores.groupby('set_id')
               .agg(mean_r=('pearson_r', 'mean'),
                    n_pairs=('pearson_r', 'count'))
               .reset_index()
               .sort_values('mean_r', ascending=False))
    summary['rank'] = range(1, len(summary) + 1)

    # Behavioral: top 50% by mean_r
    threshold = float(summary['mean_r'].quantile(0.50))
    summary['behavioral'] = summary['mean_r'] >= threshold
    n_behav = int(summary['behavioral'].sum())
    print(f"\nBehavioral threshold: r >= {threshold:.4f}")
    print(f"Behavioral sets: {n_behav}/{len(summary)}")
    print("\nTop 20 parameter sets:")
    print(summary.head(20).to_string())

    out_sum = BASE / "data" / "calibration_summary.csv"
    summary.to_csv(out_sum, index=False)
    print(f"\nSaved: {out_sum}")

    behavioral_ids = sorted(summary[summary['behavioral']]['set_id'].tolist())
    result = {
        'behavioral_set_ids': behavioral_ids,
        'threshold_r': threshold,
        'n_behavioral': n_behav,
        'n_total': len(summary),
        'n_calib_events': len(calib_events),
        'n_stations': len(STATION_NODE),
    }
    out_beh = BASE / "data" / "behavioral_sets.json"
    with open(out_beh, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_beh}")
    print(f"\nBehavioral set IDs: {behavioral_ids}")
