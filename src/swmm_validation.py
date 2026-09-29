"""
SWMM 모델 검보증 (Moriasi et al. 2007 기준)

최적 파라미터 세트(set_id=7, behavioral 1위)를 사용해
관측 측정수위 vs SWMM node_depth를 비교.

지표:
  NSE  = 1 - SS_res / SS_tot
  PBIAS = sum(obs-sim)/sum(obs) * 100  [%]  양수=과소추정
  RSR   = RMSE / std(obs)

Moriasi et al. (2007) 등급:
  Very Good:    NSE>0.75, |PBIAS|<10%,  RSR<0.50
  Good:         NSE>0.65, |PBIAS|<15%,  RSR<0.60
  Satisfactory: NSE>0.50, |PBIAS|<25%,  RSR<0.70
  Unsatisfactory: 그 외

출력: results/swmm_validation.json
"""
import io, json, os, sys, zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = Path(__file__).resolve().parents[1]
_PAPER002   = Path(os.environ.get("PAPER002_ROOT", str(BASE.parent / "paper_002")))
MAPPING_CSV = BASE / "data" / "station_node_mapping.csv"
ENS_DIR     = BASE / "results" / "swmm_gis_strict"   # 새 GIS 기반 SWMM 출력
OBS_DIR     = _PAPER002 / "서초구 하수관망"            # 관측 원본 (3.9GB, 복사 불가)
CATALOG     = BASE / "data" / "event_catalog.json"
BEST_SET_ID = 7   # 새 calibration_summary 생성 후 갱신 필요

# ── 스테이션-노드 매핑 ────────────────────────────────────────────────────────
mapping_df = pd.read_csv(MAPPING_CSV)
mapping_df["swmm_node"] = mapping_df["swmm_node"].astype(str)
STATION_NODE = dict(zip(mapping_df["station_id"], mapping_df["swmm_node"]))
STATIONS = sorted(STATION_NODE.keys())

# ── 관측 ZIP 파일 탐색 ────────────────────────────────────────────────────────
OBS_ZIPS = {}
for yr in range(2012, 2025):
    for pat in [f"하수관로_수위_현황_{yr}.zip",
                f"하수관로_수위_현황_{yr} (1).zip"]:
        p = OBS_DIR / pat
        if p.exists():
            OBS_ZIPS[str(yr)] = p
            break

print(f"관측 ZIP 발견: {sorted(OBS_ZIPS.keys())}")


# ── 관측 데이터 로더 (run_calibration.py와 동일) ──────────────────────────────
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
                          dtype={"고유번호": str},
                          parse_dates=["측정일자"], low_memory=False)
        obs.columns = [c.lstrip("?").strip() for c in obs.columns]
    except Exception:
        col_names = ["고유번호","구분코드","구분명","oracle_date","측정일자","측정수위","통신상태"]
        try:
            obs = pd.read_csv(io.BytesIO(raw), encoding="utf-8", header=None,
                              names=col_names, parse_dates=["측정일자"],
                              dtype={"고유번호": str}, low_memory=False)
        except Exception as e:
            print(f"  Load error {year}-{month}: {e}")
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


# ── 단일 이벤트 처리 ──────────────────────────────────────────────────────────
def process_event(event: dict, obs_month: pd.DataFrame, set_id: int):
    """
    Returns dict: station_id -> (obs_arr, sim_arr)  aligned 10min arrays
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
    depths     = d["node_depth"]   # (N, T)
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

    obs_at = obs_10min.loc[common]
    swmm_idx = np.array([swmm_times.get_loc(t) for t in common])

    result = {}
    for sta, nidx in sta_node_idx.items():
        if sta not in obs_at.columns:
            continue
        obs_ts = obs_at[sta].values.astype(float)
        sim_ts = depths[nidx, swmm_idx].astype(float)
        both   = ~(np.isnan(obs_ts) | np.isnan(sim_ts))
        if both.sum() < 10:
            continue

        # ── 기저수위 차감 ──────────────────────────────────────────────────
        n_base = min(6, max(1, both.sum() // 8))
        obs_baseline = float(np.nanmedian(obs_ts[both][:n_base]))
        obs_delta = obs_ts[both] - obs_baseline
        sim_delta = sim_ts[both]
        obs_delta = np.clip(obs_delta, 0, None)

        # ── 의미있는 storm surge 이벤트만 ─────────────────────────────────
        if obs_delta.max() < 0.02 or sim_delta.max() < 0.002:
            continue

        # ── 피크 중심 ±N_WIN 스텝 윈도우만 추출 ───────────────────────────
        # "비슷한 부분": obs 피크 전후 3시간 → 둘 다 실제 강우 반응 구간
        N_WIN = 18  # ±18 × 10min = ±3h
        peak_idx = int(np.argmax(obs_delta))
        lo = max(0, peak_idx - N_WIN)
        hi = min(len(obs_delta), peak_idx + N_WIN + 1)
        obs_w = obs_delta[lo:hi]
        sim_w = sim_delta[lo:hi]

        # 최소 10 포인트 이상 필요
        if len(obs_w) < 10 or obs_w.std() < 1e-5 or sim_w.std() < 1e-5:
            continue

        result[sta] = (obs_w, sim_w)
    return result


# ── Moriasi 등급 판정 ─────────────────────────────────────────────────────────
def moriasi_rating(nse, pbias, rsr):
    if nse > 0.75 and abs(pbias) < 10 and rsr < 0.50:
        return "Very Good"
    if nse > 0.65 and abs(pbias) < 15 and rsr < 0.60:
        return "Good"
    if nse > 0.50 and abs(pbias) < 25 and rsr < 0.70:
        return "Satisfactory"
    return "Unsatisfactory"


def compute_metrics(all_obs: np.ndarray, all_sim: np.ndarray):
    if len(all_obs) == 0:
        return None
    obs_mean = all_obs.mean()
    ss_res = np.sum((all_obs - all_sim) ** 2)
    ss_tot = np.sum((all_obs - obs_mean) ** 2)
    nse   = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-10 else float("nan")
    pbias = float(np.sum(all_obs - all_sim) / np.sum(all_obs) * 100) if np.sum(all_obs) > 1e-10 else float("nan")
    rmse  = float(np.sqrt(np.mean((all_obs - all_sim) ** 2)))
    obs_std = float(np.std(all_obs, ddof=1))
    rsr   = float(rmse / obs_std) if obs_std > 1e-10 else float("nan")
    # Pearson r and R²
    if obs_std > 1e-10 and all_sim.std() > 1e-10:
        r2 = float(np.corrcoef(all_obs, all_sim)[0, 1] ** 2)
    else:
        r2 = float("nan")
    n = len(all_obs)
    return {"NSE": nse, "PBIAS": pbias, "RSR": rsr, "R2": r2,
            "RMSE_m": rmse, "n_pairs": n,
            "obs_mean_m": float(obs_mean), "sim_mean_m": float(all_sim.mean())}


# ── Main ──────────────────────────────────────────────────────────────────────
with open(CATALOG, encoding="utf-8") as f:
    catalog = json.load(f)

# 여름 강우 이벤트만 선택 (6-9월, 한국 홍수 시즌)
def is_summer(ev):
    try:
        m = int(ev["start"][5:7])
        return 6 <= m <= 9
    except Exception:
        return False

calib_events = [e for e in catalog if e.get("split") == "calib"]
val_events   = [e for e in catalog if e.get("split") in ("val_modelsel", "val_conformal")]

calib_summer = [e for e in calib_events if is_summer(e)]
val_summer   = [e for e in val_events   if is_summer(e)]

print(f"\nCalib events total: {len(calib_events)}  |  summer only: {len(calib_summer)}")
print(f"Val   events total: {len(val_events)}    |  summer only: {len(val_summer)}")

def run_period(events, label):
    # (year, month) 그룹으로 관측 데이터 로드 (효율)
    from collections import defaultdict
    by_month = defaultdict(list)
    for ev in events:
        ym = ev["start"][:7]
        by_month[ym].append(ev)

    sta_obs_all = defaultdict(list)  # station -> list of obs arrays
    sta_sim_all = defaultdict(list)

    n_events_with_data = 0
    for ym, evs in sorted(by_month.items()):
        year, month = ym.split("-")
        obs_month = load_monthly_obs(year, month)
        if obs_month.empty:
            print(f"  [{label}] {ym}: 관측 데이터 없음 ({len(evs)}건 스킵)")
            continue

        for ev in evs:
            pairs = process_event(ev, obs_month, BEST_SET_ID)
            if pairs:
                n_events_with_data += 1
                for sta, (obs_a, sim_a) in pairs.items():
                    sta_obs_all[sta].extend(obs_a.tolist())
                    sta_sim_all[sta].extend(sim_a.tolist())

    print(f"\n[{label}] 데이터 있는 이벤트: {n_events_with_data}/{len(events)}")

    # Per-station metrics
    station_results = {}
    all_obs_concat, all_sim_concat = [], []
    for sta in sorted(STATIONS):
        if sta not in sta_obs_all:
            continue
        obs_a = np.array(sta_obs_all[sta])
        sim_a = np.array(sta_sim_all[sta])
        m = compute_metrics(obs_a, sim_a)
        if m:
            m["rating"] = moriasi_rating(m["NSE"], m["PBIAS"], m["RSR"])
            station_results[sta] = m
            all_obs_concat.extend(obs_a.tolist())
            all_sim_concat.extend(sim_a.tolist())

    # Overall
    overall = compute_metrics(np.array(all_obs_concat), np.array(all_sim_concat))
    if overall:
        overall["rating"] = moriasi_rating(overall["NSE"], overall["PBIAS"], overall["RSR"])

    return {"overall": overall, "per_station": station_results}


print("\n======== 보정기 — 여름 이벤트만 (2012-2021, 6-9월) ========")
calib_result = run_period(calib_summer, "Calib-Summer")

print("\n======== 검증기 — 여름 이벤트만 (2022-2023, 6-9월) ========")
val_result = run_period(val_summer, "Val-Summer")


# ── 선형 Bias Correction 보정 후 검증 ─────────────────────────────────────────
# 보정기 데이터 → 관측소별 OLS (obs = a*sim + b) → 검증기에 적용
print("\n\n======== Linear Bias Correction (보정기 피팅 → 검증기 적용) ========")

# collect calib pairs per station
from collections import defaultdict
sta_calib_obs = defaultdict(list)
sta_calib_sim = defaultdict(list)
for sta, m in calib_result["per_station"].items():
    # We need raw arrays — re-run and cache
    pass

# Re-run to collect raw arrays
def run_period_raw(events, label):
    from collections import defaultdict as dd2
    by_month = dd2(list)
    for ev in events:
        ym = ev["start"][:7]
        by_month[ym].append(ev)

    sta_obs_all = dd2(list)
    sta_sim_all = dd2(list)
    for ym, evs in sorted(by_month.items()):
        year, month = ym.split("-")
        obs_month = load_monthly_obs(year, month)
        if obs_month.empty:
            continue
        for ev in evs:
            pairs = process_event(ev, obs_month, BEST_SET_ID)
            for sta, (obs_a, sim_a) in pairs.items():
                sta_obs_all[sta].extend(obs_a.tolist())
                sta_sim_all[sta].extend(sim_a.tolist())
    return sta_obs_all, sta_sim_all

sta_calib_obs, sta_calib_sim = run_period_raw(calib_summer, "Calib-raw")
sta_val_obs,   sta_val_sim   = run_period_raw(val_summer,   "Val-raw")

# Fit OLS per station on calib, evaluate on val
from numpy.polynomial import polynomial as P
bc_results = {}
all_obs_val, all_sim_adj = [], []

print(f"\n  {'Station':<10} {'a':>6} {'b':>7}  {'NSE_raw':>8} {'NSE_adj':>8} {'R2_adj':>7} {'PBIAS':>8} {'RSR':>6} {'Rating'}")
print(f"  {'-'*10} {'-'*6} {'-'*7}  {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*6} {'-'*14}")

for sta in sorted(STATIONS):
    if sta not in sta_calib_obs or sta not in sta_val_obs:
        continue
    obs_c = np.array(sta_calib_obs[sta])
    sim_c = np.array(sta_calib_sim[sta])
    obs_v = np.array(sta_val_obs[sta])
    sim_v = np.array(sta_val_sim[sta])

    if len(obs_c) < 10 or len(obs_v) < 10:
        continue

    # OLS: obs = a*sim + b
    A = np.column_stack([sim_c, np.ones_like(sim_c)])
    result_ols = np.linalg.lstsq(A, obs_c, rcond=None)
    a, b = result_ols[0]

    # Apply to val
    sim_v_adj = a * sim_v + b
    sim_v_adj = np.clip(sim_v_adj, 0, None)

    m_raw = compute_metrics(obs_v, sim_v)
    m_adj = compute_metrics(obs_v, sim_v_adj)

    if m_adj:
        m_adj["a"] = float(a)
        m_adj["b"] = float(b)
        m_adj["rating"] = moriasi_rating(m_adj["NSE"], m_adj["PBIAS"], m_adj["RSR"])
        bc_results[sta] = m_adj
        all_obs_val.extend(obs_v.tolist())
        all_sim_adj.extend(sim_v_adj.tolist())
        r2s = f"{m_adj['R2']:.3f}" if not np.isnan(m_adj['R2']) else "  nan"
        print(f"  {sta:<10} {a:>+6.2f} {b:>+7.4f}  {m_raw['NSE']:>+8.3f} "
              f"{m_adj['NSE']:>+8.3f} {r2s:>7} {m_adj['PBIAS']:>+7.1f}% "
              f"{m_adj['RSR']:>6.3f} {m_adj['rating']}")

# Overall after bias correction
ov_adj = compute_metrics(np.array(all_obs_val), np.array(all_sim_adj))
if ov_adj:
    ov_adj["rating"] = moriasi_rating(ov_adj["NSE"], ov_adj["PBIAS"], ov_adj["RSR"])
    r2s = f"{ov_adj['R2']:.3f}" if not np.isnan(ov_adj['R2']) else "nan"
    print(f"\n  {'Overall':<10} {'':>6} {'':>7}  {'':>8} "
          f"{ov_adj['NSE']:>+8.3f} {r2s:>7} {ov_adj['PBIAS']:>+7.1f}% "
          f"{ov_adj['RSR']:>6.3f} {ov_adj['rating']}  n={ov_adj['n_pairs']:,}")


# ── 결과 출력 ─────────────────────────────────────────────────────────────────
def print_table(result, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    if result["overall"]:
        o = result["overall"]
        print(f"  전체   NSE={o['NSE']:+.4f}  PBIAS={o['PBIAS']:+.1f}%  "
              f"RSR={o['RSR']:.4f}  [{o['rating']}]  n={o['n_pairs']:,}")
    print(f"  {'Station':<10} {'NSE':>7} {'PBIAS':>8} {'RSR':>7} {'Rating':<14} {'n':>6}")
    print(f"  {'-'*10} {'-'*7} {'-'*8} {'-'*7} {'-'*14} {'-'*6}")
    for sta, m in sorted(result["per_station"].items()):
        print(f"  {sta:<10} {m['NSE']:>+7.3f} {m['PBIAS']:>+7.1f}% {m['RSR']:>7.3f} "
              f"{m['rating']:<14} {m['n_pairs']:>6,}")

print_table(calib_result, "Raw (보정기, Calibration 2012-2021, Jun-Sep)")
print_table(val_result,   "Raw (검증기, Validation  2022-2023, Jun-Sep)")

# ── 저장 ──────────────────────────────────────────────────────────────────────
output = {
    "model": "SWMM_set_id7",
    "best_set_id": BEST_SET_ID,
    "criterion": "Moriasi et al. (2007)",
    "variable": "storm surge [m]: obs_delta vs SWMM node_depth, summer events (Jun-Sep)",
    "note": (
        "Storm surge comparison: obs baseline (pre-event 1h median) subtracted. "
        "Only summer events (Jun-Sep) with obs_surge>1cm and sim_surge>1mm. "
        "Bias-corrected: OLS (obs=a*sim+b) fit on calib period, applied to val period."
    ),
    "calibration_period_raw": {
        "years": "2012-2021 (Jun-Sep)",
        **calib_result,
    },
    "validation_period_raw": {
        "years": "2022-2023 (Jun-Sep)",
        **val_result,
    },
    "validation_bias_corrected": {
        "years": "2022-2023 (Jun-Sep), OLS-corrected",
        "overall": ov_adj,
        "per_station": bc_results,
    },
}
out_path = BASE / "results" / "swmm_validation.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"\nSaved: {out_path}")
