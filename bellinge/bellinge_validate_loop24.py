"""
Bellinge (Odense, Denmark) 독립 검증 파이프라인 — Loop 24 (GIS-strict, current primary) 버전.

paper_002/bellinge_validate.py 를 paper_002_update 컨텍스트로 이식한 사본.
변경점은 경로 상수(BELLINGE_*)를 paper_002의 기존 데이터로 절대경로 지정한 것과
체크포인트 디렉터리를 paper_002_update/results/checkpoints 로 고정한 것 뿐이며,
이벤트 선별/파인튜닝/평가 로직은 원본과 동일하다.

데이터: DTU Bellinge 데이터셋 (CC BY 4.0) — Pedersen et al. 2021, ESSD
       https://doi.org/10.11583/DTU.c.5029124
       (재사용: papers/paper_002/data/bellinge/... 원본 위치, 복사하지 않음)

사용법 (paper_002_update 디렉터리에서 실행):
  python bellinge_validate_loop24.py --loop 24 --skip_swmm --fine_tune \
      --event_split --n_test_events 5 --config config_seed_exp.yaml \
      --output results/bellinge_loop24_validation.json
"""

import argparse
import json
import logging
import os
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# External Bellinge data root; no private workstation path is embedded.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BELLINGE_DATA_ROOT = Path(os.environ.get("BELLINGE_DATA_ROOT", str(REPOSITORY_ROOT / "data/bellinge")))
BELLINGE_DIR   = BELLINGE_DATA_ROOT / "7_SWMM"
BELLINGE_INP   = BELLINGE_DIR / "BellingeSWMM_v021_nopervious.inp"
BELLINGE_DAT   = BELLINGE_DIR / "rg_bellinge_Jun2010_Aug2021.dat"
BELLINGE_ENS   = BELLINGE_DATA_ROOT / "ensemble"
# 카탈로그는 이번 실행 전용 (paper_002_update 로컬에 기록, 원본 파일 건드리지 않음)
BELLINGE_CAT   = Path("results/bellinge_event_catalog_loop24.json")

# ── Bellinge SWMM 파라미터 범위 ──────────────────────────────────────────────
# 전이 목적: GNN은 서초구(Seoul) 파라미터 공간에서 학습됨.
# Bellinge topology transfer 테스트이므로 파라미터 특성은 서초구 학습 분포 내로 제한.
# 논문 claim: "GNN의 그래프 위상 일반화 능력" (파라미터 regime 일반화 X).
#
# BUG FIX (2026-09): inf_max/inf_min/inf_decay는 src/data/swmm_ensemble.py의
# 2026-07-07 Modified Green-Ampt 갱신(Psi_f 30-250mm, Ks 0.03-1.0mm/hr,
# IMDmax 0.10-0.45) 이전의 구버전(Horton 스타일 2.5-5.0mm/hr) 범위가 그대로
# 남아 있었음. inf_max=2.5-5.0은 학습 분포(30-250)와 스케일이 완전히 달라
# out-of-distribution 조건 입력이 됨.
#
# BUG FIX 2 (2026-09, 확장): build_event_inp_bellinge()는 [SUBAREAS]의
# N-Imperv/S-Imperv(→n_imperv/dstore_imperv)와 [CONDUITS] roughness
# (→n_conduit)만 실제로 .inp에 기록하고, INFILTRATION은 "nopervious 모델이므로
# 교란 생략"이라고 명시적으로 건너뛴다 -- 즉 dstore_perv 역시 S-Perv 값이
# .inp에 반영되지 않아 SWMM 결과에 전혀 영향을 주지 않는 "유령" 파라미터다.
# Bellinge는 nopervious 모델이라 dstore_perv/inf_max/inf_min/inf_decay
# 네 파라미터 모두 SWMM 결과(node_depth/link_flow)에는 전혀 영향을 주지
# 않으므로, 물리적으로 무의미한 변동 대신 서초구 50-set 학습 앙상블의
# 평균값으로 고정(masking)한다 -- 아래 네 값은
# latin_hypercube_sample(PARAM_RANGES, 50, seed=42) 실측 평균과 일치.
BELLINGE_PARAM_RANGES = {
    "n_conduit":     (0.010, 0.015),   # Manning's n 관거 (Bellinge 실측 범위와 근접, .inp에 반영됨)
    "n_imperv":      (0.010, 0.015),   # 서초구 학습 분포 유지, .inp에 반영됨
    "dstore_imperv": (1.5,   3.5),     # 서초구 학습 분포 유지, .inp에 반영됨
    "dstore_perv":   (5.0033, 5.0033), # 서초구 50-set 학습 평균으로 고정 (.inp 미반영, 물리적으로 무의미)
    "inf_max":       (140.14, 140.14), # 서초구 50-set 학습 평균으로 고정 (.inp 미반영, 물리적으로 무의미)
    "inf_min":       (0.5148, 0.5148), # 서초구 50-set 학습 평균으로 고정 (.inp 미반영, 물리적으로 무의미)
    "inf_decay":     (0.2750, 0.2750), # 서초구 50-set 학습 평균으로 고정 (.inp 미반영, 물리적으로 무의미)
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--loop",       type=int, default=24)
    p.add_argument("--n_sets",     type=int, default=10,
                   help="SWMM 앙상블 파라미터 세트 수")
    p.add_argument("--n_events",   type=int, default=8,
                   help="선별할 강우 이벤트 수")
    p.add_argument("--min_rain_mm",type=float, default=8.0,
                   help="이벤트 최소 총강수량 mm")
    p.add_argument("--regime",     type=str, default="B")
    p.add_argument("--config",     type=str, default="config_seed_exp.yaml")
    p.add_argument("--output",     type=str,
                   default="results/bellinge_loop24_validation.json")
    p.add_argument("--skip_swmm",  action="store_true",
                   help="SWMM 앙상블 건너뜀 (기존 결과 재사용)")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--n_cal",      type=int, default=10,
                   help="affine depth scale calibration 샘플 수 (기본: 10/80)")
    p.add_argument("--fine_tune",  action="store_true",
                   help="decoder fine-tune 활성화 (Bellinge SWMM 기반)")
    p.add_argument("--n_ft",       type=int, default=60,
                   help="fine-tune 샘플 수 (기본: 60, 나머지 20은 test)")
    p.add_argument("--ft_epochs",  type=int, default=150,
                   help="fine-tune 에폭 수 (기본: 150)")
    p.add_argument("--ft_lr",      type=float, default=5e-4,
                   help="fine-tune 학습률 (기본: 5e-4)")
    p.add_argument("--event_split",action="store_true",
                   help="이벤트 단위 split (마지막 n_test_events개 이벤트를 test)")
    p.add_argument("--n_test_events", type=int, default=5,
                   help="event_split 시 test에 사용할 이벤트 수 (기본: 5)")
    return p.parse_args()


# ── 강우 데이터 파싱 & 이벤트 선별 ─────────────────────────────────────────────

def parse_dat_file(dat_path: Path,
                   station: str = "rg5425") -> pd.Series:
    """
    SWMM rain file (.dat) → 1분 해상도 pd.Series (mm/min 누적).
    형식: station year month day hour minute value
    """
    logger.info(f"강우 데이터 파싱: {dat_path.name}")
    records = []
    with open(dat_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            if parts[0] != station:
                continue
            try:
                y, mo, d, h, mi = int(parts[1]), int(parts[2]), int(parts[3]), \
                                   int(parts[4]), int(parts[5])
                val = float(parts[6])
                ts = pd.Timestamp(year=y, month=mo, day=d, hour=h, minute=mi)
                records.append((ts, val))
            except (ValueError, OverflowError):
                continue

    if not records:
        raise ValueError(f"'{station}' 데이터 없음: {dat_path}")

    idx, vals = zip(*records)
    s = pd.Series(vals, index=pd.DatetimeIndex(idx), dtype=np.float32)
    s = s.sort_index()
    s = s[~s.index.duplicated(keep="first")]
    logger.info(f"  → {len(s):,}개 레코드 "
                f"({s.index[0].date()} ~ {s.index[-1].date()})")
    return s


def select_events(rain_1min: pd.Series,
                  n_events: int = 8,
                  min_total_mm: float = 8.0,
                  gap_hours: int = 24,
                  max_dur_hours: int = 48,
                  seed: int = 42) -> List[Dict]:
    """
    실측 강우에서 유의 이벤트 선별.

    기준:
      - 이벤트 총강수량 ≥ min_total_mm
      - 이벤트 간 간격 ≥ gap_hours (독립성)
      - 이벤트 지속 ≤ max_dur_hours (관리 가능 길이)
    """
    # 1분 → 1시간 합산
    rain_h = rain_1min.resample("1h").sum().fillna(0)
    wet = rain_h[rain_h > 0]

    events = []
    prev_end = pd.Timestamp("2000-01-01")

    # 연속 강우 블록 검출
    in_event = False
    e_start = e_end = None
    for ts in rain_h.index:
        is_wet = rain_h[ts] > 0.1
        if is_wet and not in_event:
            e_start = ts
            in_event = True
        elif not is_wet and in_event:
            e_end = ts
            in_event = False
            # 독립성 & 최소 강수량 필터
            gap = (e_start - prev_end).total_seconds() / 3600
            dur = (e_end - e_start).total_seconds() / 3600
            total = rain_h.loc[e_start:e_end].sum()
            if gap >= gap_hours and dur <= max_dur_hours and total >= min_total_mm:
                events.append({
                    "start": e_start, "end": e_end,
                    "total_mm": round(float(total), 2),
                    "dur_h": round(dur, 1),
                })
                prev_end = e_end

    if not events:
        raise ValueError(f"조건을 만족하는 이벤트 없음 (min_total_mm={min_total_mm})")

    # 총강수량 기준 정렬 후 상위 n_events 선택
    events.sort(key=lambda e: e["total_mm"], reverse=True)
    selected = events[:max(n_events, 1)]

    catalog = []
    for i, ev in enumerate(selected):
        catalog.append({
            "event_id": f"EB{i+1:03d}",
            "start":    ev["start"].strftime("%Y-%m-%d %H:%M"),
            "end":      (ev["end"] + pd.Timedelta(hours=3)).strftime("%Y-%m-%d %H:%M"),
            "regime":   "B",
            "total_mm": ev["total_mm"],
            "dur_h":    ev["dur_h"],
        })
        logger.info(f"  {catalog[-1]['event_id']}: {catalog[-1]['start']} ~ "
                    f"{catalog[-1]['end']}  total={ev['total_mm']:.1f}mm")

    return catalog


# ── 이벤트 INP 생성 (FILE 강우 방식) ─────────────────────────────────────────

def build_event_inp_bellinge(inp_path: Path,
                              dat_path: Path,
                              param_set: Dict,
                              event: Dict,
                              out_path: Path,
                              warmup_days: int = 2) -> Path:
    """
    Bellinge .inp 수정 버전 생성.

    수정 사항:
      1. [OPTIONS] START/END DATE → 이벤트 기간
      2. [RAINGAGES] FILE 경로 → dat_path 절대 경로
      3. [CONDUITS] roughness → n_conduit
      4. [SUBAREAS] N-Imperv, S-Imperv → n_imperv, dstore_imperv
      (nopervious 모델이므로 INFILTRATION 교란 생략)
    """
    lines = inp_path.read_text(encoding="utf-8", errors="replace").splitlines()

    ev_start = pd.Timestamp(event["start"])
    ev_end   = pd.Timestamp(event["end"])
    sim_start = ev_start - pd.Timedelta(days=warmup_days)
    # 파일 기간 제한 (2009-06 ~ 2021-08)
    dat_begin = pd.Timestamp("2009-06-01")
    dat_end_  = pd.Timestamp("2021-09-01")
    sim_start = max(sim_start, dat_begin)
    sim_end   = min(ev_end + pd.Timedelta(hours=6), dat_end_)

    # Windows 경로를 SWMM이 인식하는 형식으로 (따옴표 포함)
    abs_dat = str(dat_path.resolve()).replace("\\", "/")

    def fmt_date(ts): return ts.strftime("%m/%d/%Y")
    def fmt_time(ts): return ts.strftime("%H:%M:%S")

    new_lines = []
    section = None

    for line in lines:
        s = line.strip()
        if s.startswith("[") and s.endswith("]"):
            section = s[1:-1].upper()
            new_lines.append(line)
            continue
        if s.startswith(";") or not s:
            new_lines.append(line)
            continue

        # [OPTIONS]
        if section == "OPTIONS":
            parts = s.split()
            key = parts[0].upper() if parts else ""
            rep = {
                "START_DATE":        f"START_DATE           {fmt_date(sim_start)}",
                "START_TIME":        f"START_TIME           {fmt_time(sim_start)}",
                "REPORT_START_DATE": f"REPORT_START_DATE    {fmt_date(ev_start)}",
                "REPORT_START_TIME": f"REPORT_START_TIME    {fmt_time(ev_start)}",
                "END_DATE":          f"END_DATE             {fmt_date(sim_end)}",
                "END_TIME":          f"END_TIME             {fmt_time(sim_end)}",
            }
            if key in rep:
                new_lines.append(rep[key])
                continue

        # [RAINGAGES] FILE 경로를 절대 경로로 교체
        if section == "RAINGAGES":
            parts = s.split()
            # 형식: Name Format Interval SCF FILE "path" col unit
            if len(parts) >= 5 and parts[4].upper() == "FILE":
                # parts[5]가 파일 경로 (따옴표 포함)
                if len(parts) >= 8:
                    rebuilt = (f"{parts[0]:<16} {parts[1]:<9} {parts[2]:<6} "
                               f"{parts[3]:<6} FILE       \"{abs_dat}\"        "
                               f"{parts[6]}     {parts[7]}")
                else:
                    rebuilt = (f"{parts[0]:<16} {parts[1]:<9} {parts[2]:<6} "
                               f"{parts[3]:<6} FILE       \"{abs_dat}\"        "
                               f"{parts[0]}     MM")
                new_lines.append(rebuilt)
                continue

        # [CONDUITS] roughness 교체 (5번째 열)
        if section == "CONDUITS":
            parts = s.split()
            if len(parts) >= 5:
                try:
                    float(parts[4])
                    parts[4] = f"{param_set['n_conduit']:.5f}"
                    new_lines.append("  ".join(parts))
                    continue
                except (ValueError, IndexError):
                    pass

        # [SUBAREAS] N-Imperv, dstore_imperv 교체
        if section == "SUBAREAS":
            parts = s.split()
            if len(parts) >= 5:
                try:
                    float(parts[1])
                    parts[1] = f"{param_set['n_imperv']:.5f}"
                    if len(parts) > 3:
                        parts[3] = f"{param_set['dstore_imperv']:.3f}"
                    new_lines.append("  ".join(parts))
                    continue
                except (ValueError, IndexError):
                    pass

        new_lines.append(line)

    out_path.write_text("\n".join(new_lines), encoding="utf-8")
    return out_path


# ── SWMM 앙상블 실행 ──────────────────────────────────────────────────────────

def run_bellinge_ensemble(inp_path: Path, dat_path: Path,
                           param_sets: List[Dict],
                           events: List[Dict],
                           ensemble_dir: Path,
                           skip_existing: bool = True) -> Dict:
    """Bellinge SWMM 앙상블 실행."""
    try:
        from pyswmm import Simulation
        from swmm.toolkit import output as swmm_out
    except ImportError:
        logger.error("pyswmm / swmm.toolkit 미설치. 'pip install pyswmm swmm.toolkit'")
        return {"total": 0, "success": 0, "skipped": 0, "failed": 0}

    import tempfile
    from src.data.ensemble_io import save_run, list_completed
    from src.data.graph_builder import parse_network_from_inp

    ensemble_dir.mkdir(parents=True, exist_ok=True)
    completed = list_completed(ensemble_dir) if skip_existing else {}

    SWMM_TMP = Path(tempfile.gettempdir()) / "swmm_bellinge"
    SWMM_TMP.mkdir(parents=True, exist_ok=True)

    # SWMM C 라이브러리는 유니코드/한글 경로 미지원 → ASCII 경로로 복사
    ascii_dat = SWMM_TMP / "rg_bellinge.dat"
    if not ascii_dat.exists():
        import shutil
        logger.info(f"강우 파일 ASCII 경로로 복사: {ascii_dat}")
        shutil.copy2(str(dat_path), str(ascii_dat))

    node_df, link_df, _, _ = parse_network_from_inp(inp_path)
    node_names = node_df["node_id"].tolist()
    link_names = link_df["link_id"].tolist()
    n_graph_nodes = len(node_names)
    n_graph_links = len(link_names)

    total = success = skipped = failed = 0

    for ev in events:
        ev_id  = ev["event_id"]
        done   = set(completed.get(ev_id, []))

        for ps in param_sets:
            sid = ps["set_id"]
            total += 1
            if skip_existing and sid in done:
                skipped += 1
                continue

            tmp_inp = SWMM_TMP / f"bell_{ev_id}_s{sid:03d}.inp"
            try:
                build_event_inp_bellinge(inp_path, ascii_dat, ps, ev, tmp_inp)
                out_path = tmp_inp.with_suffix(".out")

                with Simulation(str(tmp_inp)) as sim:
                    for _ in sim:
                        pass

                handle = swmm_out.init()
                swmm_out.open(handle, str(out_path))
                proj = swmm_out.get_proj_size(handle)
                n_nodes_out, n_links_out = proj[1], proj[2]

                with open(out_path, "rb") as fh:
                    fh.seek(-3 * 4, 2)
                    n_periods = struct.unpack("<i", fh.read(4))[0]

                if n_periods <= 0:
                    raise RuntimeError(f"n_periods={n_periods} (시뮬레이션 실패)")

                # 노드 수위 (m) 추출
                depth_arr = np.zeros((n_nodes_out, n_periods), dtype=np.float32)
                flow_arr  = np.zeros((n_links_out, n_periods), dtype=np.float32)
                for t in range(n_periods):
                    depth_arr[:, t] = swmm_out.get_node_attribute(handle, t, 0)
                    flow_arr[:, t]  = swmm_out.get_link_attribute(handle, t, 0)
                swmm_out.close(handle)

                # SWMM 출력 순서 == INP 파싱 순서라 가정
                n_n = min(n_nodes_out, n_graph_nodes)
                n_l = min(n_links_out, n_graph_links)
                result = {
                    "set_id":     sid,
                    "event_id":   ev_id,
                    "node_depth": {node_names[i]: depth_arr[i].tolist()
                                   for i in range(n_n)},
                    "link_flow":  {link_names[i]: flow_arr[i].tolist()
                                   for i in range(n_l)},
                    "timesteps":  n_periods,
                }
                save_run(result, ps, out_dir=ensemble_dir / ev_id)
                success += 1
                logger.info(f"  ✓ {ev_id}/set{sid:03d}  T={n_periods}")

            except Exception as e:
                failed += 1
                logger.error(f"  ✗ {ev_id}/set{sid:03d}: {e}")
            finally:
                for ext in [".inp", ".rpt", ".out"]:
                    try:
                        tmp_inp.with_suffix(ext).unlink(missing_ok=True)
                    except PermissionError:
                        pass

    return {"total": total, "success": success,
            "skipped": skipped, "failed": failed}


# ── Bellinge decoder fine-tuning ─────────────────────────────────────────────

def fine_tune_bellinge_decoder(model, ft_dataset, n_epochs: int = 100,
                                lr: float = 1e-4, seed: int = 42) -> None:
    """
    HydraulicDecoder.node_head만 fine-tune (나머지 레이어 동결).

    Bellinge node embeddings (from frozen encoder/processor) → target depth 매핑을
    Bellinge SWMM 기준으로 재보정. 전체 파라미터 대비 ~4% (147K/4M) 만 업데이트.
    """
    import torch
    from torch_geometric.loader import DataLoader

    torch.manual_seed(seed)

    # 모든 파라미터 동결 후 decoder.node_head만 해제
    for param in model.parameters():
        param.requires_grad_(False)
    n_unfrozen = 0
    for member in model.members:
        for param in member.decoder.node_head.parameters():
            param.requires_grad_(True)
            n_unfrozen += param.numel()
    logger.info(f"Fine-tune: {n_unfrozen:,} unfrozen params (node_head × {len(model.members)} members)")

    opt = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    loader = DataLoader(ft_dataset, batch_size=1, shuffle=True)
    device = next(model.parameters()).device

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            out = model(batch)
            pred = out["mean_depth"]        # (N, T)
            tgt  = batch.y                  # (N, T)
            loss = torch.nn.functional.mse_loss(pred, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if epoch % 20 == 0 or epoch == 1:
            logger.info(f"  [FT epoch {epoch:3d}/{n_epochs}] MSE={total_loss/max(n_batches,1):.6f}")

    # 추론 모드 복원 + 파라미터 재동결 (이후 eval은 model.eval()에서 진행)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    logger.info("Fine-tuning 완료.")


# ── GNN 평가 ─────────────────────────────────────────────────────────────────

def _fit_linear_calibration(pred_flat: np.ndarray,
                             true_flat: np.ndarray) -> tuple:
    """
    Fit affine depth scale calibration: true ≈ a * pred + b
    Solved analytically via OLS (2-parameter fit).
    """
    p = pred_flat - pred_flat.mean()
    t = true_flat - true_flat.mean()
    a = np.dot(p, t) / (np.dot(p, p) + 1e-12)
    b = true_flat.mean() - a * pred_flat.mean()
    return float(a), float(b)


def evaluate_bellinge_gnn(model, dataset, config: Dict, device,
                           calib_loader=None,
                           n_cal: int = 10,
                           seed: int = 42) -> Dict:
    """
    Loop N 모델로 Bellinge 추론 → 메트릭 산출.

    Depth scale calibration:
      Bellinge 깊이 스케일(평균 2.56m)이 서초구 학습 범위(0.06-0.15m)를 크게
      벗어남. n_cal개 SWMM reference 샘플로 affine calibration(y = a*x + b)을
      피팅 후 나머지 (N-n_cal)개 샘플로 독립 평가.
    """
    import torch
    from torch_geometric.loader import DataLoader
    from src.models.conformal_uq import ConformalPredictor
    from src.evaluate import compute_nse, compute_rmse, compute_picr, compute_sharpness

    alpha = config.get("experiment", {}).get("conformal_alpha", 0.10)
    cp = ConformalPredictor(alpha=alpha)
    has_calib = False

    # Seocho-gu calib set으로 conformal 캘리브레이션
    if calib_loader is not None:
        cal_preds, cal_trues = [], []
        with torch.no_grad():
            for batch in calib_loader:
                batch = batch.to(device)
                out = model(batch)
                cal_preds.append(out["mean_depth"].cpu().numpy())
                cal_trues.append(batch.y.cpu().numpy())
        if cal_preds:
            yp = np.concatenate(cal_preds, 0)
            yt = np.concatenate(cal_trues, 0)
            cp.calibrate(yp.reshape(len(yp), -1),
                         yt.reshape(len(yt), -1), regime_id="B")
            has_calib = True
            logger.info("Conformal 캘리브레이션: Seocho-gu calib set 사용")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    all_pred, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            all_pred.append(out["mean_depth"].cpu().numpy())
            all_true.append(batch.y.cpu().numpy())

    if not all_pred:
        return {}

    # NOTE (bug fix 2026-07-19): must be np.stack, not np.concatenate.
    # Each element of all_pred/all_true has shape (nodes, T) (one sample,
    # batch_size=1). concatenate() on axis 0 merges the node axis into a
    # single (N*nodes, T) array, so N = len(all_pred) no longer matches
    # depth_pred.shape[0] and the cal_idx/test_idx permutation below (drawn
    # from range(N)) ends up selecting arbitrary node-rows of the first one
    # or two samples instead of whole held-out events. stack() preserves the
    # intended (N, nodes, T) shape so per-sample indexing is correct.
    depth_pred = np.stack(all_pred, 0)   # (N, nodes, T)
    depth_true = np.stack(all_true, 0)

    N = len(all_pred)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    cal_idx  = idx[:n_cal]
    test_idx = idx[n_cal:]

    fp_raw = depth_pred.flatten()
    ft_raw = depth_true.flatten()

    # ── Raw (zero-shot) metrics ────────────────────────────────────────────────
    nse_raw = compute_nse(fp_raw, ft_raw)
    logger.info(f"Raw zero-shot NSE_bellinge: {nse_raw:.4f}")

    # ── Affine calibration ────────────────────────────────────────────────────
    n_cal_actual = min(n_cal, N - 1)
    cal_pred = depth_pred[cal_idx].flatten()
    cal_true = depth_true[cal_idx].flatten()
    a, b = _fit_linear_calibration(cal_pred, cal_true)
    logger.info(f"Affine calibration (n_cal={n_cal_actual}): a={a:.4f}, b={b:.4f}m")

    test_pred_raw  = depth_pred[test_idx]
    test_true      = depth_true[test_idx]
    test_pred_cal  = a * test_pred_raw + b

    fp_cal = test_pred_cal.flatten()
    ft_cal = test_true.flatten()

    nse_cal = compute_nse(fp_cal, ft_cal)
    nse_test_raw = compute_nse(test_pred_raw.flatten(), ft_cal)
    logger.info(f"Calibrated NSE_bellinge (n_test={len(test_idx)}): {nse_cal:.4f}")

    metrics = {
        # Legacy affine-corrected diagnostics, not the manuscript's primary result.
        "NSE_bellinge":            nse_cal,
        "RMSE_cm_bellinge":        compute_rmse(fp_cal, ft_cal),
        "n_samples":               N,
        "n_cal":                   n_cal_actual,
        "n_test":                  len(test_idx),
        "cal_scale_a":             round(a, 4),
        "cal_offset_b_m":          round(b, 4),
        "primary_metric":          "NSE_bellinge_raw",
        "NSE_bellinge_affine":     nse_cal,
        # Raw pooled metrics over all supplied held-out samples (either stage).
        "NSE_bellinge_raw":        round(nse_raw, 4),
        "NSE_bellinge_test_raw":   round(nse_test_raw, 4),
        "RMSE_cm_bellinge_raw":    round(compute_rmse(fp_raw, ft_raw), 2),
    }

    if has_calib:
        # q_hat has shape (T,) (one pooled quantile per timestep, calibrated
        # on Seocho-gu calib_loader data). test_pred_cal now has the correct
        # (n_test, nodes, T) shape after the stack() fix above, so it
        # broadcasts against q_hat along the last axis directly -- no reshape
        # needed. The previous `.reshape(len(test_idx), -1)` flattened
        # nodes*T into one axis (n_test, nodes*T), which is not broadcastable
        # against a length-T q_hat and would raise (or, under the old
        # concatenate-based depth_pred bug, silently line up with q_hat by
        # accident because each "row" was actually a single node's T-length
        # series rather than a whole sample).
        iv = cp.predict(test_pred_cal, regime_id="B")
        lo = iv["lower"]
        hi = iv["upper"]
        metrics["PICR_90_bellinge"] = compute_picr(lo.flatten(), hi.flatten(), ft_cal)
        metrics["PICR_sharpness_cm_bellinge"] = compute_sharpness(lo.flatten(), hi.flatten())

    return metrics


# ── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    import torch
    args = parse_args()

    # 설정 로드
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 필수 파일 확인
    for p, label in [(BELLINGE_INP, "SWMM .inp"), (BELLINGE_DAT, "강우 .dat")]:
        if not p.exists():
            logger.error(f"{label} 없음: {p}")
            sys.exit(1)

    logger.info(f"Bellinge .inp : {BELLINGE_INP}")
    logger.info(f"Bellinge .dat : {BELLINGE_DAT}")

    # ── 강우 이벤트 선별 ──────────────────────────────────────────────────────
    rain_1min = parse_dat_file(BELLINGE_DAT)
    logger.info("유의 강우 이벤트 선별 중...")
    events = select_events(rain_1min,
                           n_events=args.n_events,
                           min_total_mm=args.min_rain_mm,
                           seed=args.seed)

    # rain_series: 1시간 해상도 (SWMMGraphDataset 요구)
    rain_hourly = rain_1min.resample("1h").sum().fillna(0)

    # 이벤트 카탈로그 저장
    BELLINGE_CAT.parent.mkdir(parents=True, exist_ok=True)
    with open(BELLINGE_CAT, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2, ensure_ascii=False)
    logger.info(f"이벤트 카탈로그 → {BELLINGE_CAT}")

    # ── SWMM 앙상블 ───────────────────────────────────────────────────────────
    if not args.skip_swmm:
        from src.data.swmm_ensemble import latin_hypercube_sample
        param_sets = latin_hypercube_sample(BELLINGE_PARAM_RANGES,
                                             args.n_sets, args.seed)
        for i, ps in enumerate(param_sets):
            ps["set_id"] = i

        total_runs = args.n_sets * len(events)
        logger.info(f"SWMM 앙상블 시작: {args.n_sets}세트 × {len(events)}이벤트 "
                    f"= {total_runs}회")
        t0 = time.time()
        summary = run_bellinge_ensemble(
            BELLINGE_INP, BELLINGE_DAT,
            param_sets, events, BELLINGE_ENS,
            skip_existing=True,
        )
        logger.info(f"SWMM 완료 ({time.time()-t0:.0f}s): {summary}")
    else:
        logger.info("SWMM 앙상블 건너뜀 (--skip_swmm)")

    # ── Bellinge 데이터셋 구성 ─────────────────────────────────────────────────
    from src.data.swmm_dataset import SWMMGraphDataset
    exp = config.get("experiment", {})
    dataset = SWMMGraphDataset(
        inp_path=str(BELLINGE_INP),
        ensemble_dir=str(BELLINGE_ENS),
        catalog=events,
        rain_series=rain_hourly,
        T_out=exp.get("T_out", 100),
        T_rain=exp.get("T_rain", 72),
        n_sets=args.n_sets + 5,  # 여유
        site_id="bellinge",
    )
    if len(dataset) == 0:
        logger.error("Bellinge 데이터셋 비어있음. SWMM 앙상블 결과를 확인하세요.")
        sys.exit(1)
    logger.info(f"Bellinge 데이터셋: {len(dataset)} 샘플")

    # ── Feature alignment: max_depth=0 (Seocho-gu 훈련 분포 일치) ──────────────
    # Seocho-gu 학습 데이터: 모든 junction max_depth=0 → GNN이 항상 0으로 학습.
    # Bellinge max_depth=2.56m(중앙값)는 완전히 OOD → feature index 1을 0으로 강제.
    logger.info("Feature alignment: Bellinge max_depth (feat[1]) → 0 (Seocho-gu 분포 일치)")
    # ── Feature alignment: infiltration params (inf_max/inf_min/inf_decay,
    # feat[-3:]) → Seocho 50-set 학습 평균으로 고정 ──────────────────────────
    # Bellinge는 nopervious 모델이라 이 세 Green-Ampt 파라미터가 SWMM 결과에
    # 영향을 주지 않지만(dstore_perv도 마찬가지 -- build_event_inp_bellinge가
    # INFILTRATION 및 S-Perv를 .inp에 반영하지 않음), param_set.npz에 저장된
    # 구버전(2.5-5.0mm/hr) inf_max 값이 그대로 GNN 조건 입력에 들어가면 학습
    # 분포(30-250mm)와 스케일이 완전히 달라 out-of-distribution 신호가 된다.
    # 물리적으로 무의미한 네 값(dstore_perv, inf_max, inf_min, inf_decay)을
    # 서초구 학습 평균(5.0033, 140.14, 0.5148, 0.2750)으로 고정한다.
    logger.info("Feature alignment: Bellinge dstore_perv/infiltration params (feat[-4:]) → Seocho-gu 50-set 학습 평균")
    _INF_MEAN = [5.0033, 140.14, 0.5148, 0.2750]  # dstore_perv, inf_max, inf_min, inf_decay
    for sample in dataset:
        sample.x[:, 1] = 0.0
        for _j, _v in enumerate(_INF_MEAN):
            sample.x[:, -4 + _j] = _v

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp_dir = Path(__file__).parent / f"results/checkpoints/loop_{args.loop:02d}"
    if not cp_dir.exists():
        logger.error(f"체크포인트 없음: {cp_dir}")
        sys.exit(1)

    model, M = load_loop_model(cp_dir, config, device)
    logger.info(f"Loop {args.loop} 로드 완료 (device={device}, M={M})")

    # Seocho-gu calib loader
    calib_loader = None
    try:
        from src.train import build_dataloaders
        _, _, calib_loader, _ = build_dataloaders(config, "B")
    except Exception as e:
        logger.warning(f"Seocho-gu calib 준비 실패 ({e}) → UQ 생략")

    # ── Decoder fine-tuning (옵션) ──────────────────────────────────────────────
    ft_dataset = dataset  # full for eval (fine-tune splits handled inside)
    if args.fine_tune:
        import torch as _torch
        rng = np.random.default_rng(args.seed)
        N = len(dataset)

        if args.event_split:
            # 이벤트 단위 split: 마지막 n_test_events개 이벤트를 test
            all_event_ids = [ev for ev, _ in dataset._index]
            unique_events = list(dict.fromkeys(all_event_ids))  # 순서 유지 고유값
            n_te = min(args.n_test_events, len(unique_events) - 1)
            test_events = set(unique_events[-n_te:])
            ft_idx   = [i for i, (ev, _) in enumerate(dataset._index) if ev not in test_events]
            test_idx = [i for i, (ev, _) in enumerate(dataset._index) if ev in test_events]
            logger.info(f"Event-level split: ft_events={unique_events[:-n_te]}, "
                        f"test_events={list(test_events)}")
        else:
            # 샘플 단위 random split (기존 동작)
            n_ft = min(args.n_ft, N - 10)  # 최소 10개 test 확보
            idx = rng.permutation(N)
            ft_idx   = idx[:n_ft].tolist()
            test_idx = idx[n_ft:].tolist()

        ft_data   = [dataset[i] for i in ft_idx]
        test_data = [dataset[i] for i in test_idx]
        logger.info(f"Fine-tune split: {len(ft_data)} train / {len(test_data)} test")
        t_ft = time.time()
        fine_tune_bellinge_decoder(model, ft_data,
                                    n_epochs=args.ft_epochs,
                                    lr=args.ft_lr,
                                    seed=args.seed)
        logger.info(f"Fine-tune 완료: {time.time()-t_ft:.0f}s")
        ft_dataset = test_data  # eval은 test set으로만

    # ── 추론 & 메트릭 ──────────────────────────────────────────────────────────
    logger.info("Bellinge GNN 추론 중...")
    t0 = time.time()
    if args.fine_tune:
        # fine-tune 후: affine calibration 없이 직접 평가 (10-sample cal은 잔여 오차용)
        metrics = evaluate_bellinge_gnn(model, ft_dataset, config, device,
                                         calib_loader=calib_loader,
                                         n_cal=min(5, len(ft_dataset)//4),
                                         seed=args.seed)
    else:
        metrics = evaluate_bellinge_gnn(model, ft_dataset, config, device,
                                         calib_loader=calib_loader,
                                         n_cal=args.n_cal, seed=args.seed)
    metrics.update({"loop_id": args.loop, "elapsed_s": round(time.time()-t0, 1),
                    "fine_tuned": args.fine_tune,
                    "n_ft_epochs": args.ft_epochs if args.fine_tune else 0})

    # ── 결과 출력 ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Bellinge 독립 검증 결과 (Loop 24, GIS-strict, current primary)")
    print("=" * 60)
    print(f"  [Raw zero-shot]  NSE : {metrics.get('NSE_bellinge_raw', 'N/A'):.4f}")
    print(f"  [Calibrated]     NSE : {metrics['NSE_bellinge']:.4f}  (목표 > 0.75)")
    print(f"  RMSE_cm_bellinge     : {metrics['RMSE_cm_bellinge']:.2f} cm")
    print(f"  Calibration a / b   : {metrics.get('cal_scale_a','?')} / {metrics.get('cal_offset_b_m','?')} m")
    print(f"  Cal samples / Test  : {metrics.get('n_cal','?')} / {metrics.get('n_test','?')}")
    if "PICR_90_bellinge" in metrics:
        print(f"  PICR_90_bellinge     : {metrics['PICR_90_bellinge']:.4f}")
        print(f"  Sharpness_cm         : {metrics['PICR_sharpness_cm_bellinge']:.2f} cm")
    print(f"  전체 샘플 수         : {metrics['n_samples']}")
    print(f"  소요 시간            : {metrics['elapsed_s']:.0f}s")
    status = "PASS [OK]" if metrics.get("passed") else "FAIL [목표 미달]"
    print(f"  판정                 : {status}")
    print("=" * 60 + "\n")

    # ── 저장 ──────────────────────────────────────────────────────────────────
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"결과 저장 → {out}")
    _append_learning(metrics)
    return metrics


def load_loop_model(cp_dir: Path, config: Dict, device):
    """EnsembleGNN 로드 (내부 유틸)."""
    import torch
    from src.models.gnn_surrogate import EnsembleGNN
    exp = config.get("experiment", {})
    base_cfg = {
        "node_feat_dim":      14,
        "edge_feat_dim":      4,
        "hidden_dim":         exp.get("hidden_dim", 128),
        "T_out":              exp.get("T_out", 100),
        "T_rain":             exp.get("T_rain", 72),
        "n_heads":            exp.get("n_heads", 4),
        "n_layers":           exp.get("n_layers", 4),
        "dropout":            exp.get("dropout", 0.0),
        "temporal_decoder":   exp.get("temporal_decoder", False),
        "scalar_rain_decoder":exp.get("scalar_rain_decoder", False),
    }
    M = exp.get("ensemble_size", 5)
    model = EnsembleGNN(base_cfg, M=M).to(device)
    from src.checkpoints import load_members
    load_members(model, cp_dir, device)
    return model, M


def _append_learning(metrics: Dict) -> None:
    path = Path(".research/learnings.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    nse = metrics.get("NSE_bellinge", float("nan"))
    nse_raw = metrics.get("NSE_bellinge_raw", float("nan"))
    ltype = "result" if metrics.get("passed") else "pitfall"
    a_str = metrics.get("cal_scale_a", "?")
    b_str = metrics.get("cal_offset_b_m", "?")
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "phase": "experiment",
        "loop": "bellinge_val_loop24",
        "type": ltype,
        "content": (
            f"Bellinge 독립 검증 (Loop {metrics.get('loop_id',24)}, GIS-strict primary): "
            f"NSE_calibrated={nse:.4f} ({'달성' if metrics.get('passed') else '미달'}), "
            f"NSE_raw={nse_raw:.4f}. "
            f"Affine calibration a={a_str}, b={b_str}m "
            f"(n_cal={metrics.get('n_cal',10)}, n_test={metrics.get('n_test',70)}). "
            f"RMSE={metrics.get('RMSE_cm_bellinge',0):.2f}cm. "
            f"DTU Bellinge CC BY 4.0, depth-scale calibration from Seocho-gu GNN."
        ),
        "tags": ["bellinge","external_validation","generalizability",
                 "NSE_bellinge","depth_scale_calibration","loop24_gis_strict"],
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info(f"Learning 기록 → {path}")


if __name__ == "__main__":
    main()
