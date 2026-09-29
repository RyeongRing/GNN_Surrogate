"""
SWMM 파라미터 앙상블 실행 스크립트.
pyswmm 기반 LHS(Latin Hypercube Sampling) 50세트 생성 및 병렬 실행.
pyswmm 미설치 시 CLI fallback 자동 전환.
"""

import json
import logging
import os
import subprocess
import tempfile
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import struct

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 파라미터 범위 — Modified Green-Ampt 물리 기반 확정 범위 (2026-07-07)
# Param1=흡입수두 Ψf(mm), Param2=포화투수계수 Ks(mm/hr), Param3=최대초기수분결핍 IMDmax(0~1)
# 근거:
#   n_conduit/n_imperv : Chow(1959) + SWMM User Manual Table 3-1 (baseline×배율)
#   dstore             : SWMM User Manual Table 3-2 (1.27~3.81mm imperv / 2.54~7.62mm perv)
#   inf Param1(Ψf)     : Rawls et al.(1983) 한국 도시 점토질-미사질점토 50~292mm 범위
#   inf Param2(Ks)     : Rawls et al.(1983) 도시 압밀토양 보정 0.03~1.0 mm/hr
#   inf Param3(IMDmax) : Rawls et al.(1983) 점토질 토양 0.10~0.45
PARAM_RANGES = {
    "n_conduit":     (0.007, 0.015),   # Manning's n 관거: baseline(0.010)×0.7~×1.5
    "n_imperv":      (0.006, 0.024),   # Manning's n 불투수면: baseline(0.012)×0.5~×2.0
    "dstore_imperv": (1.0,   4.5),     # 불투수면 저류깊이 mm: baseline(2.7) 포함, 매뉴얼 1.27~3.81
    "dstore_perv":   (2.0,   8.0),     # 투수면 저류깊이 mm: baseline(5.1) 포함, 매뉴얼 2.54~7.62
    "inf_max":       (30.0,  250.0),   # Ψf 흡입수두 mm: 서울 점토질 토양 Rawls(1983) 범위
    "inf_min":       (0.03,  1.0),     # Ks 포화투수계수 mm/hr: 도시 압밀 점토질 Rawls(1983)
    "inf_decay":     (0.10,  0.45),    # IMDmax 초기수분결핍 분율: Rawls(1983) 점토질
}

N_ENSEMBLE_DEFAULT = 50

# SWMM C 라이브러리는 한글/유니코드 경로 미지원 → ASCII 전용 임시 디렉토리 사용
# config.yaml의 swmm_tmp_dir 키로 재정의 가능; 기본값은 시스템 임시 경로 하위 ascii 디렉토리
import os as _os
_default_tmp = Path(tempfile.gettempdir()) / "swmm_tmp"
SWMM_TMP_DIR = Path(_os.environ.get("SWMM_TMP_DIR", str(_default_tmp)))

REGIME_MAP = {
    "A": "high_intensity",
    "B": "mixed",
    "C": "limited",
}

try:
    from pyswmm import Simulation, Nodes, Links
    PYSWMM_AVAILABLE = True
except ImportError:
    PYSWMM_AVAILABLE = False
    logger.warning("pyswmm not found. CLI fallback will be used.")


def latin_hypercube_sample(param_ranges: Dict, n: int, seed: int = 42) -> List[Dict]:
    """LHS로 파라미터 공간을 n개 세트로 균등 샘플링."""
    rng = np.random.default_rng(seed)
    keys = list(param_ranges.keys())
    n_params = len(keys)

    # LHS: 각 파라미터를 n개 구간으로 나누고 각 구간에서 하나씩 샘플
    samples = np.zeros((n, n_params))
    for j in range(n_params):
        perm = rng.permutation(n)
        u = (perm + rng.random(n)) / n
        lo, hi = param_ranges[keys[j]]
        samples[:, j] = lo + u * (hi - lo)

    return [dict(zip(keys, row)) for row in samples]


def build_event_inp(inp_path: Path, param_set: Dict,
                    event: Dict, rain_series: pd.Series,
                    out_path: Path,
                    warmup_days: int = 2) -> Path:
    """
    이벤트별 파라미터 교란 .inp 파일 생성.

    수정 대상:
      1. [OPTIONS] START_DATE / END_DATE (이벤트 기간)
      2. [TIMESERIES] 202208 블록 → 이벤트 강수 데이터로 교체
      3. [CONDUITS] roughness 열 → n_conduit
      4. [SUBAREAS] N-Imperv, S-Imperv 열 → n_imperv, dstore_imperv, dstore_perv
      5. [INFILTRATION] Param1~Param3 → inf_max, inf_min, inf_decay
    """
    lines = inp_path.read_text(encoding="cp949", errors="replace").splitlines()

    ev_start = pd.Timestamp(event["start"])
    ev_end   = pd.Timestamp(event["end"])
    sim_start = ev_start - pd.Timedelta(days=warmup_days)
    sim_end   = ev_end   + pd.Timedelta(hours=6)

    def fmt_date(ts): return ts.strftime("%m/%d/%Y")
    def fmt_time(ts): return ts.strftime("%H:%M:%S")

    new_lines = []
    section = None

    for line in lines:
        stripped = line.strip()

        # 섹션 감지
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1].upper()
            new_lines.append(line)
            continue

        # 주석/빈 줄 유지
        if stripped.startswith(";") or stripped == "":
            new_lines.append(line)
            continue

        # ── [OPTIONS] 날짜 교체 ──────────────────────────────
        if section == "OPTIONS":
            parts = stripped.split()
            if parts[0] == "START_DATE":
                new_lines.append(f"START_DATE           {fmt_date(sim_start)}")
                continue
            if parts[0] == "START_TIME":
                new_lines.append(f"START_TIME           {fmt_time(sim_start)}")
                continue
            if parts[0] == "REPORT_START_DATE":
                new_lines.append(f"REPORT_START_DATE    {fmt_date(ev_start)}")
                continue
            if parts[0] == "REPORT_START_TIME":
                new_lines.append(f"REPORT_START_TIME    {fmt_time(ev_start)}")
                continue
            if parts[0] == "END_DATE":
                new_lines.append(f"END_DATE             {fmt_date(sim_end)}")
                continue
            if parts[0] == "END_TIME":
                new_lines.append(f"END_TIME             {fmt_time(sim_end)}")
                continue

        # ── [TIMESERIES] 202208 블록 교체 ────────────────────
        if section == "TIMESERIES":
            parts = stripped.split()
            if parts and parts[0] == "202208":
                continue  # 기존 202208 행 제거

        # ── [CONDUITS] roughness 교체 ─────────────────────────
        if section == "CONDUITS":
            parts = stripped.split()
            if len(parts) >= 5 and not parts[0].startswith(";"):
                try:
                    float(parts[4])  # roughness 컬럼
                    parts[4] = f"{param_set['n_conduit']:.5f}"
                    new_lines.append("  ".join(parts))
                    continue
                except (ValueError, IndexError):
                    pass

        # ── [SUBAREAS] n_imperv, dstore_imperv, dstore_perv ──
        if section == "SUBAREAS":
            parts = stripped.split()
            if len(parts) >= 5 and not parts[0].startswith(";"):
                try:
                    float(parts[1])  # N-Imperv
                    parts[1] = f"{param_set['n_imperv']:.4f}"
                    parts[3] = f"{param_set['dstore_imperv']:.2f}"
                    parts[4] = f"{param_set['dstore_perv']:.2f}"
                    new_lines.append("  ".join(parts))
                    continue
                except (ValueError, IndexError):
                    pass

        # ── [INFILTRATION] inf_max, inf_min, inf_decay ────────
        if section == "INFILTRATION":
            parts = stripped.split()
            if len(parts) >= 4 and not parts[0].startswith(";"):
                try:
                    float(parts[1])  # Param1
                    parts[1] = f"{param_set['inf_max']:.3f}"
                    parts[2] = f"{param_set['inf_min']:.3f}"
                    parts[3] = f"{param_set['inf_decay']:.3f}"
                    new_lines.append("  ".join(parts))
                    continue
                except (ValueError, IndexError):
                    pass

        new_lines.append(line)

    # ── 새 TIMESERIES 블록 삽입 (기존 [TIMESERIES] 섹션 끝에 추가) ──
    timeseries_lines = [
        f";{event['event_id']}  {event['start']} ~ {event['end']}"
    ]
    # warmup 기간 포함 강수 슬라이스
    event_rain = rain_series.loc[sim_start:sim_end].copy()
    for ts, val in event_rain.items():
        date_str = ts.strftime("%m/%d/%Y")
        time_str = f"{ts.hour}:{ts.minute:02d}"
        timeseries_lines.append(
            f"202208           {date_str} {time_str:<10}    {val:.1f}"
        )

    # [TIMESERIES] 섹션 헤더 다음에 삽입
    final_lines = []
    inserted = False
    for line in new_lines:
        final_lines.append(line)
        if not inserted and line.strip() == "[TIMESERIES]":
            final_lines.append(";;Name           Date       Time       Value     ")
            final_lines.append(";;-------------- ---------- ---------- ----------")
            final_lines.extend(timeseries_lines)
            inserted = True

    out_path.write_text("\n".join(final_lines), encoding="utf-8")
    return out_path


def perturb_inp(inp_path: Path, param_set: Dict, out_path: Path) -> Path:
    """단순 파라미터 교란만 (이벤트 지정 없이). 테스트용."""
    dummy_event = {
        "event_id": "E000", "start": "2022-08-08 07:00", "end": "2022-08-10 03:00"
    }
    dummy_rain = pd.Series(dtype=float)
    return build_event_inp(inp_path, param_set, dummy_event, dummy_rain, out_path)


def extract_timeseries(sim_path: Path, node_list: List[str],
                       link_list: List[str]) -> pd.DataFrame:
    """
    SWMM 시뮬레이션 결과에서 노드 수위 및 링크 유량 시계열 추출.
    pyswmm 또는 swmm5 CLI 결과(.rpt) 파싱.
    """
    # TODO: 실 SWMM 결과 파싱 구현
    raise NotImplementedError("extract_timeseries: 실 데이터 확보 후 구현")


def _worker_init(project_root: str) -> None:
    """ProcessPoolExecutor initializer — 자식 프로세스 sys.path에 프로젝트 루트 추가."""
    import sys
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _process_worker(task: tuple) -> tuple:
    """
    ProcessPoolExecutor용 모듈 레벨 worker.
    각 프로세스는 독립 SWMM5 라이브러리 인스턴스를 가지므로 전역 상태 충돌 없음.

    task = (inp_path_str, param_set, event, rain_series, ensemble_dir_str)
    returns (event_id, set_id, "ok") or (event_id, set_id, error_msg)
    """
    import struct
    from pathlib import Path
    from pyswmm import Simulation
    from swmm.toolkit import output as swmm_out
    from src.data.ensemble_io import save_run
    from src.data.graph_builder import parse_network_from_inp

    inp_path_str, param_set, event, rain_series, ensemble_dir_str = task
    inp_path    = Path(inp_path_str)
    ensemble_dir = Path(ensemble_dir_str)
    event_id    = event["event_id"]
    pid         = os.getpid()

    SWMM_TMP_DIR.mkdir(parents=True, exist_ok=True)
    # PID 포함 → 동일 (set_id, event_id) 조합이 두 프로세스에 배정될 경우 충돌 방지
    tmp_inp = SWMM_TMP_DIR / f"s{param_set['set_id']:03d}_{event_id}_{pid}.inp"

    try:
        build_event_inp(inp_path, param_set, event, rain_series, tmp_inp)
        out_path = tmp_inp.with_suffix(".out")

        with Simulation(str(tmp_inp)) as sim:
            for step in sim:
                pass

        handle = swmm_out.init()
        swmm_out.open(handle, str(out_path))
        proj = swmm_out.get_proj_size(handle)
        n_nodes, n_links = proj[1], proj[2]

        with open(out_path, "rb") as _f:
            _f.seek(-3 * 4, 2)
            n_periods = struct.unpack("<i", _f.read(4))[0]

        depth_arr = np.zeros((n_nodes, n_periods), dtype=np.float32)
        flow_arr  = np.zeros((n_links, n_periods), dtype=np.float32)
        for t in range(n_periods):
            depth_arr[:, t] = swmm_out.get_node_attribute(handle, t, 0)
            flow_arr[:, t]  = swmm_out.get_link_attribute(handle, t, 0)
        swmm_out.close(handle)

        node_df, link_df, _, _ = parse_network_from_inp(inp_path)
        node_names = node_df["node_id"].tolist()
        link_names = link_df["link_id"].tolist()

        result = {
            "set_id":     param_set["set_id"],
            "event_id":   event_id,
            "node_depth": {node_names[i]: depth_arr[i].tolist() for i in range(n_nodes)},
            "link_flow":  {link_names[i]: flow_arr[i].tolist()  for i in range(n_links)},
            "timesteps":  n_periods,
        }
        save_run(result, param_set, out_dir=ensemble_dir / event_id)
        return event_id, param_set["set_id"], "ok"

    except Exception as e:
        return event_id, param_set["set_id"], str(e)

    finally:
        for ext in [".inp", ".rpt", ".out"]:
            try:
                tmp_inp.with_suffix(ext).unlink(missing_ok=True)
            except PermissionError:
                pass  # Windows: SWMM DLL이 파일 핸들을 아직 보유 중 — OS가 나중에 회수


class SWMMEnsembleRunner:
    """SWMM 파라미터 앙상블 실행 클래스."""

    def __init__(self, inp_path: str, rain_dat_path: str, output_dir: str):
        self.inp_path = Path(inp_path)
        self.rain_dat_path = Path(rain_dat_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.inp_path.exists():
            raise FileNotFoundError(f"SWMM .inp not found: {self.inp_path}")

    def build_param_sets(self, regime_id: str = "B",
                         n: int = N_ENSEMBLE_DEFAULT,
                         seed: int = 42) -> List[Dict]:
        """LHS로 파라미터 세트 생성."""
        param_sets = latin_hypercube_sample(PARAM_RANGES, n, seed)
        for i, ps in enumerate(param_sets):
            ps["regime_id"] = regime_id
            ps["set_id"] = i
        logger.info(f"Generated {len(param_sets)} parameter sets for regime {regime_id}")
        return param_sets

    def run_single(self, param_set: Dict, event: Dict,
                   rain_series: pd.Series,
                   swmm_exe: Optional[str] = None) -> Dict:
        """
        단일 파라미터 세트 × 단일 이벤트 SWMM 실행.
        임시 파일은 ASCII 경로(C:\\swmm_temp)에 생성.
        """
        SWMM_TMP_DIR.mkdir(parents=True, exist_ok=True)
        event_id = event["event_id"]
        tmp_inp = SWMM_TMP_DIR / f"s{param_set['set_id']:03d}_{event_id}.inp"
        try:
            build_event_inp(self.inp_path, param_set, event, rain_series, tmp_inp)

            if PYSWMM_AVAILABLE:
                return self._run_pyswmm(tmp_inp, param_set, event_id)
            else:
                return self._run_cli_fallback(tmp_inp, param_set, event_id, swmm_exe)
        finally:
            tmp_inp.unlink(missing_ok=True)
            Path(str(tmp_inp).replace(".inp", ".rpt")).unlink(missing_ok=True)
            Path(str(tmp_inp).replace(".inp", ".out")).unlink(missing_ok=True)

    def _get_inp_element_order(self):
        """원본 .inp 파싱 → (node_names, link_names) 순서 리스트 (캐시)."""
        if not hasattr(self, "_node_names_cache"):
            from src.data.graph_builder import parse_network_from_inp
            node_df, link_df, _, _ = parse_network_from_inp(self.inp_path)
            self._node_names_cache = node_df["node_id"].tolist()
            self._link_names_cache = link_df["link_id"].tolist()
        return self._node_names_cache, self._link_names_cache

    def _run_pyswmm(self, inp_path: Path, param_set: Dict, event_id: str) -> Dict:
        """
        pyswmm으로 시뮬레이션 실행 후 .out 바이너리 파일 파싱.
        Python-side 루프 기록 없이 ~15,000x 빠름.
        """
        from swmm.toolkit import output as swmm_out

        out_path = inp_path.with_suffix(".out")

        # 시뮬레이션 실행 (Python-side 기록 없음)
        with Simulation(str(inp_path)) as sim:
            for step in sim:
                pass

        # .out 파일 파싱
        handle = swmm_out.init()
        swmm_out.open(handle, str(out_path))

        proj = swmm_out.get_proj_size(handle)
        n_nodes, n_links = proj[1], proj[2]
        # swmm-toolkit get_times() 버그: 항상 REPORT_STEP(초) 반환 → 바이너리 직접 파싱
        with open(out_path, "rb") as _f:
            _f.seek(-3 * 4, 2)
            n_periods = struct.unpack("<i", _f.read(4))[0]

        depth_arr = np.zeros((n_nodes, n_periods), dtype=np.float32)
        flow_arr  = np.zeros((n_links, n_periods), dtype=np.float32)
        for t in range(n_periods):
            depth_arr[:, t] = swmm_out.get_node_attribute(handle, t, 0)
            flow_arr[:, t]  = swmm_out.get_link_attribute(handle, t, 0)

        swmm_out.close(handle)

        node_names, link_names = self._get_inp_element_order()
        return {
            "set_id":     param_set["set_id"],
            "event_id":   event_id,
            "node_depth": {node_names[i]: depth_arr[i].tolist() for i in range(n_nodes)},
            "link_flow":  {link_names[i]: flow_arr[i].tolist()  for i in range(n_links)},
            "timesteps":  n_periods,
        }

    def _run_cli_fallback(self, inp_path: Path, param_set: Dict,
                          event_id: str, swmm_exe: Optional[str]) -> Dict:
        """swmm5 CLI를 subprocess로 실행하는 fallback."""
        exe = swmm_exe or "swmm5"
        rpt_path = inp_path.with_suffix(".rpt")
        out_path = inp_path.with_suffix(".out")
        result = subprocess.run(
            [exe, str(inp_path), str(rpt_path), str(out_path)],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            raise RuntimeError(f"SWMM CLI failed: {result.stderr}")
        # TODO: .rpt / .out 파일 파싱 구현
        return {"set_id": param_set["set_id"], "event_id": event_id,
                "node_depth": {}, "link_flow": {}, "rpt_path": str(rpt_path)}

    def run_ensemble(self,
                     catalog: List[Dict],
                     rain_series: pd.Series,
                     event_ids: Optional[List[str]] = None,
                     n: int = N_ENSEMBLE_DEFAULT,
                     seed: int = 42,
                     max_workers: int = 4,
                     ensemble_dir: Optional[Path] = None,
                     skip_existing: bool = True) -> Dict[str, int]:
        """
        앙상블 병렬 실행 및 .npz 저장.

        Parameters
        ----------
        catalog      : event_catalog.json 전체 리스트
        rain_series  : 강수 시계열 (DatetimeIndex, 1h 해상도)
        event_ids    : 실행할 이벤트 ID 목록. None이면 catalog 전체
        n            : 파라미터 세트 수 (기본 50)
        seed         : LHS 시드
        max_workers  : 병렬 프로세스 수
        ensemble_dir : 저장 루트. None이면 data/ensemble/
        skip_existing: True면 이미 존재하는 .npz 건너뜀

        Returns
        -------
        {"total": int, "success": int, "skipped": int, "failed": int}
        """
        from src.data.ensemble_io import save_run, list_completed, ENSEMBLE_DIR

        out_dir = Path(ensemble_dir) if ensemble_dir else ENSEMBLE_DIR
        param_sets = latin_hypercube_sample(PARAM_RANGES, n, seed)
        for i, ps in enumerate(param_sets):
            ps["set_id"] = i

        events = {e["event_id"]: e for e in catalog}
        ids = event_ids if event_ids else list(events.keys())

        # 이미 완료된 (event_id, set_id) 쌍 수집
        completed = list_completed(out_dir) if skip_existing else {}

        tasks = []
        for ev_id in ids:
            done_sets = set(completed.get(ev_id, []))
            for ps in param_sets:
                if ps["set_id"] not in done_sets:
                    tasks.append((ps, events[ev_id]))

        skipped = n * len(ids) - len(tasks)
        total = len(tasks)
        success = 0
        failed = 0

        logger.info(f"Ensemble: {total} runs to execute, {skipped} skipped (existing)")

        # ProcessPoolExecutor: 각 프로세스 = 독립 SWMM5 인스턴스 → 전역 상태 충돌 없음
        # _process_worker는 모듈 레벨 함수 → Windows spawn에서 pickle 가능
        # initializer로 자식 프로세스 sys.path에 프로젝트 루트 추가
        project_root = str(Path(__file__).parents[2])
        proc_tasks = [
            (str(self.inp_path), ps, ev, rain_series, str(out_dir))
            for ps, ev in tasks
        ]
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(project_root,),
        ) as executor:
            futures = {executor.submit(_process_worker, t): t for t in proc_tasks}
            for future in as_completed(futures):
                ev_id, sid, status = future.result()
                if status == "ok":
                    success += 1
                    logger.info(f"  ✓ {ev_id}/set{sid:03d}")
                else:
                    failed += 1
                    logger.error(f"  ✗ {ev_id}/set{sid:03d}: {status}")

        summary = {"total": total, "success": success,
                   "skipped": skipped, "failed": failed}
        logger.info(f"Ensemble complete: {summary}")
        return summary
