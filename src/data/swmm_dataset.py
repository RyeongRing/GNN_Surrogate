"""
SWMM 앙상블 결과 → PyTorch Geometric Dataset.
ensemble_io(.npz) + graph_builder 정적 그래프를 결합해 학습용 Data 객체 생성.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    from torch_geometric.data import Data, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("torch_geometric not installed.")


REGIME_ONEHOT = {"A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 1]}


def _parse_report_step_seconds(inp_path: Path) -> float:
    """Read SWMM REPORT_STEP from the INP [OPTIONS] section."""
    in_options = False
    for raw_line in inp_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("["):
            in_options = line.upper() == "[OPTIONS]"
            continue
        if not in_options:
            continue
        fields = line.split()
        if fields[0].upper() != "REPORT_STEP" or len(fields) < 2:
            continue
        parts = [float(part) for part in fields[1].split(":")]
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours, minutes, seconds = 0.0, parts[0], parts[1]
        else:
            raise ValueError(f"Invalid REPORT_STEP in {inp_path}: {fields[1]}")
        report_step = hours * 3600.0 + minutes * 60.0 + seconds
        if report_step <= 0:
            raise ValueError(f"REPORT_STEP must be positive in {inp_path}")
        return report_step
    raise ValueError(f"REPORT_STEP not found in {inp_path}")


def _resampled_dt_seconds(T_in: int, T_out: int, report_step_seconds: float) -> float:
    """Physical spacing after linearly mapping the full input span to T_out."""
    if not np.isfinite(report_step_seconds) or report_step_seconds <= 0:
        raise ValueError("REPORT_STEP must be finite and positive")
    if T_in < 2 or T_out < 2:
        return float(report_step_seconds)
    return float(report_step_seconds) * (T_in - 1) / (T_out - 1)


def _resample_ts(arr: np.ndarray, T_out: int) -> np.ndarray:
    """1-D 또는 (N, T) 배열을 T_out 크기로 선형 보간 (벡터화)."""
    T_in = arr.shape[-1] if arr.ndim > 1 else len(arr)
    if T_in == T_out:
        return arr.astype(np.float32) if arr.dtype != np.float32 else arr
    # 보간 위치 (float, 0 ~ T_in-1)
    idx = np.linspace(0, T_in - 1, T_out)
    lo  = np.floor(idx).astype(np.int32)
    hi  = np.minimum(lo + 1, T_in - 1)
    t   = (idx - lo).astype(np.float32)          # [T_out]
    if arr.ndim == 1:
        arr = arr.astype(np.float32)
        return (1.0 - t) * arr[lo] + t * arr[hi]
    # (N, T) — fully vectorized
    arr = arr.astype(np.float32)
    return (1.0 - t) * arr[:, lo] + t * arr[:, hi]  # [N, T_out]


def _rain_window(rain_series: pd.Series, event: Dict,
                 T_rain: int, warmup_h: int = 48) -> np.ndarray:
    """
    이벤트 시작 warmup_h시간 전부터 이벤트 종료까지 강우 슬라이스 → T_rain 크기로 맞춤.
    hourly 강우 시계열을 가정.  짧으면 앞 zero-padding.
    """
    start = pd.Timestamp(event["start"]) - pd.Timedelta(hours=warmup_h)
    end   = pd.Timestamp(event["end"])
    rain = rain_series.loc[start:end].values.astype(np.float32)
    if len(rain) == 0 or not np.isfinite(rain).all():
        raise ValueError(f"Missing/non-finite rainfall for {event.get('event_id', 'event')} "
                         f"between {start} and {end}")
    if T_rain < 1:
        raise ValueError("T_rain must be positive")

    if len(rain) >= T_rain:
        return rain[-T_rain:]  # 마지막 T_rain 시간 사용
    # zero-padding (앞)
    padded = np.zeros(T_rain, dtype=np.float32)
    padded[-len(rain):] = rain
    return padded


class SWMMGraphDataset(Dataset):
    """
    SWMM 앙상블 결과(.npz) + 강우 시계열 → PyG Data 목록.

    Parameters
    ----------
    inp_path      : SWMM .inp 경로 (서초구 또는 Bellinge)
    ensemble_dir  : data/ensemble/ 루트
    catalog       : event_catalog.json 리스트 [{event_id, start, end, regime, ...}]
    rain_series   : 시간별 강수 pd.Series (DatetimeIndex)
    event_ids     : 사용할 이벤트 ID 목록. None이면 catalog 전체
    T_out         : 출력 타임스텝 수 (SWMM 출력 → 이 크기로 보간)
    T_rain        : 입력 강우 타임스텝 수 (hourly, zero-pad)
    n_sets        : 세트 수 (기본 50)
    site_id       : "seocho" | "bellinge"
    """

    def __init__(self,
                 inp_path: str,
                 ensemble_dir: str,
                 catalog: List[Dict],
                 rain_series: pd.Series,
                 event_ids: Optional[List[str]] = None,
                 T_out: int = 100,
                 T_rain: int = 72,
                 n_sets: int = 50,
                 site_id: str = "seocho"):
        if not TORCH_AVAILABLE:
            raise ImportError("torch_geometric required")
        super().__init__()

        self.inp_path     = Path(inp_path)
        self.ensemble_dir = Path(ensemble_dir)
        self.catalog      = {e["event_id"]: e for e in catalog}
        self.rain_series  = rain_series
        self.T_out        = T_out
        self.T_rain       = T_rain
        self.n_sets       = n_sets
        self.site_id      = site_id
        self.report_step_seconds = _parse_report_step_seconds(self.inp_path)

        ids = event_ids if event_ids else list(self.catalog.keys())
        self._build_graph_structure()
        self._index = self._build_index(ids)
        self._cache: Dict[int, "Data"] = {}
        # Lazy-initialized permutation: npz row order → graph node/link order
        self._node_perm: Optional[np.ndarray] = None
        self._link_perm: Optional[np.ndarray] = None

    # ── 정적 그래프 구조 (1회만 파싱) ─────────────────────────────────────────

    def _build_graph_structure(self):
        from src.data.graph_builder import parse_network_from_inp
        node_df, link_df, xsec_df, _ = parse_network_from_inp(self.inp_path)

        if not xsec_df.empty:
            link_df = link_df.merge(xsec_df[["link_id", "geom1"]], on="link_id", how="left")
        if "geom1" not in link_df.columns:
            link_df["geom1"] = 0.0
        link_df["geom1"] = link_df["geom1"].fillna(0.0)

        self._node_df   = node_df.reset_index(drop=True)
        self._link_df   = link_df.reset_index(drop=True)
        self._node_index: Dict[str, int] = {
            nid: i for i, nid in enumerate(node_df["node_id"])
        }

        # edge_index / edge_attr (고정) — node 미발견 관거 제외
        src_list, dst_list, attr_list, link_ids_used = [], [], [], []
        for _, row in link_df.iterrows():
            s = self._node_index.get(row["from_node"])
            d = self._node_index.get(row["to_node"])
            if s is None or d is None:
                continue
            length    = max(float(row["length"]), 1.0)
            roughness = float(row["roughness"])
            geom1     = float(row["geom1"])
            slope     = geom1 / length
            src_list.append(s)
            dst_list.append(d)
            attr_list.append([length, roughness, geom1, slope])
            link_ids_used.append(row["link_id"])

        self._edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
        self._edge_attr   = torch.tensor(attr_list, dtype=torch.float32)
        self._link_ids    = link_ids_used  # 실제 edge_index에 포함된 관거 ID 순서

        # 노드 정적 피처 (elev, max_depth, ponded_area, is_storage) — shape (N, 4)
        elev   = node_df["invert_elev"].fillna(0.0).values.astype(np.float32)
        depth  = node_df["max_depth"].fillna(0.0).values.astype(np.float32)
        ponded = node_df["ponded_area"].fillna(0.0).values.astype(np.float32)
        is_st  = (node_df["type"] == "storage").astype(np.float32).values
        self._node_static = np.stack([elev, depth, ponded, is_st], axis=1)  # (N, 4)

        logger.info(f"Graph: {len(node_df)} nodes, {len(link_df)} links")

    # ── 인덱스 구축 ────────────────────────────────────────────────────────────

    def _build_index(self, event_ids: List[str]) -> List[Tuple[str, int]]:
        """완료된 (event_id, set_id) 쌍 목록."""
        from src.data.ensemble_io import list_completed
        completed = list_completed(self.ensemble_dir)
        index = []
        for ev_id in event_ids:
            if ev_id not in self.catalog:
                continue
            done_sets = completed.get(ev_id, [])
            for sid in done_sets:
                if sid < self.n_sets:
                    index.append((ev_id, sid))
        logger.info(f"Dataset: {len(index)} samples "
                    f"({len(event_ids)} events × up to {self.n_sets} sets)")
        return index

    # ── PyG Dataset interface ──────────────────────────────────────────────────

    def len(self) -> int:
        return len(self._index)

    def _build_permutations(self, node_names: List[str],
                            link_names: List[str]) -> None:
        """npz 저장 순서 → 그래프 인덱스 순서 매핑 (최초 1회만 실행)."""
        link_name_to_idx = {lid: j for j, lid in enumerate(self._link_ids)}
        self._node_perm = np.array(
            [self._node_index.get(n, -1) for n in node_names], dtype=np.int32)
        self._link_perm = np.array(
            [link_name_to_idx.get(l, -1) for l in link_names], dtype=np.int32)

    def get(self, idx: int) -> "Data":
        if idx in self._cache:
            return self._cache[idx]
        from src.data.ensemble_io import load_run_arrays
        ev_id, set_id = self._index[idx]
        event = self.catalog[ev_id]
        regime = event.get("regime", "B")

        result = load_run_arrays(ev_id, set_id, ensemble_dir=self.ensemble_dir)

        # build permutation once using first sample's name order
        if self._node_perm is None:
            self._build_permutations(result["node_names"], result["link_names"])

        # ── 강우 입력 (T_rain, 1) ──
        rain_arr = _rain_window(self.rain_series, event, self.T_rain)
        rain = torch.tensor(rain_arr, dtype=torch.float32).unsqueeze(-1)

        # ── 수위 타겟 y (N, T_out) — numpy fancy-index, no Python loop ──
        n_nodes  = len(self._node_df)
        n_links  = len(self._link_ids)
        T_swmm   = result["timesteps"]

        nd = result["node_depth"]            # (N_npz, T_npz) float32
        T_use = min(nd.shape[1], T_swmm)
        dt_seconds = _resampled_dt_seconds(
            T_use, self.T_out, self.report_step_seconds)
        y_raw = np.zeros((n_nodes, T_use), dtype=np.float32)
        valid_n = self._node_perm >= 0
        y_raw[self._node_perm[valid_n]] = nd[valid_n, :T_use]
        y = torch.tensor(_resample_ts(y_raw, self.T_out))

        # ── 유량 타겟 y_flow (E, T_out) — numpy fancy-index, no Python loop ──
        lf = result["link_flow"]             # (E_npz, T_npz) float32
        f_raw = np.zeros((n_links, T_use), dtype=np.float32)
        valid_l = self._link_perm >= 0
        f_raw[self._link_perm[valid_l]] = lf[valid_l, :T_use]
        y_flow = torch.tensor(_resample_ts(f_raw, self.T_out))

        # ── 노드 피처 x (N, 14) ──
        regime_tile = np.tile(REGIME_ONEHOT[regime],
                              (n_nodes, 1)).astype(np.float32)
        param_vec = np.array([result["param_set"].get(k, 0.0)
                               for k in ["n_conduit", "n_imperv", "dstore_imperv",
                                         "dstore_perv", "inf_max", "inf_min", "inf_decay"]],
                              dtype=np.float32)
        param_tile = np.tile(param_vec, (n_nodes, 1))
        x_np = np.hstack([self._node_static, regime_tile, param_tile])
        x = torch.tensor(x_np, dtype=torch.float32)

        max_depth_arr = self._node_static[:, 1].clip(min=0.1).astype(np.float32)
        data = Data(
            x=x,
            edge_index=self._edge_index,
            edge_attr=self._edge_attr,
            y=y,
            y_flow=y_flow,
            rain=rain,
            event_id=ev_id,
            set_id=set_id,
            dt_seconds=torch.tensor([dt_seconds], dtype=torch.float32),
            max_depth=torch.tensor(max_depth_arr, dtype=torch.float32),
        )
        self._cache[idx] = data
        return data


def build_splits(dataset: "SWMMGraphDataset",
                 val_frac: float = 0.15,
                 test_frac: float = 0.15,
                 seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    이벤트 단위로 train/val/test 분할 (데이터 누수 방지).
    같은 이벤트의 모든 파라미터 세트는 같은 split에 배정.
    """
    rng = np.random.default_rng(seed)
    events = sorted(set(ev for ev, _ in dataset._index))
    rng.shuffle(events)

    n_val  = max(1, int(len(events) * val_frac))
    n_test = max(1, int(len(events) * test_frac))
    val_events  = set(events[:n_val])
    test_events = set(events[n_val:n_val + n_test])

    train_idx, val_idx, test_idx = [], [], []
    for i, (ev_id, _) in enumerate(dataset._index):
        if ev_id in test_events:
            test_idx.append(i)
        elif ev_id in val_events:
            val_idx.append(i)
        else:
            train_idx.append(i)

    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    return train_idx, val_idx, test_idx
