"""
SWMM 네트워크 → PyTorch Geometric 그래프 변환 파이프라인.
서초구 및 Bellinge 공통 인터페이스.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import torch
    from torch_geometric.data import Data
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    logger.warning("torch_geometric not found.")

try:
    import networkx as nx
    NX_AVAILABLE = True
except ImportError:
    NX_AVAILABLE = False

REGIME_ONEHOT = {"A": [1, 0, 0], "B": [0, 1, 0], "C": [0, 0, 1]}


# ── INP 직접 파싱 ─────────────────────────────────────────────────────────────

def _parse_inp_sections(inp_path: Path) -> Dict[str, List[List[str]]]:
    """
    .inp 파일을 섹션별로 파싱.
    반환: {section_name: [[col1, col2, ...], ...]} (주석·빈 줄 제외)
    """
    sections: Dict[str, List[List[str]]] = {}
    current = None
    text = inp_path.read_text(encoding="cp949", errors="replace")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1].upper()
            sections.setdefault(current, [])
        elif current is not None and stripped and not stripped.startswith(";"):
            sections[current].append(stripped.split())
    return sections


def _section_to_df(rows: List[List[str]], columns: List[str]) -> pd.DataFrame:
    """섹션 행 리스트 → DataFrame (열 수 불일치는 앞부터 채움)."""
    parsed = []
    for row in rows:
        if len(row) >= len(columns):
            parsed.append(row[:len(columns)])
        else:
            padded = row + ["0"] * (len(columns) - len(row))
            parsed.append(padded)
    return pd.DataFrame(parsed, columns=columns)


def parse_network_from_inp(inp_path: Path):
    """
    SWMM .inp → (node_df, link_df, xsec_df, coord_df)
    node_df: node_id, type, invert_elev, max_depth, ponded_area
    link_df: link_id, from_node, to_node, length, roughness
    xsec_df: link_id, shape, geom1
    coord_df: node_id, x, y
    """
    secs = _parse_inp_sections(inp_path)

    # ── 노드 ──────────────────────────────────────────────────────────────────
    node_rows = []

    # JUNCTIONS: Name InvertElev MaxDepth InitDepth SurDepth Aponded
    for row in secs.get("JUNCTIONS", []):
        if len(row) >= 2:
            node_rows.append({
                "node_id":    row[0],
                "type":       "junction",
                "invert_elev": float(row[1]) if len(row) > 1 else 0.0,
                "max_depth":  float(row[2]) if len(row) > 2 else 0.0,
                "ponded_area": float(row[5]) if len(row) > 5 else 0.0,
            })

    # OUTFALLS: Name Elevation Type ...
    for row in secs.get("OUTFALLS", []):
        if len(row) >= 2:
            node_rows.append({
                "node_id":    row[0],
                "type":       "outfall",
                "invert_elev": float(row[1]) if len(row) > 1 else 0.0,
                "max_depth":  0.0,
                "ponded_area": 0.0,
            })

    # STORAGE: Name InvertEl MaxDepth InitDepth Shape ...
    for row in secs.get("STORAGE", []):
        if len(row) >= 2:
            node_rows.append({
                "node_id":    row[0],
                "type":       "storage",
                "invert_elev": float(row[1]) if len(row) > 1 else 0.0,
                "max_depth":  float(row[2]) if len(row) > 2 else 0.0,
                "ponded_area": 0.0,
            })

    node_df = pd.DataFrame(node_rows)
    if node_df.empty:
        node_df = pd.DataFrame(columns=["node_id", "type", "invert_elev", "max_depth", "ponded_area"])
    node_df = node_df.drop_duplicates("node_id").reset_index(drop=True)

    # ── 관거 ──────────────────────────────────────────────────────────────────
    # CONDUITS: Name FromNode ToNode Length Roughness InOffset OutOffset InitFlow MaxFlow
    link_rows = []
    for row in secs.get("CONDUITS", []):
        if len(row) >= 5:
            try:
                link_rows.append({
                    "link_id":   row[0],
                    "from_node": row[1],
                    "to_node":   row[2],
                    "length":    float(row[3]),
                    "roughness": float(row[4]),
                })
            except ValueError:
                pass

    link_df = pd.DataFrame(link_rows)
    if link_df.empty:
        link_df = pd.DataFrame(columns=["link_id", "from_node", "to_node", "length", "roughness"])

    # XSECTIONS: Link Shape Geom1 Geom2 Geom3 Geom4 Barrels Culvert
    xsec_rows = []
    for row in secs.get("XSECTIONS", []):
        if len(row) >= 3:
            try:
                xsec_rows.append({
                    "link_id": row[0],
                    "shape":   row[1],
                    "geom1":   float(row[2]),
                })
            except ValueError:
                pass

    xsec_df = pd.DataFrame(xsec_rows)
    if xsec_df.empty:
        xsec_df = pd.DataFrame(columns=["link_id", "shape", "geom1"])

    # COORDINATES: Name X-Coord Y-Coord
    coord_rows = []
    for row in secs.get("COORDINATES", []):
        if len(row) >= 3:
            try:
                coord_rows.append({
                    "node_id": row[0],
                    "x": float(row[1]),
                    "y": float(row[2]),
                })
            except ValueError:
                pass

    coord_df = pd.DataFrame(coord_rows)
    if coord_df.empty:
        coord_df = pd.DataFrame(columns=["node_id", "x", "y"])

    return node_df, link_df, xsec_df, coord_df


# ── 메인 클래스 ───────────────────────────────────────────────────────────────

class SWMMGraphBuilder:
    """SWMM 네트워크 정보를 PyG Data 객체로 변환."""

    def __init__(self, inp_path: str, site_id: str = "seocho",
                 node_catalog_csv: Optional[str] = None):
        self.inp_path = Path(inp_path)
        self.site_id = site_id
        self.node_catalog_csv = node_catalog_csv
        self._node_df: Optional[pd.DataFrame] = None
        self._link_df: Optional[pd.DataFrame] = None
        self._xsec_df: Optional[pd.DataFrame] = None
        self._coord_df: Optional[pd.DataFrame] = None
        self._node_index: Dict[str, int] = {}  # node_id → integer index
        self._graph = None

    def parse_network(self):
        """SWMM .inp 직접 파싱 → networkx DiGraph."""
        if not NX_AVAILABLE:
            raise ImportError("networkx required: pip install networkx")
        if not self.inp_path.exists():
            raise FileNotFoundError(f"SWMM .inp not found: {self.inp_path}")

        logger.info(f"Parsing SWMM network: {self.inp_path.name}")
        node_df, link_df, xsec_df, coord_df = parse_network_from_inp(self.inp_path)

        # geom1 (diameter) 관거에 병합
        if not xsec_df.empty:
            link_df = link_df.merge(xsec_df[["link_id", "geom1"]], on="link_id", how="left")
        if "geom1" not in link_df.columns:
            link_df["geom1"] = 0.0
        link_df["geom1"] = link_df["geom1"].fillna(0.0)

        self._node_df = node_df
        self._link_df = link_df
        self._xsec_df = xsec_df
        self._coord_df = coord_df
        self._node_index = {nid: i for i, nid in enumerate(node_df["node_id"])}

        # networkx DiGraph 구성
        G = nx.DiGraph()
        for _, row in node_df.iterrows():
            G.add_node(row["node_id"], **row.to_dict())
        for _, row in link_df.iterrows():
            G.add_edge(row["from_node"], row["to_node"],
                       link_id=row["link_id"],
                       length=row["length"],
                       roughness=row["roughness"],
                       geom1=row["geom1"])
        self._graph = G

        logger.info(f"  Nodes: {len(node_df)}, Links (conduits): {len(link_df)}")
        return G

    def build_node_features(self, regime_id: str = "B",
                             param_summary_vec: Optional[np.ndarray] = None,
                             n_params: int = 7) -> "torch.Tensor":
        """
        노드 피처 텐서:
          static (4): invert_elev, max_depth, ponded_area, is_storage
          regime (3): one-hot A/B/C
          param_summary (n_params): LHS 파라미터 요약벡터
        shape: (N_nodes, 4 + 3 + n_params)
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required")
        if self._node_df is None or self._node_df.empty:
            raise RuntimeError("parse_network() must be called first")

        n_nodes = len(self._node_df)
        elev   = self._node_df["invert_elev"].fillna(0.0).values.astype(np.float32)
        depth  = self._node_df["max_depth"].fillna(0.0).values.astype(np.float32)
        ponded = self._node_df["ponded_area"].fillna(0.0).values.astype(np.float32)
        is_storage = (self._node_df["type"] == "storage").astype(np.float32).values

        static = np.stack([elev, depth, ponded, is_storage], axis=1)  # (N, 4)
        regime_feat = np.tile(REGIME_ONEHOT[regime_id], (n_nodes, 1)).astype(np.float32)

        if param_summary_vec is None:
            param_summary_vec = np.zeros(n_params, dtype=np.float32)
        param_tile = np.tile(param_summary_vec.astype(np.float32), (n_nodes, 1))

        features = np.hstack([static, regime_feat, param_tile])  # (N, 4+3+n_params)
        return torch.tensor(features, dtype=torch.float32)

    def build_edge_features(self) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """
        엣지 피처:
          length, roughness, geom1 (diameter), slope=geom1/length (4)
        반환: (edge_index [2, E], edge_attr [E, 4])
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("torch_geometric required")
        if self._link_df is None or self._link_df.empty:
            raise RuntimeError("parse_network() must be called first")

        src_indices, dst_indices, attrs = [], [], []
        for _, row in self._link_df.iterrows():
            src = self._node_index.get(row["from_node"])
            dst = self._node_index.get(row["to_node"])
            if src is None or dst is None:
                continue
            length = float(row["length"]) if row["length"] > 0 else 1.0
            roughness = float(row["roughness"])
            geom1 = float(row["geom1"])
            slope = geom1 / length  # Diameter-to-length ratio, not an elevation gradient.
            src_indices.append(src)
            dst_indices.append(dst)
            attrs.append([length, roughness, geom1, slope])

        if not src_indices:
            return (torch.zeros((2, 0), dtype=torch.long),
                    torch.zeros((0, 4), dtype=torch.float32))

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_attr  = torch.tensor(attrs, dtype=torch.float32)
        return edge_index, edge_attr

    def build_pyg_data(self, rainfall_hyetograph: np.ndarray,
                        ensemble_output: Dict,
                        regime_id: str = "B",
                        param_summary_vec: Optional[np.ndarray] = None) -> "Data":
        """
        단일 이벤트 × 단일 파라미터 세트 → PyG Data 객체.
        x: node features (N, F)
        edge_index, edge_attr: 관거 연결
        y: 수위 시계열 (N, T)
        rain: 강우 hyetograph (T, 1)
        """
        x = self.build_node_features(regime_id, param_summary_vec)
        edge_index, edge_attr = self.build_edge_features()

        T = len(rainfall_hyetograph)
        n_nodes = x.shape[0]

        # ensemble_output["node_depth"]: {node_id: [depth_t0, depth_t1, ...]}
        node_depth = ensemble_output.get("node_depth", {})
        y_arr = np.zeros((n_nodes, T), dtype=np.float32)
        for node_id, idx in self._node_index.items():
            ts = node_depth.get(node_id, [])
            ts_len = min(len(ts), T)
            if ts_len > 0:
                y_arr[idx, :ts_len] = ts[:ts_len]

        rain = torch.tensor(rainfall_hyetograph, dtype=torch.float32).unsqueeze(-1)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor(y_arr), rain=rain,
                    regime_id=regime_id,
                    set_id=ensemble_output.get("set_id", -1))

    def build_dataset(self, event_list: List[str],
                      ensemble_dir: str,
                      rain_series: Optional[pd.Series] = None) -> List["Data"]:
        """이벤트 리스트 전체 → PyG Data 리스트."""
        if self._graph is None:
            self.parse_network()
        dataset = []
        ens_dir = Path(ensemble_dir)
        for event_id in event_list:
            ens_file = ens_dir / f"{event_id}.json"
            if not ens_file.exists():
                logger.warning(f"Missing ensemble file: {ens_file}")
                continue
            import json
            with open(ens_file) as f:
                ens_outputs = json.load(f)
            for ens_out in ens_outputs:
                if rain_series is not None:
                    start = ens_out.get("event_start", "")
                    end   = ens_out.get("event_end", "")
                    try:
                        rain_arr = rain_series.loc[start:end].values.astype(np.float32)
                    except Exception:
                        rain_arr = np.zeros(100, dtype=np.float32)
                else:
                    T = max((len(v) for v in ens_out.get("node_depth", {}).values()), default=100)
                    rain_arr = np.zeros(T, dtype=np.float32)
                data = self.build_pyg_data(rain_arr, ens_out)
                dataset.append(data)
        return dataset

    def save_dataset(self, dataset: List["Data"], out_path: str):
        with open(out_path, "wb") as f:
            pickle.dump(dataset, f)
        logger.info(f"Saved dataset ({len(dataset)} samples) → {out_path}")

    @property
    def n_nodes(self) -> int:
        return len(self._node_df) if self._node_df is not None else 0

    @property
    def n_edges(self) -> int:
        return len(self._link_df) if self._link_df is not None else 0


def validate_graph_connectivity(G) -> Dict:
    """고립 노드, 역방향 엣지 등 그래프 품질 검사."""
    if not NX_AVAILABLE:
        return {}
    isolated = list(nx.isolates(G))
    n_components = nx.number_weakly_connected_components(G)
    return {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "isolated_nodes": len(isolated),
        "n_weakly_connected_components": n_components,
    }
