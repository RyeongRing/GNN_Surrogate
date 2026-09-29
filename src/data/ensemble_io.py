"""
앙상블 실행 결과 저장/로드 유틸리티.
형식: data/ensemble/{event_id}/{set_id:03d}.npz
  - node_depth : (n_nodes, T) float32
  - link_flow  : (n_links, T) float32
  - node_names : (n_nodes,) str array
  - link_names : (n_links,) str array
  - param_set  : (7,) float32  [n_conduit, n_imperv, dstore_imperv, dstore_perv,
                                  inf_max, inf_min, inf_decay]
  - meta       : JSON string (event_id, set_id, timesteps, elapsed_s, ...)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

PARAM_KEYS = ["n_conduit", "n_imperv", "dstore_imperv", "dstore_perv",
              "inf_max", "inf_min", "inf_decay"]

ENSEMBLE_DIR = Path(__file__).parents[2] / "data" / "ensemble"


def save_run(result: Dict, param_set: Dict,
             out_dir: Optional[Path] = None) -> Path:
    """
    단일 SWMM 실행 결과를 .npz로 저장.

    Parameters
    ----------
    result : _run_pyswmm / run_event 반환 딕셔너리
        keys: set_id, event_id, node_depth, link_flow, timesteps
    param_set : LHS 파라미터 딕셔너리
    out_dir : 저장 디렉토리 (기본: data/ensemble/{event_id}/)
    """
    event_id = result["event_id"]
    set_id   = result["set_id"]

    if out_dir is None:
        out_dir = ENSEMBLE_DIR / event_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{set_id:03d}.npz"

    node_depth_dict = result["node_depth"]
    link_flow_dict  = result["link_flow"]

    node_names = np.array(list(node_depth_dict.keys()), dtype=object)
    link_names = np.array(list(link_flow_dict.keys()),  dtype=object)

    depth_arr = np.array(list(node_depth_dict.values()), dtype=np.float32)  # (N, T)
    flow_arr  = np.array(list(link_flow_dict.values()),  dtype=np.float32)  # (E, T)

    param_vec = np.array([param_set.get(k, 0.0) for k in PARAM_KEYS], dtype=np.float32)

    meta = {
        "event_id":  event_id,
        "set_id":    set_id,
        "timesteps": result.get("timesteps", depth_arr.shape[1]),
        "elapsed_s": result.get("elapsed_s", 0),
    }

    np.savez_compressed(
        out_path,
        node_depth=depth_arr,
        link_flow=flow_arr,
        node_names=node_names,
        link_names=link_names,
        param_set=param_vec,
        meta=np.array([json.dumps(meta)]),
    )

    size_mb = out_path.stat().st_size / 1e6
    logger.info(f"Saved {event_id}/set{set_id:03d} → {out_path.name} ({size_mb:.1f} MB)")
    return out_path


def load_run(event_id: str, set_id: int,
             ensemble_dir: Optional[Path] = None) -> Dict:
    """단일 실행 결과 로드."""
    if ensemble_dir is None:
        ensemble_dir = ENSEMBLE_DIR
    path = ensemble_dir / event_id / f"{set_id:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No ensemble result: {path}")

    data = np.load(path, allow_pickle=True)
    node_names = data["node_names"].tolist()
    link_names = data["link_names"].tolist()
    meta = json.loads(data["meta"][0])

    return {
        "event_id":   meta["event_id"],
        "set_id":     meta["set_id"],
        "timesteps":  meta["timesteps"],
        "node_depth": {node_names[i]: data["node_depth"][i].tolist()
                       for i in range(len(node_names))},
        "link_flow":  {link_names[i]: data["link_flow"][i].tolist()
                       for i in range(len(link_names))},
        "param_set":  dict(zip(PARAM_KEYS, data["param_set"].tolist())),
    }


def load_run_arrays(event_id: str, set_id: int,
                    ensemble_dir: Optional[Path] = None) -> Dict:
    """단일 실행 결과를 numpy 배열로 로드 (Python 루프 없음, 빠름)."""
    if ensemble_dir is None:
        ensemble_dir = ENSEMBLE_DIR
    path = ensemble_dir / event_id / f"{set_id:03d}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No ensemble result: {path}")

    raw = np.load(path, allow_pickle=True)
    meta = json.loads(raw["meta"][0])
    return {
        "event_id":      meta["event_id"],
        "set_id":        meta["set_id"],
        "timesteps":     meta["timesteps"],
        "node_depth":    raw["node_depth"],        # (N, T) float32 — whole array
        "link_flow":     raw["link_flow"],         # (E, T) float32
        "node_names":    raw["node_names"].tolist(),
        "link_names":    raw["link_names"].tolist(),
        "param_set":     dict(zip(PARAM_KEYS, raw["param_set"].tolist())),
    }


def load_event_array(event_id: str, n_sets: int = 50,
                     ensemble_dir: Optional[Path] = None
                     ) -> Dict[str, np.ndarray]:
    """
    이벤트의 모든 파라미터 세트를 배열로 로드.
    반환:
      node_depth : (n_sets, n_nodes, T) float32
      link_flow  : (n_sets, n_links, T) float32
      param_sets : (n_sets, 7) float32
      node_names / link_names
    """
    if ensemble_dir is None:
        ensemble_dir = ENSEMBLE_DIR
    event_dir = ensemble_dir / event_id

    depths, flows, params = [], [], []
    node_names = link_names = None

    for sid in range(n_sets):
        path = event_dir / f"{sid:03d}.npz"
        if not path.exists():
            logger.warning(f"Missing: {path}")
            continue
        data = np.load(path, allow_pickle=True)
        depths.append(data["node_depth"])
        flows.append(data["link_flow"])
        params.append(data["param_set"])
        if node_names is None:
            node_names = data["node_names"].tolist()
            link_names = data["link_names"].tolist()

    return {
        "node_depth": np.stack(depths, axis=0),   # (n_sets, N, T)
        "link_flow":  np.stack(flows,  axis=0),   # (n_sets, E, T)
        "param_sets": np.stack(params, axis=0),   # (n_sets, 7)
        "node_names": node_names,
        "link_names": link_names,
    }


def list_completed(ensemble_dir: Optional[Path] = None) -> Dict[str, List[int]]:
    """완료된 (event_id, set_ids) 목록 반환."""
    if ensemble_dir is None:
        ensemble_dir = ENSEMBLE_DIR
    completed = {}
    if not ensemble_dir.exists():
        return completed
    for event_dir in sorted(ensemble_dir.iterdir()):
        if event_dir.is_dir():
            sets = sorted(int(f.stem) for f in event_dir.glob("*.npz")
                          if f.stem.isdigit())
            if sets:
                completed[event_dir.name] = sets
    return completed
