"""
평가 스크립트 — 전체 메트릭 산출 및 결과 저장.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def compute_nse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency."""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    if denominator < 1e-12:
        return float("nan")
    return float(1.0 - numerator / denominator)


def compute_rmse(y_pred: np.ndarray, y_true: np.ndarray,
                 unit_scale: float = 100.0) -> float:
    """RMSE (단위: cm, unit_scale=100 for m→cm)."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) * unit_scale)


def compute_mae(y_pred: np.ndarray, y_true: np.ndarray,
                unit_scale: float = 100.0) -> float:
    """MAE (단위: cm)."""
    return float(np.mean(np.abs(y_true - y_pred)) * unit_scale)


def compute_picr(lower: np.ndarray, upper: np.ndarray,
                 y_true: np.ndarray, alpha: float = 0.10) -> float:
    """Prediction Interval Coverage Rate."""
    covered = (y_true >= lower) & (y_true <= upper)
    return float(covered.mean())


def compute_sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
    """평균 interval width (cm)."""
    return float(np.mean(upper - lower) * 100.0)


def compute_continuity_violation(depth_pred: np.ndarray,
                                  flow_pred: np.ndarray,
                                  edge_index: np.ndarray,
                                  dt: float = 300.0,
                                  storage_area: Optional[np.ndarray] = None,
                                  tol: float = 0.01) -> float:
    """
    정규화 연속방정식 잔차가 tol을 초과하는 (노드, 시간) 비율.
    depth_pred: [N, T], flow_pred: [E, T], edge_index: [2, E]
    """
    N, T = depth_pred.shape
    if storage_area is None:
        storage_area = np.ones((N, 1))

    dS_dt = storage_area * (depth_pred[:, 1:] - depth_pred[:, :-1]) / dt

    src, dst = edge_index[0], edge_index[1]
    Q_in  = np.zeros((N, T - 1))
    Q_out = np.zeros((N, T - 1))
    np.add.at(Q_in,  dst, flow_pred[:, :T - 1])
    np.add.at(Q_out, src, flow_pred[:, :T - 1])

    # Per-node normalization: scale by peak flow magnitude at each node.
    # Dry nodes (Q_in≈Q_out≈0) get floored at 1e-3 so their near-zero
    # residuals never spuriously trigger violations.
    node_scale = np.maximum(
        np.abs(Q_in).max(axis=1, keepdims=True)
        + np.abs(Q_out).max(axis=1, keepdims=True),
        1e-3,
    )  # [N, 1]
    residual = np.abs(dS_dt - (Q_in - Q_out)) / node_scale
    violation_pct = float((residual > tol).mean() * 100.0)
    return violation_pct


def compute_primary_continuity_violation(depth_pred, flow_pred, edge_index,
                                         dt_seconds, tol=0.01):
    """Training-consistent proxy: A_i=1, event dt, no peak-flux normalization.

    Neither this diagnostic nor the normalized diagnostic validates mass
    conservation or agreement with SWMM conduit flows.
    """
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("dt_seconds must be finite and positive")
    net_flux = np.zeros((depth_pred.shape[0], depth_pred.shape[1] - 1))
    np.add.at(net_flux, edge_index[1], flow_pred[:, :-1])
    np.add.at(net_flux, edge_index[0], -flow_pred[:, :-1])
    residual = np.abs(np.diff(depth_pred, axis=1) / dt_seconds - net_flux)
    return float((residual > tol).mean() * 100.0)


def evaluate_all(depth_pred: np.ndarray, flow_pred: np.ndarray,
                 y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                 edge_index: np.ndarray,
                 regime_id: str = "B") -> Dict[str, float]:
    """research.targets 메트릭 전체 산출."""
    metrics = {
        "NSE":   compute_nse(depth_pred, y_true),
        "RMSE_cm": compute_rmse(depth_pred, y_true),
        "MAE_cm":  compute_mae(depth_pred, y_true),
        "PICR_90": compute_picr(lower, upper, y_true, alpha=0.10),
        "PICR_sharpness_cm": compute_sharpness(lower, upper),
        "continuity_violation_pct": compute_continuity_violation(
            depth_pred, flow_pred, edge_index
        ),
        "regime_id": regime_id,
    }
    return metrics


def plot_prediction_vs_truth(depth_pred: np.ndarray, depth_true: np.ndarray,
                              node_ids: List[str], out_path: str,
                              max_nodes: int = 6) -> None:
    """선택 노드의 예측 vs 실측 시계열 플롯."""
    if not MPL_AVAILABLE:
        logger.warning("matplotlib not available.")
        return
    n_plot = min(len(node_ids), max_nodes)
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 2 * n_plot))
    if n_plot == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(depth_true[i], label="SWMM reference", color="black", lw=1.5)
        ax.plot(depth_pred[i], label="Predicted", color="steelblue", lw=1.5)
        ax.set_title(f"Node: {node_ids[i]}")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info(f"Saved prediction plot → {out_path}")


def plot_interval_coverage(lower: np.ndarray, upper: np.ndarray,
                            y_true: np.ndarray, out_path: str,
                            sample_node: int = 0) -> None:
    """예측 구간 및 실측 플롯 (단일 노드)."""
    if not MPL_AVAILABLE:
        return
    t = np.arange(lower.shape[1])
    plt.figure(figsize=(10, 4))
    plt.fill_between(t, lower[sample_node], upper[sample_node],
                     alpha=0.3, label="90% PI", color="steelblue")
    plt.plot(t, y_true[sample_node], "k-", lw=1.5, label="SWMM reference")
    plt.legend()
    plt.xlabel("Time step")
    plt.ylabel("Water depth (m)")
    plt.title(f"Prediction Interval Coverage — Node {sample_node}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_results(metrics: Dict, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     type=str, default="configs/config_gauge401_fix_v1.yaml")
    p.add_argument("--checkpoint", type=str, default="results/checkpoints",
                   help="체크포인트 디렉토리 (member_*.pt 파일 포함)")
    p.add_argument("--regime",     type=str, default="B")
    p.add_argument("--site",       type=str, default="seocho",
                   choices=["seocho", "bellinge"])
    p.add_argument("--output",     type=str, default="results/eval.json")
    p.add_argument("--loop_id",    type=int, default=1)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--eval_split", type=str, default="test_temporal",
                   choices=["val_modelsel", "test_temporal"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def _load_ensemble_model(cp_dir: Path, base_cfg: dict, M: int, device):
    import torch
    from src.models.gnn_surrogate import PhysicsInformedGNN, EnsembleGNN
    model = EnsembleGNN(base_cfg, M=M).to(device)
    from src.checkpoints import load_members
    return load_members(model, cp_dir, device)


def main():
    import yaml
    import sys
    import torch

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    args = parse_args()
    if Path(args.output).exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {args.output}; use --overwrite explicitly")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_cfg  = config.get("experiment", {})
    hidden_dim       = exp_cfg.get("hidden_dim", 128)
    n_heads          = exp_cfg.get("n_heads",    4)
    n_layers         = exp_cfg.get("n_layers",   4)
    T_out            = exp_cfg.get("T_out",      100)
    M                = exp_cfg.get("ensemble_size", 5)
    alpha            = exp_cfg.get("conformal_alpha", 0.10)
    dropout              = exp_cfg.get("dropout", 0.0)
    temporal_decoder     = exp_cfg.get("temporal_decoder", False)
    scalar_rain_decoder  = exp_cfg.get("scalar_rain_decoder", False)
    conv_type            = exp_cfg.get("conv_type", "gat")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── データ ──────────────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(args.config).parent))
    from src.train import build_dataloaders
    _, val_loader, calib_loader, test_loader = build_dataloaders(config, args.regime)
    eval_loader = val_loader if args.eval_split == "val_modelsel" else test_loader

    if eval_loader is None or not len(eval_loader) or calib_loader is None or not len(calib_loader):
        raise ValueError("Evaluation and interval-calibration loaders must be nonempty")

    # ── モデル ─────────────────────────────────────────────────────────────────
    base_cfg = {
        "node_feat_dim": 14,
        "edge_feat_dim": 4,
        "hidden_dim":    hidden_dim,
        "T_out":         T_out,
        "n_heads":       n_heads,
        "n_layers":      n_layers,
        "dropout":       dropout,
        "temporal_decoder":    temporal_decoder,
        "scalar_rain_decoder": scalar_rain_decoder,
        "conv_type":           conv_type,
    }
    cp_dir = Path(args.checkpoint)
    model = _load_ensemble_model(cp_dir, base_cfg, M, device)

    # ── Conformal Calibration ──────────────────────────────────────────────────
    from src.models.conformal_uq import ConformalPredictor
    cp = ConformalPredictor(alpha=alpha)

    calib_preds, calib_trues = [], []
    with torch.no_grad():
        for batch in calib_loader:
            batch = batch.to(device)
            out = model(batch)
            calib_preds.append(out["mean_depth"].cpu().numpy())
            calib_trues.append(batch.y.cpu().numpy())

    if calib_preds:
        # PyG batches concatenate node dimensions flat, so out["mean_depth"]
        # per batch is already (batch_graphs * N_nodes, T), not (N_nodes, T);
        # concatenating across batches gives (n_calib_graphs * N_nodes, T)
        # directly -- there is no separate node axis left to reshape away.
        y_pred_cal = np.concatenate(calib_preds, axis=0)   # (n_calib_graphs * N_nodes, T)
        y_true_cal = np.concatenate(calib_trues, axis=0)
        # .reshape(len(y_pred_cal), -1) is a no-op here (already 2D); kept for
        # robustness if a caller ever passes a 3D array. ConformalPredictor
        # then takes a per-timestep quantile over axis=0, i.e. pooling over
        # all (calibration graph, node) pairs sharing a timestep -> q_hat has
        # shape (T,), one value per normalized timestep, not per node.
        cp.calibrate(y_pred_cal.reshape(len(y_pred_cal), -1),
                     y_true_cal.reshape(len(y_true_cal), -1),
                     regime_id=args.regime)

    # ── Inference on test set ──────────────────────────────────────────────────
    all_depth_pred, all_depth_true = [], []
    continuity_rates, primary_rates = [], []

    with torch.no_grad():
        for batch in eval_loader:
            batch = batch.to(device)
            out = model(batch)
            if getattr(batch, "num_graphs", 1) != 1:
                raise ValueError("Continuity diagnostics require one graph per batch")
            all_depth_pred.append(out["mean_depth"].cpu().numpy())
            all_depth_true.append(batch.y.cpu().numpy())
            flow_key = "mean_flow" if "mean_flow" in out else "flow_pred"
            if flow_key in out and torch.is_tensor(out[flow_key]):
                flow = out[flow_key].cpu().numpy()
            else:
                raise ValueError("Auxiliary edge-flux output is required for continuity diagnostics")
            depth = all_depth_pred[-1]
            edge_index = batch.edge_index.cpu().numpy()
            continuity_rates.append(compute_continuity_violation(depth, flow, edge_index))
            dt = float(batch.dt_seconds.detach().cpu().reshape(-1)[0])
            primary_rates.append(compute_primary_continuity_violation(depth, flow, edge_index, dt))

    depth_pred = np.concatenate(all_depth_pred, axis=0)
    depth_true = np.concatenate(all_depth_true, axis=0)

    # ── Prediction Intervals ───────────────────────────────────────────────────
    lower = upper = None
    if calib_preds:
        interval = cp.predict(
            depth_pred.reshape(len(depth_pred), -1),
            regime_id=args.regime
        )
        lower = interval["lower"].reshape(depth_pred.shape)
        upper = interval["upper"].reshape(depth_pred.shape)

    # ── Metrics ────────────────────────────────────────────────────────────────
    metrics: dict = {
        "NSE":     compute_nse(depth_pred.flatten(), depth_true.flatten()),
        "RMSE_cm": compute_rmse(depth_pred.flatten(), depth_true.flatten()),
        "MAE_cm":  compute_mae(depth_pred.flatten(), depth_true.flatten()),
        "regime":  args.regime,
        "site":    args.site,
        "loop_id": args.loop_id,
        "seed":    args.seed,
        "eval_split": args.eval_split,
    }
    if lower is not None:
        metrics["PICR_90"] = compute_picr(lower.flatten(), upper.flatten(),
                                           depth_true.flatten())
        metrics["PICR_sharpness_cm"] = compute_sharpness(lower.flatten(), upper.flatten())

    metrics["continuity_violation_pct"] = float(np.mean(continuity_rates))
    metrics["primary_continuity_violation_pct"] = float(np.mean(primary_rates))
    metrics["continuity_n_graphs"] = len(continuity_rates)
    metrics["continuity_definition"] = "Mean over ALL graphs; dt=300s, peak-flux normalized, tol=0.01"
    metrics["primary_continuity_definition"] = "Mean over ALL graphs; event dt, A_i=1, unnormalized, tol=0.01"
    metrics["config"] = args.config
    metrics["checkpoint"] = str(cp_dir)

    logger.info("=== Evaluation Results ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")

    save_results(metrics, args.output)
    logger.info(f"Results saved → {args.output}")


if __name__ == "__main__":
    main()
