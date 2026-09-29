"""Per-node conformal coverage disaggregation for the gauge-and-timestamp
-corrected primary model (loop 300, seed 42), replacing the stale loop-24
(pre-correction) analysis in results/pernode_coverage.json.

Same methodology as src/pernode_coverage.py and src/evaluate.py's own
ConformalPredictor-based calibration (val_conformal 2023 -> test_temporal
2024), but run through the gauge-and-timestamp-corrected dataset loader and
additionally exporting a per-node table (node_id, x, y, mean_depth, degree,
PICR_90) for spatial mapping in QGIS.

Usage (from project root, via the timecorrected entrypoint so the dataset
loader points at the gauge-fixed SWMM ensemble):
    python scripts/run_timecorrected_entrypoint.py src/pernode_coverage_gauge401fix.py --device auto
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LOOP_ID = 300
CONFIG_PATH = "configs/config_gauge401_fix_v1.yaml"
ALPHA = 0.10
INP_PATH = "inp_versions/seocho_imperv_landcover_strict.inp"


def load_model(config, device):
    from src.models.gnn_surrogate import EnsembleGNN
    exp_cfg = config["experiment"]
    base_cfg = {
        "node_feat_dim": 14, "edge_feat_dim": 4,
        "hidden_dim": exp_cfg.get("hidden_dim", 128),
        "T_out": exp_cfg.get("T_out", 100),
        "T_rain": exp_cfg.get("T_rain", 72),
        "n_heads": exp_cfg.get("n_heads", 4),
        "n_layers": exp_cfg.get("n_layers", 4),
        "dropout": exp_cfg.get("dropout", 0.0),
        "temporal_decoder": exp_cfg.get("temporal_decoder", False),
        "scalar_rain_decoder": exp_cfg.get("scalar_rain_decoder", False),
        "conv_type": exp_cfg.get("conv_type", "gat"),
    }
    M = exp_cfg.get("ensemble_size", 5)
    model = EnsembleGNN(base_cfg, M=M).to(device)
    cp_dir = Path("results") / "checkpoints" / f"loop_{LOOP_ID:02d}"
    from src.checkpoints import load_members
    load_members(model, cp_dir, device)
    return model


def collect(model, loader, device):
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out["mean_depth"].cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    if not preds:
        return None, None
    return np.stack(preds, axis=0), np.stack(trues, axis=0)  # (n_events, N, T)


def quartile_stats(per_node_picr, node_mask, n_nodes_total):
    n = int(node_mask.sum())
    if n == 0:
        return {"n_nodes": 0, "mean_picr_90": None}
    picr = float(per_node_picr[node_mask].mean())
    below_target = int((per_node_picr[node_mask] < (1 - ALPHA)).sum())
    return {
        "n_nodes": n,
        "frac_total": float(n / n_nodes_total),
        "mean_picr_90": picr,
        "nodes_below_target": below_target,
        "frac_nodes_below_target": float(below_target / n),
    }


def compute_node_degree(edge_index_np, n_nodes):
    degree = np.zeros(n_nodes, dtype=int)
    for i in range(edge_index_np.shape[1]):
        degree[edge_index_np[0, i]] += 1
        degree[edge_index_np[1, i]] += 1
    return degree


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--config", default=CONFIG_PATH)
    args = p.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    logger.info(f"Device: {device}")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = load_model(config, device)

    from src.train import build_dataloaders
    _, val_loader, calib_loader, test_loader = build_dataloaders(config, regime_id="B")
    if calib_loader is None or test_loader is None or len(test_loader) == 0:
        logger.error("Dataloaders missing/empty; aborting.")
        return

    # ── Conformal calibration (val_conformal 2023), matching evaluate.py ──────
    from src.models.conformal_uq import ConformalPredictor
    cp = ConformalPredictor(alpha=ALPHA)

    # NOTE: calibrate/predict must flatten to (n_samples*N_nodes, T) -- one row
    # per (event, node) pair, matching evaluate.py's own convention exactly --
    # NOT (n_events, N*T). The latter silently changes the calibration unit
    # from per-node-per-event to per-event and does not reproduce evaluate.py's
    # headline PICR_90 (verified against results/eval_gauge401fix_loop300_test_temporal.json).
    calib_pred, calib_true = collect(model, calib_loader, device)
    T_calib = calib_pred.shape[-1]
    cp.calibrate(calib_pred.reshape(-1, T_calib),
                 calib_true.reshape(-1, T_calib), regime_id="B")
    logger.info("Calibrated on val_conformal (2023, gauge-and-timestamp-corrected)")

    # ── Inference on test_temporal (2024) ──────────────────────────────────────
    test_pred, test_true = collect(model, test_loader, device)
    n_events, N_nodes, T = test_pred.shape
    logger.info(f"test_temporal shape: {test_pred.shape}")

    interval = cp.predict(test_pred.reshape(-1, T), regime_id="B")
    lower = interval["lower"].reshape(test_pred.shape)
    upper = interval["upper"].reshape(test_pred.shape)

    overall_picr = float(((test_true >= lower) & (test_true <= upper)).mean())
    logger.info(f"Overall PICR_90 (test_temporal, gauge401fix primary): {overall_picr:.4f}")

    covered = (test_true >= lower) & (test_true <= upper)
    per_node_picr = covered.mean(axis=(0, 2))  # (N,)
    logger.info(f"Per-node PICR_90: mean={per_node_picr.mean():.4f}, "
                f"min={per_node_picr.min():.4f}, max={per_node_picr.max():.4f}")

    # ── Depth quartiles ─────────────────────────────────────────────────────────
    mean_depth_per_node = test_true.mean(axis=(0, 2))
    quartile_edges = np.percentile(mean_depth_per_node, [0, 25, 50, 75, 100])
    labels = ["Q1 (shallowest)", "Q2", "Q3", "Q4 (deepest)"]
    quartile_results = {}
    quartile_id_per_node = np.zeros(N_nodes, dtype=int)
    for qi, (lo, hi, lab) in enumerate(zip(quartile_edges[:-1], quartile_edges[1:], labels)):
        if qi == 0:
            mask = mean_depth_per_node <= hi
        elif qi == 3:
            mask = mean_depth_per_node > lo
        else:
            mask = (mean_depth_per_node > lo) & (mean_depth_per_node <= hi)
        quartile_id_per_node[mask] = qi + 1
        stats = quartile_stats(per_node_picr, mask, N_nodes)
        stats["depth_range_m"] = [float(lo), float(hi)]
        quartile_results[lab] = stats
        logger.info(f"  {lab}: n={stats['n_nodes']}, PICR_90={stats['mean_picr_90']:.4f}")

    # ── Node degree ──────────────────────────────────────────────────────────────
    edge_index_np = None
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            edge_index_np = batch.edge_index.cpu().numpy()
            break
    degree_results = {}
    degree = np.zeros(N_nodes, dtype=int)
    if edge_index_np is not None:
        degree = compute_node_degree(edge_index_np, N_nodes)
        for d_label, d_mask in [
            ("degree_1 (inlets/outlets)", degree == 1),
            ("degree_2 (intermediate)", degree == 2),
            ("degree_3+ (junctions/trunks)", degree >= 3),
        ]:
            degree_results[d_label] = quartile_stats(per_node_picr, d_mask, N_nodes)

    # ── Node coordinates (same canonical order as the graph) ─────────────────────
    from src.data.graph_builder import parse_network_from_inp
    node_df, _, _, coord_df = parse_network_from_inp(Path(config["data"]["inp_path"]))
    node_ids = list(node_df["node_id"])
    if len(node_ids) != N_nodes:
        raise ValueError(f"Node mapping has {len(node_ids)} rows, expected {N_nodes}")

    # ── Save summary JSON (same schema as the old loop-24 file) ──────────────────
    out = {
        "loop_id": LOOP_ID,
        "protocol": "gauge-and-timestamp-corrected",
        "alpha": ALPHA,
        "eval_split": "test_temporal (2024, 5 events)",
        "n_eval_samples": int(n_events),
        "n_nodes": int(N_nodes),
        "T_out": int(T),
        "overall_picr_90": overall_picr,
        "per_node_picr_mean": float(per_node_picr.mean()),
        "per_node_picr_std": float(per_node_picr.std()),
        "per_node_picr_min": float(per_node_picr.min()),
        "per_node_picr_max": float(per_node_picr.max()),
        "per_node_picr_median": float(np.median(per_node_picr)),
        "depth_quartile_results": quartile_results,
        "node_degree_results": degree_results,
        "note": ("Recomputed on the gauge-and-timestamp-corrected primary model "
                 "(loop 300, seed 42), replacing the stale loop-24 pre-correction "
                 "analysis. Conformal quantile calibrated on val_conformal (2023, "
                 "gauge-corrected), evaluated on test_temporal (2024, gauge-corrected)."),
    }
    out_path = Path("results") / "pernode_coverage_gauge401fix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out_path}")

    # ── Save per-node table (for shapefile / fig6 / fig7 regeneration) ───────────
    import csv
    csv_path = Path("results") / "pernode_coverage_gauge401fix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "mean_depth_m", "degree",
                          "depth_quartile", "picr_90"])
        coord_map = {row["node_id"]: (row["x"], row["y"]) for _, row in coord_df.iterrows()}
        for i, nid in enumerate(node_ids):
            xy = coord_map.get(nid, (None, None))
            writer.writerow([nid, xy[0], xy[1], float(mean_depth_per_node[i]),
                              int(degree[i]) if len(degree) == N_nodes else "",
                              int(quartile_id_per_node[i]), float(per_node_picr[i])])
    logger.info(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
