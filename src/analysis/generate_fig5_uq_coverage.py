"""
Generate new Fig 5 (2-panel UQ coverage figure):
  Panel (a): 90% PI time series — Q1 node vs Q4 node side-by-side (same test event)
  Panel (b): Per-node PICR₉₀ vs mean depth scatter (all 8,480 nodes, coloured by quartile)

Saves: paper/figures/fig5_prediction_intervals.png

Usage (from project root):
    python src/analysis/generate_fig5_uq_coverage.py --device auto
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LOOP_ID   = 24
HIDDEN    = 128
CONFIG    = "config_seed_exp.yaml"
ALPHA     = 0.10
OUT_PATH  = Path("paper") / "figures" / "fig5_prediction_intervals.png"
CONFORMAL_PATH = Path("results") / "conformal" / "conformal_loop24.json"

# SCI-style colour palette (colourblind-safe)
Q_COLORS = {
    "Q1": "#6F96B5",
    "Q2": "#7FA49B",
    "Q3": "#C2A15A",
    "Q4": "#B9685A",
}
SWMM_COLOR   = "#000000"
MEAN_COLOR   = "#5B7C99"
BAND_COLOR   = "#DCE8ED"
BAND_ALPHA   = 0.55
TARGET_COLOR = "#7A8087"

DPI = 300
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "mathtext.fontset": "stix",
    "mathtext.rm": "Times New Roman",
    "figure.dpi": DPI,
})


def get_device(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def load_model(config, device):
    from src.models.gnn_surrogate import EnsembleGNN
    exp_cfg = config["experiment"]
    base_cfg = {
        "node_feat_dim": 14, "edge_feat_dim": 4,
        "hidden_dim": HIDDEN,
        "T_out":  exp_cfg.get("T_out", 100),
        "T_rain": exp_cfg.get("T_rain", 72),
        "n_heads": exp_cfg.get("n_heads", 4),
        "n_layers": exp_cfg.get("n_layers", 4),
        "dropout": exp_cfg.get("dropout", 0.0),
        "temporal_decoder":    exp_cfg.get("temporal_decoder", False),
        "scalar_rain_decoder": exp_cfg.get("scalar_rain_decoder", False),
    }
    M = exp_cfg.get("ensemble_size", 5)
    model = EnsembleGNN(base_cfg, M=M).to(device)
    cp_dir = Path("results") / "checkpoints" / f"loop_{LOOP_ID:02d}"
    for mid in range(M):
        path = cp_dir / f"member_{mid:02d}.pt"
        if not path.exists():
            continue
        ckpt = torch.load(str(path), map_location=device, weights_only=False)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.members[mid].load_state_dict(state)
    model.eval()
    return model, M


def run_inference(model, loader, device):
    """Returns preds (n, N, T), trues (n, N, T)."""
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out["mean_depth"].cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    if not preds:
        return None, None
    return np.stack(preds, axis=0), np.stack(trues, axis=0)


def load_headline_q_hat() -> float:
    """Load the scalar quantile used by the confirmatory conformal analysis."""
    result = json.loads(CONFORMAL_PATH.read_text(encoding="utf-8"))
    if int(result.get("loop_id", -1)) != LOOP_ID:
        raise ValueError(f"Unexpected conformal loop in {CONFORMAL_PATH}")
    if result.get("conformal_split") != "val_conformal (2023, 7 events)":
        raise ValueError(f"Unexpected conformal split in {CONFORMAL_PATH}")
    return float(result["q_hat_m"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = get_device(args.device)
    logger.info(f"Device: {device}")

    with open(CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model, M = load_model(config, device)

    from src.train import build_dataloaders
    _, val_loader, _, test_loader = build_dataloaders(config, regime_id="B")

    eval_loader = test_loader if (test_loader and len(test_loader) > 0) else val_loader

    # Reuse the scalar quantile from the confirmatory analysis rather than
    # recalibrating a time-wise vector inside the figure generator.
    q_hat = load_headline_q_hat()
    logger.info(f"Headline scalar q_hat: {q_hat * 100:.2f} cm")

    # ── Inference on test split ───────────────────────────────────────────────
    logger.info("Running test inference...")
    pred_test, true_test = run_inference(model, eval_loader, device)
    # pred_test, true_test: (n_events, N_nodes, T)
    n_events, N_nodes, T = pred_test.shape
    logger.info(f"Test shape: {pred_test.shape}")

    # ── Per-node statistics ───────────────────────────────────────────────────
    mean_depth_per_node = true_test.mean(axis=(0, 2))   # (N,)  mean over events & time
    lower = pred_test - q_hat                            # scalar broadcast
    upper = pred_test + q_hat
    covered = (true_test >= lower) & (true_test <= upper)
    picr_per_node = covered.mean(axis=(0, 2))            # (N,)

    # Depth quartile boundaries
    q_edges = np.percentile(mean_depth_per_node, [0, 25, 50, 75, 100])
    masks = {
        "Q1": mean_depth_per_node <= q_edges[1],
        "Q2": (mean_depth_per_node > q_edges[1]) & (mean_depth_per_node <= q_edges[2]),
        "Q3": (mean_depth_per_node > q_edges[2]) & (mean_depth_per_node <= q_edges[3]),
        "Q4": mean_depth_per_node > q_edges[3],
    }
    for q, mask in masks.items():
        logger.info(f"  {q}: n={mask.sum()}, PICR={picr_per_node[mask].mean():.4f}, "
                    f"mean_depth={mean_depth_per_node[mask].mean()*100:.1f} cm")

    # Representative nodes minimise standardised distance from their quartile
    # means in both mean depth and PICR; no target coverage is hard-coded.
    def representative_node(mask):
        nodes = np.where(mask)[0]
        depths = mean_depth_per_node[nodes]
        coverages = picr_per_node[nodes]
        depth_scale = max(float(np.std(depths)), 1e-12)
        coverage_scale = max(float(np.std(coverages)), 1e-12)
        distance = (
            ((depths - float(np.mean(depths))) / depth_scale) ** 2
            + ((coverages - float(np.mean(coverages))) / coverage_scale) ** 2
        )
        return int(nodes[np.argmin(distance)])

    q1_node_idx = representative_node(masks["Q1"])
    q4_node_idx = representative_node(masks["Q4"])

    logger.info(f"  Q1 representative node {q1_node_idx}: "
                f"PICR={picr_per_node[q1_node_idx]:.4f}, "
                f"mean_depth={mean_depth_per_node[q1_node_idx]*100:.1f} cm")
    logger.info(f"  Q4 representative node {q4_node_idx}: "
                f"PICR={picr_per_node[q4_node_idx]:.4f}, "
                f"mean_depth={mean_depth_per_node[q4_node_idx]*100:.1f} cm")

    # Use first test event for time series display
    ev = 0
    t_axis = np.arange(T) * 5   # 5-min resolution → minutes

    def node_series(node_idx):
        p  = pred_test[ev, node_idx, :] * 100    # cm
        y  = true_test[ev, node_idx, :] * 100    # cm
        lo = (pred_test[ev, node_idx, :] - q_hat) * 100
        hi = (pred_test[ev, node_idx, :] + q_hat) * 100
        return p, y, lo, hi

    q1_pred, q1_true, q1_lo, q1_hi = node_series(q1_node_idx)
    q4_pred, q4_true, q4_lo, q4_hi = node_series(q4_node_idx)

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))

    # Layout: 2 rows — top row = panel (a) with two subplots, bottom row = panel (b)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.2], hspace=0.38, wspace=0.28)
    ax_q1  = fig.add_subplot(gs[0, 0])
    ax_q4  = fig.add_subplot(gs[0, 1])
    ax_sct = fig.add_subplot(gs[1, :])

    # ── Panel (a-left): Q1 node time series ──────────────────────────────────
    ax_q1.fill_between(t_axis, q1_lo, q1_hi, color=BAND_COLOR, alpha=BAND_ALPHA,
                       label="90% PI", zorder=1)
    ax_q1.plot(t_axis, q1_true, color=SWMM_COLOR, lw=1.5, label="SWMM", zorder=3)
    ax_q1.plot(t_axis, q1_pred, color=MEAN_COLOR, lw=1.2, ls="--",
               label="GNN mean", zorder=2)
    ax_q1.set_xlabel("Time (min)", fontsize=10)
    ax_q1.set_ylabel("Water depth (cm)", fontsize=10)
    ax_q1.set_title(
        f"(a-i) Q1 node: PICR$_{{90}}$ = {picr_per_node[q1_node_idx]:.3f}\n"
        f"mean depth = {mean_depth_per_node[q1_node_idx]*100:.1f} cm "
        f"(shallowest quartile)",
        fontsize=9, pad=6)
    ax_q1.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax_q1.tick_params(labelsize=9)
    ax_q1.set_xlim(t_axis[0], t_axis[-1])

    # ── Panel (a-right): Q4 node time series ─────────────────────────────────
    ax_q4.fill_between(t_axis, q4_lo, q4_hi, color="#EAD7D2", alpha=0.55,
                       label="90% PI", zorder=1)
    ax_q4.plot(t_axis, q4_true, color=SWMM_COLOR, lw=1.5, label="SWMM", zorder=3)
    ax_q4.plot(t_axis, q4_pred, color=MEAN_COLOR, lw=1.2, ls="--",
               label="GNN mean", zorder=2)
    ax_q4.set_xlabel("Time (min)", fontsize=10)
    ax_q4.set_ylabel("Water depth (cm)", fontsize=10)
    ax_q4.set_title(
        f"(a-ii) Q4 node: PICR$_{{90}}$ = {picr_per_node[q4_node_idx]:.3f}\n"
        f"mean depth = {mean_depth_per_node[q4_node_idx]*100:.1f} cm "
        f"(deepest quartile)",
        fontsize=9, pad=6)
    ax_q4.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax_q4.tick_params(labelsize=9)
    ax_q4.set_xlim(t_axis[0], t_axis[-1])

    # ── Panel (b): Scatter PICR vs mean depth ────────────────────────────────
    q_labels = ["Q1", "Q2", "Q3", "Q4"]
    for q in q_labels:
        mask = masks[q]
        ax_sct.scatter(
            mean_depth_per_node[mask] * 100,
            picr_per_node[mask],
            c=Q_COLORS[q], s=3, alpha=0.35, linewidths=0,
            label=f"{q} (n={mask.sum():,})"
        )
    # Quartile mean markers
    for q in q_labels:
        mask = masks[q]
        mx = mean_depth_per_node[mask].mean() * 100
        my = picr_per_node[mask].mean()
        ax_sct.scatter(mx, my, c=Q_COLORS[q], s=120, marker="D",
                       edgecolors="white", linewidths=0.8, zorder=5)
        ax_sct.annotate(
            f"{q}\nPICR={my:.3f}",
            xy=(mx, my), xytext=(mx + 0.5, my - 0.06),
            fontsize=8, color=Q_COLORS[q],
            arrowprops=dict(arrowstyle="-", color=Q_COLORS[q], lw=0.8),
        )

    # Reference lines
    ax_sct.axhline(0.90, color=TARGET_COLOR, lw=1.0, ls="--",
                   label="Nominal 90% target")

    ax_sct.set_xlabel("Mean observed depth (cm)", fontsize=10)
    ax_sct.set_ylabel("PICR$_{90}$ per node", fontsize=10)
    ax_sct.set_title(
        "(b)  Per-node PICR$_{90}$ vs mean water depth  "
        f"(all {N_nodes:,} nodes; 2024 test events)",
        fontsize=10, pad=6)
    ax_sct.set_ylim(-0.05, 1.08)
    ax_sct.tick_params(labelsize=9)

    handles = [
        mpatches.Patch(
            color=Q_COLORS[q],
            label=(
                f"{q} (n={masks[q].sum():,}, "
                f"mean PICR={picr_per_node[masks[q]].mean():.3f})"
            ),
        )
        for q in q_labels
    ]
    handles += [
        plt.Line2D([0], [0], color=TARGET_COLOR, ls="--", lw=1.0,
                   label="Nominal 90% target"),
    ]
    ax_sct.legend(handles=handles, fontsize=8, loc="lower left",
                  ncol=2, framealpha=0.85)

    # Panel label
    fig.text(0.01, 0.97, "(a)", fontsize=12, fontweight="bold", va="top")
    fig.text(0.01, 0.50, "(b)", fontsize=12, fontweight="bold", va="top")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PATH), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(str(OUT_PATH.with_suffix(".pdf")), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Saved: {OUT_PATH}")

    # ── Save per-node arrays for record ──────────────────────────────────────
    np.save("results/pernode_picr_array.npy",  picr_per_node)
    np.save("results/pernode_depth_array.npy", mean_depth_per_node * 100)
    logger.info("Saved per-node arrays to results/pernode_*.npy")


if __name__ == "__main__":
    main()
