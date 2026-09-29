"""Fig 6 (Q1/Q4 representative-node hydrographs with 90% PI band), rerun on
the gauge-and-timestamp-corrected primary model (Loop 300, seed 42).

Replaces the earlier style-only draft (generate_fig6_prediction_intervals_style.py,
which used loop24 pre-correction placeholder data and never finished a
successful run). This version:
  - Loads the actual gauge401fix backbone (config_gauge401_fix_v1.yaml, Loop 300).
  - Patches in TimeCorrectedSWMMGraphDataset (report_lead_steps trimmed per
    run_manifest.json) BEFORE importing src.train.build_dataloaders, matching
    the documented usage of the sibling "primary" script
    src/pernode_coverage_gauge401fix.py (normally run via
    scripts/run_timecorrected_entrypoint.py) and the inline-patch convention
    used by scripts/recompute_primary_continuity_diag_final.py and siblings.
    2026-09-16 fix: an earlier version of this script called
    src.train.build_dataloaders directly, without this patch, so it evaluated
    on windows that still included the 6 pre-event report-lead steps
    (results/swmm_timecorrected_v2_gauge401_fix/run_manifest.json:
    report_lead_steps=6) instead of the locked evaluation window used by the
    headline pipeline (results/pernode_coverage_gauge401fix.json,
    results/timecorrected_v2/eval_gauge401fix_loop300_test_temporal.json).
  - Calibrates the conformal quantile on val_conformal (2023) using the same
    per-timestep-vector convention as src/evaluate.py and
    src/pernode_coverage_gauge401fix.py (NOT a single pooled scalar), so the
    reported overall PICR_90 can be checked against the manuscript's headline
    number (0.869 overall on test_temporal).
  - Selects representative Q1 (shallowest) / Q4 (deepest) nodes by the same
    depth+coverage-distance rule used throughout this project
    (generate_uq_template_matched.representative_node): within the quartile,
    the node whose (mean depth, network-aggregate PICR_90) pair is closest in
    standardised (z-score) units to the quartile's own mean of both -- i.e.
    the most typical node of the quartile, not hand-picked for appearance.
  - Reports, and prints for caption use: the actual SWMM node_id (not the
    graph array index) for each plotted node, the event_id/set_id of the
    plotted trajectory, and both (a) that single trajectory's own PICR_90 and
    (b) the plotted node's PICR_90 aggregated over the full test_temporal
    evaluation set (5 events) -- these are different quantities and are
    labelled separately in the figure and printed separately here.
  - Reconstructs the physical time axis from each sample's own
    ``dt_seconds`` (Methods: events are resampled to 100 steps, so the
    real per-step duration varies by event); falls back to a normalised
    0-100% axis only if dt_seconds is unavailable.

Time series + shaded prediction-interval band is kept as the chart type
(the standard convention in rainfall-runoff UQ literature, e.g. Klotz et al.
2022 HESS) -- only the color scheme and annotation placement were refreshed
to match the rest of the corrected-v2 figure set.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# The training loader selects the corrected adapter from the configuration.
DATASET_CLASS_USED = "TimeCorrectedSWMMGraphDataset (report_lead_steps trimmed per run_manifest.json)"

from src.analysis.generate_uq_template_matched import representative_node  # noqa: E402
from src.train import build_dataloaders  # noqa: E402
from src.models.conformal_uq import ConformalPredictor  # noqa: E402

CONFIG_PATH = "configs/config_gauge401_fix_v1.yaml"
LOOP_ID = 300
ALPHA = 0.10
OUT = PROJECT / "paper" / "figures" / "fig6_prediction_intervals.png"

SWMM_COLOR = "#000000"
GNN_COLOR = "#33393E"
BAND_Q1 = "#B7CBE0"
BAND_Q4 = "#D2A9AF"
GRID = "#E5E9EC"


def style_axis(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_device(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


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
    """Run inference and also collect per-sample event_id/set_id/dt_seconds,
    index-aligned with the stacked prediction/target arrays (assumes the
    loader's batch_size is 1, matching this project's existing convention
    for evaluation loaders over full-network graphs)."""
    preds, trues, event_ids, set_ids, dt_seconds_list = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out["mean_depth"].cpu().numpy())
            trues.append(batch.y.cpu().numpy())
            event_ids.append(batch.event_id[0] if isinstance(batch.event_id, list) else batch.event_id)
            set_ids.append(int(batch.set_id[0]) if isinstance(batch.set_id, list) else int(batch.set_id))
            dt_seconds_list.append(float(batch.dt_seconds.reshape(-1)[0].cpu()))
    if not preds:
        return None, None, [], [], []
    return (np.stack(preds, axis=0), np.stack(trues, axis=0),
            event_ids, set_ids, dt_seconds_list)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--config", default=CONFIG_PATH)
    args = p.parse_args()
    device = get_device(args.device)
    print(f"Device: {device}")
    print(f"Dataset class: {DATASET_CLASS_USED}")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = load_model(config, device)
    _, val_loader, calib_loader, test_loader = build_dataloaders(config, regime_id="B")
    node_df = test_loader.dataset.dataset._node_df  # Subset(dataset, idx).dataset -> array index -> real SWMM node_id

    cp = ConformalPredictor(alpha=ALPHA)
    calib_pred, calib_true = collect(model, calib_loader, device)[:2]
    T_calib = calib_pred.shape[-1]
    cp.calibrate(calib_pred.reshape(-1, T_calib), calib_true.reshape(-1, T_calib), regime_id="B")

    test_pred, test_true, event_ids, set_ids, dt_seconds_list = collect(model, test_loader, device)
    n_events, N_nodes, T = test_pred.shape
    print(f"test_temporal shape: {test_pred.shape}")

    interval = cp.predict(test_pred.reshape(-1, T), regime_id="B")
    lower = interval["lower"].reshape(test_pred.shape)
    upper = interval["upper"].reshape(test_pred.shape)
    covered = (test_true >= lower) & (test_true <= upper)

    # --- sanity check against the headline pipeline (results/pernode_coverage_gauge401fix.json,
    # results/timecorrected_v2/eval_gauge401fix_loop300_test_temporal.json: PICR_90=0.8690,
    # mean width 20.39 cm) ---
    overall_picr = float(covered.mean())
    overall_width_cm = float((upper - lower).mean()) * 100
    print(f"Overall PICR_90 (this run) = {overall_picr:.4f}  (headline: 0.8690)")
    print(f"Overall mean interval width (this run) = {overall_width_cm:.2f} cm  (headline: 20.39 cm)")
    if abs(overall_picr - 0.8690) > 0.01:
        print("WARNING: overall PICR_90 differs from the headline value by more than 0.01 -- "
              "investigate before treating this figure as consistent with Section 4.4.")

    mean_depth = test_true.mean(axis=(0, 2))
    picr_node_aggregate = covered.mean(axis=(0, 2))  # per node, pooled over all 5 events + 100 steps
    edges = np.percentile(mean_depth, [0, 25, 50, 75, 100])
    masks = {
        "Q1": mean_depth <= edges[1],
        "Q4": mean_depth > edges[3],
    }
    q1_node = representative_node(masks["Q1"], mean_depth, picr_node_aggregate)
    q4_node = representative_node(masks["Q4"], mean_depth, picr_node_aggregate)

    q_hat = cp.q_hat_dict["B"]  # (T,) per-timestep vector, m below/above
    q_hat_cm = q_hat * 100

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 12,
        "axes.labelsize": 17,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "mathtext.fontset": "stix",
    })

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    event = 0  # first sample in test_temporal loader order; identity printed/logged below
    caption_meta = []

    # Shared y-axis range across both panels (professor feedback: unify axes so
    # panel-to-panel width comparisons aren't visually distorted by different
    # auto-scaled ranges). Computed from the actual plotted series (SWMM,
    # GATv2Conv mean, and the PI band) across both nodes so nothing is clipped.
    y_lo, y_hi = np.inf, -np.inf
    for node in (q1_node, q4_node):
        y_pred_n = test_pred[event, node] * 100
        y_true_n = test_true[event, node] * 100
        band_lo = y_pred_n - q_hat_cm
        band_hi = y_pred_n + q_hat_cm
        y_lo = min(y_lo, band_lo.min(), y_true_n.min())
        y_hi = max(y_hi, band_hi.max(), y_true_n.max())
    pad = 0.06 * (y_hi - y_lo)
    shared_ylim = (y_lo - pad, y_hi + pad)

    for ax, node, label, band in [
        (axes[0], q1_node, "Q1 (shallowest quartile)", BAND_Q1),
        (axes[1], q4_node, "Q4 (deepest quartile)", BAND_Q4),
    ]:
        node_id = node_df.iloc[node]["node_id"]
        dt_s = dt_seconds_list[event]
        if dt_s and dt_s > 0:
            t = np.arange(T) * dt_s / 60.0  # minutes, this event's own resampled step
            xlabel = "Time since event start (min)"
        else:
            t = np.arange(T) / (T - 1) * 100.0  # normalised 0-100% fallback
            xlabel = "Normalised event time (%)"

        picr_single_curve = float(covered[event, node].mean())
        picr_node_agg = float(picr_node_aggregate[node])
        print(f"{label}: array_idx={node}, node_id={node_id}, event={event_ids[event]}, "
              f"set={set_ids[event]}, dt_seconds={dt_s:.1f}, "
              f"depth={mean_depth[node]*100:.1f}cm, "
              f"PICR_single_curve={picr_single_curve:.3f}, "
              f"PICR_node_aggregate(5 events)={picr_node_agg:.3f}")
        caption_meta.append(
            (label, node_id, event_ids[event], set_ids[event], picr_single_curve, picr_node_agg)
        )

        y_pred = test_pred[event, node] * 100
        y_true = test_true[event, node] * 100
        style_axis(ax)
        ax.fill_between(t, y_pred - q_hat_cm, y_pred + q_hat_cm, color=band, alpha=0.35,
                         label="90% PI", zorder=1)
        ax.plot(t, y_true, color=SWMM_COLOR, linewidth=1.7, label="SWMM", zorder=3)
        ax.plot(t, y_pred, color=GNN_COLOR, linewidth=1.4, linestyle="--",
                 label="GATv2Conv mean", zorder=2)
        ax.text(0.03, 0.95,
                f"this curve PICR$_{{90}}$={picr_single_curve:.2f}\n"
                f"node PICR$_{{90}}$ (5 events)={picr_node_agg:.2f}",
                transform=ax.transAxes,
                ha="left", va="top", fontsize=10.5,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#B0B6BB", alpha=0.9))
        ax.set_title(label, loc="left", fontsize=15, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Water depth (cm)")
        ax.set_xlim(float(t[0]), float(t[-1]))
        ax.set_ylim(*shared_ylim)
        ax.legend(loc="upper right", frameon=True, fancybox=False,
                  edgecolor="#B0B6BB", framealpha=0.95)

    fig.text(0.005, 0.98, "(a)", fontsize=17, fontweight="bold", va="top")
    fig.text(0.505, 0.98, "(b)", fontsize=17, fontweight="bold", va="top")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white")
    print("Saved ->", OUT)
    print("Caption metadata (label, node_id, event_id, set_id, PICR_single_curve, PICR_node_aggregate):")
    for row in caption_meta:
        print("  ", row)


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    main()
