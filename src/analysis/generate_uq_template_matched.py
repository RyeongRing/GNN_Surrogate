"""Generate the confirmatory UQ figure in the project reference style."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from src.analysis.generate_fig5_uq_coverage import (  # noqa: E402
    get_device,
    load_headline_q_hat,
    load_model,
    run_inference,
)
from src.train import build_dataloaders  # noqa: E402


OUT = PROJECT / "paper" / "figures" / "fig5_prediction_intervals"
Q_COLORS = {"Q1": "#6F96B5", "Q2": "#7FA49B", "Q3": "#C2A15A", "Q4": "#B9685A"}
SWMM = "#222222"
GNN = "#B17762"
BAND_Q1 = "#DCE8E8"
BAND_Q4 = "#EADDD9"
GRID = "#DDE1E4"
INK = "#303B48"


def style(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#777D82")
    ax.spines["bottom"].set_color("#777D82")
    ax.tick_params(length=3, width=0.7, colors=INK)


def representative_node(mask: np.ndarray, depths: np.ndarray, coverage: np.ndarray) -> int:
    nodes = np.where(mask)[0]
    node_depth = depths[nodes]
    node_coverage = coverage[nodes]
    depth_scale = max(float(np.std(node_depth)), 1e-12)
    coverage_scale = max(float(np.std(node_coverage)), 1e-12)
    distance = (
        ((node_depth - float(np.mean(node_depth))) / depth_scale) ** 2
        + ((node_coverage - float(np.mean(node_coverage))) / coverage_scale) ** 2
    )
    return int(nodes[np.argmin(distance)])


def add_series(
    ax: plt.Axes,
    t_axis: np.ndarray,
    pred: np.ndarray,
    true: np.ndarray,
    q_hat_cm: float,
    band_color: str,
    title: str,
) -> None:
    lower = pred - q_hat_cm
    upper = pred + q_hat_cm
    ax.fill_between(t_axis, lower, upper, color=band_color, alpha=0.52, label="90% PI", zorder=1)
    ax.plot(t_axis, true, color=SWMM, linewidth=1.55, label="SWMM", zorder=3)
    ax.plot(t_axis, pred, color=GNN, linewidth=1.4, linestyle="--", label="GNN mean", zorder=2)
    ax.set_title(title, loc="left", pad=7)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Water depth (cm)")
    ax.set_xlim(float(t_axis[0]), float(t_axis[-1]))
    style(ax)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = get_device(args.device)

    with (PROJECT / "config_seed_exp.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model, _ = load_model(config, device)
    _, val_loader, _, test_loader = build_dataloaders(config, regime_id="B")
    eval_loader = test_loader if test_loader and len(test_loader) else val_loader
    pred, true = run_inference(model, eval_loader, device)
    if pred is None or true is None:
        raise RuntimeError("No temporal-test samples were available for UQ rendering")

    q_hat = load_headline_q_hat()
    lower = pred - q_hat
    upper = pred + q_hat
    covered = (true >= lower) & (true <= upper)
    mean_depth = true.mean(axis=(0, 2))
    picr = covered.mean(axis=(0, 2))
    edges = np.percentile(mean_depth, [0, 25, 50, 75, 100])
    masks = {
        "Q1": mean_depth <= edges[1],
        "Q2": (mean_depth > edges[1]) & (mean_depth <= edges[2]),
        "Q3": (mean_depth > edges[2]) & (mean_depth <= edges[3]),
        "Q4": mean_depth > edges[3],
    }
    q1_node = representative_node(masks["Q1"], mean_depth, picr)
    q4_node = representative_node(masks["Q4"], mean_depth, picr)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 9,
            "mathtext.fontset": "stix",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 5.2), gridspec_kw={"width_ratios": [1, 1, 1.12]})
    event = 0
    t_axis = np.arange(pred.shape[2]) * 5
    q_hat_cm = q_hat * 100
    add_series(
        axes[0],
        t_axis,
        pred[event, q1_node] * 100,
        true[event, q1_node] * 100,
        q_hat_cm,
        BAND_Q1,
        f"(a) Q1 shallow node (PICR$_{{90}}$ = {picr[q1_node]:.3f})",
    )
    add_series(
        axes[1],
        t_axis,
        pred[event, q4_node] * 100,
        true[event, q4_node] * 100,
        q_hat_cm,
        BAND_Q4,
        f"(b) Q4 deep node (PICR$_{{90}}$ = {picr[q4_node]:.3f})",
    )
    axes[1].legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#777D82")

    for quartile, mask in masks.items():
        axes[2].scatter(
            mean_depth[mask] * 100,
            picr[mask],
            s=4,
            alpha=0.24,
            linewidths=0,
            color=Q_COLORS[quartile],
            label=quartile,
        )
        mx = float(mean_depth[mask].mean() * 100)
        my = float(picr[mask].mean())
        axes[2].scatter(mx, my, s=82, marker="D", color=Q_COLORS[quartile], edgecolor="white", linewidth=0.8, zorder=4)
        axes[2].annotate(f"{quartile}: {my:.3f}", (mx, my), xytext=(5, -12), textcoords="offset points", fontsize=8.5, color=Q_COLORS[quartile])
    axes[2].axhline(0.90, color="#777D82", linestyle="--", linewidth=1.0, label="Nominal 0.90")
    axes[2].set_title("(c) Coverage across 8,480 nodes", loc="left", pad=7)
    axes[2].set_xlabel("Mean SWMM reference depth (cm)")
    axes[2].set_ylabel("Per-node PICR$_{90}$")
    axes[2].set_ylim(-0.03, 1.04)
    style(axes[2])
    axes[2].legend(loc="lower left", ncol=2, frameon=True, framealpha=0.93, edgecolor="#777D82")

    fig.tight_layout(w_pad=2.2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    np.save(PROJECT / "results" / "pernode_picr_array.npy", picr)
    np.save(PROJECT / "results" / "pernode_depth_array.npy", mean_depth * 100)
    print(
        f"Saved {OUT.with_suffix('.png')} | "
        f"Q1={picr[q1_node]:.4f}, Q4={picr[q4_node]:.4f}, "
        f"q_hat={q_hat_cm:.2f} cm"
    )


if __name__ == "__main__":
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    main()
