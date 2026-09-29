"""Redraw fig7 (per-node PICR_90 vs depth) using the real 8,480-node data
from the gauge-and-timestamp-corrected primary model.

Style pass 5: replaced the boxplot/broken-axis panel (a) with a plain bar
chart of mean PICR_90 per depth bin -- the same chart type as panel (b),
just with finer (six-bin) resolution. This avoids the box-compression /
broken-axis complexity entirely: bar height always reads clearly
regardless of how small the underlying variance is, and the two panels
now share one consistent, simple visual language.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "results" / "pernode_coverage_gauge401fix.csv"
OUTPUT = ROOT / "paper" / "figures" / "fig7_pernode_coverage.png"

BAR_COLOR = "#3D6EA6"
BAR_HIGHLIGHT = "#8C4A52"
LINE_COLOR = "#2C5480"
TARGET_COLOR = "#33393E"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 15,
    "axes.labelsize": 18,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})


def style_axis(ax):
    ax.grid(axis="y", color="#E5E9EC", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    df = pd.read_csv(CSV_PATH)
    depth_cm = df["mean_depth_m"].to_numpy() * 100.0
    picr = df["picr_90"].to_numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # ---- Panel (a): depth-binned bar chart ----
    ax = axes[0]
    style_axis(ax)

    n_bins = 6
    edges = np.quantile(depth_cm, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    n_bins = len(edges) - 1
    bin_idx = np.clip(np.digitize(depth_cm, edges[1:-1]), 0, n_bins - 1)

    bin_means, bin_labels, bin_colors = [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        bin_means.append(picr[mask].mean())
        lo, hi = edges[b], edges[b + 1]
        bin_labels.append(f"{lo:.0f}–{hi:.0f}")
        bin_colors.append(BAR_HIGHLIGHT if bin_means[-1] <= 0.90 else BAR_COLOR)

    positions = np.arange(1, len(bin_means) + 1)
    ax.plot(positions, bin_means, color="#000000", linewidth=1.1, zorder=2)
    ax.scatter(positions, bin_means, s=130, color=bin_colors, edgecolor="#33393E",
               linewidth=1.1, zorder=3)
    ax.axhline(0.90, color=TARGET_COLOR, linestyle=(0, (4, 2)), linewidth=1.2, zorder=1)
    n_pts = len(positions)
    for i, (x, val) in enumerate(zip(positions, bin_means)):
        # Points sitting close to the 0.90 target line need extra clearance
        # so the value label does not sit on top of the dashed line.
        offset = 22 if abs(val - 0.90) < 0.10 else 10
        # The last point's label sits where the steep incoming line segment
        # arrives from the left; center-aligning it there makes the label
        # overlap that line, so anchor it to the right of the point instead.
        if i == n_pts - 1:
            ax.annotate(f"{val:.3f}", xy=(x, val), xytext=(8, offset),
                        textcoords="offset points", ha="left",
                        va="bottom", fontsize=12.5)
        else:
            ax.annotate(f"{val:.3f}", xy=(x, val), xytext=(0, offset),
                        textcoords="offset points", ha="center",
                        va="bottom", fontsize=12.5)

    ax.set_xticks(positions, bin_labels)
    ax.set_xlabel("Mean SWMM reference depth bin (cm)")
    ax.set_ylabel("Mean PICR$_{90}$")
    ax.set_ylim(0, 1.12)

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=9,
               markerfacecolor=BAR_COLOR, markeredgecolor="#33393E", label="Above 0.90 target"),
        Line2D([0], [0], marker="o", linestyle="None", markersize=9,
               markerfacecolor=BAR_HIGHLIGHT, markeredgecolor="#33393E", label="At/below 0.90 target"),
        Line2D([0], [0], color=TARGET_COLOR, linestyle=(0, (4, 2)), linewidth=1.2,
               label="0.90 target"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", frameon=True, framealpha=0.95, fontsize=10.5)
    ax.set_title("(a)", loc="left", fontsize=17, fontweight="bold")

    # ---- Panel (b): quartile bar chart ----
    ax = axes[1]
    style_axis(ax)
    q_labels = ["Q1", "Q2", "Q3", "Q4"]
    q_means = []
    quartile_col = df["depth_quartile"].to_numpy()
    for qi in range(1, 5):
        mask = quartile_col == qi
        q_means.append(picr[mask].mean())

    colors = [BAR_HIGHLIGHT if mean <= 0.90 else BAR_COLOR for mean in q_means]
    bars = ax.bar(q_labels, q_means, color=colors, edgecolor="#33393E",
                   linewidth=0.9, width=0.55, zorder=3)
    ax.axhline(0.90, color=TARGET_COLOR, linestyle=(0, (4, 2)), linewidth=1.2, zorder=1)
    for bar, mean_v in zip(bars, q_means):
        ax.annotate(f"{mean_v:.3f}", xy=(bar.get_x() + bar.get_width() / 2, mean_v),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    va="bottom", fontsize=13)

    ax.set_ylabel("Mean PICR$_{90}$")
    ax.set_ylim(0, 1.18)
    ax.set_xlabel("Depth quartile")
    ax.set_xlim(-0.6, 3.6)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    b_legend_handles = [
        Patch(facecolor=BAR_COLOR, edgecolor="#33393E", label="Above 0.90 target"),
        Patch(facecolor=BAR_HIGHLIGHT, edgecolor="#33393E", label="At/below 0.90 target"),
        Line2D([0], [0], color=TARGET_COLOR, linestyle=(0, (4, 2)), linewidth=1.2,
               label="0.90 target"),
    ]
    ax.legend(handles=b_legend_handles, loc="upper right",
              ncol=1, frameon=True, framealpha=0.95, fontsize=10,
              handletextpad=0.6)
    ax.set_title("(b)", loc="left", fontsize=17, fontweight="bold")

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    print("Saved ->", OUTPUT)


if __name__ == "__main__":
    main()
