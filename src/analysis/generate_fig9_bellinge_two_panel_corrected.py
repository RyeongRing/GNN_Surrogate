"""Generate the corrected two-panel Bellinge transfer figure.

Panel (a) reports the designated catchment-wide transfer metrics. Panel (b)
shows one held-out event/node series and labels the NSE computed from exactly
that displayed series. This avoids the previous mismatch where a pooled
per-node metric was placed next to a single event-series hydrograph.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import src.analysis.generate_fig9_bellinge as base


DEFAULT_OUTPUT = Path("output/figures/그림_update") / (
    "fig9_bellinge_transfer_arrow_deep_magenta_corrected.png"
)
DEFAULT_METRICS = Path("results") / "fig9_bellinge_two_panel_corrected_metrics.json"

COLORS = {
    "stage1": "#C86F52",
    "stage2": "#3F7A5E",
    "reference": "#2C2C2C",
    "moriasi": "#9A7258",
    "seocho": "#376F9E",
    "accent": "#9A1766",
    "grid": "#DEE5EA",
}

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 15,
        "axes.labelsize": 19,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "mathtext.fontset": "stix",
    }
)


def nse_with_sums(true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Return NSE, residual sum of squares, and total sum of squares."""
    y = np.asarray(true, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    if ss_tot <= 0:
        raise ValueError("Cannot compute NSE for a constant displayed series.")
    return 1.0 - ss_res / ss_tot, ss_res, ss_tot


def select_display_series(
    true_ft: np.ndarray,
    pred_ft: np.ndarray,
    test_event_ids: list[str],
) -> tuple[int, int, np.ndarray]:
    """Apply the original representative-node selection rule."""
    per_node_nse = []
    for node_idx in range(true_ft.shape[1]):
        true_node = np.asarray(true_ft[:, node_idx, :], dtype=np.float64).reshape(-1)
        pred_node = np.asarray(pred_ft[:, node_idx, :], dtype=np.float64).reshape(-1)
        ss_tot = float(np.sum((true_node - true_node.mean()) ** 2))
        if ss_tot <= 0:
            score = np.nan
        else:
            score = 1.0 - float(np.sum((true_node - pred_node) ** 2)) / ss_tot
        per_node_nse.append(score)
    per_node_nse = np.asarray(per_node_nse)

    peak_depths_cm = true_ft.max(axis=(0, 2)) * 100.0
    candidate_mask = (peak_depths_cm >= 20.0) & np.isfinite(per_node_nse)
    if not candidate_mask.any():
        candidate_mask = np.isfinite(per_node_nse)
    candidates = np.where(candidate_mask)[0]
    target = np.median(per_node_nse[candidates])
    node_idx = int(candidates[np.argmin(np.abs(per_node_nse[candidates] - target))])

    peak_by_sample = true_ft[:, node_idx, :].max(axis=1)
    sample_idx = int(np.argmax(peak_by_sample))
    if sample_idx >= len(test_event_ids):
        raise IndexError("Selected sample has no matching held-out event ID.")
    return node_idx, sample_idx, per_node_nse


def style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.9, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#65717C")
    ax.spines["bottom"].set_color("#65717C")
    ax.tick_params(width=0.9, length=5, color="#65717C")


def plot_summary_panel(ax: plt.Axes) -> None:
    values = [base.STAGE1_NSE, base.STAGE2_NSE]
    labels = ["Zero-shot", "Fine-tuned"]
    colors = [COLORS["stage1"], COLORS["stage2"]]
    x = np.arange(2)
    bars = []
    for xi, value, label, color in zip(x, values, labels, colors):
        bar = ax.bar(
            xi,
            value,
            width=0.48,
            color=color,
            edgecolor="#4C4C4C",
            linewidth=0.9,
            zorder=2,
            label=label,
        )
        bars.append(bar[0])
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
        )
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.04)
    ax.set_ylabel("NSE")
    ax.legend(loc="upper right", frameon=True, borderaxespad=0.8)
    style_axis(ax)


def plot_series_panel(
    ax: plt.Axes,
    time_min: np.ndarray,
    true_cm: np.ndarray,
    zero_shot_cm: np.ndarray,
    fine_tuned_cm: np.ndarray,
    displayed_nse: float,
    event_id: str,
    node_idx: int,
) -> None:
    ax.plot(
        time_min,
        true_cm,
        color=COLORS["reference"],
        linewidth=2.4,
        label="SWMM",
        zorder=4,
    )
    ax.plot(
        time_min,
        zero_shot_cm,
        color=COLORS["stage1"],
        linewidth=2.0,
        label="Zero-shot",
        zorder=3,
    )
    ax.plot(
        time_min,
        fine_tuned_cm,
        color=COLORS["stage2"],
        linewidth=2.0,
        label="Fine-tuned",
        zorder=3,
    )
    ax.set_xlim(float(time_min[0]), float(time_min[-1]))
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Water depth (cm)")
    ax.legend(loc="upper right", frameon=True, borderaxespad=0.8)
    style_axis(ax)
    _ = (displayed_nse, event_id, node_idx)  # retained for signature compatibility


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--ft-epochs", type=int, default=base.FT_EPOCHS)
    parser.add_argument("--selected-node", type=int)
    parser.add_argument("--selected-sample-index", type=int)
    args = parser.parse_args()
    if (args.selected_node is None) != (args.selected_sample_index is None):
        parser.error("--selected-node and --selected-sample-index must be provided together.")

    device = base.get_device(args.device)
    with open(base.CONFIG, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    checkpoint_dir = Path("results/checkpoints") / f"loop_{base.BACKBONE_LOOP:02d}"
    model, _ = base.bv.load_loop_model(checkpoint_dir, config, device)
    ft_data, test_data, test_event_ids = base.build_bellinge_event_split_dataset(
        config, device
    )

    pred_zero_shot, true_zero_shot = base.run_inference(model, test_data, device)
    base.bv.fine_tune_bellinge_decoder(
        model,
        ft_data,
        n_epochs=args.ft_epochs,
        lr=base.FT_LR,
        seed=42,
    )
    pred_fine_tuned, true_fine_tuned = base.run_inference(model, test_data, device)

    node_idx, sample_idx, per_node_nse = select_display_series(
        true_fine_tuned, pred_fine_tuned, test_event_ids
    )
    if args.selected_node is not None:
        node_idx = args.selected_node
        sample_idx = args.selected_sample_index
        if not 0 <= node_idx < true_fine_tuned.shape[1]:
            raise IndexError(f"Selected node {node_idx} is outside the dataset.")
        if not 0 <= sample_idx < true_fine_tuned.shape[0]:
            raise IndexError(f"Selected sample {sample_idx} is outside the held-out set.")
    event_id = test_event_ids[sample_idx]

    true_series = true_fine_tuned[sample_idx, node_idx, :]
    zero_shot_series = pred_zero_shot[sample_idx, node_idx, :]
    fine_tuned_series = pred_fine_tuned[sample_idx, node_idx, :]
    displayed_nse, displayed_ss_res, displayed_ss_tot = nse_with_sums(
        true_series, fine_tuned_series
    )

    event_mask = np.asarray(test_event_ids) == event_id
    event_nse, _, _ = nse_with_sums(
        true_fine_tuned[event_mask, node_idx, :],
        pred_fine_tuned[event_mask, node_idx, :],
    )
    global_nse = base.compute_nse(pred_fine_tuned, true_fine_tuned)

    time_min = np.arange(true_series.shape[0]) * 5
    series_output = args.output.with_suffix(".npz")
    series_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        series_output,
        time_min=time_min,
        true_cm=true_series * 100.0,
        zero_shot_cm=zero_shot_series * 100.0,
        fine_tuned_cm=fine_tuned_series * 100.0,
        displayed_series_nse=displayed_nse,
        selected_event=event_id,
        selected_node=node_idx,
        selected_sample_index=sample_idx,
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(19.8, 8.6),
        gridspec_kw={"width_ratios": [1.0, 1.95], "wspace": 0.27},
    )
    plot_summary_panel(axes[0])
    plot_series_panel(
        axes[1],
        time_min,
        true_series * 100.0,
        zero_shot_series * 100.0,
        fine_tuned_series * 100.0,
        displayed_nse,
        event_id,
        node_idx,
    )
    axes[0].text(-0.18, 1.03, "(a)", transform=axes[0].transAxes, fontsize=25, fontweight="bold")
    axes[1].text(-0.07, 1.03, "(b)", transform=axes[1].transAxes, fontsize=25, fontweight="bold")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metrics = {
        "figure": str(args.output),
        "series_cache": str(series_output),
        "checkpoint_loop": base.BACKBONE_LOOP,
        "fine_tune_epochs": args.ft_epochs,
        "fine_tune_seed": 42,
        "held_out_events": sorted(set(test_event_ids)),
        "n_test_samples": int(true_fine_tuned.shape[0]),
        "selected_event": event_id,
        "selected_node": node_idx,
        "selected_sample_index": sample_idx,
        "displayed_series_nse": displayed_nse,
        "displayed_series_ss_res": displayed_ss_res,
        "displayed_series_ss_tot": displayed_ss_tot,
        "selected_event_pooled_nse": event_nse,
        "selected_node_pooled_nse": float(per_node_nse[node_idx]),
        "rerun_global_fine_tuned_nse": global_nse,
        "designated_stage2_nse": base.STAGE2_NSE,
        "true_peak_cm": float(true_series.max() * 100.0),
        "zero_shot_peak_cm": float(zero_shot_series.max() * 100.0),
        "fine_tuned_peak_cm": float(fine_tuned_series.max() * 100.0),
        "annotation_scope": "NSE is computed from exactly the displayed event/node series.",
    }
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
