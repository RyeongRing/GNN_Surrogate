"""Regenerate fig_glue_screening for the timestamp-corrected v2 protocol.

Reads results/timecorrected_v2/calibration/calibration_multimetric_50sets.csv
(mean_r per set, behavioral flag) and the threshold from
results/timecorrected_v2/calibration/calibration_audit.json, replacing the
legacy 0.458-threshold figure. Simple ranked-bar style matching the caption
in main_corrected_v2.tex: 50th-percentile threshold separates 25 behavioural
(blue) from 25 non-behavioural (grey) sets. Output filename is suffixed
_corrected_v2 so the original figure is left untouched.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "results" / "timecorrected_v2" / "calibration" / "calibration_multimetric_50sets.csv"
AUDIT_PATH = ROOT / "results" / "timecorrected_v2" / "calibration" / "calibration_audit.json"
OUTPUT = ROOT / "paper" / "figures" / "fig_glue_screening_corrected_v2.png"

SELECTED_SET_ID = 7
BEHAVIORAL_COLOR = "#335C81"
NONBEHAVIORAL_COLOR = "#B7BEC4"
SELECTED_EDGE = "#C0392B"
THRESHOLD_COLOR = "#A95A3A"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12.5,
        "axes.labelsize": 15.5,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 12.8,
        "legend.fontsize": 11.8,
    }
)


def main() -> None:
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    threshold = float(audit["behavioral_screening"]["threshold_r"])
    n_behavioral = int(audit["behavioral_screening"]["n_behavioral"])
    n_total = int(audit["behavioral_screening"]["n_total"])

    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows = [
        {"set_id": int(r["set_id"]), "mean_r": float(r["mean_r"]), "behavioral": r["behavioral"] == "True"}
        for r in rows
    ]
    rows.sort(key=lambda r: r["mean_r"], reverse=True)

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    x = range(len(rows))
    colors = [BEHAVIORAL_COLOR if r["behavioral"] else NONBEHAVIORAL_COLOR for r in rows]
    edgecolors = [
        SELECTED_EDGE if r["set_id"] == SELECTED_SET_ID else "#33393E" for r in rows
    ]
    linewidths = [2.2 if r["set_id"] == SELECTED_SET_ID else 0.6 for r in rows]
    values = [r["mean_r"] for r in rows]
    ax.bar(x, values, color=colors, edgecolor=edgecolors, linewidth=linewidths, zorder=3)

    ax.axhline(threshold, color=THRESHOLD_COLOR, linestyle=(0, (3, 2)), linewidth=1.5, zorder=2,
               label=f"50th-percentile threshold ($r$ = {threshold:.3f})")

    ax.set_xlabel("Parameter set rank (by mean Pearson $r$)")
    ax.set_ylabel("Mean Pearson $r$ (61 usable calibration events)")
    ax.set_xticks([])
    ax.set_ylim(min(values) - 0.02, max(values) + 0.02)
    ax.grid(axis="y", color="#E5E9EC", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=BEHAVIORAL_COLOR, edgecolor="#33393E", label=f"Behavioral ({n_behavioral})"),
        Patch(facecolor=NONBEHAVIORAL_COLOR, edgecolor="#33393E", label=f"Non-behavioral ({n_total - n_behavioral})"),
        plt.Line2D([0], [0], color=THRESHOLD_COLOR, linestyle=(0, (3, 2)), linewidth=1.5,
                   label=f"Threshold ($r$ = {threshold:.3f})"),
    ]
    ax.legend(handles=handles, loc="lower left", frameon=True, framealpha=0.95)

    ax.set_title(
        "Timestamp-corrected protocol: 738 station-event pairs, 61 usable calibration events",
        fontsize=11, color="#555555",
    )

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    print(OUTPUT)


if __name__ == "__main__":
    main()
