"""
Generate Fig 9 (3-panel Bellinge cross-catchment transfer figure), Loop 24
(current primary model, GIS-strict), using the corrected (bug-fixed) Bellinge
evaluation pipeline.

Background: this figure previously showed stale results from a decommissioned
"Legacy-L15" backbone. bellinge_validate_loop24.py (project root) had two bugs
(array-axis mismatch: np.concatenate instead of np.stack, breaking the
cal/test index split; and a conformal-interval broadcast mismatch) that were
fixed on 2026-07-19. This script regenerates the figure from the corrected
Loop-24 results:
  - results/bellinge_loop24_zeroshot_n20.json      (Stage 1, zero-shot)
  - results/bellinge_loop24_validation_n20_fixed.json (Stage 2 fine-tuned /
    Stage 3 affine-calibrated)

Panel (c) requires real prediction curves (not fabricated), so this script
reuses bellinge_validate_loop24.py's proven data-loading / model-loading /
fine-tuning functions (same Bellinge .inp + SWMM ensemble paths, same
Seocho-gu-distribution feature alignment step `x[:, 1] = 0`) to run actual
zero-shot and fine-tuned inference on the same 20-event Bellinge catalog
(EB001-EB015 fine-tune / EB016-EB020 held-out test, matching
bellinge_validate_loop24.py's --event_split --n_test_events 5 default),
and picks a representative held-out test node/event for the hydrograph.

Saves: paper/figures/fig9_bellinge_transfer.png  (300 dpi, overwritten in place)

Usage (from project root, paper_002_update):
    python src/analysis/generate_fig9_bellinge.py --device auto
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
import numpy as np
import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Reuse the proven, bug-fixed Bellinge pipeline (paths, model loader,
#    dataset construction pattern, decoder fine-tuning) ─────────────────────
from bellinge import bellinge_validate_loop24 as bv

BACKBONE_LOOP = 24
CONFIG = "config_seed_exp.yaml"
STRICT_CATALOG = Path("results/bellinge_event_catalog_loop24_20event_strict.json")
FT_EPOCHS = 150       # matches bellinge_validate_loop24.py default --ft_epochs
FT_LR = 5e-4          # matches default --ft_lr
N_TEST_EVENTS = 5     # matches default --n_test_events (EB016-EB020 held out)
OUT_PATH = Path("paper") / "figures" / "fig9_bellinge_transfer.png"

# ── Corrected metrics (from the two designated result files) ───────────────
# results/bellinge_loop24_zeroshot_n20.json:
#   NSE_bellinge_raw = 0.1019, RMSE_cm_bellinge_raw = 34.56 cm
#   (the file's top-level "RMSE_cm_bellinge" = 52.18 cm is NOT used here: that
#   field is the *post-hoc affine-calibrated* zero-shot RMSE, i.e. after
#   fitting a 10-sample affine correction (cal_scale_a=3.6) that overcorrects
#   and generalizes poorly (paired NSE_bellinge=-0.14) -- pairing it with the
#   raw NSE=0.102 would mismatch two different computations. The raw pair
#   (NSE=0.102, RMSE=34.56 cm) is corroborated by two independent reruns
#   today: bellinge_loop24_zeroshot_strict_posthoc.json (NSE=0.1024,
#   RMSE=34.10 cm) and the 20-event strict rerun (NSE=0.1061, RMSE=34.74 cm).
STAGE1_NSE = 0.1019
STAGE1_RMSE = 34.56

# results/bellinge_exact_matched_v2.json (pretrained_per_seed."42"), the
# genuinely exact-matched rerun with decoder init matched to the
# random-backbone arm and the uniform 100-test-sample protocol -- supersedes
# the earlier bellinge_loop24_validation_n20_fixed.json (75-sample protocol):
STAGE2_NSE = 0.7456    # NSE_bellinge_raw (fine-tuned, uncontaminated)
STAGE2_RMSE = 18.53    # RMSE_cm_bellinge_raw
STAGE3_NSE = 0.7453    # NSE_bellinge (Stage 3, post-hoc affine, upper bound)
CAL_SCALE_A = 0.9604
CAL_OFFSET_B = 0.0011

SEOCHO_NSE = 0.626      # in-domain primary-model NSE (results/eval_loop24_val.json)
SEOCHO_RMSE = 12.15     # in-domain primary-model RMSE_cm
MORIASI_NSE = 0.65      # surrogate-to-SWMM "Good" reference threshold

# ── Okabe-Ito colorblind-safe palette ───────────────────────────────────────
OKABE_ITO = {
    "stage1": "#D55E00",       # vermillion
    "stage2": "#0072B2",       # blue
    "stage3": "#56B4E9",       # sky blue (lighter, visually distinct from stage2)
    "seocho_ref": "#009E73",   # bluish green
    "moriasi_ref": "#000000",  # black (dashed)
    "observed": "#000000",     # black
    "note": "#555555",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def get_device(s):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def compute_nse(pred, true):
    y = true.flatten()
    p = pred.flatten()
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def build_bellinge_event_split_dataset(config, device):
    """
    Builds the 20-event Bellinge dataset using the SAME paths / feature
    alignment as bellinge_validate_loop24.py, then splits it event-wise:
    EB001-EB015 -> fine-tune, EB016-EB020 -> held-out test (matches
    --event_split --n_test_events 5 default in that script).
    """
    from bellinge.bellinge_exact_matched_corrected_v2 import build_dataset

    with open(STRICT_CATALOG, encoding="utf-8") as f:
        catalog = json.load(f)
    logger.info(f"Loaded {len(catalog)}-event Bellinge catalog: {STRICT_CATALOG}")

    dataset = build_dataset(catalog, config)
    logger.info(f"Bellinge dataset: {len(dataset)} samples")

    all_event_ids = [ev for ev, _ in dataset._index]
    unique_events = list(dict.fromkeys(all_event_ids))
    test_events = set(unique_events[-N_TEST_EVENTS:])
    ft_idx = [i for i, (ev, _) in enumerate(dataset._index) if ev not in test_events]
    test_idx = [i for i, (ev, _) in enumerate(dataset._index) if ev in test_events]
    logger.info(f"Fine-tune events: {unique_events[:-N_TEST_EVENTS]}")
    logger.info(f"Held-out test events: {sorted(test_events)}")

    ft_data = [dataset[i] for i in ft_idx]
    test_data = [dataset[i] for i in test_idx]
    test_event_ids = [dataset._index[i][0] for i in test_idx]

    return ft_data, test_data, test_event_ids


def run_inference(model, data_list, device):
    """(n, N, T) for mean depth and truth, batch-by-batch (preserves sample axis)."""
    from torch_geometric.loader import DataLoader
    loader = DataLoader(data_list, batch_size=1, shuffle=False)
    preds, trues = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out["mean_depth"].cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    return np.stack(preds, axis=0), np.stack(trues, axis=0)


def style_axes(ax):
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.8, length=4)


def plot_nse_panel(ax):
    labels = ["Stage 1\nZero-shot", "Stage 2\nFine-tuned", "Stage 3\nAffine-calibrated"]
    values = [STAGE1_NSE, STAGE2_NSE, STAGE3_NSE]
    colors = [OKABE_ITO["stage1"], OKABE_ITO["stage2"], OKABE_ITO["stage3"]]
    x = np.arange(3)
    bars = ax.bar(x, values, width=0.55, color=colors, edgecolor="#303030",
                  linewidth=0.8, zorder=2)
    ax.axhline(MORIASI_NSE, color=OKABE_ITO["moriasi_ref"], ls="--", lw=1.4,
              zorder=1, label=f"Surrogate-to-SWMM reference (NSE={MORIASI_NSE})")
    ax.axhline(SEOCHO_NSE, color=OKABE_ITO["seocho_ref"], ls=":", lw=1.8,
              zorder=1, label=f"Seocho-gu in-domain (NSE={SEOCHO_NSE})")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
               ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x, labels)
    ax.set_ylabel("NSE")
    style_axes(ax)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.annotate("Stage 3 is a post-hoc, test-fitted\naffine upper-bound estimate,\nnot a generalization metric.",
               xy=(2, STAGE3_NSE), xytext=(1.15, 0.32),
               fontsize=7.5, color=OKABE_ITO["note"],
               arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8))


def plot_rmse_panel(ax):
    labels = ["Stage 1\nZero-shot", "Stage 2\nFine-tuned"]
    values = [STAGE1_RMSE, STAGE2_RMSE]
    colors = [OKABE_ITO["stage1"], OKABE_ITO["stage2"]]
    x = np.arange(2)
    bars = ax.bar(x, values, width=0.45, color=colors, edgecolor="#303030",
                  linewidth=0.8, zorder=2)
    ax.axhline(SEOCHO_RMSE, color=OKABE_ITO["seocho_ref"], ls=":", lw=1.8,
              zorder=1, label=f"Seocho-gu in-domain (RMSE={SEOCHO_RMSE} cm)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.6, f"{val:.1f} cm",
               ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("RMSE (cm)")
    ax.set_xticks(x, labels)
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.text(0.98, 0.05,
           "Stage 3 RMSE omitted: affine\nparameters are test-fitted\n(not independently evaluable).",
           ha="right", va="bottom", transform=ax.transAxes, fontsize=7.5,
           color=OKABE_ITO["note"],
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))


def plot_timeseries_panel(ax, t_axis, y_swmm, y_zs, y_ft, nse_ft_node, event_id, node_idx):
    ax.plot(t_axis, y_swmm, color=OKABE_ITO["observed"], lw=2.2, ls="-",
           marker=None, label="SWMM reference", zorder=4)
    ax.plot(t_axis, y_zs, color=OKABE_ITO["stage1"], lw=1.8, ls="--",
           label="Stage 1 - zero-shot", zorder=3)
    ax.plot(t_axis, y_ft, color=OKABE_ITO["stage2"], lw=1.8, ls="-.",
           label="Stage 2 - fine-tuned", zorder=3)
    ax.set_xlabel("Time (min, 5-min resolution)")
    ax.set_ylabel("Water depth (cm)")
    ax.set_xlim(0, int(t_axis[-1]))
    style_axes(ax)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    peak_idx = int(np.argmax(y_swmm))
    ax.annotate("Zero-shot depth-scale\nunder-prediction",
               xy=(t_axis[peak_idx], float(y_zs[peak_idx])),
               xytext=(min(t_axis[peak_idx] + 40, t_axis[-1] * 0.7), float(y_zs[peak_idx]) + max(y_swmm) * 0.18),
               fontsize=8.5, color=OKABE_ITO["stage1"],
               arrowprops=dict(arrowstyle="->", color=OKABE_ITO["stage1"], lw=1.0))
    ax.text(0.015, 0.95,
           f"Bellinge test event {event_id}, held-out node #{node_idx}\nFine-tuned per-node NSE = {nse_ft_node:.3f}",
           transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#333333")


def build_figure(t_axis, y_swmm, y_zs, y_ft, nse_ft_node, event_id, node_idx):
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1], hspace=0.38, wspace=0.30)
    ax_nse = fig.add_subplot(gs[0, 0])
    ax_rmse = fig.add_subplot(gs[0, 1])
    ax_ts = fig.add_subplot(gs[1, :])

    plot_nse_panel(ax_nse)
    plot_rmse_panel(ax_rmse)
    plot_timeseries_panel(ax_ts, t_axis, y_swmm, y_zs, y_ft, nse_ft_node, event_id, node_idx)

    ax_nse.text(-0.14, 1.05, "(a)", transform=ax_nse.transAxes, fontsize=15, fontweight="bold")
    ax_rmse.text(-0.14, 1.05, "(b)", transform=ax_rmse.transAxes, fontsize=15, fontweight="bold")
    ax_ts.text(-0.045, 1.04, "(c)", transform=ax_ts.transAxes, fontsize=15, fontweight="bold")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = get_device(args.device)
    logger.info(f"Device: {device}")

    import yaml
    with open(CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    exp_cfg = config["experiment"]
    M = exp_cfg.get("ensemble_size", 5)

    # ── Load pristine Loop-24 model, build event-disjoint dataset ────────────
    cp_dir = Path("results/checkpoints") / f"loop_{BACKBONE_LOOP:02d}"
    model, _ = bv.load_loop_model(cp_dir, config, device)

    ft_data, test_data, test_event_ids = build_bellinge_event_split_dataset(config, device)
    logger.info(f"Fine-tune samples: {len(ft_data)}, held-out test samples: {len(test_data)}")

    # ── Stage 1: zero-shot inference on held-out test events (pristine model) ─
    logger.info("Stage 1: zero-shot inference on held-out test events...")
    pred_zs, true_zs = run_inference(model, test_data, device)
    nse_zs = compute_nse(pred_zs, true_zs)
    rmse_zs = float(np.sqrt(((pred_zs - true_zs) ** 2).mean())) * 100
    logger.info(f"  Zero-shot (this rerun, held-out only) NSE={nse_zs:.4f}, RMSE={rmse_zs:.2f} cm")

    # ── Stage 2: fine-tune decoder node_head on the other 15 events ─────────
    logger.info(f"Stage 2: fine-tuning decoder node_head ({FT_EPOCHS} epochs)...")
    bv.fine_tune_bellinge_decoder(model, ft_data, n_epochs=FT_EPOCHS, lr=FT_LR, seed=42)

    pred_ft, true_ft = run_inference(model, test_data, device)
    nse_ft = compute_nse(pred_ft, true_ft)
    rmse_ft = float(np.sqrt(((pred_ft - true_ft) ** 2).mean())) * 100
    logger.info(f"  Fine-tuned (this rerun, held-out only) NSE={nse_ft:.4f}, RMSE={rmse_ft:.2f} cm")

    # ── Pick a representative node/event for panel (c) ───────────────────────
    per_node_nse_ft = []
    for ni in range(true_ft.shape[1]):
        y = true_ft[:, ni, :].flatten()
        pr = pred_ft[:, ni, :].flatten()
        ss_res = ((y - pr) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        per_node_nse_ft.append(1.0 - ss_res / (ss_tot + 1e-9))
    per_node_nse_ft = np.array(per_node_nse_ft)

    # Prefer a node with reasonable peak depth (>= 20 cm) and FT NSE close to
    # the pooled Stage-2 value, for a representative (not cherry-picked-best)
    # illustration.
    peak_depths_cm = true_ft.max(axis=(0, 2)) * 100
    candidate_mask = peak_depths_cm >= 20.0
    if not candidate_mask.any():
        candidate_mask = np.ones_like(peak_depths_cm, dtype=bool)
    candidate_idx = np.where(candidate_mask)[0]
    target_nse = np.median(per_node_nse_ft[candidate_idx])
    best_node = int(candidate_idx[np.argmin(np.abs(per_node_nse_ft[candidate_idx] - target_nse))])

    # Prefer the test sample (event x param-set) with the largest peak depth
    # at that node for visual clarity. test_event_ids is in per-sample order.
    node_peak_per_sample = true_ft[:, best_node, :].max(axis=1)
    best_ev = int(np.argmax(node_peak_per_sample))
    event_id = test_event_ids[best_ev]

    T = true_ft.shape[2]
    t_axis = np.arange(T) * 5
    y_swmm = true_ft[best_ev, best_node, :] * 100
    y_zs = pred_zs[best_ev, best_node, :] * 100
    y_ft = pred_ft[best_ev, best_node, :] * 100

    logger.info(f"Panel (c): event={event_id}, node_idx={best_node}, "
               f"per-node FT NSE={per_node_nse_ft[best_node]:.3f}, "
               f"peak depth={y_swmm.max():.1f} cm")

    fig = build_figure(t_axis, y_swmm, y_zs, y_ft, per_node_nse_ft[best_node], event_id, best_node)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_PATH), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
