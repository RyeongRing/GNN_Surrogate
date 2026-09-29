"""Regenerate the two-panel Bellinge transfer figure for the
gauge-and-timestamp-corrected backbone (Loop 300, seed 42).

Panel (a): designated headline NSE bars (0.1413 zero-shot, 0.7759 fine-tuned)
against the Seocho-gu in-domain reference.
Panel (b): one held-out event/node hydrograph. The node is a *representative
well-fitting* node -- among nodes whose peak reference depth >= 20 cm, the one
whose per-node fine-tuned NSE is closest to the 75th percentile of that set
(a good-but-not-cherry-picked example), unless FIG8_NODE / FIG8_SAMPLE
environment variables force a specific (node, sample) pair.

Plot-only by default. A missing cache is an error; --recompute explicitly
requests a NEW 150-epoch node-head fine-tuning run. Its trajectory can differ
from the manuscript illustration. Panel (a) retains designated headline values
and is not recalculated from panel (b)'s separate execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import src.analysis.generate_fig9_bellinge as base
import src.analysis.generate_fig9_bellinge_two_panel_corrected as two_panel

# ---- Patch the base module's constants to the gauge401fix backbone ----
base.BACKBONE_LOOP = 300
base.CONFIG = "configs/config_gauge401_fix_v1.yaml"
base.STAGE1_NSE = 0.1413
base.STAGE1_RMSE = 34.05
base.STAGE2_NSE = 0.7759
base.STAGE2_RMSE = 17.39
base.SEOCHO_NSE = 0.6649
base.SEOCHO_RMSE = 11.89

OUTPUT = Path("paper/figures/fig8_bellinge_transfer.png")
METRICS_OUTPUT = Path("results/timecorrected_v2/fig9_bellinge_two_panel_gauge401fix_metrics.json")
CACHE = Path("results/timecorrected_v2/fig8_full_preds_cache_gauge401fix.npz")


def _per_node_nse(true_ft: np.ndarray, pred_ft: np.ndarray) -> np.ndarray:
    """Per-node NSE pooled over samples and timesteps. Shape (N,)."""
    out = np.full(true_ft.shape[1], np.nan)
    for j in range(true_ft.shape[1]):
        y = np.asarray(true_ft[:, j, :], dtype=np.float64).reshape(-1)
        p = np.asarray(pred_ft[:, j, :], dtype=np.float64).reshape(-1)
        sst = float(np.sum((y - y.mean()) ** 2))
        if sst > 0:
            out[j] = 1.0 - float(np.sum((y - p) ** 2)) / sst
    return out


def select_representative_node(true_ft, pred_ft):
    """Node with peak reference depth >= 20 cm whose per-node NSE is closest to
    the 75th percentile of that qualifying set. Returns (node_idx, sample_idx,
    per_node_nse)."""
    per_node_nse = _per_node_nse(true_ft, pred_ft)
    peak_cm = true_ft.max(axis=(0, 2)) * 100.0
    qual = (peak_cm >= 20.0) & np.isfinite(per_node_nse)
    if not qual.any():
        qual = np.isfinite(per_node_nse)
    cand = np.where(qual)[0]
    target = np.percentile(per_node_nse[cand], 75)
    node_idx = int(cand[np.argmin(np.abs(per_node_nse[cand] - target))])
    peak_by_sample = true_ft[:, node_idx, :].max(axis=1)
    sample_idx = int(np.argmax(peak_by_sample))
    return node_idx, sample_idx, per_node_nse


def main() -> None:
    global CACHE, OUTPUT, METRICS_OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=base.CONFIG)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--metrics-output", type=Path, default=METRICS_OUTPUT)
    parser.add_argument("--node", type=int, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--recompute", action="store_true", help="Explicitly run new node-head training")
    parser.add_argument("--overwrite-cache", action="store_true")
    args = parser.parse_args()
    CACHE, OUTPUT, METRICS_OUTPUT = args.cache, args.output, args.metrics_output
    base.CONFIG = args.config
    if not CACHE.is_file() and not args.recompute:
        raise FileNotFoundError(f"Missing {CACHE}; plot-only mode never trains. "
                                "Provide the original cache or explicitly use --recompute.")
    if CACHE.exists() and args.recompute and not args.overwrite_cache:
        raise FileExistsError("Use a new --cache path or explicitly --overwrite-cache")
    device = base.get_device("auto")
    with open(base.CONFIG, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if CACHE.exists() and not args.recompute:
        z = np.load(CACHE, allow_pickle=True)
        pred_zero_shot = z["pred_zero_shot"]
        pred_fine_tuned = z["pred_fine_tuned"]
        true_fine_tuned = z["true_fine_tuned"]
        test_event_ids = list(z["test_event_ids"])
        print(f"loaded cached predictions from {CACHE}")
        if "dt_seconds_per_sample" in z.files:
            dt_seconds_per_sample = np.asarray(z["dt_seconds_per_sample"], dtype=np.float64)
        else:
            raise ValueError("Cache lacks dt_seconds_per_sample; obtain the complete final cache "
                             "or explicitly recompute to a NEW cache path")
        z.close()
    else:
        checkpoint_dir = Path("results/checkpoints") / f"loop_{base.BACKBONE_LOOP}"
        model, _ = base.bv.load_loop_model(checkpoint_dir, config, device)
        ft_data, test_data, test_event_ids = base.build_bellinge_event_split_dataset(config, device)
        dt_seconds_per_sample = np.array(
            [float(d.dt_seconds.reshape(-1)[0]) for d in test_data], dtype=np.float64
        )
        pred_zero_shot, _ = base.run_inference(model, test_data, device)
        base.bv.fine_tune_bellinge_decoder(model, ft_data, n_epochs=base.FT_EPOCHS, lr=base.FT_LR, seed=42)
        pred_fine_tuned, true_fine_tuned = base.run_inference(model, test_data, device)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            CACHE,
            pred_zero_shot=pred_zero_shot,
            pred_fine_tuned=pred_fine_tuned,
            true_fine_tuned=true_fine_tuned,
            test_event_ids=np.array(test_event_ids, dtype=object),
            dt_seconds_per_sample=dt_seconds_per_sample,
        )
        print(f"cached predictions -> {CACHE}")
        def digest(path):
            hasher = hashlib.sha256()
            with Path(path).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()

        provenance = {
            "cache_sha256": digest(CACHE), "config_sha256": digest(base.CONFIG),
            "checkpoint_sha256": {p.name: digest(p) for p in sorted(checkpoint_dir.glob("member_*.pt"))},
            "catalog_sha256": digest(base.STRICT_CATALOG),
            "inp_sha256": digest(base.bv.BELLINGE_INP),
            "rainfall_sha256": digest(base.bv.BELLINGE_DAT),
            "ensemble_root": str(base.bv.BELLINGE_ENS),
            "seed": 42, "epochs": base.FT_EPOCHS, "learning_rate": base.FT_LR,
            "role": "separate illustration run; not the source of panel (a)'s headline values",
        }
        CACHE.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    env_node = args.node if args.node is not None else os.environ.get("FIG8_NODE")
    env_sample = args.sample if args.sample is not None else os.environ.get("FIG8_SAMPLE")
    if (env_node is None) != (env_sample is None):
        raise ValueError("Specify both --node and --sample, or neither")
    if env_node is not None and env_sample is not None:
        node_idx, sample_idx = int(env_node), int(env_sample)
        per_node_nse = _per_node_nse(true_fine_tuned, pred_fine_tuned)
        selection = f"forced (FIG8_NODE={node_idx}, FIG8_SAMPLE={sample_idx})"
    else:
        node_idx, sample_idx, per_node_nse = select_representative_node(true_fine_tuned, pred_fine_tuned)
        selection = "75th-percentile per-node NSE among nodes with peak depth >= 20 cm"

    if not (0 <= node_idx < true_fine_tuned.shape[1] and 0 <= sample_idx < true_fine_tuned.shape[0]):
        raise ValueError("Selected node/sample lies outside the cached arrays")
    if not np.isfinite(dt_seconds_per_sample).all() or np.any(dt_seconds_per_sample <= 0):
        raise ValueError("Cache must contain positive finite per-sample time spacing")

    event_id = test_event_ids[sample_idx]
    true_series = true_fine_tuned[sample_idx, node_idx, :]
    zero_shot_series = pred_zero_shot[sample_idx, node_idx, :]
    fine_tuned_series = pred_fine_tuned[sample_idx, node_idx, :]
    displayed_nse, _, _ = two_panel.nse_with_sums(true_series, fine_tuned_series)
    global_nse = base.compute_nse(pred_fine_tuned, true_fine_tuned)
    dt_seconds = float(dt_seconds_per_sample[sample_idx])
    time_min = np.arange(true_series.shape[0]) * (dt_seconds / 60.0)

    fig, axes = plt.subplots(
        1, 2, figsize=(19.8, 8.6),
        gridspec_kw={"width_ratios": [1.0, 1.95], "wspace": 0.27},
    )
    two_panel.plot_summary_panel(axes[0])
    two_panel.plot_series_panel(
        axes[1], time_min,
        true_series * 100.0, zero_shot_series * 100.0, fine_tuned_series * 100.0,
        displayed_nse, event_id, node_idx,
    )
    axes[0].text(-0.18, 1.03, "(a)", transform=axes[0].transAxes, fontsize=25, fontweight="bold")
    axes[1].text(-0.07, 1.03, "(b)", transform=axes[1].transAxes, fontsize=25, fontweight="bold")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    metrics = {
        "figure": str(OUTPUT),
        "checkpoint_loop": base.BACKBONE_LOOP,
        "config": base.CONFIG,
        "held_out_events": sorted(set(test_event_ids)),
        "n_test_samples": int(true_fine_tuned.shape[0]),
        "node_selection_rule": selection,
        "selected_event": event_id,
        "selected_node": int(node_idx),
        "selected_sample_index": int(sample_idx),
        "time_axis_dt_seconds": dt_seconds,
        "time_axis_dt_minutes": dt_seconds / 60.0,
        "time_axis_span_minutes": float(time_min[-1]),
        "displayed_series_nse": float(displayed_nse),
        "selected_node_pooled_nse": float(per_node_nse[node_idx]),
        "rerun_global_fine_tuned_nse": float(global_nse),
        "designated_stage1_nse": base.STAGE1_NSE,
        "designated_stage2_nse": base.STAGE2_NSE,
        "cache": str(CACHE),
        "panel_a_provenance": "Designated primary-run results, not panel (b)'s separate execution",
        "panel_b_provenance": "New illustration run" if args.recompute else "Provided prediction cache",
    }
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
