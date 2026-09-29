"""Recompute the "reuse Seocho-gu empirical residual quantiles on Bellinge"
interval-transfer result under the FINAL protocol (Loop 300, gauge401fix,
paramfix2 feature-masking fix) -- no retraining, inference + calibration only.

The manuscript previously cited PICR90=0.997 / width=242cm, which comes from
results/bellinge_loop24_validation_n20_fixed.json (Loop 24, 75-sample,
pre-gauge401fix backbone) -- not the final protocol. This script:

  1. Loads the Loop 300 checkpoint (the final primary backbone).
  2. Runs it on the Seocho val_conformal (2023) calibration split to get the
     per-timestep q_t array exactly as in src/evaluate.py / Methods Eq. (2):
     q_t = quantile_{ceil((1-alpha)(n+1))/n} of pooled |y - yhat| at timestep t.
  3. Applies q_t to the ALREADY-CACHED final Bellinge fine-tuned predictions
     (results/timecorrected_v2/fig8_full_preds_cache_gauge401fix.npz, raw /
     uncalibrated, same convention as the reported headline NSE 0.7759) to
     get PICR90 and mean width for Bellinge under the reused Seocho quantile.

Output: results/timecorrected_v2/bellinge_interval_transfer_final.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASET_CLASS_USED = "TimeCorrectedSWMMGraphDataset (report_lead_steps trimmed per run_manifest.json)"

from src.train import build_dataloaders
from src.models.conformal_uq import ConformalPredictor

CONFIG = ROOT / "configs/config_gauge401_fix_v1.yaml"
CACHE = ROOT / "results/timecorrected_v2/fig8_full_preds_cache_gauge401fix.npz"
OUT = ROOT / "results/timecorrected_v2/bellinge_interval_transfer_final_v2.json"
ALPHA = 0.10


def load_ensemble(cp_dir, base_cfg, M, device):
    from src.models.gnn_surrogate import EnsembleGNN
    model = EnsembleGNN(base_cfg, M=M).to(device)
    from src.checkpoints import load_members
    load_members(model, cp_dir, device)
    return model


def main():
    global CONFIG, CACHE, OUT
    parser = argparse.ArgumentParser(description="Evaluate interval transfer from cached Bellinge predictions")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    CONFIG, CACHE, OUT = args.config, args.cache, args.output
    if not CACHE.is_file():
        raise FileNotFoundError(f"Missing cached Bellinge predictions: {CACHE}")
    if OUT.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {OUT}; use --overwrite explicitly")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    exp = config.get("experiment", {})
    base_cfg = {
        "node_feat_dim": 14, "edge_feat_dim": 4,
        "hidden_dim": exp.get("hidden_dim", 128), "T_out": exp.get("T_out", 100),
        "n_heads": exp.get("n_heads", 4), "n_layers": exp.get("n_layers", 4),
        "dropout": exp.get("dropout", 0.0),
        "temporal_decoder": exp.get("temporal_decoder", False),
        "scalar_rain_decoder": exp.get("scalar_rain_decoder", False),
        "conv_type": exp.get("conv_type", "gat"),
    }
    M = exp.get("ensemble_size", 5)
    cp_dir = ROOT / "results/checkpoints/loop_300"
    model = load_ensemble(cp_dir, base_cfg, M, device)

    print("Building dataloaders (regime B) for Seocho val_conformal calibration split...")
    _, _, calib_loader, _ = build_dataloaders(config, "B")
    print(f"  calib graphs: {len(calib_loader.dataset)}")

    calib_preds, calib_trues = [], []
    with torch.no_grad():
        for batch in calib_loader:
            batch = batch.to(device)
            out = model(batch)
            calib_preds.append(out["mean_depth"].cpu().numpy())
            calib_trues.append(batch.y.cpu().numpy())
    y_pred_cal = np.concatenate(calib_preds, axis=0)
    y_true_cal = np.concatenate(calib_trues, axis=0)
    n_cal = y_pred_cal.shape[0]
    print(f"  n_calib_graphs x N_nodes = {n_cal}  (T={y_pred_cal.shape[1]})")

    cp = ConformalPredictor(alpha=ALPHA)
    cp.calibrate(y_pred_cal.reshape(n_cal, -1), y_true_cal.reshape(n_cal, -1), regime_id="B")
    q_t = cp.q_hat_dict["B"]  # shape (T,)
    print(f"  Seocho q_t: shape={q_t.shape}, mean_half_width_m={q_t.mean():.4f} "
          f"({q_t.mean()*100:.2f} cm), min={q_t.min()*100:.2f}cm max={q_t.max()*100:.2f}cm")

    print(f"Loading cached final Bellinge (Loop 300, paramfix2 masking) predictions from {CACHE}")
    z = np.load(CACHE, allow_pickle=True)
    true_bellinge = z["true_fine_tuned"]   # (S, N, T), absolute depth in metres, same as headline raw-NSE convention
    pred_bellinge = z["pred_fine_tuned"]   # (S, N, T)
    S, N, T = true_bellinge.shape
    assert T == q_t.shape[0], f"timestep mismatch: Bellinge T={T} vs Seocho q_t T={q_t.shape[0]}"

    lower = pred_bellinge - q_t[None, None, :]
    upper = pred_bellinge + q_t[None, None, :]
    covered = (true_bellinge >= lower) & (true_bellinge <= upper)
    picr90 = float(covered.mean())
    mean_width_m = float(2.0 * q_t.mean())
    mean_width_cm = mean_width_m * 100.0

    result = {
        "description": "Reusing Seocho-gu empirical residual quantiles (Loop 300, final "
                        "gauge401fix/paramfix2 protocol) on the final Bellinge fine-tuned "
                        "predictions (same checkpoint/backbone as the headline NSE=0.7759).",
        "dataset_class_used": DATASET_CLASS_USED,
        "seocho_calibration_split": "val_conformal (2023, 7 events x 50 sets)",
        "seocho_q_t_mean_half_width_cm": float(q_t.mean() * 100.0),
        "seocho_q_t_min_cm": float(q_t.min() * 100.0),
        "seocho_q_t_max_cm": float(q_t.max() * 100.0),
        "bellinge_n_samples": int(S),
        "bellinge_n_nodes": int(N),
        "bellinge_PICR_90": picr90,
        "bellinge_mean_width_cm": mean_width_cm,
        "note": "Supersedes the pre-gauge401fix Loop-24 value (PICR90=0.997, width=242cm) "
                "from results/bellinge_loop24_validation_n20_fixed.json, which used a "
                "different backbone/protocol and is not the final experiment. Also "
                "supersedes results/timecorrected_v2/bellinge_interval_transfer_final.json "
                "(v1), which called build_dataloaders() directly and therefore used the "
                "legacy (untrimmed) SWMMGraphDataset instead of the final "
                "TimeCorrectedSWMMGraphDataset adapter (6 pre-event lead steps not "
                "removed) -- this v2 run installs that adapter explicitly.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
