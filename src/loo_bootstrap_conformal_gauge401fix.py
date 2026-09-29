"""Leave-one-event-out calibration and event-bootstrap sensitivity for the
conformal quantile, rerun on the gauge-and-timestamp-corrected primary model
(loop 300, seed 42), replacing the pre-correction diagnostics cited in the
manuscript (q_hat 8.8-11.4cm across LOO folds; bootstrap CI 7.1-14.4cm for
q_hat; 95% CI 0.820-0.919 for test coverage).

Methodology (matches the pre-correction analysis being replaced):
  - 7 val_conformal (2023) events, ~50 samples/event.
  - Leave-one-event-out: calibrate on the other 6 events' samples, report the
    resulting scalar q_hat (mean of the per-timestep quantile vector, in cm)
    for each of the 7 folds -> range across folds.
  - Event-bootstrap of q_hat: resample the 7 calibration events with
    replacement (n_boot draws), recalibrate each time, take the 2.5/97.5
    percentiles of the resulting q_hat distribution.
  - Event-bootstrap of test coverage: using the full-7-event calibration
    q_hat (the headline quantile), resample the 5 test_temporal events with
    replacement, recompute pooled PICR_90 each time, take the 2.5/97.5
    percentiles.

Usage:
    python src/loo_bootstrap_conformal_gauge401fix.py --device auto
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

LOOP_ID = 300
CONFIG_PATH = "configs/config_gauge401_fix_v1.yaml"
ALPHA = 0.10
N_BOOT = 500
RNG_SEED = 42


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


def collect_by_event(model, loader, device):
    """Returns {event_id: (pred (n,N,T), true (n,N,T))}."""
    by_event = {}
    with torch.no_grad():
        for batch in loader:
            ev_id = getattr(batch, "event_id", None)
            if isinstance(ev_id, (list, tuple)):
                ev_id = ev_id[0]
            ev_id = str(ev_id)
            batch_dev = batch.to(device)
            out = model(batch_dev)
            pred = out["mean_depth"].cpu().numpy()
            true = batch.y.cpu().numpy()
            by_event.setdefault(ev_id, {"pred": [], "true": []})
            by_event[ev_id]["pred"].append(pred)
            by_event[ev_id]["true"].append(true)
    return {ev: (np.stack(d["pred"], axis=0), np.stack(d["true"], axis=0))
            for ev, d in by_event.items()}


def q_hat_from_events(cp_cls, events_dict, event_ids):
    """Calibrate on the union of samples from the given events; return
    scalar q_hat (mean of the per-timestep vector) in metres."""
    preds = np.concatenate([events_dict[e][0] for e in event_ids], axis=0)
    trues = np.concatenate([events_dict[e][1] for e in event_ids], axis=0)
    T = preds.shape[-1]
    cp = cp_cls(alpha=ALPHA)
    cp.calibrate(preds.reshape(-1, T), trues.reshape(-1, T), regime_id="B")
    q_hat_vec = cp.q_hat_dict["B"]
    return float(q_hat_vec.mean()), q_hat_vec


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
    from src.models.conformal_uq import ConformalPredictor
    _, val_loader, calib_loader, test_loader = build_dataloaders(config, regime_id="B")

    logger.info("Collecting calibration (val_conformal, 2023) predictions by event...")
    calib_by_event = collect_by_event(model, calib_loader, device)
    calib_events = sorted(calib_by_event.keys())
    logger.info(f"Calibration events ({len(calib_events)}): {calib_events}")

    logger.info("Collecting test (test_temporal, 2024) predictions by event...")
    test_by_event = collect_by_event(model, test_loader, device)
    test_events = sorted(test_by_event.keys())
    logger.info(f"Test events ({len(test_events)}): {test_events}")

    # ---- Headline: calibrate on all 7 events -------------------------------
    headline_q_hat_m, headline_q_hat_vec = q_hat_from_events(
        ConformalPredictor, calib_by_event, calib_events)
    logger.info(f"Headline q_hat (all {len(calib_events)} calib events): "
                f"{headline_q_hat_m * 100:.2f} cm")

    # ---- Leave-one-event-out ------------------------------------------------
    loo_q_hats_cm = []
    for held_out in calib_events:
        fold_events = [e for e in calib_events if e != held_out]
        q_m, _ = q_hat_from_events(ConformalPredictor, calib_by_event, fold_events)
        loo_q_hats_cm.append(q_m * 100)
        logger.info(f"  LOO fold (holding out {held_out}): q_hat = {q_m * 100:.2f} cm")
    loo_min, loo_max = min(loo_q_hats_cm), max(loo_q_hats_cm)
    logger.info(f"LOO q_hat range: {loo_min:.2f}--{loo_max:.2f} cm")

    # ---- Event-bootstrap of q_hat --------------------------------------------
    rng = np.random.default_rng(RNG_SEED)
    n_calib = len(calib_events)
    boot_q_hats_cm = []
    for _ in range(N_BOOT):
        sampled = rng.choice(calib_events, size=n_calib, replace=True).tolist()
        q_m, _ = q_hat_from_events(ConformalPredictor, calib_by_event, sampled)
        boot_q_hats_cm.append(q_m * 100)
    boot_q_hats_cm = np.array(boot_q_hats_cm)
    q_hat_ci = (float(np.percentile(boot_q_hats_cm, 2.5)),
                float(np.percentile(boot_q_hats_cm, 97.5)))
    logger.info(f"Event-bootstrap 95% CI for q_hat ({N_BOOT} draws): "
                f"{q_hat_ci[0]:.2f}--{q_hat_ci[1]:.2f} cm")

    # ---- Event-bootstrap of test coverage (using the headline q_hat) --------
    n_test = len(test_events)
    boot_coverages = []
    for _ in range(N_BOOT):
        sampled = rng.choice(test_events, size=n_test, replace=True).tolist()
        preds = np.concatenate([test_by_event[e][0] for e in sampled], axis=0)
        trues = np.concatenate([test_by_event[e][1] for e in sampled], axis=0)
        lower = preds - headline_q_hat_vec
        upper = preds + headline_q_hat_vec
        covered = (trues >= lower) & (trues <= upper)
        boot_coverages.append(float(covered.mean()))
    boot_coverages = np.array(boot_coverages)
    coverage_ci = (float(np.percentile(boot_coverages, 2.5)),
                   float(np.percentile(boot_coverages, 97.5)))
    logger.info(f"Event-bootstrap 95% CI for test coverage ({N_BOOT} draws): "
                f"{coverage_ci[0]:.3f}--{coverage_ci[1]:.3f}")

    out = {
        "loop_id": LOOP_ID,
        "protocol": "gauge-and-timestamp-corrected",
        "alpha": ALPHA,
        "n_boot": N_BOOT,
        "rng_seed": RNG_SEED,
        "calib_events": calib_events,
        "test_events": test_events,
        "headline_q_hat_cm": headline_q_hat_m * 100,
        "loo_q_hats_cm": dict(zip(calib_events, loo_q_hats_cm)),
        "loo_q_hat_range_cm": [loo_min, loo_max],
        "bootstrap_q_hat_ci_cm": list(q_hat_ci),
        "bootstrap_test_coverage_ci": list(coverage_ci),
    }
    out_path = Path("results") / "timecorrected_v2" / "loo_bootstrap_conformal_gauge401fix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
