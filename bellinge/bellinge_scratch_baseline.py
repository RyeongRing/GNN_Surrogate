"""Random-backbone control for the Bellinge decoder-fine-tuning transfer result.

Isolates whether the PRETRAINED Seocho-gu backbone specifically drives the
Stage-2 transfer result (NSE=0.765), or whether any fixed backbone -- even a
randomly initialised, never-trained one -- would do about as well once only
the decoder is fit on the 15 Bellinge fine-tuning events. Trainable
parameter count is identical in both arms (decoder.node_head only, backbone
frozen); only the backbone's origin (pretrained vs random) differs.

Reuses the exact same 20-event catalog, 15/5 event split, fine-tuning
procedure (150 epochs, lr=5e-4), and evaluation code as
bellinge_validate_loop24.py so results are directly comparable to the
existing Loop 24 result (results/bellinge_loop24_validation_n20_fixed.json).

Usage:
    python bellinge_scratch_baseline.py --seeds 42,100,200,300,400
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from bellinge.bellinge_validate_loop24 import (
    BELLINGE_INP, BELLINGE_DAT, BELLINGE_ENS, BELLINGE_CAT,
    parse_dat_file, fine_tune_bellinge_decoder, evaluate_bellinge_gnn,
)


def build_dataset(events, config):
    from src.data.swmm_dataset import SWMMGraphDataset
    exp = config.get("experiment", {})
    rain_1min = parse_dat_file(BELLINGE_DAT)
    rain_hourly = rain_1min.resample("1h").sum().fillna(0)
    dataset = SWMMGraphDataset(
        inp_path=str(BELLINGE_INP),
        ensemble_dir=str(BELLINGE_ENS),
        catalog=events,
        rain_series=rain_hourly,
        T_out=exp.get("T_out", 100),
        T_rain=exp.get("T_rain", 72),
        n_sets=20,
        site_id="bellinge",
    )
    _INF_MEAN = [5.0033, 140.14, 0.5148, 0.2750]  # dstore_perv, inf_max, inf_min, inf_decay
    for sample in dataset:
        sample.x[:, 1] = 0.0
        for _j, _v in enumerate(_INF_MEAN):
            sample.x[:, -4 + _j] = _v
    return dataset


def random_backbone_model(config, seed, device):
    import torch
    from src.models.gnn_surrogate import EnsembleGNN
    torch.manual_seed(seed)
    exp = config.get("experiment", {})
    base_cfg = {
        "node_feat_dim": 14, "edge_feat_dim": 4,
        "hidden_dim": exp.get("hidden_dim", 128),
        "T_out": exp.get("T_out", 100), "T_rain": exp.get("T_rain", 72),
        "n_heads": exp.get("n_heads", 4), "n_layers": exp.get("n_layers", 4),
        "dropout": exp.get("dropout", 0.0),
        "temporal_decoder": exp.get("temporal_decoder", False),
        "scalar_rain_decoder": exp.get("scalar_rain_decoder", False),
    }
    M = exp.get("ensemble_size", 5)
    model = EnsembleGNN(base_cfg, M=M).to(device)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_seed_exp.yaml")
    p.add_argument("--seeds", default="42,100,200,300,400")
    p.add_argument("--ft_epochs", type=int, default=150)
    p.add_argument("--ft_lr", type=float, default=5e-4)
    p.add_argument("--n_test_events", type=int, default=5)
    p.add_argument("--output", default="results/bellinge_scratch_baseline.json")
    args = p.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(BELLINGE_CAT, encoding="utf-8") as f:
        events = json.load(f)
    logger.info(f"Loaded existing catalog: {len(events)} events (identical to Loop 24 run)")

    dataset = build_dataset(events, config)
    logger.info(f"Bellinge dataset: {len(dataset)} samples")

    all_event_ids = [ev for ev, _ in dataset._index]
    unique_events = list(dict.fromkeys(all_event_ids))
    n_te = min(args.n_test_events, len(unique_events) - 1)
    test_events = set(unique_events[-n_te:])
    ft_idx = [i for i, (ev, _) in enumerate(dataset._index) if ev not in test_events]
    test_idx = [i for i, (ev, _) in enumerate(dataset._index) if ev in test_events]
    logger.info(f"ft_events={unique_events[:-n_te]}")
    logger.info(f"test_events={list(test_events)}")

    ft_data = [dataset[i] for i in ft_idx]
    test_data = [dataset[i] for i in test_idx]
    logger.info(f"Split: {len(ft_data)} fine-tune samples / {len(test_data)} test samples")

    seeds = [int(s) for s in args.seeds.split(",")]
    results = {}
    for seed in seeds:
        logger.info(f"=== Random-backbone seed {seed} ===")
        t0 = time.time()
        model = random_backbone_model(config, seed, device)
        fine_tune_bellinge_decoder(model, ft_data, n_epochs=args.ft_epochs,
                                    lr=args.ft_lr, seed=seed)
        metrics = evaluate_bellinge_gnn(model, test_data, config, device,
                                          calib_loader=None,
                                          n_cal=min(5, len(test_data) // 4),
                                          seed=seed)
        metrics["elapsed_s"] = round(time.time() - t0, 1)
        results[seed] = metrics
        logger.info(f"  seed {seed}: NSE_raw={metrics.get('NSE_bellinge_raw')}, "
                    f"RMSE_raw={metrics.get('RMSE_cm_bellinge_raw')}, "
                    f"elapsed={metrics['elapsed_s']}s")

    nse_raws = [results[s]["NSE_bellinge_raw"] for s in seeds]
    rmse_raws = [results[s]["RMSE_cm_bellinge_raw"] for s in seeds]
    summary = {
        "nse_raw_mean": float(np.mean(nse_raws)),
        "nse_raw_std": float(np.std(nse_raws, ddof=1)) if len(seeds) > 1 else 0.0,
        "nse_raw_values": nse_raws,
        "rmse_raw_mean": float(np.mean(rmse_raws)),
        "rmse_raw_std": float(np.std(rmse_raws, ddof=1)) if len(seeds) > 1 else 0.0,
        "pretrained_backbone_reference_nse_raw": 0.7648,
        "pretrained_backbone_reference_rmse_raw": 17.86,
    }
    out = {"per_seed": results, "summary": summary}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved -> {args.output}")
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
