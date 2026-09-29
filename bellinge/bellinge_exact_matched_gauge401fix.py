"""Bellinge exact-matched pretrained-vs-random backbone control, rerun against
the gauge-and-timestamp-corrected Seocho-gu backbone.

Identical protocol to bellinge_exact_matched_corrected_v2.py (same 20-event
catalog, same 15/5 split, same n_sets=20, same 150-epoch/5e-4 decoder
fine-tuning, same matched-decoder-init fix), except:

1. Backbone checkpoints are the gauge401fix loops (300, 301-307) trained on
   the gauge-and-timestamp-corrected SWMM ensemble (AWS 401/Seocho explicit
   selection on top of the -60min KMA end-of-interval fix), not the
   timestamp-only-corrected loops (124, 127-133).
2. Default config is config_gauge401_fix_v1.yaml (architecture params are
   identical to config_timecorrected_v2.yaml -- hidden_dim=128, n_heads=4,
   n_layers=4, lambda_cont=0.1, ensemble_size=5 -- so checkpoints load
   without changes).

Bellinge's own SWMM ensemble/rainfall data (DTU, Denmark) is unaffected by
either Seocho-gu-specific bug -- only the backbone being fine-tuned changes.
"""
import copy
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
    BELLINGE_CAT,
    fine_tune_bellinge_decoder, evaluate_bellinge_gnn,
    load_loop_model,
)
from bellinge.bellinge_scratch_baseline import random_backbone_model
from bellinge.bellinge_exact_matched_corrected_v2 import build_dataset

SEED_TO_LOOP = {
    42: 300, 100: 301, 200: 302, 300: 303,
    400: 304, 500: 305, 600: 306, 700: 307,
}


def main():
    import argparse
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config_gauge401_fix_v1.yaml")
    p.add_argument("--seeds", default="42,100,200,300,400,500,600,700")
    p.add_argument("--ft_epochs", type=int, default=150)
    p.add_argument("--ft_lr", type=float, default=5e-4)
    p.add_argument("--n_test_events", type=int, default=5)
    p.add_argument("--output", default="results/timecorrected_v2/bellinge_exact_matched_gauge401fix.json")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if Path(args.output).exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {args.output}; use --overwrite explicitly")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open(BELLINGE_CAT, encoding="utf-8") as f:
        events = json.load(f)
    logger.info(f"Loaded existing catalog: {len(events)} events")

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
    logger.info(f"Split: {len(ft_data)} fine-tune samples / {len(test_data)} test samples "
                f"(identical for both arms, all seeds)")

    seeds = [int(s) for s in args.seeds.split(",")]
    pretrained_results = {}
    random_results = {}

    for seed in seeds:
        loop_id = SEED_TO_LOOP[seed]
        logger.info(f"=== seed {seed} (gauge401fix Loop {loop_id}) ===")

        t0 = time.time()
        random_model = random_backbone_model(config, seed, device)
        matched_decoder_sds = [
            copy.deepcopy(m.decoder.node_head.state_dict())
            for m in random_model.members
        ]

        fine_tune_bellinge_decoder(random_model, ft_data, n_epochs=args.ft_epochs,
                                    lr=args.ft_lr, seed=seed)
        rand_metrics = evaluate_bellinge_gnn(random_model, test_data, config, device,
                                              calib_loader=None,
                                              n_cal=min(5, len(test_data) // 4),
                                              seed=seed)
        rand_metrics["elapsed_s"] = round(time.time() - t0, 1)
        random_results[seed] = rand_metrics
        logger.info(f"  [random]     seed {seed}: NSE_raw={rand_metrics.get('NSE_bellinge_raw')}")

        t1 = time.time()
        pretrained_model, M = load_loop_model(Path(f"results/checkpoints/loop_{loop_id}"),
                                               config, device)
        for i, m in enumerate(pretrained_model.members):
            m.decoder.node_head.load_state_dict(copy.deepcopy(matched_decoder_sds[i]))
        fine_tune_bellinge_decoder(pretrained_model, ft_data, n_epochs=args.ft_epochs,
                                    lr=args.ft_lr, seed=seed)
        pre_metrics = evaluate_bellinge_gnn(pretrained_model, test_data, config, device,
                                             calib_loader=None,
                                             n_cal=min(5, len(test_data) // 4),
                                             seed=seed)
        pre_metrics["elapsed_s"] = round(time.time() - t1, 1)
        pre_metrics["loop_id"] = loop_id
        pretrained_results[seed] = pre_metrics
        logger.info(f"  [pretrained] seed {seed}: NSE_raw={pre_metrics.get('NSE_bellinge_raw')}")

    pre_nse = [pretrained_results[s]["NSE_bellinge_raw"] for s in seeds]
    rand_nse = [random_results[s]["NSE_bellinge_raw"] for s in seeds]
    diffs = [pv - rv for pv, rv in zip(pre_nse, rand_nse)]

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    out = {
        "protocol": "gauge401fix backbone rerun of the exact-matched pretrained-vs-random "
                    "control: same 20-event Bellinge catalog, same 15/5 split, same "
                    "n_sets=20 (100 test samples), same 150-epoch/5e-4 node-head fine-tuning, "
                    "decoder.node_head given an IDENTICAL random init in both arms before "
                    "fine-tuning; only the frozen backbone's origin (pretrained-on-gauge401fix "
                    "vs random) differs. All 8 matched seeds (42-700) used, backbones are "
                    "results/checkpoints/loop_{300,301-307} trained on the gauge-and-"
                    "timestamp-corrected SWMM ensemble.",
        "seed_to_loop": SEED_TO_LOOP,
        "seeds": seeds,
        "pretrained_per_seed": pretrained_results,
        "random_per_seed": random_results,
        "pretrained_nse_raw": pre_nse,
        "random_nse_raw": rand_nse,
        "paired_diffs": diffs,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "positive_delta_count": int(np.sum(np.asarray(diffs) > 0)),
        "n_seeds": len(seeds),
        "pretrained_mean": float(np.mean(pre_nse)),
        "pretrained_sd": float(np.std(pre_nse, ddof=1)) if len(seeds) > 1 else None,
        "random_mean": float(np.mean(rand_nse)),
        "random_sd": float(np.std(rand_nse, ddof=1)) if len(seeds) > 1 else None,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved -> {args.output}")
    logger.info(json.dumps({
        "pretrained_nse_raw": pre_nse, "random_nse_raw": rand_nse,
        "mean_diff": mean_diff, "sd_diff": sd_diff,
        "positive_delta_count": out["positive_delta_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
