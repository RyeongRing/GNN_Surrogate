"""Primary-seed Bellinge zero-shot + decoder-fine-tuned headline numbers,
rerun against the gauge-and-timestamp-corrected Seocho-gu backbone
(Loop 300, seed 42), replacing bellinge_primary_corrected_v2.py's
timestamp-only-corrected backbone (Loop 124).

Identical protocol otherwise: same 20-event Bellinge catalog, same 15/5
split, same 150-epoch/5e-4 node-head fine-tuning (loads the existing catalog
so results are directly comparable to the timestamp-only-corrected run).

Bellinge's own SWMM ensemble/rainfall data (DTU, Denmark) is unaffected by
either the KMA timestamp bug or the AWS 400/401 gauge-mixup bug, both of
which were specific to Seocho-gu (Seoul) rainfall preprocessing -- only the
backbone being fine-tuned changes.
"""
import json
import logging
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from bellinge.bellinge_validate_loop24 import (
    BELLINGE_CAT,
    fine_tune_bellinge_decoder, evaluate_bellinge_gnn,
    load_loop_model,
)
from bellinge.bellinge_exact_matched_corrected_v2 import build_dataset


def main():
    import argparse
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config_gauge401_fix_v1.yaml")
    p.add_argument("--loop_id", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ft_epochs", type=int, default=150)
    p.add_argument("--ft_lr", type=float, default=5e-4)
    p.add_argument("--n_test_events", type=int, default=5)
    p.add_argument("--output", default="results/timecorrected_v2/bellinge_primary_gauge401fix.json")
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

    t0 = time.time()
    model, M = load_loop_model(Path(f"results/checkpoints/loop_{args.loop_id}"), config, device)
    logger.info(f"gauge401fix Loop {args.loop_id} loaded (M={M})")

    zero_shot = evaluate_bellinge_gnn(model, test_data, config, device,
                                       calib_loader=None,
                                       n_cal=min(5, len(test_data) // 4),
                                       seed=args.seed)
    logger.info(f"[zero-shot] NSE_raw={zero_shot.get('NSE_bellinge_raw')}")

    fine_tune_bellinge_decoder(model, ft_data, n_epochs=args.ft_epochs,
                                lr=args.ft_lr, seed=args.seed)
    fine_tuned = evaluate_bellinge_gnn(model, test_data, config, device,
                                        calib_loader=None,
                                        n_cal=min(5, len(test_data) // 4),
                                        seed=args.seed)
    logger.info(f"[fine-tuned] NSE_raw={fine_tuned.get('NSE_bellinge_raw')}")

    out = {
        "protocol": "gauge-and-timestamp-corrected primary-seed Bellinge headline: "
                    "zero-shot then node-head-only fine-tuning on the gauge401fix "
                    "Loop 300 backbone (AWS 401/Seocho explicit gauge selection on "
                    "top of the -60min KMA end-of-interval fix), same 20-event "
                    "catalog / 15-5 event split as bellinge_exact_matched_gauge401fix.py.",
        "loop_id": args.loop_id,
        "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 1),
        "zero_shot": zero_shot,
        "fine_tuned": fine_tuned,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
