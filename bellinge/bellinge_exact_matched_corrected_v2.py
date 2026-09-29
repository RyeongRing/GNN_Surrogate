"""Bellinge exact-matched pretrained-vs-random backbone control, rerun against
the timestamp-corrected v2 Seocho-gu backbone (see CODEX_HANDOFF_20260805.md).

Identical protocol to bellinge_exact_matched_v2.py (same 20-event catalog,
same 15/5 split, same n_sets=20, same 150-epoch/5e-4 decoder fine-tuning,
same matched-decoder-init fix), except:

1. Backbone checkpoints are the corrected-v2 loops (124, 127-133) trained on
   the timestamp-corrected SWMM ensemble, not the legacy loops (24, 27-33).
2. All 8 matched seeds are used (42,100,200,300,400,500,600,700), matching
   the corrected-v2 primary replication and confirmatory ablations, instead
   of the legacy 5-seed subset.
3. Default config is config_timecorrected_v2.yaml (architecture params are
   identical to config_seed_exp.yaml -- hidden_dim=128, n_heads=4, n_layers=4,
   lambda_cont=0.1, ensemble_size=5 -- so checkpoints load without changes).

Bellinge's own SWMM ensemble/rainfall data (DTU, Denmark) is unaffected by
the KMA end-of-interval timestamp bug, which was specific to the Seocho-gu
(Seoul) rainfall preprocessing -- only the backbone being fine-tuned changes.
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
    BELLINGE_INP, BELLINGE_DAT, BELLINGE_ENS, BELLINGE_CAT,
    parse_dat_file, fine_tune_bellinge_decoder, evaluate_bellinge_gnn,
    load_loop_model,
)
from bellinge.bellinge_scratch_baseline import random_backbone_model

SEED_TO_LOOP = {
    42: 124, 100: 127, 200: 128, 300: 129,
    400: 130, 500: 131, 600: 132, 700: 133,
}


def build_dataset(events, config):
    from src.data.swmm_dataset import SWMMGraphDataset
    from src.data.protocol import validate_completed_runs
    if len(events) != 20 or len({e["event_id"] for e in events}) != 20:
        raise ValueError("The final Bellinge protocol requires 20 distinct locked events")
    exp = config.get("experiment", {})
    rain_1min = parse_dat_file(BELLINGE_DAT)
    rain_hourly = rain_1min.resample("1h").sum().fillna(0)
    dataset = SWMMGraphDataset(
        inp_path=str(BELLINGE_INP), ensemble_dir=str(BELLINGE_ENS),
        catalog=events, rain_series=rain_hourly,
        T_out=exp.get("T_out", 100), T_rain=exp.get("T_rain", 72),
        n_sets=20, site_id="bellinge",
    )
    validate_completed_runs(dataset._index, events, 20)
    # inf_max/inf_min/inf_decay (feat[-3:]) are Green-Ampt infiltration
    # params that do not affect Bellinge's nopervious SWMM output, but the
    # stored param_set uses a stale pre-Green-Ampt-migration scale
    # (2.5-5.0 vs the Seocho training range 30-250 for inf_max); fix to the
    # Seocho 50-set training mean rather than feeding an out-of-distribution
    # conditioning signal.
    _INF_MEAN = [5.0033, 140.14, 0.5148, 0.2750]  # dstore_perv, inf_max, inf_min, inf_decay
    for sample in dataset:
        sample.x[:, 1] = 0.0
        for _j, _v in enumerate(_INF_MEAN):
            sample.x[:, -4 + _j] = _v
    return dataset


def copy_matched_decoder_init(pretrained_model, random_model):
    """Overwrite pretrained_model's decoder.node_head with random_model's
    (already seed-initialised) decoder.node_head, member-for-member, so both
    arms start fine-tuning from an identical decoder init."""
    n = len(pretrained_model.members)
    assert n == len(random_model.members)
    for i in range(n):
        src_sd = random_model.members[i].decoder.node_head.state_dict()
        pretrained_model.members[i].decoder.node_head.load_state_dict(copy.deepcopy(src_sd))
    return pretrained_model


def main():
    import argparse
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_timecorrected_v2.yaml")
    p.add_argument("--seeds", default="42,100,200,300,400,500,600,700")
    p.add_argument("--ft_epochs", type=int, default=150)
    p.add_argument("--ft_lr", type=float, default=5e-4)
    p.add_argument("--n_test_events", type=int, default=5)
    p.add_argument("--output", default="results/timecorrected_v2/bellinge_exact_matched_corrected_v2.json")
    args = p.parse_args()

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
        logger.info(f"=== seed {seed} (corrected-v2 Loop {loop_id}) ===")

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

    from scipy import stats

    pre_nse = [pretrained_results[s]["NSE_bellinge_raw"] for s in seeds]
    rand_nse = [random_results[s]["NSE_bellinge_raw"] for s in seeds]
    diffs = [pv - rv for pv, rv in zip(pre_nse, rand_nse)]

    t_stat, t_p = stats.ttest_rel(pre_nse, rand_nse)
    try:
        wres = stats.wilcoxon(pre_nse, rand_nse, alternative="two-sided", mode="exact")
        wilcoxon_stat, wilcoxon_p = float(wres.statistic), float(wres.pvalue)
        wilcoxon_method = "exact"
    except Exception as e:
        wres = stats.wilcoxon(pre_nse, rand_nse, alternative="two-sided", mode="approx")
        wilcoxon_stat, wilcoxon_p = float(wres.statistic), float(wres.pvalue)
        wilcoxon_method = f"approx (exact failed: {e})"

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    cohens_dz = mean_diff / sd_diff if sd_diff > 0 else None

    out = {
        "protocol": "corrected-v2 backbone rerun of the exact-matched pretrained-vs-random "
                    "control: same 20-event Bellinge catalog, same 15/5 split, same "
                    "n_sets=20 (100 test samples), same 150-epoch/5e-4 decoder fine-tuning, "
                    "decoder.node_head given an IDENTICAL random init in both arms before "
                    "fine-tuning; only the frozen backbone's origin (pretrained-on-corrected-v2 "
                    "vs random) differs. All 8 matched seeds (42-700) used, backbones are "
                    "results/checkpoints/loop_{124,127-133} trained on the timestamp-corrected "
                    "(-60min KMA end-of-interval fix) SWMM ensemble.",
        "seed_to_loop": SEED_TO_LOOP,
        "seeds": seeds,
        "pretrained_per_seed": pretrained_results,
        "random_per_seed": random_results,
        "pretrained_nse_raw": pre_nse,
        "random_nse_raw": rand_nse,
        "paired_diffs": diffs,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "cohens_dz": cohens_dz,
        "paired_ttest": {"t": float(t_stat), "p": float(t_p), "n": len(seeds)},
        "wilcoxon_signed_rank": {"statistic": wilcoxon_stat, "p": wilcoxon_p,
                                  "method": wilcoxon_method, "n": len(seeds)},
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved -> {args.output}")
    logger.info(json.dumps({
        "pretrained_nse_raw": pre_nse, "random_nse_raw": rand_nse,
        "paired_ttest": out["paired_ttest"],
        "wilcoxon_signed_rank": out["wilcoxon_signed_rank"],
    }, indent=2))


if __name__ == "__main__":
    main()
