"""
Extended matched ablation: n=4 → n=8 per group.

Trains BOTH arms (lambda_cont=0.1 and lambda_cont=0) for four additional seeds
(400, 500, 600, 700) on the 92-event dataset with the same architecture as
Loop 24 (hidden_dim=128, L=4, H=4, M=5).

After training, combines the new seeds with existing results:
  - with-cont: seed_sensitivity.json (seeds 42, 100, 200, 300)
  - no-cont:   ablation_matched_92ev.json (seeds 42, 100, 200, 300)

Outputs:
  results/ablation_matched_92ev_n8.json   -- combined n=8 per group

Usage (from project root):
    python src/run_extended_ablation.py --device auto
    python src/run_extended_ablation.py --device auto --eval_only
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import stats
from src.evaluate import compute_continuity_violation

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ── New seeds only — existing 4 seeds already trained ─────────────────────────
NEW_SEEDS   = [400, 500, 600, 700]
NEW_RUN_IDS = [4,   5,   6,   7]
HIDDEN      = 128
CONFIG      = "config_seed_exp.yaml"

# Checkpoint directories
CP_BASE_NOCONT   = Path("results") / "checkpoints" / "ablation_92ev"       # no-cont runs 0-3 exist here
CP_BASE_WITHCONT = Path("results") / "checkpoints" / "ablation_92ev_wcont" # new with-cont runs go here


def get_device(s: str):
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


def load_rain_series(data_cfg: dict):
    import pandas as pd
    rain_dir = Path(data_cfg["rain_dir"])
    frames = []
    for f in sorted(rain_dir.glob("AWS_4*.csv")):
        df = pd.read_csv(f, encoding="cp949", header=0,
                         names=["stn_id", "stn_name", "datetime", "rain_mm"])
        df["rain_mm"] = pd.to_numeric(df["rain_mm"], errors="coerce").fillna(0).clip(lower=0)
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])
        frames.append(df[["datetime", "rain_mm"]])
    combined = pd.concat(frames).drop_duplicates("datetime").sort_values("datetime")
    return combined.set_index("datetime")["rain_mm"].resample("1h").sum().fillna(0)


def build_loaders(config, seed):
    import random
    from torch_geometric.loader import DataLoader
    from torch.utils.data import Subset
    from src.data.swmm_dataset import SWMMGraphDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_cfg = config["data"]
    exp_cfg  = config["experiment"]
    rain_series = load_rain_series(data_cfg)

    with open(data_cfg["catalog_path"], encoding="utf-8") as f:
        catalog = json.load(f)

    dataset = SWMMGraphDataset(
        inp_path=data_cfg["inp_path"],
        ensemble_dir=data_cfg["ensemble_dir"],
        catalog=catalog,
        rain_series=rain_series,
        T_out=exp_cfg.get("T_out", 100),
        T_rain=exp_cfg.get("T_rain", 72),
        n_sets=data_cfg.get("n_sets", 50),
    )

    catalog_dict = {e["event_id"]: e for e in catalog}
    train_idx, val_idx = [], []
    for i, (ev_id, _) in enumerate(dataset._index):
        sp = catalog_dict.get(ev_id, {}).get("split", "")
        if sp in ("calib", "train_noobs"):
            train_idx.append(i)
        elif sp in ("val_modelsel", "val_conformal"):
            val_idx.append(i)
    val_idx_sel = val_idx[: max(1, len(val_idx) // 2)]

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx_sel), batch_size=1)
    test_loader  = DataLoader(Subset(dataset, val_idx),  batch_size=1)  # full val for NSE eval
    return train_loader, val_loader, test_loader


def make_base_cfg(config):
    exp_cfg = config["experiment"]
    return {
        "node_feat_dim": 14, "edge_feat_dim": 4,
        "hidden_dim": HIDDEN,
        "T_out":  exp_cfg.get("T_out", 100),
        "T_rain": exp_cfg.get("T_rain", 72),
        "n_heads": exp_cfg.get("n_heads", 4),
        "n_layers": exp_cfg.get("n_layers", 4),
        "dropout": exp_cfg.get("dropout", 0.0),
        "temporal_decoder":    exp_cfg.get("temporal_decoder", False),
        "scalar_rain_decoder": exp_cfg.get("scalar_rain_decoder", False),
    }


def train_one(run_id, seed, lambda_cont, cp_dir, config, device):
    from src.models.gnn_surrogate import EnsembleGNN
    import torch.optim as optim

    exp_cfg = config["experiment"]
    M        = exp_cfg.get("ensemble_size", 5)
    lr       = exp_cfg.get("lr", 5e-4)
    wd       = exp_cfg.get("weight_decay", 1e-4)
    epochs   = exp_cfg.get("epochs", 200)
    patience = exp_cfg.get("patience", 50)

    base_cfg = make_base_cfg(config)
    train_loader, val_loader, _ = build_loaders(config, seed)

    model     = EnsembleGNN(base_cfg, M=M).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 3)

    cp_dir.mkdir(parents=True, exist_ok=True)

    best_val    = float("inf")
    best_states = None
    no_improve  = 0
    t0 = time.time()
    label = f"run_{run_id:02d} seed={seed} lam={lambda_cont}"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.compute_total_loss(batch, lambda_cont=lambda_cont,
                                            use_nse_loss=False)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred = model(batch)
                val_loss += F.mse_loss(pred["mean_depth"], batch.y).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val    = val_loss
            best_states = [{k: v.clone() for k, v in m.state_dict().items()}
                           for m in model.members]
            no_improve  = 0
        else:
            no_improve += 1

        if epoch % 20 == 0:
            logger.info(f"  [{label}] Epoch {epoch:3d} train={train_loss:.4f} "
                        f"val={val_loss:.4f} best={best_val:.4f} "
                        f"patience={no_improve}/{patience}")

        if no_improve >= patience:
            logger.info(f"  [{label}] Early stopping at epoch {epoch}")
            break

    if best_states:
        for mid, (member, state) in enumerate(zip(model.members, best_states)):
            member.load_state_dict(state)
        for mid, member in enumerate(model.members):
            torch.save({
                "model_state": member.state_dict(),
                "metrics": {"best_val_loss": best_val},
                "seed": seed, "run_id": run_id,
                "lambda_cont": lambda_cont,
            }, str(cp_dir / f"member_{mid:02d}.pt"))

    elapsed = (time.time() - t0) / 60
    logger.info(f"  [{label}] Done in {elapsed:.1f} min (best_val={best_val:.5f})")
    return True


def compute_nse(model, loader, device) -> float:
    nse, _ = compute_nse_and_violation(model, loader, device)
    return nse


def compute_nse_and_violation(model, loader, device):
    """Returns (nse, continuity_violation_pct) over all batches in loader."""
    model.eval()
    all_nse, all_viol = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            mean_d = pred["mean_depth"]
            mean_f = pred["mean_flow"]
            y = batch.y.cpu().numpy().flatten()
            p = mean_d.cpu().numpy().flatten()
            ss_res = ((y - p) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            if ss_tot > 1e-6:
                all_nse.append(float(1.0 - ss_res / ss_tot))
            # continuity violation per batch (shape [N,T] / [E,T])
            d_np = mean_d.cpu().numpy()
            f_np = mean_f.cpu().numpy()
            ei = batch.edge_index.cpu().numpy()
            all_viol.append(compute_continuity_violation(d_np, f_np, ei))
    nse = float(np.mean(all_nse)) if all_nse else float("nan")
    viol = float(np.mean(all_viol)) if all_viol else float("nan")
    return nse, viol


def load_model(cp_dir, base_cfg, M, device):
    from src.models.gnn_surrogate import EnsembleGNN
    model = EnsembleGNN(base_cfg, M=M).to(device)
    from src.checkpoints import load_members
    load_members(model, cp_dir, device)
    return model


def main():
    raise RuntimeError("Legacy experiment only; use scripts/run_final_protocol.py for final results")
    p = argparse.ArgumentParser()
    p.add_argument("--device",    default="auto")
    p.add_argument("--eval_only", action="store_true",
                   help="Skip training; evaluate existing checkpoints only")
    args = p.parse_args()

    device = get_device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Extended ablation: new seeds={NEW_SEEDS}")

    with open(CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    exp_cfg  = config["experiment"]
    base_cfg = make_base_cfg(config)
    M        = exp_cfg.get("ensemble_size", 5)

    # ── Training phase ───────────────────────────────────────────────────────
    if not args.eval_only:
        for run_id, seed in zip(NEW_RUN_IDS, NEW_SEEDS):
            # --- with-cont (lambda=0.1) ---
            wcp = CP_BASE_WITHCONT / f"run_{run_id:02d}"
            if wcp.exists() and len(list(wcp.glob("member_*.pt"))) == M:
                logger.info(f"With-cont run_{run_id:02d} exists, skipping.")
            else:
                logger.info(f"\n{'='*60}\n"
                            f"Training WITH-CONT run_{run_id:02d} (seed={seed}, λ=0.1)\n"
                            f"{'='*60}")
                train_one(run_id, seed, lambda_cont=0.1, cp_dir=wcp,
                          config=config, device=device)

            # --- no-cont (lambda=0.0) ---
            ncp = CP_BASE_NOCONT / f"run_{run_id:02d}"
            if ncp.exists() and len(list(ncp.glob("member_*.pt"))) == M:
                logger.info(f"No-cont run_{run_id:02d} exists, skipping.")
            else:
                logger.info(f"\n{'='*60}\n"
                            f"Training NO-CONT run_{run_id:02d} (seed={seed}, λ=0.0)\n"
                            f"{'='*60}")
                train_one(run_id, seed, lambda_cont=0.0, cp_dir=ncp,
                          config=config, device=device)

    # ── Evaluation phase ─────────────────────────────────────────────────────
    logger.info("\n" + "="*60 + "\nEvaluating extended ablation\n" + "="*60)

    with open(config["data"]["catalog_path"], encoding="utf-8") as f:
        catalog = json.load(f)

    from torch_geometric.loader import DataLoader
    from torch.utils.data import Subset
    from src.data.swmm_dataset import SWMMGraphDataset
    rain_series = load_rain_series(config["data"])
    dataset = SWMMGraphDataset(
        inp_path=config["data"]["inp_path"],
        ensemble_dir=config["data"]["ensemble_dir"],
        catalog=catalog,
        rain_series=rain_series,
        T_out=exp_cfg.get("T_out", 100),
        T_rain=exp_cfg.get("T_rain", 72),
        n_sets=config["data"].get("n_sets", 50),
    )
    catalog_dict = {e["event_id"]: e for e in catalog}

    val_ids     = {e["event_id"] for e in catalog if e.get("split") in ("val_modelsel", "val_conformal")}
    holdout_ids = {e["event_id"] for e in catalog if e.get("split") == "test_temporal"}
    val_idx     = [i for i, (ev_id, _) in enumerate(dataset._index) if ev_id in val_ids]
    holdout_idx = [i for i, (ev_id, _) in enumerate(dataset._index) if ev_id in holdout_ids]

    val_loader     = DataLoader(Subset(dataset, val_idx),     batch_size=1)
    holdout_loader = DataLoader(Subset(dataset, holdout_idx), batch_size=1)
    logger.info(f"Val loader: {len(val_idx)} samples | Holdout loader: {len(holdout_idx)} samples")

    # Checkpoint paths for old seeds (42,100,200,300)
    OLD_SEED_LOOP = {42: "loop_24", 100: "loop_27", 200: "loop_28", 300: "loop_29"}
    OLD_SEED_NOCONT_RUN = {42: "run_00", 100: "run_01", 200: "run_02", 300: "run_03"}
    CP_BASE_LOOPS = Path("results") / "checkpoints"

    # --- Evaluate NEW with-cont seeds (400,500,600,700) ---
    new_withcont_nse, new_withcont_viol = {}, {}
    for run_id, seed in zip(NEW_RUN_IDS, NEW_SEEDS):
        wcp = CP_BASE_WITHCONT / f"run_{run_id:02d}"
        model = load_model(wcp, base_cfg, M, device)
        nse, viol = compute_nse_and_violation(model, val_loader, device)
        new_withcont_nse[seed] = nse
        new_withcont_viol[seed] = viol
        logger.info(f"  With-cont run_{run_id:02d} seed={seed}: NSE={nse:.4f} viol={viol:.2f}%")

    # --- Evaluate NEW no-cont seeds (400,500,600,700) ---
    new_nocont_nse, new_nocont_viol = {}, {}
    for run_id, seed in zip(NEW_RUN_IDS, NEW_SEEDS):
        ncp = CP_BASE_NOCONT / f"run_{run_id:02d}"
        model = load_model(ncp, base_cfg, M, device)
        nse, viol = compute_nse_and_violation(model, val_loader, device)
        new_nocont_nse[seed] = nse
        new_nocont_viol[seed] = viol
        logger.info(f"  No-cont  run_{run_id:02d} seed={seed}: NSE={nse:.4f} viol={viol:.2f}%")

    # --- Load NSE for OLD seeds from existing JSONs ---
    with open("results/seed_sensitivity.json") as f:
        ss = json.load(f)
    old_withcont_nse = {int(v): ss["nse_per_loop"][k] for k, v in ss["seeds"].items()}

    with open("results/ablation_matched_92ev.json") as f:
        am = json.load(f)
    old_nocont_nse = {int(k): v for k, v in am["no_cont_nse"].items()}

    # --- Compute violation rates for OLD seeds (requires loading checkpoints) ---
    old_withcont_viol, old_nocont_viol = {}, {}
    for seed, loop_name in OLD_SEED_LOOP.items():
        cp = CP_BASE_LOOPS / loop_name
        if cp.exists():
            model = load_model(cp, base_cfg, M, device)
            _, viol = compute_nse_and_violation(model, val_loader, device)
            old_withcont_viol[seed] = viol
            logger.info(f"  With-cont {loop_name} seed={seed}: viol={viol:.2f}%")
    for seed, run_name in OLD_SEED_NOCONT_RUN.items():
        cp = CP_BASE_NOCONT / run_name
        if cp.exists():
            model = load_model(cp, base_cfg, M, device)
            _, viol = compute_nse_and_violation(model, val_loader, device)
            old_nocont_viol[seed] = viol
            logger.info(f"  No-cont  {run_name} seed={seed}: viol={viol:.2f}%")

    # --- Combine n=4 + n=4 = n=8 ---
    all_withcont_nse = {**old_withcont_nse, **new_withcont_nse}
    all_nocont_nse   = {**old_nocont_nse,   **new_nocont_nse}
    all_withcont_viol = {**old_withcont_viol, **new_withcont_viol}
    all_nocont_viol   = {**old_nocont_viol,   **new_nocont_viol}

    all_seeds = sorted(all_withcont_nse.keys())
    pairs = [(all_withcont_nse[s], all_nocont_nse[s])
             for s in all_seeds if s in all_withcont_nse and s in all_nocont_nse]
    with_values = [p[0] for p in pairs]
    no_values   = [p[1] for p in pairs]
    n_per_group = len(with_values)

    # Paired ΔNSE
    delta_nse  = [w - n for w, n in zip(with_values, no_values)]
    delta_arr  = np.array(delta_nse)
    delta_mean = float(np.mean(delta_arr))
    delta_std  = float(np.std(delta_arr, ddof=1))
    delta_med  = float(np.median(delta_arr))

    # Two-sided exact paired Wilcoxon signed-rank test
    wilcoxon_result = stats.wilcoxon(delta_arr, alternative="two-sided", method="exact")
    wilcoxon_stat   = float(wilcoxon_result.statistic)
    wilcoxon_p      = float(wilcoxon_result.pvalue)

    # Rank-biserial correlation r_rb = (T+ - T-) / (n*(n+1)/2)
    nonzero = delta_arr[delta_arr != 0]
    n_nz = len(nonzero)
    if n_nz > 0:
        ranks_nz = stats.rankdata(np.abs(nonzero))
        T_plus  = float(np.sum(ranks_nz[nonzero > 0]))
        T_minus = float(np.sum(ranks_nz[nonzero < 0]))
        r_rb = (T_plus - T_minus) / (n_nz * (n_nz + 1) / 2.0)
    else:
        T_plus, T_minus, r_rb = 0.0, 0.0, 0.0

    # Paired Cohen's d_z
    d_z = delta_mean / (delta_std + 1e-9)

    min_with   = min(with_values)
    max_no     = max(no_values)
    perfect_sep = min_with > max_no

    logger.info("\n" + "="*60)
    logger.info(f"EXTENDED MATCHED ABLATION — PAIRED ANALYSIS (n={n_per_group} pairs)")
    logger.info("="*60)
    logger.info(f"  Seed | With-cont NSE | No-cont NSE | ΔNSE | With-viol% | No-viol%")
    for s in all_seeds:
        wn = all_withcont_nse.get(s, float("nan"))
        nn = all_nocont_nse.get(s, float("nan"))
        dn = wn - nn
        wv = all_withcont_viol.get(s, float("nan"))
        nv = all_nocont_viol.get(s, float("nan"))
        logger.info(f"   {s:>4} | {wn:.4f}        | {nn:.4f}      | {dn:+.4f} | {wv:>9.2f}% | {nv:.2f}%")
    logger.info(f"  ΔNSE: mean={delta_mean:+.4f}  SD={delta_std:.4f}  median={delta_med:+.4f}")
    logger.info(f"  Paired Wilcoxon (two-sided, exact): W={wilcoxon_stat:.1f}, p={wilcoxon_p:.4f}")
    logger.info(f"  Rank-biserial r_rb={r_rb:.3f}  (T+={T_plus:.0f}, T-={T_minus:.0f})")
    logger.info(f"  Paired Cohen's d_z={d_z:.3f}")
    logger.info(f"  Perfect rank separation: {perfect_sep} "
                f"(min_with={min_with:.4f} vs max_no={max_no:.4f})")
    logger.info("="*60)

    # ── Holdout evaluation (unseen-event surrogate fidelity) ─────────────────
    holdout_results = {}
    if holdout_idx:
        logger.info(f"\nHoldout evaluation ({len(holdout_ids)} events, SWMM as ground truth)")
        # Old with-cont seeds
        for seed, loop_name in OLD_SEED_LOOP.items():
            cp = CP_BASE_LOOPS / loop_name
            if cp.exists():
                model = load_model(cp, base_cfg, M, device)
                nse_h, viol_h = compute_nse_and_violation(model, holdout_loader, device)
                holdout_results[f"with_cont_{seed}"] = {"nse": nse_h, "viol_pct": viol_h}
                logger.info(f"  Holdout with-cont seed={seed}: NSE={nse_h:.4f} viol={viol_h:.2f}%")
        # Old no-cont seeds
        for seed, run_name in OLD_SEED_NOCONT_RUN.items():
            cp = CP_BASE_NOCONT / run_name
            if cp.exists():
                model = load_model(cp, base_cfg, M, device)
                nse_h, viol_h = compute_nse_and_violation(model, holdout_loader, device)
                holdout_results[f"no_cont_{seed}"] = {"nse": nse_h, "viol_pct": viol_h}
                logger.info(f"  Holdout no-cont  seed={seed}: NSE={nse_h:.4f} viol={viol_h:.2f}%")
        # New seeds
        for run_id, seed in zip(NEW_RUN_IDS, NEW_SEEDS):
            wcp = CP_BASE_WITHCONT / f"run_{run_id:02d}"
            model = load_model(wcp, base_cfg, M, device)
            nse_h, viol_h = compute_nse_and_violation(model, holdout_loader, device)
            holdout_results[f"with_cont_{seed}"] = {"nse": nse_h, "viol_pct": viol_h}
            logger.info(f"  Holdout with-cont seed={seed}: NSE={nse_h:.4f} viol={viol_h:.2f}%")
            ncp = CP_BASE_NOCONT / f"run_{run_id:02d}"
            model = load_model(ncp, base_cfg, M, device)
            nse_h, viol_h = compute_nse_and_violation(model, holdout_loader, device)
            holdout_results[f"no_cont_{seed}"] = {"nse": nse_h, "viol_pct": viol_h}
            logger.info(f"  Holdout no-cont  seed={seed}: NSE={nse_h:.4f} viol={viol_h:.2f}%")

        # Paired holdout stats
        h_with = [holdout_results[f"with_cont_{s}"]["nse"]
                  for s in all_seeds if f"with_cont_{s}" in holdout_results]
        h_no   = [holdout_results[f"no_cont_{s}"]["nse"]
                  for s in all_seeds if f"no_cont_{s}"   in holdout_results]
        if len(h_with) >= 2 and len(h_with) == len(h_no):
            h_delta = [w - n for w, n in zip(h_with, h_no)]
            h_res = stats.wilcoxon(h_delta, alternative="two-sided", method="exact")
            logger.info(f"\n  Holdout ΔNSE mean={np.mean(h_delta):+.4f} SD={np.std(h_delta, ddof=1):.4f}")
            logger.info(f"  Holdout paired Wilcoxon: W={h_res.statistic:.1f}, p={h_res.pvalue:.4f}")

    result = {
        "seeds": all_seeds,
        "n_per_group": n_per_group,
        "with_cont_nse":  {str(s): all_withcont_nse.get(s) for s in all_seeds},
        "no_cont_nse":    {str(s): all_nocont_nse.get(s)   for s in all_seeds},
        "with_cont_violation_pct": {str(s): all_withcont_viol.get(s) for s in all_seeds},
        "no_cont_violation_pct":   {str(s): all_nocont_viol.get(s)   for s in all_seeds},
        "with_cont_mean": float(np.mean(with_values)),
        "with_cont_std":  float(np.std(with_values, ddof=1)),
        "no_cont_mean": float(np.mean(no_values)),
        "no_cont_std":  float(np.std(no_values, ddof=1)),
        "delta_nse_per_seed": {str(s): float(w - n)
                               for s, w, n in zip(all_seeds, with_values, no_values)},
        "delta_nse_mean":   delta_mean,
        "delta_nse_std":    delta_std,
        "delta_nse_median": delta_med,
        "wilcoxon_W":       wilcoxon_stat,
        "wilcoxon_p":       wilcoxon_p,
        "wilcoxon_method":  "two-sided exact paired signed-rank",
        "rank_biserial_r":  float(r_rb),
        "T_plus":  float(T_plus),
        "T_minus": float(T_minus),
        "cohens_d_z": float(d_z),
        "perfect_rank_separation": bool(perfect_sep),
        "min_with_cont": float(min_with),
        "max_no_cont":   float(max_no),
        "holdout_per_seed": holdout_results,
        "note": (
            f"Extended 92-event matched ablation: n={n_per_group} paired seeds. "
            f"Statistic: two-sided exact paired Wilcoxon signed-rank (unit=seed pair). "
            f"Holdout=unseen 2024 events evaluated against SWMM output (surrogate fidelity)."
        ),
    }

    out_path = Path("results") / "ablation_matched_92ev_n8.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
