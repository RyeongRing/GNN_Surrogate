"""
Matched ablation experiment: 92-event dataset, lambda_cont=0 (no continuity loss).
Same architecture as Loop 24 (hidden_dim=128, L=4, H=4), same seeds (42, 100, 200, 300).

Compares directly with seed_sensitivity.json (lambda_cont=0.1, same 4 seeds).
Runs Mann-Whitney U test on n=4 vs n=4 to test for NSE difference.

Usage (from project root):
    python src/run_matched_ablation.py --device auto
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SEEDS    = [42, 100, 200, 300]
RUN_IDS  = [0,  1,   2,   3]
HIDDEN   = 128   # must match Loop 24 / seed_sensitivity runs
CONFIG   = "config_seed_exp.yaml"
CP_BASE  = Path("results") / "checkpoints" / "ablation_92ev"


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


def compute_nse(model, loader, device) -> float:
    model.eval()
    all_nse = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            mean_d = pred["mean_depth"]
            y = batch.y.cpu().numpy().flatten()
            p = mean_d.cpu().numpy().flatten()
            ss_res = ((y - p) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            if ss_tot > 1e-6:
                all_nse.append(float(1.0 - ss_res / ss_tot))
    return float(np.mean(all_nse)) if all_nse else float("nan")


def build_val_loader(config, val_events, device):
    import pandas as pd
    from torch_geometric.loader import DataLoader
    from torch.utils.data import Subset
    from src.data.swmm_dataset import SWMMGraphDataset

    data_cfg = config["data"]
    exp_cfg  = config["experiment"]
    rain_series = load_rain_series(data_cfg)

    with open("data/event_catalog.json", encoding="utf-8") as f:
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

    val_set = set(val_events)
    idx = [i for i, (ev_id, _) in enumerate(dataset._index) if ev_id in val_set]
    loader = DataLoader(Subset(dataset, idx), batch_size=1)
    logger.info(f"Val loader: {len(idx)} samples")
    return loader


def load_model(cp_dir: Path, base_cfg: dict, M: int, device):
    from src.models.gnn_surrogate import EnsembleGNN
    model = EnsembleGNN(base_cfg, M=M).to(device)
    from src.checkpoints import load_members
    load_members(model, cp_dir, device)
    return model


def train_seed(run_id: int, seed: int, config: dict, device) -> bool:
    """Train M=5 EnsembleGNN with lambda_cont=0 (no continuity loss)."""
    import random
    import json as _json

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    exp_cfg  = config["experiment"]
    data_cfg = config["data"]

    n_heads      = exp_cfg.get("n_heads", 4)
    n_layers     = exp_cfg.get("n_layers", 4)
    lr           = exp_cfg.get("lr", 5e-4)
    weight_decay = exp_cfg.get("weight_decay", 1e-4)
    epochs       = exp_cfg.get("epochs", 200)
    patience     = exp_cfg.get("patience", 50)
    M            = exp_cfg.get("ensemble_size", 5)
    dropout      = exp_cfg.get("dropout", 0.0)

    # NOTE: lambda_cont = 0 — this is the ablation condition
    lambda_cont = 0.0

    base_cfg = {
        "node_feat_dim": 14, "edge_feat_dim": 4,
        "hidden_dim": HIDDEN,
        "T_out":  exp_cfg.get("T_out", 100),
        "T_rain": exp_cfg.get("T_rain", 72),
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "temporal_decoder":    exp_cfg.get("temporal_decoder", False),
        "scalar_rain_decoder": exp_cfg.get("scalar_rain_decoder", False),
    }

    from torch_geometric.loader import DataLoader
    from torch.utils.data import Subset
    from src.data.swmm_dataset import SWMMGraphDataset

    rain_series = load_rain_series(data_cfg)
    with open(data_cfg["catalog_path"], encoding="utf-8") as f:
        catalog_full = _json.load(f)

    dataset = SWMMGraphDataset(
        inp_path=data_cfg["inp_path"],
        ensemble_dir=data_cfg["ensemble_dir"],
        catalog=catalog_full,
        rain_series=rain_series,
        T_out=exp_cfg.get("T_out", 100),
        T_rain=exp_cfg.get("T_rain", 72),
        n_sets=data_cfg.get("n_sets", 50),
    )

    catalog_dict = {e["event_id"]: e for e in catalog_full}
    train_idx, val_idx = [], []
    for i, (ev_id, _) in enumerate(dataset._index):
        sp = catalog_dict.get(ev_id, {}).get("split", "")
        if sp in ("calib", "train_noobs"):
            train_idx.append(i)
        elif sp in ("val_modelsel", "val_conformal"):
            val_idx.append(i)
    val_idx = val_idx[: max(1, len(val_idx) // 2)]   # model-selection half only

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=1, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=1)
    logger.info(f"Run {run_id} seed={seed} lambda_cont=0: "
                f"train={len(train_idx)} val={len(val_idx)}")

    from src.models.gnn_surrogate import EnsembleGNN
    import torch.optim as optim

    model = EnsembleGNN(base_cfg, M=M).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 3)

    cp_dir = CP_BASE / f"run_{run_id:02d}"
    cp_dir.mkdir(parents=True, exist_ok=True)

    best_val    = float("inf")
    best_states = None
    no_improve  = 0
    t0 = time.time()

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
            logger.info(f"  Epoch {epoch:3d} train={train_loss:.4f} "
                        f"val={val_loss:.4f} best={best_val:.4f} "
                        f"patience={no_improve}/{patience}")

        if no_improve >= patience:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    if best_states:
        for mid, (member, state) in enumerate(zip(model.members, best_states)):
            member.load_state_dict(state)
        for mid, member in enumerate(model.members):
            ckpt = cp_dir / f"member_{mid:02d}.pt"
            torch.save({
                "model_state": member.state_dict(),
                "metrics": {"best_val_loss": best_val},
                "seed": seed, "run_id": run_id,
                "lambda_cont": 0.0,
            }, str(ckpt))

    elapsed = (time.time() - t0) / 60
    logger.info(f"Run {run_id} done in {elapsed:.1f} min (best_val={best_val:.5f})")
    return True


def main():
    raise RuntimeError("Legacy experiment only; use scripts/run_final_protocol.py for final results")
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--eval_only", action="store_true")
    args = p.parse_args()

    device = get_device(args.device)
    logger.info(f"Device: {device}")
    logger.info(f"Matched ablation: lambda_cont=0, 92-event dataset, seeds={SEEDS}")

    with open(CONFIG, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    with open("data/event_catalog.json", encoding="utf-8") as f:
        catalog = json.load(f)
    val_events = [e["event_id"] for e in catalog if e.get("split") in ("val_modelsel", "val_conformal")]
    logger.info(f"Val events ({len(val_events)}): {val_events}")

    if not args.eval_only:
        for run_id, seed in zip(RUN_IDS, SEEDS):
            cp_dir = CP_BASE / f"run_{run_id:02d}"
            if cp_dir.exists() and len(list(cp_dir.glob("member_*.pt"))) == 5:
                logger.info(f"Run {run_id} checkpoints exist, skipping training.")
                continue
            logger.info(f"\n{'='*60}\nTraining run_{run_id:02d} (seed={seed}, lambda_cont=0)\n{'='*60}")
            ok = train_seed(run_id, seed, config, device)
            if not ok:
                logger.error(f"Training failed for run {run_id}. Aborting.")
                sys.exit(1)

    logger.info("\n" + "="*60 + "\nEvaluating all ablation runs\n" + "="*60)

    # Build architecture config
    exp_cfg = config["experiment"]
    base_cfg = {
        "node_feat_dim": 14, "edge_feat_dim": 4,
        "hidden_dim": HIDDEN,
        "T_out": exp_cfg.get("T_out", 100),
        "T_rain": exp_cfg.get("T_rain", 72),
        "n_heads": exp_cfg.get("n_heads", 4),
        "n_layers": exp_cfg.get("n_layers", 4),
        "dropout": exp_cfg.get("dropout", 0.0),
        "temporal_decoder": exp_cfg.get("temporal_decoder", False),
        "scalar_rain_decoder": exp_cfg.get("scalar_rain_decoder", False),
    }
    M = exp_cfg.get("ensemble_size", 5)

    val_loader = build_val_loader(config, val_events, device)

    no_cont_nse = {}
    for run_id, seed in zip(RUN_IDS, SEEDS):
        cp_dir = CP_BASE / f"run_{run_id:02d}"
        model = load_model(cp_dir, base_cfg, M, device)
        nse = compute_nse(model, val_loader, device)
        no_cont_nse[seed] = nse
        logger.info(f"  run_{run_id:02d} (seed={seed}, no cont): NSE = {nse:.4f}")

    # Load with-cont results
    with open("results/seed_sensitivity.json") as f:
        with_cont = json.load(f)

    # Map seeds to NSE for with-cont
    seed_map = with_cont["seeds"]   # {"24": 42, "27": 100, "28": 200, "29": 300}
    with_cont_nse = {}
    for loop_str, seed_val in seed_map.items():
        with_cont_nse[int(seed_val)] = with_cont["nse_per_loop"][loop_str]

    # Mann-Whitney U test
    no_values   = [no_cont_nse[s]   for s in SEEDS]
    with_values = [with_cont_nse[s] for s in SEEDS]
    U, p_val = stats.mannwhitneyu(with_values, no_values, alternative="greater")
    d = (np.mean(with_values) - np.mean(no_values)) / (
        np.std(with_values + no_values, ddof=1) + 1e-9)

    logger.info("\n" + "="*60)
    logger.info("MATCHED ABLATION RESULTS (92-event dataset)")
    logger.info("="*60)
    logger.info(f"  WITH continuity loss (lambda=0.1):")
    for s, v in zip(SEEDS, with_values):
        logger.info(f"    seed={s}: NSE = {v:.4f}")
    logger.info(f"  Mean = {np.mean(with_values):.4f}, SD = {np.std(with_values, ddof=1):.4f}")
    logger.info(f"  WITHOUT continuity loss (lambda=0):")
    for s, v in zip(SEEDS, no_values):
        logger.info(f"    seed={s}: NSE = {v:.4f}")
    logger.info(f"  Mean = {np.mean(no_values):.4f}, SD = {np.std(no_values, ddof=1):.4f}")
    logger.info(f"  ΔNSE (with - no) = {np.mean(with_values)-np.mean(no_values):.4f}")
    logger.info(f"  Mann-Whitney U={U:.0f}, p={p_val:.4f} (one-sided, H1: with > no)")
    logger.info(f"  Cohen's d = {d:.3f}")
    logger.info("="*60)

    result = {
        "no_cont_nse": no_cont_nse,
        "with_cont_nse": with_cont_nse,
        "seeds": SEEDS,
        "no_cont_mean": float(np.mean(no_values)),
        "no_cont_std":  float(np.std(no_values, ddof=1)),
        "with_cont_mean": float(np.mean(with_values)),
        "with_cont_std":  float(np.std(with_values, ddof=1)),
        "delta_nse": float(np.mean(with_values) - np.mean(no_values)),
        "mannwhitney_U": float(U),
        "mannwhitney_p": float(p_val),
        "cohens_d": float(d),
        "n_per_group": len(SEEDS),
        "note": "92-event matched ablation: same arch (H=128,L=4,H=4), same seeds, same dataset",
    }

    out_path = Path("results") / "ablation_matched_92ev.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
