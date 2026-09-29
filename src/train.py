"""
학습 루프.
EnsembleGNN 학습 + ConformalPredictor 캘리브레이션.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

# ensure project root is on sys.path when running as `python src/train.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("PyTorch not installed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train physics-informed GNN surrogate")
    parser.add_argument("--config", type=str, default="configs/config_gauge401_fix_v1.yaml")
    parser.add_argument("--regime", type=str, default="B",
                        choices=["A", "B", "C"])
    parser.add_argument("--loop_id", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly allow replacing checkpoints for this loop")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> "torch.device":
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def build_dataloaders(config: Dict, regime_id: str):
    """SWMM 앙상블 결과 → PyG DataLoader 구성."""
    import json
    import pandas as pd
    from torch_geometric.loader import DataLoader
    from src.data.swmm_dataset import SWMMGraphDataset
    from src.data.protocol import validate_catalog, validate_completed_runs

    data_cfg = config.get("data", {})
    inp_path     = data_cfg.get("inp_path")
    ensemble_dir = data_cfg.get("ensemble_dir")
    catalog_path = data_cfg.get("catalog_path")
    rain_dir     = data_cfg.get("rain_dir")

    if not inp_path or not ensemble_dir:
        raise ValueError("data.inp_path and data.ensemble_dir must be set")

    # 강수 시계열 로드 — 명시적으로 AWS 401(서초)만 선택 (station 400/Gangnam과의
    # 타임스탬프 중복을 drop_duplicates() 행 순서에 맡기던 버그 수정,
    # timestamp_and_gauge_input_audit_20260824.md 참고)
    from src.data.rainfall import load_hourly_rainfall
    rain_series = load_hourly_rainfall(rain_dir, timestamp_shift_minutes=0)

    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)
    validate_catalog(catalog, data_cfg.get("expected_split_counts"))

    exp_cfg  = config.get("experiment", {})
    data_cfg = config.get("data", {})
    T_out  = exp_cfg.get("T_out",  100)
    T_rain = exp_cfg.get("T_rain",  72)
    n_sets = data_cfg.get("n_sets", exp_cfg.get("n_sets", 50))
    batch  = exp_cfg.get("batch_size", 1)
    if batch != 1:
        raise ValueError("The released final protocol requires batch_size=1")

    dataset_cls = SWMMGraphDataset
    adapter_kwargs = {}
    if "report_lead_steps" in data_cfg:
        from src.data.swmm_dataset_timecorrected import TimeCorrectedSWMMGraphDataset
        dataset_cls = TimeCorrectedSWMMGraphDataset
        adapter_kwargs["report_lead_steps"] = data_cfg["report_lead_steps"]

    dataset = dataset_cls(
        inp_path=inp_path,
        ensemble_dir=ensemble_dir,
        catalog=catalog,
        rain_series=rain_series,
        T_out=T_out,
        T_rain=T_rain,
        n_sets=n_sets,
        **adapter_kwargs,
    )

    validate_completed_runs(dataset._index, catalog, n_sets)

    # event-level split (year-based, no sample-index slicing):
    #   calib/train_noobs → training
    #   val_modelsel (2022, 6 events) → model selection / early stopping
    #   val_conformal (2023, 7 events) → conformal calibration
    #   test_temporal (2024, 5 events) → temporal holdout test
    catalog_dict = {e["event_id"]: e for e in catalog}
    train_idx, val_idx, calib_idx, test_idx = [], [], [], []
    for i, (ev_id, _) in enumerate(dataset._index):
        sp = catalog_dict.get(ev_id, {}).get("split", "")
        if sp in ("calib", "train_noobs"):
            train_idx.append(i)
        elif sp == "val_modelsel":
            val_idx.append(i)
        elif sp == "val_conformal":
            calib_idx.append(i)
        elif sp == "test_temporal":
            test_idx.append(i)

    from torch.utils.data import Subset
    if not all((train_idx, val_idx, calib_idx, test_idx)):
        raise ValueError("Training, selection, calibration and evaluation splits must be nonempty")
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch, shuffle=True)
    val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=batch)
    calib_loader = DataLoader(Subset(dataset, calib_idx), batch_size=batch)
    test_loader  = DataLoader(Subset(dataset, test_idx),  batch_size=batch)

    logger.info(f"DataLoaders: train={len(train_idx)} val={len(val_idx)} "
                f"calib={len(calib_idx)} test={len(test_idx)}")
    return train_loader, val_loader, calib_loader, test_loader


def train_epoch(model: "nn.Module", loader,
                optimizer: "optim.Optimizer",
                config: Dict, device: "torch.device",
                epoch: int = 0) -> Dict:
    """단일 epoch 학습."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    exp_cfg = config.get("experiment", {})
    lambda_cont = exp_cfg.get("lambda_cont", 0.1)
    alpha_peak = exp_cfg.get("alpha_peak", 0.0)
    use_cont_loss = exp_cfg.get("use_cont_loss", False)
    log_interval = exp_cfg.get("log_interval", 100)
    t0 = time.time()

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        use_nse_loss = exp_cfg.get("use_nse_loss", False)
        if use_cont_loss and hasattr(model, "compute_total_loss"):
            loss = model.compute_total_loss(batch, lambda_cont=lambda_cont,
                                            alpha_peak=alpha_peak, use_nse_loss=use_nse_loss)
        else:
            pred = model(batch)
            depth_pred = pred.get("mean_depth", pred.get("depth_pred"))
            loss = nn.functional.mse_loss(depth_pred, batch.y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

        if n_batches % log_interval == 0:
            elapsed = time.time() - t0
            logger.info(f"  Epoch {epoch} batch {n_batches}/{len(loader)} "
                        f"loss={loss.item():.4f} elapsed={elapsed:.1f}s")

    return {"train_loss": total_loss / max(n_batches, 1)}


def validate(model: "nn.Module", loader, device: "torch.device") -> Dict:
    """검증 셋 평가."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            depth_pred = pred.get("mean_depth", pred.get("depth_pred"))
            loss = nn.functional.mse_loss(depth_pred, batch.y)
            total_loss += loss.item()
            n_batches += 1
    return {"val_loss": total_loss / max(n_batches, 1)}


def save_checkpoint(model: "nn.Module", metrics: Dict,
                    loop_id: int, member_id: int, cp_dir: Path) -> None:
    cp_dir.mkdir(parents=True, exist_ok=True)
    path = cp_dir / f"member_{member_id:02d}.pt"
    torch.save({"model_state": model.state_dict(), "metrics": metrics}, str(path))
    logger.info(f"Checkpoint saved → {path}")


def save_experiment_record(loop_id: int, regime_id: str, metrics: Dict,
                            changes: List[Dict], wall_time_s: float,
                            status: str, history_path: Path,
                            hypothesis: str = "") -> None:
    """experiment_history.json에 현재 loop 기록 추가 (기존 동일 loop 항목은 업데이트)."""
    record = {
        "loop": loop_id,
        "experiment_type": "exploration",
        "category": "B",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hypothesis": hypothesis or f"Loop {loop_id} GNN 학습",
        "regime_id": regime_id,
        "changes": changes,
        "metrics": metrics,
        "improvement_pct": None,
        "status": status,
        "error": None,
        "wall_time_s": round(wall_time_s, 1),
        "git_commit": None,
    }
    history = []
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
    # 동일 loop의 기존 항목 교체 (in_progress → success/failed)
    replaced = False
    for i, entry in enumerate(history):
        if entry.get("loop") == loop_id:
            history[i] = record
            replaced = True
            break
    if not replaced:
        history.append(record)
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    logger.info(f"Experiment record saved → {history_path}")


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(args.seed)
    cp_dir = Path("results/checkpoints") / f"loop_{args.loop_id:02d}"
    if cp_dir.exists() and any(cp_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to replace {cp_dir}; use a new loop ID or --overwrite")

    if not TORCH_AVAILABLE:
        logger.error("PyTorch not available. Install torch + torch_geometric.")
        sys.exit(1)

    device = get_device(args.device)
    logger.info(f"Device: {device}")

    exp_config = config.get("experiment", {})
    hidden_dim       = exp_config.get("hidden_dim", 128)
    n_heads          = exp_config.get("n_heads", 4)
    n_layers         = exp_config.get("n_layers", 4)
    lr               = exp_config.get("lr", 1e-3)
    epochs           = exp_config.get("epochs", 100)
    patience         = exp_config.get("patience", 15)
    M                = exp_config.get("ensemble_size", 5)
    T_out            = exp_config.get("T_out", 100)
    dropout          = exp_config.get("dropout", 0.0)
    temporal_decoder = exp_config.get("temporal_decoder", False)

    train_loader, val_loader, calib_loader, test_loader = \
        build_dataloaders(config, args.regime)

    if train_loader is None:
        raise ValueError("No training data available")

    # 실 학습 (Loop 2+)
    from src.models.gnn_surrogate import EnsembleGNN
    from src.models.conformal_uq import ConformalPredictor

    T_rain = exp_config.get("T_rain", 72)
    scalar_rain_decoder = exp_config.get("scalar_rain_decoder", False)
    conv_type = exp_config.get("conv_type", "gat")
    base_cfg = {
        "node_feat_dim": 14,
        "edge_feat_dim": 4,
        "hidden_dim": hidden_dim,
        "T_out": T_out,
        "T_rain": T_rain,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "dropout": dropout,
        "temporal_decoder": temporal_decoder,
        "scalar_rain_decoder": scalar_rain_decoder,
        "conv_type": conv_type,
    }
    model = EnsembleGNN(base_cfg, M=M).to(device)
    logger.info(f"Model: EnsembleGNN M={M}, "
                f"params={sum(p.numel() for p in model.parameters()):,}")

    weight_decay = exp_config.get("weight_decay", 0.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_cosine_wr = exp_config.get("use_cosine_wr", False)
    if use_cosine_wr:
        T_0 = exp_config.get("cosine_T0", 50)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=2, eta_min=1e-6
        )
    else:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience // 3
        )

    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, config, device, epoch=epoch)
        val_metrics   = validate(model, val_loader, device)
        if use_cosine_wr:
            scheduler.step()
        else:
            scheduler.step(val_metrics["val_loss"])

        val_loss = val_metrics["val_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            for mid, member in enumerate(model.members):
                save_checkpoint(member, val_metrics, args.loop_id, mid, cp_dir)
        else:
            patience_counter += 1

        if epoch % 10 == 0 or patience_counter == 0:
            logger.info(f"Epoch {epoch:3d} | train={train_metrics['train_loss']:.4f} "
                        f"val={val_loss:.4f} | best={best_val_loss:.4f} | "
                        f"patience={patience_counter}/{patience}")

        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    wall_time = time.time() - start_time

    # 실험 기록
    history_path = Path(args.config).parent / "logs" / "experiment_history.json"
    history_path.parent.mkdir(exist_ok=True)
    use_cont = exp_config.get("use_cont_loss", False)
    save_experiment_record(
        loop_id=args.loop_id,
        regime_id=args.regime,
        metrics={"best_val_loss": best_val_loss, "train_loss": train_metrics["train_loss"]},
        changes=[{"file": "src/train.py", "type": "modify"},
                 {"file": "config.yaml", "type": "modify"}],
        wall_time_s=wall_time,
        status="success",
        history_path=history_path,
        hypothesis=(
            f"EnsembleGNN M={M} + continuity_loss={'on' if use_cont else 'off'} "
            f"학습 (Loop {args.loop_id})"
        ),
    )
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}, "
                f"wall_time={wall_time:.0f}s")


if __name__ == "__main__":
    main()
