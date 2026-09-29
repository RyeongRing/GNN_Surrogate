"""Plan or execute the final eight-seed protocol; never train by default."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 100, 200, 300, 400, 500, 600, 700)
ARMS = {
    "with_continuity": ("config_gauge401_fix_v1.yaml", 300),
    "no_continuity": ("config_gauge401_fix_v1_no_cont.yaml", 310),
    "gcn": ("config_gauge401_fix_v1_gcn.yaml", 320),
}
SPLITS = ("val_modelsel", "test_temporal")


def build_plan(action, arms, seeds, output_dir):
    commands = []
    for arm in arms:
        config, base_loop = ARMS[arm]
        for seed in seeds:
            loop = base_loop + SEEDS.index(seed)
            common = ["--config", f"configs/{config}", "--loop_id", str(loop),
                      "--seed", str(seed), "--regime", "B"]
            if action == "train":
                commands.append([sys.executable, "-m", "src.train", *common])
            else:
                for split in SPLITS:
                    output = output_dir / f"{arm}_seed{seed}_{split}.json"
                    commands.append([sys.executable, "-m", "src.evaluate", *common,
                                     "--checkpoint", f"results/checkpoints/loop_{loop}",
                                     "--eval_split", split, "--output", str(output)])
    return commands


def summarize(arms, seeds, output_dir):
    import numpy as np

    metrics = ("NSE", "RMSE_cm", "MAE_cm", "PICR_90", "PICR_sharpness_cm",
               "continuity_violation_pct", "primary_continuity_violation_pct")
    summary = {"seeds": list(seeds), "dispersion": "sample SD across outer seeds (ddof=1)",
               "ensemble_members_per_seed": 5, "splits": {}}
    for split in SPLITS:
        records = {}
        summary["splits"][split] = by_arm = {}
        for arm in arms:
            records[arm] = []
            for seed in seeds:
                path = output_dir / f"{arm}_seed{seed}_{split}.json"
                record = json.loads(path.read_text(encoding="utf-8"))
                expected_loop = ARMS[arm][1] + SEEDS.index(seed)
                if (record.get("seed"), record.get("loop_id"), record.get("eval_split")) != (seed, expected_loop, split):
                    raise ValueError(f"Result provenance mismatch: {path}")
                if record.get("continuity_n_graphs") != (300 if split == "val_modelsel" else 250):
                    raise ValueError(f"Full-split diagnostic missing or incomplete: {path}")
                records[arm].append(record)
            by_arm[arm] = {}
            for metric in metrics:
                values = np.asarray([r[metric] for r in records[arm]], dtype=float)
                if not np.isfinite(values).all():
                    raise ValueError(f"Non-finite {arm}/{split}/{metric}")
                by_arm[arm][metric] = {"per_seed": values.tolist(), "mean": float(values.mean()),
                                      "sd": float(values.std(ddof=1)) if len(values) > 1 else None}
        if "with_continuity" in arms:
            for other in ("no_continuity", "gcn"):
                if other not in arms:
                    continue
                delta = np.asarray(by_arm["with_continuity"]["NSE"]["per_seed"]) - np.asarray(by_arm[other]["NSE"]["per_seed"])
                by_arm[f"with_continuity_minus_{other}"] = {
                    "paired_NSE_delta": delta.tolist(), "mean_delta": float(delta.mean()),
                    "sd_delta": float(delta.std(ddof=1)) if len(delta) > 1 else None,
                    "positive_delta_count": int((delta > 0).sum()), "n_seeds": len(seeds),
                }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("evaluate", "train", "summarize"), default="evaluate")
    parser.add_argument("--arms", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/release_reproduction")
    parser.add_argument("--execute", action="store_true", help="Run the printed commands; training is expensive")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.arms)) != len(args.arms):
        parser.error("Duplicate arms/seeds are not allowed")
    output_dir = args.output_dir.resolve()
    if args.action == "summarize":
        print(json.dumps(summarize(args.arms, args.seeds, output_dir), indent=2))
        return
    for command in build_plan(args.action, args.arms, args.seeds, output_dir):
        print(shlex.join(command), flush=True)
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
