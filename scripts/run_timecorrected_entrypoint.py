"""Run an existing GNN script with the corrected SWMM dataset adapter.

Example
-------
python scripts/run_timecorrected_entrypoint.py src/train.py \
  --config configs/config_gauge401_fix_v1.yaml --loop_id 300 --seed 42 --regime B
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def parse_args() -> tuple[Path, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("target")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    namespace = parser.parse_args()
    target = Path(namespace.target)
    if not target.is_absolute():
        target = PROJECT / target
    arguments = list(namespace.arguments)
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    return target, arguments


def main() -> None:
    target, arguments = parse_args()
    if not target.exists():
        raise FileNotFoundError(target)

    # build_dataloaders selects the adapter from data.report_lead_steps.
    # Avoid global monkeypatches: Bellinge archives do not have Seocho's lead.
    os.chdir(PROJECT)
    os.environ["PAPER002_TIMECORRECTED_DATASET"] = "1"
    sys.argv = [str(target), *arguments]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
