"""Validate final input files without running SWMM, training or inference."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from src.data.ensemble_io import list_completed
from src.data.protocol import validate_catalog, validate_completed_runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config_gauge401_fix_v1.yaml")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data = config["data"]

    def resolve(path):
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    for key in ("inp_path", "catalog_path", "ensemble_dir", "rain_dir"):
        path = resolve(data[key])
        if not path.exists():
            raise FileNotFoundError(f"Missing {key}: {path}. See docs/DATA_PROTOCOL.md")
    catalog = json.loads(resolve(data["catalog_path"]).read_text(encoding="utf-8"))
    counts = validate_catalog(catalog, data.get("expected_split_counts"))
    archive = resolve(data["ensemble_dir"])
    manifest = json.loads((archive / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("report_lead_steps") != data["report_lead_steps"]:
        raise ValueError("Archive/config report_lead_steps mismatch")
    if manifest.get("event_specific_lag_optimisation") is not False:
        raise ValueError("Archive must explicitly disable event-specific lag optimization")
    completed = list_completed(archive)
    index = [(e["event_id"], sid) for e in catalog
             for sid in completed.get(e["event_id"], []) if sid < data["n_sets"]]
    validate_completed_runs(index, catalog, data["n_sets"])
    rainfall_manifest = resolve(data["rain_dir"]) / "rainfall_transform_manifest.json"
    rain = json.loads(rainfall_manifest.read_text(encoding="utf-8"))
    if rain.get("timestamp_shift_minutes") != -60:
        raise ValueError("Expected a once-shifted (-60 min) rainfall mirror")
    if config["experiment"]["batch_size"] != 1:
        raise ValueError("The final protocol requires batch_size=1")
    if args.checkpoint:
        for member in range(config["experiment"]["ensemble_size"]):
            path = resolve(args.checkpoint) / f"member_{member:02d}.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
    print(json.dumps({"event_counts": counts, "completed_runs": len(index),
                      "status": "file-level preflight passed; no inference performed"}, indent=2))


if __name__ == "__main__":
    main()
