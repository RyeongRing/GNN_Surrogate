"""Dataset adapter for SWMM archives that include a pre-event report lead."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.swmm_dataset import SWMMGraphDataset


class TimeCorrectedSWMMGraphDataset(SWMMGraphDataset):
    """Expose only the locked event window to the GNN.

    The corrected SWMM archive retains six pre-event 10-minute steps for
    observation-based baseline evaluation. This adapter removes those steps
    before the legacy dataset resamples targets to ``T_out``.
    """

    def __init__(self, *args, report_lead_steps: int | None = None, **kwargs):
        ensemble_dir = kwargs.get("ensemble_dir")
        if ensemble_dir is None and len(args) >= 2:
            ensemble_dir = args[1]
        if ensemble_dir is None:
            raise ValueError("ensemble_dir is required")

        manifest_path = Path(ensemble_dir) / "run_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Corrected archive requires {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "report_lead_steps" not in manifest:
            raise ValueError("run_manifest.json must declare report_lead_steps")
        manifest_lead = manifest["report_lead_steps"]
        if isinstance(manifest_lead, bool) or not isinstance(manifest_lead, int):
            raise ValueError("report_lead_steps must be an integer")
        if report_lead_steps is None:
            report_lead_steps = manifest_lead
        if report_lead_steps != manifest_lead:
            raise ValueError("Config and archive disagree on report_lead_steps")
        self._timecorrected_report_lead_steps = int(report_lead_steps)
        if self._timecorrected_report_lead_steps < 0:
            raise ValueError("report_lead_steps must be non-negative")
        if self._timecorrected_report_lead_steps and (
            manifest.get("event_specific_lag_optimisation") is not False
        ):
            raise RuntimeError(
                "Corrected ensemble manifest must explicitly disable "
                "event-specific lag optimisation"
            )
        super().__init__(*args, **kwargs)

    def get(self, idx):
        lead = self._timecorrected_report_lead_steps
        if lead == 0:
            return super().get(idx)

        import src.data.ensemble_io as ensemble_io

        original_loader = ensemble_io.load_run_arrays

        def load_trimmed(event_id, set_id, ensemble_dir=None):
            result = original_loader(
                event_id,
                set_id,
                ensemble_dir=ensemble_dir,
            )
            available = min(
                result["node_depth"].shape[1],
                result["link_flow"].shape[1],
                int(result["timesteps"]),
            )
            if available <= lead:
                raise ValueError(
                    f"{event_id}/set{set_id}: cannot remove {lead} "
                    f"lead steps from {available} outputs"
                )
            result["node_depth"] = result["node_depth"][:, lead:available]
            result["link_flow"] = result["link_flow"][:, lead:available]
            result["timesteps"] = available - lead
            return result

        ensemble_io.load_run_arrays = load_trimmed
        try:
            return super().get(idx)
        finally:
            ensemble_io.load_run_arrays = original_loader
