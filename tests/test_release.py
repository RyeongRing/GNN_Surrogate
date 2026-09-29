"""Small CPU regressions; no study data, trained weights or SWMM required."""

import ast
import importlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Data

from src.checkpoints import load_members
from src.data.protocol import validate_catalog, validate_completed_runs
from src.data.swmm_dataset import _rain_window, _resample_ts, _resampled_dt_seconds
from src.data.swmm_dataset_timecorrected import TimeCorrectedSWMMGraphDataset
from src.models.conformal_uq import ConformalPredictor
from src.models.gnn_surrogate import EnsembleGNN
from scripts.run_final_protocol import build_plan, ARMS, SEEDS

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_python_syntax_and_yaml(self):
        for folder in ("src", "bellinge", "scripts", "tests"):
            for path in (ROOT / folder).rglob("*.py"):
                with self.subTest(file=str(path)):
                    ast.parse(path.read_text(encoding="utf-8-sig"))
        for path in (ROOT / "configs").glob("*.yaml"):
            self.assertIsInstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)

    def test_final_configs(self):
        for name in ARMS:
            config = yaml.safe_load((ROOT / "configs" / ARMS[name][0]).read_text(encoding="utf-8"))
            self.assertEqual(config["experiment"]["ensemble_size"], 5)
            self.assertEqual(config["data"]["expected_split_counts"]["val_conformal"], 7)
            self.assertIn("gauge401_fix", config["data"]["ensemble_dir"])
            self.assertEqual(config["experiment"].get("conv_type", "gat"), "gcn" if name == "gcn" else "gat")

    def test_complete_checkpoints_required(self):
        model = SimpleNamespace(members=[torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)], eval=lambda: None)
        for available in (0, 1):
            with self.subTest(available=available), \
                 patch.object(Path, "is_file", side_effect=[True] * available + [False] * (2 - available)), \
                 patch("torch.load") as read:
                with self.assertRaises(FileNotFoundError):
                    load_members(model, Path("missing"), "cpu")
                read.assert_not_called()
        states = [{"model_state": member.state_dict()} for member in model.members]
        with patch.object(Path, "is_file", return_value=True), patch("torch.load", side_effect=states):
            self.assertIs(load_members(model, Path("mock"), "cpu"), model)

    def test_all_evaluation_loaders_are_strict(self):
        config = {"experiment": {"ensemble_size": 2, "hidden_dim": 8, "n_heads": 2, "n_layers": 1, "T_out": 3}}
        for module in ("src.pernode_coverage_gauge401fix", "src.loo_bootstrap_conformal_gauge401fix",
                       "src.analysis.generate_fig6_prediction_intervals_gauge401fix"):
            loader = importlib.import_module(module).load_model
            with self.subTest(module=module), patch.object(Path, "is_file", return_value=False):
                with self.assertRaises(FileNotFoundError):
                    loader(config, torch.device("cpu"))
        from bellinge.bellinge_validate_loop24 import load_loop_model
        with patch.object(Path, "is_file", return_value=False), self.assertRaises(FileNotFoundError):
            load_loop_model(Path("missing"), config, torch.device("cpu"))

    def test_timestep_quantiles(self):
        true = np.arange(3000, dtype=float).reshape(30, 100) / 1000
        pred = np.zeros_like(true)
        cp = ConformalPredictor(alpha=0.1)
        cp.calibrate(pred, true)
        expected = np.quantile(true, np.ceil(0.9 * 31) / 30, axis=0, method="linear")
        self.assertEqual(cp.q_hat_dict["B"].shape, (100,))
        np.testing.assert_allclose(cp.q_hat_dict["B"], expected)
        np.testing.assert_allclose(cp.predict(pred)["upper"], np.broadcast_to(expected, true.shape))
        with self.assertRaises(ValueError):
            cp.calibrate(np.empty((0, 100)), np.empty((0, 100)))

    def test_two_graph_continuity_aggregation(self):
        import src.evaluate as evaluate
        batches = [Data(y=torch.arange(6, dtype=torch.float32).reshape(2, 3),
                        fixture_depth=torch.zeros(2, 3), fixture_flow=torch.full((1, 3), float(value)),
                        edge_index=torch.tensor([[0], [1]]), dt_seconds=torch.tensor([300.]))
                   for value in (0, 1)]

        class Model:
            def __call__(self, batch):
                return {"mean_depth": batch.fixture_depth, "mean_flow": batch.fixture_flow}

        args = SimpleNamespace(config="unused.yaml", checkpoint="unused", output="unused.json", overwrite=False,
                               regime="B", site="seocho", loop_id=300, seed=42, eval_split="test_temporal")
        with patch.object(evaluate, "parse_args", return_value=args), \
             patch("builtins.open", return_value=io.StringIO("experiment: {T_out: 3}")), \
             patch.object(Path, "exists", return_value=False), \
             patch("src.train.build_dataloaders", return_value=(batches, batches, batches, batches)), \
             patch.object(evaluate, "_load_ensemble_model", return_value=Model()), \
             patch.object(evaluate, "save_results") as save:
            evaluate.main()
        metrics = save.call_args.args[0]
        self.assertEqual(metrics["continuity_violation_pct"], 50.0)
        self.assertEqual(metrics["primary_continuity_violation_pct"], 50.0)
        self.assertEqual(metrics["continuity_n_graphs"], 2)

    def test_resampling_and_empty_rainfall(self):
        actual = _resample_ts(np.array([[2., 4.]]), 100)
        np.testing.assert_equal(actual[:, [0, -1]], [[2., 4.]])
        self.assertAlmostEqual(_resampled_dt_seconds(103, 100, 600), 618.18181818)
        rain = pd.Series([1.], index=pd.to_datetime(["2020-01-01"]))
        with self.assertRaisesRegex(ValueError, "Missing/non-finite rainfall"):
            _rain_window(rain, {"event_id": "synthetic", "start": "2024-01-01", "end": "2024-01-02"}, 72)

    def test_manifest_required_and_consistent(self):
        with patch.object(Path, "is_file", return_value=False), self.assertRaises(FileNotFoundError):
            TimeCorrectedSWMMGraphDataset(ensemble_dir="missing")
        for manifest in ({}, {"report_lead_steps": 0}, {"report_lead_steps": 6.5}):
            with patch.object(Path, "is_file", return_value=True), \
                 patch.object(Path, "read_text", return_value=json.dumps(manifest)), \
                 self.assertRaises(ValueError):
                TimeCorrectedSWMMGraphDataset(ensemble_dir="mock", report_lead_steps=6)

    def test_locked_catalog_and_archive(self):
        event = {"event_id": "synthetic", "split": "train", "regime": "B", "start": "2024-01-01", "end": "2024-01-02"}
        with self.assertRaises(ValueError):
            validate_catalog([event])
        event["split"] = "test_temporal"
        self.assertEqual(validate_catalog([event]), {"test_temporal": 1})
        with self.assertRaises(ValueError):
            validate_catalog([event, event])
        with self.assertRaises(ValueError):
            validate_completed_runs([("synthetic", 0)], [event], 2)
        validate_completed_runs([("synthetic", 0), ("synthetic", 1)], [event], 2)

    def test_final_runner_plan(self):
        commands = build_plan("evaluate", list(ARMS), list(SEEDS), Path("results/test"))
        self.assertEqual(len(commands), 48)
        self.assertTrue(all(command[2] == "src.evaluate" for command in commands))
        self.assertIn("results/checkpoints/loop_327", commands[-1])
        self.assertNotIn("run_matched_ablation.py", str(commands))

    def test_figure8_missing_cache_never_trains(self):
        module = importlib.import_module("src.analysis.generate_fig9_bellinge_two_panel_gauge401fix")
        with patch("sys.argv", ["figure8"]), patch.object(Path, "is_file", return_value=False), \
             patch.object(module.base.bv, "fine_tune_bellinge_decoder") as train:
            with self.assertRaisesRegex(FileNotFoundError, "plot-only mode never trains"):
                module.main()
            train.assert_not_called()

    def test_small_graph_forward(self):
        model = EnsembleGNN({"node_feat_dim": 14, "edge_feat_dim": 4, "hidden_dim": 8,
                             "n_heads": 2, "n_layers": 1, "T_out": 100, "T_rain": 72}, M=2)
        data = Data(x=torch.ones(4, 14), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
                    edge_attr=torch.ones(3, 4), rain=torch.zeros(72, 1), y=torch.zeros(4, 100),
                    dt_seconds=torch.tensor([600.]))
        with torch.no_grad():
            result = model(data)
            self.assertEqual(tuple(result["mean_depth"].shape), (4, 100))
            self.assertEqual(tuple(result["mean_flow"].shape), (3, 100))
            self.assertTrue(torch.isfinite(model.compute_total_loss(data)))


if __name__ == "__main__":
    unittest.main()
