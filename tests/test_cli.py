"""Documented command-line smoke tests; --help never starts an experiment."""

from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_documented_help_commands(self):
        targets = [
            ["-m", "src.train"], ["-m", "src.evaluate"],
            ["-m", "src.pernode_coverage_gauge401fix"],
            ["-m", "src.loo_bootstrap_conformal_gauge401fix"],
            ["-m", "bellinge.bellinge_primary_gauge401fix"],
            ["-m", "bellinge.bellinge_exact_matched_gauge401fix"],
            ["-m", "src.analysis.generate_fig6_prediction_intervals_gauge401fix"],
            ["-m", "src.analysis.generate_fig9_bellinge_two_panel_gauge401fix"],
            ["scripts/run_final_protocol.py"], ["scripts/check_release_inputs.py"],
            ["scripts/prepare_timecorrected_rainfall_v2.py"],
            ["scripts/run_full_swmm_ensemble_timecorrected_v2.py"],
            ["scripts/evaluate_calibration_timecorrected_v2.py"],
            ["scripts/recompute_bellinge_interval_transfer_final.py"],
            ["scripts/run_timecorrected_entrypoint.py", "src/pernode_coverage_gauge401fix.py"],
        ]
        for target in targets:
            with self.subTest(target=target):
                result = subprocess.run([sys.executable, "-B", *target, "--help"],
                                        cwd=ROOT, capture_output=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
                self.assertIn(b"usage:", result.stdout.lower())

    def test_default_runner_only_plans(self):
        result = subprocess.run([sys.executable, "-B", "scripts/run_final_protocol.py"],
                                cwd=ROOT, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        self.assertEqual(len(result.stdout.splitlines()), 48)


if __name__ == "__main__":
    unittest.main()
