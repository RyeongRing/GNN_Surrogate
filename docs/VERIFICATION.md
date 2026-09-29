# Release verification

Checked on 2026-09-29. This records software checks, not new scientific results.
Original manuscript files, archived result JSONs, network inputs and trained
checkpoints were not edited. Verification used local checks without training or
new SWMM runs; publication is a separate source-only step.

## Completed

- `python -B -m unittest discover -s tests -v`: 18 tests passed, including 15
  command-line `--help` invocations and the default dry-run evaluation planner.
- Python source parsing and all six YAML configurations passed.
- A two-graph regression with violation rates 0% and 100% reports 50%, not the
  first graph's 0%, for both continuity diagnostic aggregations.
- Missing 0/2 and 1/2 checkpoints fail before loading any weights; complete
  mocked checkpoint sets load. Final per-node, LOO, Figure 6 and Bellinge
  loaders also reject incomplete ensembles.
- Synthetic calibration produces a length-100 timestep quantile vector matching
  the direct absolute-residual NumPy calculation.
- Invalid catalogs, incomplete archives, absent/inconsistent lead manifests,
  empty rainfall windows and implicit Figure 8 fine-tuning are rejected.
- Tiny-graph forward/loss checks pass without optimizer steps.
- Source-only export tests reject disallowed paths, binary content and recognizable
  credential patterns, and confirm that unlisted files and Git history are excluded.
- A read-only integration check used authorized local study artifacts outside
  this release: all 4,600 Seocho samples were indexed into 3,700 training, 300
  model-selection, 350 interval-calibration and 250 temporal-evaluation samples.
  All five original Loop 300 checkpoint files loaded successfully. One E024
  evaluation sample yielded finite depth `(8480, 100)` and auxiliary edge-flux
  `(9966, 100)` arrays, with `dt_seconds = 618.1818...`. No prediction file was saved.

The test environment was the existing Python 3.11.15 CPU environment with
torch 2.14.0+cpu, torch-geometric 2.8.0.post1, NumPy 2.2.6 and pandas 3.0.5.
An upstream torch.jit deprecation warning was emitted; tests completed normally.

## Not certified by these checks

These checks do not reproduce the full manuscript metrics, retrain the models,
remeasure the runtime benchmark, rerun Bellinge fine-tuning, execute SWMM,
inspect freshly rendered study figures, or certify installation in a clean
environment. The declared dependency ranges are portability guidance, not a
claim that every combination in those ranges has been tested.

Restricted data and checkpoints remain outside the public release. Their
absence is distinct from source-code execution errors. Verify publication rights
and the exact study-artifact versions before making any additional files public.

## Changes from the audited source snapshot

The final evaluator now aggregates both residual diagnostics over all graphs,
using the same two definitions as the original corrected analyses. Common
checkpoint loading fails closed. Final data loaders explicitly select the
lead-trimming adapter rather than relying on a process-global monkeypatch.
README commands, package imports, eight-seed loop mappings and input schemas
are aligned. Required SWMM preparation/screening helpers and interval-transfer
code are included, while legacy ablation/selectors are marked and disabled as
final entry points. Figure 8 separates plot-only use from explicit recomputation;
new caches receive a provenance sidecar. The final matched-control summaries
use descriptive comparisons rather than hypothesis tests. Historical environment
records and prior scientific result files are preserved.
