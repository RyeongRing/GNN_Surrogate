# Study workflows and result-to-code map

The final protocol is gauge401fix, with the corrected rainfall timestamps and
explicit AWS 401 selection. Filenames containing `timecorrected_v2` can be
historical names retained by final diagnostics; use this map, not the filename
alone, to identify the calculation.

| Result | Entry point | Inputs |
| --- | --- | --- |
| Primary depth metrics and empirical intervals | `src.evaluate` | gauge401 config; loop 300, seed 42; `val_modelsel` or `test_temporal`; `val_conformal` for intervals |
| Matched continuity/operator comparison | `scripts/run_final_protocol.py` | eight seeds, three final configs, loops below |
| Fixed-300s, peak-flux-normalized continuity diagnostic | `src.evaluate:compute_continuity_violation` | ALL graphs in chosen split; `continuity_violation_pct` |
| Event-dt, unnormalized continuity diagnostic (S6) | `src.evaluate:compute_primary_continuity_violation` | ALL graphs in chosen split; `primary_continuity_violation_pct` |
| Figure 6 trajectories | `src.analysis.generate_fig6_prediction_intervals_gauge401fix` | loop 300 and final loaders |
| Node-depth quartiles / Figure 7 | `src.pernode_coverage_gauge401fix`, then `src.analysis.generate_fig7_pernode_coverage_gauge401fix` | final 2024 predictions, node-level CSV |
| Calibration sensitivity | `src.loo_bootstrap_conformal_gauge401fix` | event-block LOO/bootstrap; not a significance test |
| Bellinge primary zero-shot/adaptation | `bellinge.bellinge_primary_gauge401fix` | loop 300; locked 20-event target archive |
| Matched pretrained/random control | `bellinge.bellinge_exact_matched_gauge401fix` | loops 300-307; identical node-head initialization within each pair |
| Figure 8 | `src.analysis.generate_fig9_bellinge_two_panel_gauge401fix` | original cache by default; panel (a) designated primary values, panel (b) separate illustration execution |
| Reuse of Seocho intervals on Bellinge | `scripts/recompute_bellinge_interval_transfer_final.py` | loop 300 calibration and Bellinge prediction cache |
| Observational behavioral screening | `scripts/evaluate_calibration_timecorrected_v2.py` | all 50 corrected SWMM sets and restricted sensor records |

| Outer seed | GATv2 + continuity | GATv2 without continuity | GCN + continuity |
| --- | --- | --- | --- |
| 42 | 300 | 310 | 320 |
| 100 | 301 | 311 | 321 |
| 200 | 302 | 312 | 322 |
| 300 | 303 | 313 | 323 |
| 400 | 304 | 314 | 324 |
| 500 | 305 | 315 | 325 |
| 600 | 306 | 316 | 326 |
| 700 | 307 | 317 | 327 |

Every outer seed is a five-member ensemble. The release consolidates the original
`recompute_continuity_fullsplit_gauge401fix.py`,
`recompute_primary_continuity_diag_final.py`, and
`eval_gcn_ablation_batch_gauge401fix.py` tasks into `src.evaluate` and the final
runner. Existing manuscript JSONs are not overwritten by this consolidation.
The two residual definitions remain distinct and neither validates physical
conduit flows. Pooled NSE is computed over all evaluated node-time samples.

## Running the workflows

Run these commands from the repository root after supplying the authorized inputs
listed in [Data and Protocol](DATA_PROTOCOL.md). Final configurations are
`configs/config_gauge401_fix_v1*.yaml`.

### Primary and matched evaluation

Check input files and the primary checkpoint ensemble:

```sh
python scripts/check_release_inputs.py --checkpoint results/checkpoints/loop_300
```

Evaluate the primary model on the temporal split:

```sh
python -m src.evaluate --config configs/config_gauge401_fix_v1.yaml --checkpoint results/checkpoints/loop_300 --loop_id 300 --seed 42 --eval_split test_temporal --output results/release_reproduction/primary_test.json
```

The evaluator requires a complete checkpoint ensemble and calculates pooled depth
metrics, timestep-conditioned intervals and both continuity diagnostics over the
chosen split.

Preview all eight-seed, three-arm evaluations, then summarize completed runs:

```sh
python scripts/run_final_protocol.py
python scripts/run_final_protocol.py --action summarize
```

Add `--execute` to the first command to run evaluation, or use
`--arms with_continuity --seeds 42` to select the primary arm and seed. Outputs
are saved in `results/release_reproduction`. Summaries report means, sample
standard deviations and paired differences.

### Training

Preview training for the primary ensemble:

```sh
python scripts/run_final_protocol.py --action train --arms with_continuity --seeds 42
```

Add `--execute` to start training. Replacing existing training or evaluation
artifacts requires `--overwrite` on the corresponding direct command.

### Diagnostics and figures

```sh
python -m src.pernode_coverage_gauge401fix --config configs/config_gauge401_fix_v1.yaml
python -m src.loo_bootstrap_conformal_gauge401fix --config configs/config_gauge401_fix_v1.yaml
python -m src.analysis.generate_fig6_prediction_intervals_gauge401fix --config configs/config_gauge401_fix_v1.yaml
python -m src.analysis.generate_fig7_pernode_coverage_gauge401fix
```

These commands perform inference or post-processing. Figure 6 uses the primary
checkpoint ensemble; Figure 7 reads the node-coverage CSV from the first command.

### Cross-catchment adaptation

The following commands perform node-head fine-tuning:

```sh
python -m bellinge.bellinge_primary_gauge401fix --config configs/config_gauge401_fix_v1.yaml
python -m bellinge.bellinge_exact_matched_gauge401fix --config configs/config_gauge401_fix_v1.yaml
```

Set `BELLINGE_DATA_ROOT` to use an alternative target-data location. The matched
control uses identical node-head initial weights within each pair and reports
raw pooled NSE as its primary metric.

Recreate Figure 8 from its original prediction cache:

```sh
python -m src.analysis.generate_fig9_bellinge_two_panel_gauge401fix --node 430 --sample 70
```

The default is plot-only and requires the cache. `--recompute` explicitly starts
a new 150-epoch fine-tuning run; specify a new `--cache` path to preserve the
original. Panel (a) uses designated primary-run values, while panel (b) displays
a separate execution. The displayed series NSE is a single-trajectory metric,
distinct from the node-level aggregate.

Evaluate reuse of source-catchment interval widths on Bellinge:

```sh
python scripts/recompute_bellinge_interval_transfer_final.py
```

## Historical modules

`config_timecorrected_v2*.yaml`, `src/run_matched_ablation.py`,
`src/run_extended_ablation.py`, and `src/data/select_events.py` are historical,
not final protocol instructions. Their old selectors/ablation CLIs are disabled;
helpers remain available to avoid breaking imports. Older Bellinge and figure
modules are also retained as supporting code for final modules. Run only the
entry points listed above for final results. Historical significance-test code
is not used by the final descriptive-summary runner.

The earlier-checkpoint runtime benchmark is not remeasured by this release.
Availability of a generator is not evidence that the manuscript numbers have
been reproduced in a new environment. Check hashes, checkpoint lineage, data
versions and split membership before comparing new outputs with the manuscript.
