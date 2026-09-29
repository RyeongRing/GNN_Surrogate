# GNN-SWMM Surrogate for Urban Drainage

Research code accompanying the manuscript:

**A Graph Neural Network Surrogate for Urban Drainage Simulation: Uncertainty Quantification and Cross-Catchment Transfer**  
Hyeryeong Yun and Junsuk Kang

This repository implements a graph neural network (GNN) surrogate for the Storm
Water Management Model (SWMM). It combines network-wide nodal-depth emulation,
internal-consistency regularization, empirical prediction intervals, and
cross-catchment adaptation in a common evaluation framework.

## Overview

- **Depth emulation:** a five-member GATv2Conv ensemble predicts SWMM nodal-depth trajectories.
- **Internal consistency:** an auxiliary edge-flux branch supports a soft depth-flux regularizer.
- **Uncertainty assessment:** timestep-conditioned empirical intervals are evaluated at network and node levels.
- **Cross-catchment transfer:** node-head fine-tuning is evaluated on the published Bellinge SWMM model with matched pretrained and random-backbone controls.

The prediction target is SWMM-simulated depth. The auxiliary edge-flux output
supports the internal-consistency regularizer; it is not a validated conduit-flow
prediction. Prediction intervals are evaluated empirically.

## Release Contents

This is a source-code release, not a data/checkpoint bundle. Restricted Seocho
inputs, trained weights and cached predictions are not included. Synthetic CPU
tests can run without them; numerical reproduction requires authorized study
artifacts.

| Directory | Contents |
| --- | --- |
| `src/` | GNN models, data loaders, training, evaluation and diagnostic code |
| `configs/` | Final model configurations and labeled historical configurations |
| `scripts/` | Protocol runners, input validation and source-only export tools |
| `bellinge/` | Cross-catchment adaptation and matched-control experiments |
| `tests/` | Synthetic CPU and command-line regression tests |
| `docs/` | Input requirements, result-to-code mapping and verification details |

See [data and protocol requirements](docs/DATA_PROTOCOL.md),
[the result-to-code map](docs/RESULTS_MAP.md), and
[verification scope](docs/VERIFICATION.md) for reproducibility details.

## Quick Start

Run commands from the repository root. Use Python 3.11 in a virtual environment.
Install a CPU or CUDA PyTorch wheel
appropriate to your machine before the remaining direct dependencies:
```sh
python -m venv .venv
```
Activate `.venv` using `.venv\Scripts\Activate.ps1` on PowerShell or
`source .venv/bin/activate` on POSIX. A CPU installation example is:
```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
```
See the official [PyTorch wheel options](https://pytorch.org/get-started/locally/)
and [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
for CUDA builds. torchvision and torchaudio are not required. Optional SWMM
simulation dependencies are in `requirements-swmm.txt`; processed NPZ loading
and GNN tests do not require a running SWMM engine.

The tests use synthetic inputs and do not require study data. Dependency ranges
and the tested environment are documented in [verification scope](docs/VERIFICATION.md).
`environment/original_freeze.txt` is a historical record, not an installation file.

## Study Workflows

The workflows below require the authorized inputs and, where applicable, trained
checkpoints listed in [DATA_PROTOCOL.md](docs/DATA_PROTOCOL.md). For a data-free
preview of the evaluation commands, run:

```sh
python scripts/run_final_protocol.py
```

This prints the evaluation plan without training or inference.

<details>
<summary>Evaluation, training, diagnostics and cross-catchment commands</summary>

## Final Protocol

Run all commands below from the repository root. Final configs are
`configs/config_gauge401_fix_v1*.yaml`. `src.train.build_dataloaders` selects
the corrected adapter directly from `data.report_lead_steps`; the old wrapper
is optional. Bellinge data are not globally patched with Seocho lead removal.

File-level preflight, with an optional checkpoint check:
```sh
python scripts/check_release_inputs.py --checkpoint results/checkpoints/loop_300
```

Primary evaluation, without training:
```sh
python -m src.evaluate --config configs/config_gauge401_fix_v1.yaml --checkpoint results/checkpoints/loop_300 --loop_id 300 --seed 42 --eval_split test_temporal --output results/release_reproduction/primary_test.json
```
This calculates pooled accuracy, timestep-conditioned intervals, and both
continuity diagnostics over the entire split. Missing inputs or any missing
ensemble member cause an error. It does not substitute random weights.

Plan the full eight-seed, three-arm evaluation:
```sh
python scripts/run_final_protocol.py
```
The default only prints commands. Add `--execute` to run inference. Use
`--arms with_continuity --seeds 42` for one outer seed. Outputs go to
`results/release_reproduction`, separate from manuscript result files.
After evaluation, obtain means, sample SDs and paired deltas without significance tests:
```sh
python scripts/run_final_protocol.py --action summarize
```

Training is an explicit, potentially expensive separate action:
```sh
python scripts/run_final_protocol.py --action train --arms with_continuity --seeds 42
```
This also prints only; add `--execute` deliberately. Existing checkpoints and
evaluation JSONs are protected from replacement unless a direct command is
explicitly given `--overwrite`. Old `run_matched_ablation.py` and
`run_extended_ablation.py` are not the final ablation runners.

## Diagnostics and Figures

```sh
python -m src.pernode_coverage_gauge401fix --config configs/config_gauge401_fix_v1.yaml
python -m src.loo_bootstrap_conformal_gauge401fix --config configs/config_gauge401_fix_v1.yaml
python -m src.analysis.generate_fig6_prediction_intervals_gauge401fix --config configs/config_gauge401_fix_v1.yaml
python -m src.analysis.generate_fig7_pernode_coverage_gauge401fix
```
Figure 6 runs inference; Figure 7 reads the node-coverage CSV. Both require
the study inputs described in the data protocol. These commands do not train.

Bellinge primary and matched-control commands DO perform node-head fine-tuning:
```sh
python -m bellinge.bellinge_primary_gauge401fix --config configs/config_gauge401_fix_v1.yaml
python -m bellinge.bellinge_exact_matched_gauge401fix --config configs/config_gauge401_fix_v1.yaml
```
Do not change into `bellinge/`. Set `BELLINGE_DATA_ROOT` if the target archive
is elsewhere. The pretrained/random control uses identical node-head initial
weights within each pair and reports raw pooled NSE, not affine-corrected NSE.

Figure 8 is plot-only by default, using its original cache:
```sh
python -m src.analysis.generate_fig9_bellinge_two_panel_gauge401fix --node 430 --sample 70
```
A missing cache is an error, not permission to train. `--recompute` explicitly
requests a new 150-epoch fine-tuning run; use a new `--cache` path to preserve
the original. Panel (a) uses designated primary-run values, while panel (b)
illustrates a separate execution. Its per-series NSE is not a representative
node-level aggregate. Reusing source-catchment interval widths can be evaluated with:
```sh
python scripts/recompute_bellinge_interval_transfer_final.py
```

</details>

## Data Availability

The Seocho-gu SWMM model and water-level observations were provided by the
Seoul Metropolitan Government and cannot be publicly redistributed by the authors.
Access is subject to the provider's approval. The published
[Bellinge data](https://doi.org/10.5194/essd-13-4779-2021) are available separately;
the study's processed catalogs/archives are additional inputs. No automatic
download or public redistribution of restricted data is performed by this repository.

**No sewer-network datasets are distributed here.** SWMM input files, network
geometry and topology, GIS layers, sensor mappings, observations, simulation
outputs, checkpoints and prediction caches are excluded from this release.

## Citation

If you use this code, please cite the manuscript named above. Publication details
and a manuscript DOI will be added when available.

## License

Source code is distributed under the [MIT License](LICENSE). The code license
does not grant redistribution rights to third-party datasets.

## Publishing Safely

Only files explicitly listed in `PUBLIC_FILES.txt` belong in the public release.
Use the [source-only export procedure](docs/PUBLISHING.md), not an upload of the
working directory or its Git history. Network geometry/topology, GIS layers,
sensor mappings, observations, simulation outputs, trained weights and prediction
caches must remain outside the public repository.
