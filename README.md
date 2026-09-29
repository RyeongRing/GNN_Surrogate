# GNN-SWMM Surrogate for Urban Drainage

Research code for **A Graph Neural Network Surrogate for Urban Drainage Simulation:
Uncertainty Quantification and Cross-Catchment Transfer**

Hyeryeong Yun and Junsuk Kang

A graph neural network (GNN) framework for emulating Storm Water Management Model
(SWMM) nodal-depth dynamics and evaluating internal consistency, uncertainty, and
cross-catchment transfer.

## Features

- **Depth emulation:** five-member GATv2Conv ensembles predict network-wide SWMM nodal depths.
- **Internal consistency:** an auxiliary edge-flux branch regularizes the relationship between depth changes and latent fluxes.
- **Uncertainty assessment:** timestep-conditioned empirical prediction intervals are evaluated across nodes and depth groups.
- **Cross-catchment transfer:** node-head adaptation is assessed on Bellinge SWMM outputs using matched pretrained and random-backbone controls.

## Quick Start

Use Python 3.11 and run commands from the repository root:

```sh
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on PowerShell or
`source .venv/bin/activate` on POSIX. For a CPU installation:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -v
```

The tests use synthetic inputs. For GPU installation, follow the
[PyTorch](https://pytorch.org/get-started/locally/) and
[PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
installation guides. SWMM simulation dependencies are listed in
`requirements-swmm.txt`; the tested environment is recorded in
[Verification](docs/VERIFICATION.md).

## Study Workflows

Preview the matched evaluation plan:

```sh
python scripts/run_final_protocol.py
```

This command prints the plan; adding `--execute` starts evaluation.
Study runs require the authorized inputs and checkpoints specified in
[Data and Protocol](docs/DATA_PROTOCOL.md). Detailed evaluation, training, and
figure-generation commands are provided in
[Workflows and Result Mapping](docs/RESULTS_MAP.md).

## Repository Structure

| Directory | Purpose |
| --- | --- |
| `src/` | Models, data processing, training, evaluation and diagnostics |
| `configs/` | Experiment configurations |
| `scripts/` | Protocol runners and validation utilities |
| `bellinge/` | Cross-catchment adaptation and matched controls |
| `tests/` | Synthetic and command-line tests |
| `docs/` | Reproducibility and usage documentation |

## Data Availability

Access to the Seocho-gu sewer-network data and water-level observations requires approval from the Seoul Metropolitan Government. The SWMM model was constructed by the authors using these data. These restricted data, their processed and simulated derivatives, trained weights and prediction caches are excluded from this code-only release.

The published [Bellinge source data](https://doi.org/10.5194/essd-13-4779-2021) are
available separately. Study-specific input requirements are described in
[Data and Protocol](docs/DATA_PROTOCOL.md).

## Citation and License

Please cite the manuscript named above when using this code. Publication details
will be added when available.

The [MIT License](LICENSE) applies to the source code. Dataset access and reuse
follow the respective providers' terms.

Maintainer guidance: [Source-only publication](docs/PUBLISHING.md).
