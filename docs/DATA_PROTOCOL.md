# Input data and locked protocol

Run commands from the repository root. Paths in YAML files are relative to that
root; absolute paths to authorized local data also work. Data and checkpoints
are not bundled. Do not copy restricted files into a public Git commit.

## Seocho inputs

| Input | Default location | Required content |
| --- | --- | --- |
| Network | `inp_versions/seocho_imperv_landcover_strict.inp` | Authorized study INP, including its original infiltration-method settings |
| Catalog | `data/event_catalog.json` | Locked event records, not a newly selected event list |
| Raw rainfall | `data/rainfall_raw/AWS_4*.csv` | CP949 CSV: station ID, station name, datetime, hourly rainfall |
| Shifted rainfall | `data/rainfall_timecorrected_v2/` | Once-shifted files and `rainfall_transform_manifest.json` |
| SWMM archive | `results/swmm_timecorrected_v2_gauge401_fix/` | `run_manifest.json` and `{event_id}/{set_id:03d}.npz` |
| Checkpoints | `results/checkpoints/loop_300/` | `member_00.pt` through `member_04.pt` |
| Observations | `data/observations/` | Provider's water-level records, only needed for reference-model screening |
| Sensor mapping | `data/station_node_mapping.csv` | `station_id`, `swmm_node`, for observational screening |

Each catalog record includes `event_id`, `start`, `end`, `regime` (A/B/C), and
`split`. The locked catalog contains 64 `calib`, 10 `train_noobs`, six
`val_modelsel` (2022), seven `val_conformal` (2023), and five `test_temporal`
(2024) events. `calib` here names the observational-screening event list; both
`calib` and `train_noobs` are used for surrogate training. It is distinct from
the `val_conformal` interval-calibration split. The final YAMLs check these counts.

The exact dates, IDs, event selection and observational availability are part
of the study's locked data package. Obtain that package through authorized
access for numerical reproduction. The historical `src/data/select_events.py`
does not recreate those splits and its command-line execution is disabled.
Counts alone do not establish equivalence of two catalogs; retain file hashes.

Each NPZ has `node_depth` (N,T), `link_flow` (E,T), `node_names`, `link_names`,
`param_set` (seven values), and `meta` (one JSON string with event_id, set_id,
timesteps). `src/data/ensemble_io.py` defines serialization. Parameter order:
`n_conduit`, `n_imperv`, `dstore_imperv`, `dstore_perv`, `inf_max`, `inf_min`,
`inf_decay`. The last three legacy field names encode Green-Ampt suction head,
hydraulic conductivity and initial moisture deficit, not Horton parameters.
Preserve the original INP method settings; changing method names is not a
documentation-only edit.

Every event must have sets 0-49. The corrected archive manifest declares
`report_lead_steps: 6` and `event_specific_lag_optimisation: false`. The adapter
checks these values, removes the six 10-minute pre-event reports, and then
resamples the retained depth/auxiliary-flow reference arrays to 100 timesteps.
Physical spacing is `REPORT_STEP * (T_in - 1) / 99` seconds. The Bellinge loader
does not apply this Seocho-specific lead removal.

Generate the rainfall mirror and SWMM archive separately from the RAW rainfall:

```sh
python scripts/prepare_timecorrected_rainfall_v2.py --source data/rainfall_raw
python scripts/run_full_swmm_ensemble_timecorrected_v2.py --rain-dir data/rainfall_raw --dry-run
```

Remove `--dry-run` only to explicitly start SWMM ensemble generation. Both
commands apply the fixed -60-minute shift once; do not give the SWMM runner the
already shifted mirror with its default shift. AWS 401 is selected explicitly.
All 50 parameter sets are retained for training, regardless of observational
screening rank. Optional screening:

```sh
python scripts/evaluate_calibration_timecorrected_v2.py --observations data/observations
```

## Bellinge inputs

Download the published Bellinge source data separately. Set `BELLINGE_DATA_ROOT`
to a folder containing `7_SWMM/BellingeSWMM_v021_nopervious.inp`,
`7_SWMM/rg_bellinge_Jun2010_Aug2021.dat`, and `ensemble/{event_id}/{set_id:03d}.npz`.
The default is `data/bellinge`. The study's processed INP variant, 20-event
catalog and SWMM archive are additional study artifacts, not automatically
supplied by the publication DOI.

Primary/control runners require `results/bellinge_event_catalog_loop24.json`;
the figure helper uses `results/bellinge_event_catalog_loop24_20event_strict.json`.
Provide the corresponding locked catalogs, preserving event order: 15 fine-tuning
events and the last five held-out events, each with sets 0-19. Do not regenerate
or substitute event lists while claiming exact reproduction. The helper checks
20 distinct events and 400 samples. Both final arms mask four inactive Bellinge
conditioning features to the Seocho training means and adapt only `decoder.node_head`.

Figure 8 additionally needs its original prediction cache with
`pred_zero_shot`, `pred_fine_tuned`, `true_fine_tuned` (sample,node,time),
`test_event_ids`, and `dt_seconds_per_sample`. The cache supports plot-only
reproduction. A new fine-tuning run is a separate execution, not the original
illustration. Only load trusted NPZ/checkpoint files: the legacy formats use pickle.
