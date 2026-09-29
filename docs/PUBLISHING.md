# Source-only publication

This repository publishes software, configurations, documentation and synthetic
tests only. The MIT License applies to the source code, not to third-party data.

Do not publish Seocho network inputs or derived network data. This includes SWMM
INP files, node/link geometry or topology, coordinates, GIS layers, station-node
mappings, water-level observations, processed graph tensors, simulation outputs,
trained checkpoints, prediction caches and figures containing restricted data.
Compressed copies and renamed files are still restricted. No study datasets are
included, including Bellinge inputs; obtain those separately from their provider.

## Build an isolated snapshot

From the repository root:

```sh
python scripts/build_source_release.py --check
python scripts/build_source_release.py --output ../gnn_surrogate_public_snapshot
```

The output directory must be new and outside this working directory. The exporter
copies only the exact paths in `PUBLIC_FILES.txt`, plus a generated
`SHA256SUMS.txt`. It never copies `.git`, ignored files or directory contents
recursively. It rejects linked paths, non-UTF-8/binary files, unexpected file types,
large files and recognizable credential patterns. These mechanical checks do not
establish publication rights or detect every possible secret or embedded dataset;
review file contents before expanding the allowlist.

Upload only that inspected snapshot as a new repository history. Do not push the
local research repository or its old commits. Review future commits individually;
`.gitignore` does not protect already tracked files and can be overridden.

Code may reference private input paths and describe schemas. These references do
not include the files themselves. Authorized users must supply inputs locally;
ordinary users can run the synthetic tests without any sewer-network data.
