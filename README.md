# AI Computing

Research prototype for **specification-driven execution with large language models**.

The project explores whether a function can be executed directly from its specification and concrete inputs without first generating and running an explicit implementation.

```text
Traditional:
specification -> implementation -> execution -> output

AI Computing:
specification + input -> model-mediated runtime -> output
```

The repository contains the Python runtime and the experimental infrastructure used to evaluate this idea.

## Repository structure

```text
.
├── aicomputing/   # Python package and benchmark adapters
├── scripts/       # Experiment setup, execution, verification, and aggregation
├── manifests/     # Frozen benchmark selections and experiment configurations
└── data/          # Derived benchmark artifacts required for reproduction
```

For installation, API usage, architecture, strategies, backends, validation, and CLI documentation, see:

**[`aicomputing/README.md`](aicomputing/README.md)**

## Installation

From the repository root:

```bash
pip install -e "./aicomputing[hf,openai,benchmarks]"
```

For a smaller installation, select only the extras you need.

## Experiment setup

Run synthetic plumbing checks without loading datasets or making model API calls:

```bash
python scripts/verify_experiment.py
```

Current manifests require integer schema version **2** and selection compatibility version **4**, using the `pilot-v3` / `main-v2` experiment presets. Preset names alone do not establish compatibility. Earlier manifests must be rebuilt, or used with their corresponding historical code; they are not automatically converted. See the package README for model configuration and manifest creation.

Inspect a newly generated manifest without loading datasets: 

```bash
python scripts/verify_experiment.py --manifest manifests/pilot-v3-run.json
```

Add `--check-datasets` to resolve real items and rebuild selections; that mode may download data, but still makes no real model calls.

Run an experiment from a finalized current-version manifest:

```bash
python scripts/run_experiment.py --manifest manifests/pilot-v3-run.json
```

New reported-run manifests freeze benchmark selections, generation settings, provider/backend/URL identity and explicit native reasoning overrides before execution. Credentials remain outside the manifest.

## Status

The framework and reproducible experiment pipeline are implemented. Final model configurations and full experimental results are not included yet.

## License

MIT
