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

Verify the frozen experiment definition without making model API calls:

```bash
python scripts/verify_experiment.py --manifest manifests/pilot-v2.json
```

Run an experiment from a finalized manifest:

```bash
python scripts/run_experiment.py --manifest manifests/pilot-v2-run.json
```

The committed manifests define the benchmark samples and experimental configuration. Model/provider configurations for a reported run are frozen separately before execution.

## Status

The framework and reproducible experiment pipeline are implemented. Final model configurations and full experimental results are not included yet.

## License

MIT
