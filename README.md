# AI Computing (`aicomputing`)

`aicomputing` is a Python oracle package: provide a function signature, behavioral specification, and concrete inputs, and receive a model-predicted Python return value with validation diagnostics. You supply no implementation or reference answer.

This repository also contains separate evaluation tooling for measuring model and strategy performance. Ordinary package use requires no benchmarks, datasets, manifests, or scoring setup.

## Installation

Requires Python 3.10+. From a cloned repository's root, install the core package:

```bash
python -m pip install ./aicomputing
```

The core installation supports Ollama. Install only the extra needed for another backend:

```bash
# OpenAI-compatible hosted APIs or local servers
python -m pip install "./aicomputing[openai]"

# Local Hugging Face inference
python -m pip install "./aicomputing[hf]"
```

## Use the oracle

For this example, Ollama must be running with the selected model available. Replace `YOUR_INSTALLED_MODEL` with that model's identifier.

```python
from aicomputing import ExecuteRequest, VirtualRuntime

with VirtualRuntime(
    backend="ollama",
    model="YOUR_INSTALLED_MODEL",
    strategy="direct",
    validation_mode="strict",
) as oracle:
    prediction = oracle.execute(ExecuteRequest(
        signature="initials(name: str) -> str",
        docstring=(
            "Return the uppercase first letter of each word, "
            "separated by dots, without a trailing dot."
        ),
        inputs={"name": "ada lovelace"},
    ))

print(prediction.result)             # Expected: "A.L"
print(prediction.validation_status)
```

The package returns a prediction and diagnostics. Structural validation checks types and optional constraints; it does not prove that the model's answer is correct.

## Documentation

- [Package guide](aicomputing/README.md): API, strategies, backends, validation, caching, tracing, and CLI.
- [Evaluation workflow](evaluation/README.md): prepare, verify, execute and audit frozen experiments.
- [Benchmark reference](docs/evaluation.md): benchmark data, selection and scoring.

All seven package strategies are available independently of the subset selected for an experiment.

## Repository structure

```text
aicomputing/src/aicomputing/  # Installable oracle package
aicomputing/benchmarks/      # Data adapters, grading and selection
evaluation/                 # Audited preparation, execution and analysis
docs/evaluation.md          # Benchmark data and scoring
scripts/                    # Data preparation and local-model evaluation
manifests/prepared/          # Frozen experiment inputs (local, ignored)
runs/                       # Run snapshots and results (local, ignored)
```

## Status

The oracle runtime and evaluation tooling are implemented. Model predictions remain probabilistic. Future evaluations use the [reproducible workflow](evaluation/README.md), with frozen input bundles and separate run directories. Historical evaluations remain local and unchanged.

## License

MIT
