# AI Computing (`aicomputing`)

`aicomputing` is a Python oracle package: provide a function signature, behavioral specification, and concrete inputs, and receive a model-predicted Python return value with validation diagnostics. You supply no implementation or reference answer.

This repository also contains separate evaluation tooling for measuring model and strategy performance. Ordinary package use requires no benchmarks, datasets, manifests, or scoring setup.

The package also offers `strategy="code_execute"`: generate a Python function
and run it in a standard Python worker. The evaluator uses this same strategy.
Results retain `generated_code` and `execution_details` alongside the return value.

## Installation

Requires Python 3.10+. From a cloned repository's root, install the core package:

```bash
python -m pip install ./aicomputing
```

The core runtime has no required third-party dependencies. Install the extra
for each backend you want to use:

```bash
# Local Ollama server
python -m pip install "./aicomputing[ollama]"

# OpenAI-compatible hosted APIs or local servers
python -m pip install "./aicomputing[openai]"

# Local Hugging Face inference
python -m pip install "./aicomputing[hf]"
```

The OpenAI backend and evaluator require OpenAI SDK 2.30.0 or newer within major version 2.

## Development checks

Install both packages and their offline test dependencies from the repository root:

```powershell
python -m pip install -e "./aicomputing[dev,benchmarks]" -e "./case_study[dev]"
python -m pytest -q
python -m ruff check .
python -m build ./aicomputing
python -m build ./case_study
```

Tests cover the runtime, provider adapters, evaluation checkpoints and scoring,
and the case-study app. Provider calls are mocked; no API keys are required.
The GitHub Checks workflow runs these checks on Windows and Linux with Python
3.10 and 3.14. Generated results, caches and build artifacts are ignored.

## Use the oracle

For this example, install the `ollama` extra and run Ollama with the selected
model available. Replace `YOUR_INSTALLED_MODEL` with that model's identifier.
Pass the backend explicitly to `VirtualRuntime`; the `aicomp` command likewise
requires `--backend`.

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

The package checks caller inputs and defaults to strict output validation.
Declare the output type in the function signature, for example `-> str` or
`-> list[int]`. A signature without a return annotation allows any decoded type.
Use `validation_mode="warn"` to return invalid outputs with diagnostics or
`"none"` to preserve parsed values without output checks. Structural validation
checks types and optional constraints; it does not prove that an answer is correct.

Result caching is disabled by default. For repeated identical requests in a
long-lived runtime, opt in with a positive `cache_size`, such as `cache_size=128`.

## Documentation

- [Package guide](aicomputing/README.md): API, strategies, backends, validation, caching, tracing, and CLI.
- [Evaluation workflow](evaluation/README.md): run selected benchmarks with one command and receive one detailed CSV and one JSON.
- [Benchmark reference](docs/evaluation.md): benchmark data, selection and scoring.
- [Reviewed dataset](data/README.md): bundled release, checksum, provenance and upstream notices.
- [TDMSci case study](case_study/README.md): application, corpus, evaluation, and annotation-guideline review.
- [Published results](results/README.md): the four full benchmark experiments and the latest TDMSci v2 case-study run, with checksums and verification notes.

The package's registered strategies are available independently of the subset selected for an experiment.

## Repository structure

```text
aicomputing/src/aicomputing/  # Installable oracle package
evaluation/                 # Single-command evaluation and shared helpers
evaluation/benchmarks/      # Reviewed data, task selection, grading and result rows
docs/evaluation.md          # Benchmark data and scoring
data/dataset-release.json   # Bundled reviewed benchmark data
results/                    # Published result bundle; new raw runs remain local
```

## Status

The oracle runtime and evaluation tooling are implemented. Model predictions remain probabilistic. The [evaluation command](evaluation/README.md) selects reviewed inputs, runs the requested approaches, and saves results and resumable progress in two files.

## License

The package and evaluation code are [MIT-licensed](aicomputing/LICENSE).
Benchmark data retains the upstream terms
documented in [data/README.md](data/README.md).
