# AI Computing

Research prototype for specification-driven execution with large language models.

Instead of executing an explicit implementation of a function, the runtime receives a function specification and concrete inputs and asks a language model to realize the corresponding output.

## Repository structure

- `aicomputing/` — Python package and benchmark adapters
- `scripts/` — experiment construction, execution, verification, and aggregation
- `manifests/` — frozen benchmark selections and experiment configurations
- `data/` — frozen derived benchmark artifacts required for reproduction

## Installation

```bash
pip install -e "./aicomputing[benchmarks,openai]"