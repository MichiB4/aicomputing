# AI Computing (`aicomputing`)

> **Predictive execution with language models:** describe a function by its interface and behaviour, provide concrete inputs, and let a model-mediated runtime produce the return value — without generating or executing a target implementation first.

`aicomputing` is a research prototype for exploring **specification-driven execution**. In conventional programming, a specification is translated into an implementation and that implementation is executed. Here, the specification itself is the persistent program artifact:

```text
Conventional:  specification -> implementation -> interpreter/runtime -> output
AI Computing:  specification + input -> model-mediated runtime -> output
```

The distinction is not that the model “does no computation.” The distinction is architectural: no separately represented executable implementation of the target function is required at invocation time.

> **Research status:** the framework and experimental harness are implemented and tested. The final model configurations and full benchmark runs are still to be completed, so this repository does not yet claim final empirical results.

---

## Quick start

```python
from aicomputing import ExecuteRequest, VirtualRuntime

runtime = VirtualRuntime(
    backend="ollama",
    model="qwen3:4b",
    strategy="direct",
    validation_mode="strict",
)

request = ExecuteRequest(
    signature="repeat(s: str, n: int) -> str",
    docstring="Return the string s repeated n times.",
    inputs={"s": "ab", "n": 3},
    return_type=str,
)

result = runtime.execute(request)

print(result.result)             # model-produced Python value, e.g. 'ababab'
print(result.validation_status)  # structural runtime status
print(result.raw_text)           # original model response
```

No Python implementation of `repeat` is supplied to the runtime. The model receives the function specification and concrete arguments and predicts the value the function should return.

This is a **probabilistic execution mechanism**, not a semantic guarantee. Passing structural validation means that the value fits the declared interface; it does not prove that the value is correct.

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/MichiB4/aicomputing.git
cd aicomputing

# Core package (Ollama backend)
pip install -e .

# Local Hugging Face inference
pip install -e ".[hf]"

# OpenAI-compatible APIs
pip install -e ".[openai]"

# Benchmark dependencies (HumanEval+, MBPP+, GSM8K, BBH)
pip install -e ".[benchmarks]"

# Test dependencies
pip install -e ".[test]"

# Typical research/development install
pip install -e ".[hf,openai,benchmarks,test]"
```

### Ollama

Ollama must be installed and running locally:

```bash
ollama pull qwen3:4b
ollama serve
```

### Hugging Face

The `hf` backend runs `transformers` models locally. A CUDA-capable GPU is strongly recommended for non-trivial models.

### OpenAI-compatible APIs

The `openai` backend works with services exposing a compatible chat-completions API, including hosted providers and local servers such as vLLM or LM Studio. Provider credentials should be supplied through the environment or CLI and must **not** be committed to manifests.

---

## Architecture

The framework separates the function specification from the mechanism that realizes it.

```text
                         ExecuteRequest
                 signature + docstring + inputs
                                |
                                v
                     +--------------------+
                     |   VirtualRuntime   |
                     | orchestration,     |
                     | cache, retries     |
                     +---------+----------+
                               |
                               v
                     +--------------------+
                     |      Strategy      |
                     | execution policy   |
                     +---------+----------+
                               |
                        builds | prompt(s)
                               v
                     +--------------------+
                     | Prompting helpers  |
                     +---------+----------+
                               |
                               v
                     +--------------------+
                     |      Backend       |
                     | provider/model I/O |
                     +---------+----------+
                               |
                               v
                           LLM / model
                               |
                          raw response
                               |
                               v
                     +--------------------+
                     | Parsing + coercion |
                     +---------+----------+
                               |
                               v
                     +--------------------+
                     |     Validation     |
                     | type + constraints |
                     +---------+----------+
                               |
                  invalid + retry | valid / exhausted
                        +----------+----------+
                        |                     |
                        +--> Strategy         v
                                       ExecuteResult
```

The separation is deliberate:

- **`ExecuteRequest`** describes *what* should be computed and the concrete invocation.
- **`VirtualRuntime`** coordinates execution, caching, retries, parsing, and validation.
- **Strategies** determine *how* the model is asked to realize the specification.
- **Prompt builders** serialize the specification independently of the provider.
- **Backends** normalize communication with different inference engines.
- **Parsing** converts model text into Python values.
- **Validation** checks the declared structural interface and optional constraints.
- **`ExecuteResult`** returns the value together with execution and diagnostic metadata.

This makes model choice and execution strategy independently replaceable: a function specification does not need to know which provider, model, or prompting strategy will realize it.

### Execution lifecycle

For a call `runtime.execute(request)`, the runtime roughly performs:

1. parse the function signature and validate the supplied arguments;
2. check the result cache;
3. resolve the selected strategy and model;
4. build the strategy-specific prompt;
5. call the backend/model;
6. parse the model response into a candidate Python value;
7. apply type and constraint validation;
8. retry with validation feedback if permitted;
9. return a structured `ExecuteResult` and optionally write a trace.

The model output is therefore a **candidate** until it crosses the parsing and validation boundary.

---

## Public API

### `ExecuteRequest`

The stable core of a request is:

| Field | Purpose |
|---|---|
| `signature` | Python-style function interface |
| `docstring` | Natural-language behavioural specification |
| `inputs` | Concrete arguments keyed by parameter name |
| `return_type` | Optional expected Python return type |
| `constraints` | Optional predicates checked on the returned value |
| `strategy` | Optional per-call strategy override |
| `model` | Optional per-call model override |
| `temperature`, `top_p` | Generation controls |
| `max_new_tokens` | Optional token-budget override |
| `max_retries` | Maximum validation retries |
| `think` | Optional explicit thinking-mode override |
| `validation_mode` | `"warn"` or `"strict"` structural-validation policy |

The legacy `strict_types` option remains supported for compatibility; `validation_mode` is the clearer interface for output-validation policy.

### `ExecuteResult`

`ExecuteResult` contains the returned value plus enough metadata to make model-mediated execution observable. Important fields include:

| Field | Meaning |
|---|---|
| `result` | Parsed return value; may be withheld under strict validation |
| `raw_text` | Raw model response |
| `model`, `strategy` | Effective model and strategy |
| `parse_status` | `parsed`, `fallback_parsed`, or `failed` |
| `validation_status` | `valid`, `invalid_type`, `invalid_constraint`, or `unparsed` |
| `failure_type` | Structured reason for an execution failure, if any |
| `scorable` | Whether the outcome represents model behaviour rather than infrastructure failure |
| `requested_think` / `effective_think` | Requested vs. realizable model reasoning mode |
| `thinking_honoured` | Whether the model family could honor the requested mode |
| `input_tokens`, `output_tokens`, `total_tokens` | Provider-reported token usage when available |
| `model_calls` | Total model calls consumed by the invocation |
| `retries`, `token_retries` | Validation and token-budget retries |
| `latency_ms` | End-to-end wall-clock latency |
| `prompts` | Prompts actually sent during the strategy execution |
| `revised`, `pre_revision_value` | Contract-strategy revision metadata |
| `cached` | Whether the value was returned from the runtime cache |

Token fields remain `None` when the provider does not report them. Missing usage is never converted to zero or estimated silently.

---

## Execution strategies

Seven strategies are registered and runnable, but the primary experiment uses **five** of them.

| Strategy | Main experiment | Typical model calls | Idea |
|---|:---:|:---:|---|
| `direct` | yes | 1 | Predict the result directly from specification + inputs |
| `cot` | yes | 1 | Produce a structured execution trace before the result |
| `pseudocode` | yes | 1 | Commit to a non-executable procedure, then simulate it |
| `confidence` | yes | 1 | Let the model choose whether to answer directly or reason explicitly |
| `contract` | yes | 2 | Direct-like prediction, then semantic check and optional revision |
| `consistency` | no | 5 by default | Sample several CoT executions and majority-vote |
| `hybrid` | no | 1 | Legacy keyword router between Direct and CoT |

All primary strategies receive the same underlying information: function signature, behavioural specification, and concrete inputs. They differ in the inference procedure applied to that information.

### `direct`

The minimal execution condition. The model is asked for the return value without requiring an explicit procedure or trace. Unless overridden, Direct requests `think=False`.

### `cot`

Requests an explicit state-oriented reasoning trace before the final value. Unless overridden, the strategy requests `think=True`.

### `pseudocode`

First asks the model to formulate a **non-executable** procedure in plain language, then to simulate that procedure on the concrete input. This is intentionally different from code generation: no generated program is executed by Python.

### `confidence`

Uses one call and lets the model choose between an immediate answer and an explicit reasoning trace. It is an adaptive-computation strategy rather than a calibrated probability estimator.

### `contract`

Implements **execute -> check -> revise**:

1. produce the initial candidate under the same Direct-like prediction condition;
2. present that candidate back to the model as unverified;
3. accept it or return a replacement.

By default the prediction stage requests `think=False` and the checking stage requests `think=True`. The initial and final values are retained so beneficial and harmful revisions can be measured separately.

Contract checking is **model-based semantic self-checking**, not formal verification.

### Non-primary strategies

`consistency` remains available for a separate sampling study but is not one of the five primary treatments: it changes repeated sampling and voting and costs roughly five model calls per invocation.

`hybrid` remains available for legacy/routing analysis but is excluded from the primary matrix because its behaviour depends on a hand-written task taxonomy and can be reconstructed from Direct/CoT results plus the router.

---

## Prompt construction

Prompt wording is kept separate from both strategies and backends.

Each primary request serializes the same core information:

```text
Signature: <function signature>
Docstring: <behavioural specification>
Inputs: <concrete arguments>
```

Strategy-specific instructions determine whether the model should answer directly, expose a trace, construct pseudocode, or check a candidate. Final answers use a structured `result` representation where possible, while the parser retains controlled fallback behaviour for outputs that do not exactly follow the requested wrapper.

The prompt layer is intentionally provider-independent: changing from Ollama to Hugging Face or an OpenAI-compatible API does not redefine the function or execution strategy.

---

## Backends and model reasoning modes

All backends implement a common `Backend` interface and return a normalized `GenerateResponse`.

### Ollama

```python
runtime = VirtualRuntime(backend="ollama", model="qwen3:4b")
```

A custom Ollama host can be supplied with `host=...`.

### Hugging Face

```python
runtime = VirtualRuntime(
    backend="hf",
    model="Qwen/Qwen3-4B-Instruct-2507",
)
```

The model is loaded lazily and kept in memory across calls.

### OpenAI-compatible

```python
runtime = VirtualRuntime(
    backend="openai",
    model="<provider-model-id>",
    base_url="https://<provider>/v1",
    api_key="...",
)
```

The backend normalizes provider responses, token usage, and finish reasons. Provider-side tool use is treated as contamination of the specification-only execution condition and is rejected rather than silently scored as model reasoning.

### Thinking-mode resolution

Model families differ in whether reasoning can be enabled or disabled. `model_config.resolve_thinking()` is the single place where a strategy's requested mode is reconciled with the model family's capability.

The runtime records:

```text
requested_think -> effective_think -> thinking_honoured
```

This matters experimentally. For example, a model that always reasons cannot realize a genuinely non-thinking Direct condition; the result metadata makes that mismatch explicit rather than silently treating the conditions as equivalent.

---

## Parsing, validation, and retries

### Parsing

The parser tries to recover the requested structured result, then handles supported literal/scalar forms. It records how the value was recovered:

- `parsed` — structured/literal value recovered;
- `fallback_parsed` — raw text retained as a string fallback;
- `failed` — no value recoverable.

### Structural validation

The runtime keeps three concepts separate:

| Property | Decided by |
|---|---|
| structural/type validity | mechanical check against the declared interface |
| constraint validity | caller-supplied executable predicates |
| semantic correctness | external ground truth or application-specific oracle |

`WARN` mode preserves permissive library behaviour and returns structurally problematic values with diagnostics.

`STRICT` mode applies only meaning-preserving coercions. After retries are exhausted, a known-invalid value is withheld and reported as an execution failure rather than returned as a successful result. The runtime does **not** claim that a structurally valid value is semantically correct.

### Two retry mechanisms

The framework distinguishes:

1. **validation retries** — a complete response was produced but did not satisfy the required structure or constraints;
2. **token-budget retries** — generation ended because the model hit its output limit.

Token usage and model-call counts accumulate across all calls and retries.

---

## Caching and tracing

`VirtualRuntime` includes an in-memory LRU cache. The cache key includes the specification, inputs, model, strategy, thinking request, and relevant generation settings, so experimentally different invocations do not silently share results.

Optional JSONL tracing records the observable execution path, including request data, prompts, raw responses, result metadata, retries, and provider information.

```python
runtime = VirtualRuntime(
    backend="ollama",
    model="qwen3:4b",
    strategy="direct",
    trace_file="traces/run.jsonl",
)
```

Tracing is useful for debugging and reproducibility but is not required for ordinary library use.

---

## CLI

The package installs the `aicomp` command:

```bash
aicomp \
  "max_of_two(a: int, b: int) -> int" \
  "Return the larger of the two integers." \
  --inputs '{"a": 7, "b": 3}' \
  --backend ollama \
  --model qwen3:4b \
  --strategy direct
```

Use:

```bash
aicomp --help
```

for the current full option list. Registered strategies are sourced from the package registry rather than duplicated in the CLI.

---

# Experimental evaluation

The experiment is deliberately separated from the core runtime. Benchmark adapters, ground-truth construction, manifests, scoring, aggregation, and the code-generation baseline are **evaluation infrastructure**, not requirements of the `aicomputing` abstraction.

## Primary benchmark set

| Benchmark | Group | Main `n` | Role |
|---|---|---:|---|
| HumanEval+ | core | 158 | specification execution on programming functions |
| MBPP+ | core | 200 | specification execution on programming functions |
| GSM8K | generalization | 200 | numerical reasoning through the same runtime interface |
| BBH-5 | generalization | 200 | symbolic/logical reasoning through the same runtime interface |

The benchmark groups are reported separately; the experiment does **not** construct one pooled accuracy across all four datasets.

### HumanEval+

- 158 of 164 tasks expose usable precomputed expected outputs without executing a reference implementation during evaluation.
- One invocation is selected per function.
- Invocation selection draws from the combined base + plus input pool.
- Worked `>>>` doctest examples are stripped from the public docstring for the experiment so the target value is not partially demonstrated in the specification.

The frozen main selection contains 153 plus-pool and 5 base-pool invocations.

### MBPP+

MBPP+ does not publish expected outputs for its augmented `plus_input` cases. To include these harder invocations without exposing reference implementations to the evaluated system, the project constructs a frozen ground-truth artifact **offline**.

`scripts/build_mbpp_ground_truth.py` executes the official canonical solution once per task in an isolated preprocessing process and stores only the invocation inputs, expected outputs, pool labels, and provenance. The canonical implementation itself is not serialized into the artifact and is never available to the runtime under evaluation.

Current preprocessing statistics:

- 378 source tasks;
- 345 tasks with at least one successfully resolved canonical invocation;
- 1,040 resolved base inputs;
- 35,633 resolved plus inputs;
- 374 tasks usable by the final adapter after combining frozen augmented outputs with available base cases.

The main experiment selects 200 MBPP+ functions, with 172 plus-pool and 28 base-pool invocations.

Because the original MBPP prompt does not contain a Python signature, the adapter reads the **function interface only** from the reference solution's `def` line. The implementation body is not shown to the evaluated model.

### GSM8K

A fixed 200-item sample is selected from the 1,319-problem test split. GSM8K is treated as a generalization benchmark rather than pooled with the core programming functions.

### BBH-5

The study uses five BIG-Bench Hard tasks rather than the full suite:

- `boolean_expressions`
- `date_understanding`
- `logical_deduction_five_objects`
- `tracking_shuffled_objects_three_objects`
- `web_of_lies`

The main sample is stratified to 40 items per category (200 total).

HotpotQA and the historical per-benchmark oracle scripts are retained for legacy/pilot reproducibility but are not part of the primary experiment.

---

## One invocation per core function

HumanEval+ and MBPP+ contain many test inputs per underlying function. Treating every test case as an independent observation would make functions with deeper generated test suites receive more statistical weight.

The primary experiment therefore uses the **function as the unit of analysis** and selects one concrete invocation per function.

Selection is deterministic and based on `(seed, item_id)` using `blake2b`, not Python's process-salted `hash()`. The selected invocation is stored in the experiment manifest so every model and strategy receives the same item.

---

## Code-generation baseline

The generate-then-execute baseline is run on **HumanEval+ and MBPP+ only**.

```text
Specification execution:
  specification + input -> model/runtime -> output

Code-generation baseline:
  specification -> model -> Python implementation -> execute on same input -> output
```

The baseline receives the same signature and behavioural specification as the specification-execution strategies. It does not receive hidden tests, expected outputs, or canonical implementations.

Generated programs are compiled and executed in a restricted sandbox. Syntax errors, runtime errors, timeouts, and invalid outputs are recorded separately and remain scorable model outcomes.

---

## Reproducible experiment manifests

Benchmark selection happens **once**, when a manifest is built. Runs consume the frozen manifest rather than sampling again.

The manifest records, among other fields:

- experiment and selection-algorithm versions;
- seed (`42` for the current protocol);
- benchmark item and invocation IDs;
- base/plus pool labels;
- dataset revisions and content fingerprints;
- validation mode;
- primary strategies;
- generation settings;
- strategy thinking policy;
- model IDs, providers, and model-specific limits once the run is finalized.

Content fingerprints are recomputed when a run starts. Dataset drift is treated as an error rather than silently evaluating different content under the same item ID.

### Template manifests vs. frozen run manifests

The shipped `pilot-v2.json` and `main-v1.json` define the benchmark selection but intentionally leave `models` / `model_configs` empty until the final provider/model choices are verified.

For a reportable run, create a model configuration file such as:

```json
{
  "model/id-a": {
    "provider": "groq",
    "max_new_tokens": 4096
  },
  "model/id-b": {
    "provider": "cerebras",
    "max_new_tokens": 8192
  }
}
```

Do **not** store API keys in this file.

Then build a frozen manifest from the workspace root:

```bash
python scripts/make_manifest.py \
  --preset pilot \
  --model-configs model_configs.json \
  --out manifests/pilot-v2-run.json
```

The runner refuses an unfrozen model configuration for a reportable run. Provider information is taken from the manifest, and a contradictory CLI provider is rejected.

---

## Running the experiment

The following commands assume the workspace layout containing `aicomputing/`, `scripts/`, `manifests/`, and `data/`.

### 1. Verify the manifest offline

```bash
python scripts/verify_experiment.py \
  --manifest manifests/pilot-v2-run.json
```

This checks the experiment plumbing without making provider calls.

### 2. Inspect the run plan

```bash
python scripts/run_experiment.py \
  --manifest manifests/pilot-v2-run.json \
  --dry-run
```

### 3. Run pilot-v2

```bash
python scripts/run_experiment.py \
  --manifest manifests/pilot-v2-run.json
```


---

## Scoring and failure taxonomy

Capability accuracy is computed over **scorable** outcomes only:

```text
accuracy = correct scorable invocations / all scorable invocations
```

Scorable failures include model-attributable outcomes such as:

- semantic wrong answer;
- parse failure;
- structural validation failure;
- constraint validation failure;
- generated-code syntax/runtime/timeout/invalid-output failures.

Infrastructure failures such as provider errors, provider-side timeouts, or harness failures are reported separately and excluded from the capability denominator. A rate limit must not look like a model-capability failure.

The shared benchmark result schema also records full cost information and Contract revision metadata, enabling paired Direct-vs-enhanced comparisons and reliability-vs-cost analysis.

---

## Result aggregation

Experiment results are written to the experiment-specific results directory and can be processed with the shared analysis scripts:

```bash
python scripts/aggregate_results.py ...
python scripts/make_tables.py ...
```

The result format supports:

- per-model / per-benchmark accuracy;
- paired Direct-vs-enhanced comparisons;
- Contract revision rates and wrong->correct / correct->wrong flips;
- code-generation failure decomposition;
- model calls, latency, and token usage;
- infrastructure-failure reporting.

Exact statistical reporting is performed separately in the thesis analysis rather than embedded into the runtime.

---

## Testing

From the workspace root:

```bash
pytest
```

The test suite covers the core runtime, parsing and validation, strategy behaviour, thinking-mode propagation, cost accounting, benchmark selection, manifests, EvalPlus adaptation, failure classification, and integration paths.

The experiment also has an offline consistency verifier:

```bash
python scripts/verify_experiment.py --manifest manifests/pilot-v2.json
```

No provider calls are required for this verification step.

---

## Workspace layout

The research workspace is currently organized as:

```text
workspace/
├── aicomputing/                 # installable Python package
│   ├── src/aicomputing/         # runtime implementation
│   ├── benchmarks/              # adapters + unified benchmark runner
│   ├── tests/                   # package/experiment tests
│   ├── pyproject.toml
│   └── README.md
├── scripts/                     # manifest, experiment, analysis scripts
├── manifests/                   # frozen benchmark selections
├── data/
    └── mbpp_plus_ground_truth.json

```

Before archival/public release, these components should be versioned together so that an experimental result points to the exact runtime, scripts, manifest, and frozen ground-truth artifact that produced it.

---

## Current status

Implemented:

- specification-driven `VirtualRuntime`;
- modular strategy, prompting, backend, parsing, and validation layers;
- Ollama, Hugging Face, and OpenAI-compatible backends;
- five primary execution strategies plus two non-primary/legacy strategies;
- explicit model reasoning-mode resolution and reporting;
- strict vs. warning validation policies;
- retries, caching, tracing, and full model-call/token accounting;
- unified benchmark adapters and result schema;
- deterministic manifest-based benchmark selection;
- HumanEval+ and MBPP+ augmented-input handling;
- offline MBPP+ ground-truth construction;
- generate-then-execute baseline for the core programming benchmarks;
- automated tests and offline experiment verification.


---

## Research scope

This project investigates whether sufficiently capable language models can act as a **virtual runtime** that realizes function behaviour from specification and input without a separately represented target implementation.

It does **not** claim that current language models provide the determinism, security, efficiency, or semantic guarantees of conventional runtimes. The empirical study measures how close current systems are to that abstraction and how execution strategies trade additional inference cost for reliability.

---

## License

MIT.
