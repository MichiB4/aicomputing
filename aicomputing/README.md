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
pip install -e "./aicomputing"

# Local Hugging Face inference
pip install -e "./aicomputing[hf]"

# OpenAI-compatible APIs
pip install -e "./aicomputing[openai]"

# Benchmark dependencies (HumanEval+, MBPP+, GSM8K, BBH)
pip install -e "./aicomputing[benchmarks]"

# Test dependencies
pip install -e "./aicomputing[test]"

# Typical research/development install
pip install -e "./aicomputing[hf,openai,benchmarks,test]"
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

### System certificates for benchmark commands

The benchmark runners, manifest builder, and MBPP ground-truth builder configure
system certificates at command startup when the optional `truststore` package is
installed. The verifier does so only with `--check-datasets`; its default checks
remain offline. Importing these modules or `bench_args` does not change SSL
configuration.

If `truststore` is absent, Python's normal HTTPS configuration is retained.
Errors importing an installed package or configuring its certificates propagate
so a broken setup is visible. Applications embedding these helpers can call
`bench_args.configure_system_certificates()` explicitly at startup, before
importing HTTPS clients. The core prediction package does not configure SSL.

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
| `max_new_tokens` | `None` by default: no application output-token cap; a positive starting allowance opts in, and truncation retries may increase it |
| `max_retries` | Maximum validation retries |
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
| `reasoning_requests` | Ordered native override records per returned response; not proof of model compliance |
| `reasoning_observations` | Ordered response measurements, recorded separately from requested settings |
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

The minimal execution condition. The model is asked for the return value without requiring an explicit procedure or trace. Direct does not enable or disable native reasoning.

### `cot`

Requests an explicit state-oriented reasoning trace before the final value. The trace instruction does not configure native reasoning.

### `pseudocode`

First asks the model to formulate a **non-executable** procedure in plain language, then to simulate that procedure on the concrete input. This is intentionally different from code generation: no generated program is executed by Python.

### `confidence`

Uses one call and lets the model choose between an immediate answer and an explicit reasoning trace. It is an adaptive-computation strategy rather than a calibrated probability estimator.

### `contract`

Implements **execute -> check -> revise**:

1. produce the initial candidate under the same Direct-like prediction condition;
2. present that candidate back to the model as unverified;
3. accept it or return a replacement.

Both stages use the same configured backend-native reasoning settings. Neither stage has an application output-token cap by default. An explicit positive `max_new_tokens` starts both stages at that allowance; each stage can increase it independently after truncation. The initial and final values are retained so beneficial and harmful revisions can be measured separately.

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

### Explicit native reasoning settings

Prompting strategy and native reasoning configuration are independent. The
package does not infer support, activation or successful disabling from model
names. Configure only the native option understood by the selected backend:

| Backend | Runtime constructor option | CLI option |
|---|---|---|
| OpenAI-compatible | `reasoning_effort="<value>"` | `--reasoning-effort VALUE` |
| Ollama | `think=False`, `True`, or a level string | `--ollama-think false`, `true`, or `VALUE` |
| HuggingFace | `enable_thinking=False` or `True` | `--no-hf-thinking` or `--hf-thinking` |

Omitting an option sends no override. Explicit unsupported settings surface as
errors; they are not silently removed or retried without the setting. Whether
a provider/model honors a setting requires separate verification.


All adapters expose the same capability API, independently of model names:

```python
from aicomputing.backends import create_backend

backend = create_backend("ollama", model="<model-id>", host="http://127.0.0.1:11434")
capabilities = backend.reasoning_capabilities(discover=True)
native_options = backend.reasoning_options(False, capabilities=capabilities)
# Pass native_options into a new backend/runtime constructor for this same model/endpoint.
```

A capability record identifies its backend, model, native parameter, endpoint,
source, accepted values and advertised default. Its kind is unknown, unsupported,
toggle, fixed, levels or choices. These describe accepted controls, not verified
internal behavior. A singleton value means one accepted setting.

Discovery is explicit. Without discover=True it performs no metadata requests.
Ollama reads raw /api/show JSON with a finite timeout, preserving fields older
SDK response models discard. An omitted thinking object leaves support unknown; malformed controls raise. HF does
not load a tokenizer/model for discovery; a loaded tokenizer without a chat
template makes this adapter's template control unavailable. Other HF cases and
generic OpenAI-compatible endpoints remain unknown unless usable controls are
supplied as a ReasoningCapabilities declaration. Declarations are labelled
user_declared and are not provider verification.

reasoning_options returns native constructor keywords without changing the
backend. It checks value types and model/endpoint scope. None returns no override.
Unknown support requires allow_unknown=True for an experimental request; this
cannot bypass known unavailable controls or unsupported values. Native strings
are preserved literally; low is not interpreted as off.

Existing native constructor/CLI options remain unverified passthroughs. This
additive API does not yet enforce discovery in those existing execution paths
or qualify a model's observed thinking behavior.

`ExecuteResult.reasoning_requests` preserves one entry for each returned model
response, in order: `{}` means no native override was sent, and `None` means
the backend did not record it. A cache hit has `[]` because it made no new model
call. These records do not include failed internal SDK attempts and do not
prove the model's internal reasoning behavior.

`reasoning_observations` records Ollama's thinking-field type and character
count, and whether the original content contains opening or closing `<think>`
tags. Observations are captured before text cleanup and preserve returned-call
order through retries, traces, and canonical result CSVs. Other backends report
`None` per response until they implement observation recording. Cache hits have
`[]`. These measurements do not infer an actual mode or certify compliance;
ordinary CoT text can be requested independently of native thinking.

`ExecuteRequest.think`, the inferred result fields and universal CLI
`--think`/`--no-think` flags are removed. Output-token limits are separate from
native reasoning: model names and prompting strategies do not introduce a cap.
The compatibility helper `get_token_budget(model, extended=...)` returns `None`.

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

Both modes use the same limited conversions; neither truncates a fractional float to an integer. Under `STRICT`, after retries are exhausted, a known-invalid value is withheld and reported as an execution failure rather than returned as a successful result. The runtime does **not** claim that a structurally valid value is semantically correct.

### Two retry mechanisms

The framework distinguishes:

1. **validation retries** — output recovery, type/constraint validation, or generated-program execution failed;
2. **token-budget retries** — generation ended because the model hit its output limit.

Token usage and model-call counts accumulate across all calls and retries.

### Output-token allowances

`max_new_tokens=None` (the default), or JSON `null` in a manifest, means no
application output-token cap. OpenAI-compatible requests omit the output-limit
field; Ollama receives `num_predict=-1`. The Hugging Face backend must provide a
finite generation stopping bound: it uses the model's declared context capacity
minus the prompt length, falling back to tokenizer metadata only when the model
does not declare a capacity. An explicit allowance is required if neither source
provides a finite capacity. Provider defaults, model output limits, and context windows
still apply; omitting an application cap does not guarantee unlimited generation.

A positive `max_new_tokens` or `--max-tokens` value is an explicit **starting
allowance**, not a hard ceiling. A token-limited response can trigger up to five
additional calls, doubling that allowance each time. Each call starts generation
again and its usage is counted. With no application cap there is no allowance to
increase, so a token-limited response is kept as truncated without a token-budget
retry or parsing its incomplete prefix. Normal validation retries are separate
and can still request a new answer with feedback.

---

## Caching and tracing

`VirtualRuntime` includes an in-memory LRU cache. The cache key includes the specification, inputs, model, strategy, relevant generation settings, backend class and declared backend configuration (including native reasoning overrides). A backend must explicitly supply a cache state; unknown custom backends bypass caching by default. Credentials are excluded from that state, so this is not an account- or environment-isolation guarantee.

Optional JSONL tracing records request and result metadata, including the ordered `reasoning_requests`. Prompt/raw-text fields describe the strategy result; they are not a complete history of every internal call or retry.

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
| HumanEval+ | core | 157 | specification execution on programming functions |
| MBPP+ | core | 200 | specification execution on programming functions |
| GSM8K | generalization | 200 | numerical reasoning through the same runtime interface |
| BBH-5 | generalization | 200 | symbolic/logical reasoning through the same runtime interface |

The benchmark groups are reported separately; the experiment does **not** construct one pooled accuracy across all four datasets.

### HumanEval+

- 158 of 164 tasks expose precomputed expected outputs without executing a reference implementation during evaluation.
- 157 of those tasks have signatures supported by the runtime. `HumanEval/12` uses `Optional[str]` and is ineligible.
- One invocation is selected per function.
- Invocation selection draws from the combined base + plus input pool.
- Worked `>>>` doctest examples are stripped from the public docstring for the experiment so the target value is not partially demonstrated in the specification.

The frozen main selection contains 152 plus-pool and 5 base-pool invocations.

### MBPP+

MBPP+ does not publish expected outputs for its augmented `plus_input` cases. To include these harder invocations without exposing reference implementations to the evaluated system, the project constructs a frozen ground-truth artifact **offline**.

`scripts/build_mbpp_ground_truth.py` executes the official canonical solution once per task in an isolated preprocessing process and stores only the invocation inputs, expected outputs, pool labels, and provenance. The canonical implementation itself is not serialized into the artifact and is never available to the runtime under evaluation.

Current preprocessing statistics (cache format 3, `typed-v1`):

- 378 source tasks;
- 362 tasks with at least one successfully resolved canonical invocation;
- 1,120 resolved base inputs;
- 37,992 resolved plus inputs;
- 375 tasks usable by the final adapter after combining frozen augmented outputs with available base cases.

The main experiment selects 200 MBPP+ functions, with 184 plus-pool and 16 base-pool invocations.

Because the original MBPP prompt does not contain a Python signature, the adapter reads the **function interface only** from the reference solution's `def` line. The implementation body is not shown to the evaluated model.

#### Restore the frozen cache

The published artifact is `data/mbpp_plus_ground_truth.json.gz`.
`data/mbpp_plus_ground_truth.sha256` records SHA-256 checksums for both the
archive and the extracted JSON. The benchmark loader reads the extracted
`data/mbpp_plus_ground_truth.json`; restore it before dataset verification or
experiment runs.

From the repository root, run this PowerShell block:

```powershell
$ErrorActionPreference = 'Stop'
$hashes = @{}
Get-Content -LiteralPath 'data/mbpp_plus_ground_truth.sha256' | ForEach-Object {
    $hash, $name = $_ -split '\s+', 2
    $hashes[$name] = $hash
}
$archive = 'data/mbpp_plus_ground_truth.json.gz'
$cache = 'data/mbpp_plus_ground_truth.json'
if ((Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash -ne $hashes['mbpp_plus_ground_truth.json.gz']) {
    throw 'Compressed MBPP cache checksum mismatch'
}
if (-not (Test-Path -LiteralPath $cache)) {
    python -m gzip -d $archive
    if ($LASTEXITCODE -ne 0) { throw 'MBPP cache extraction failed' }
}
if ((Get-FileHash -LiteralPath $cache -Algorithm SHA256).Hash -ne $hashes['mbpp_plus_ground_truth.json']) {
    throw 'Extracted MBPP cache checksum mismatch'
}
```

An existing JSON cache is verified and reused. A checksum mismatch raises an
error without replacing an existing file. Extraction retains the gzip archive.
Version the gzip and checksum files; keep the extracted JSON and obsolete
`mbpp_plus_ground_truth.v1.backup.json` local. Rebuilding ground truth is a
separate preprocessing operation and can change the frozen observations.

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

Each generated program uses one shared namespace, so its functions can access
sibling helpers, recursive definitions, constants and import aliases. The existing
restricted builtins and allowed modules remain available.
Standard built-in exception classes are available for `raise` and `try/except`.
Other builtin and import restrictions remain in effect.
Allowed absolute imports use Python's standard importer, so nonexistent
submodules raise normal import errors. Relative imports are rejected because
the generated program has no package context.
Helper class definitions and custom exceptions are supported, including ordinary
`object` inheritance. The namespace has a stable `__name__` for class metadata.
`callable`, `bytes` and `getattr` are also available. Returning custom class
instances remains subject to the result-transfer restrictions described below.
Source extraction removes surrounding code fences without cutting at main-guard
text. A normal `if __name__ == "__main__":` block remains inactive in the named
worker namespace; its source is still parsed, so syntax errors remain failures.
The baseline validates its declared Python function header before model generation
and uses Python's normalized function name. Invalid headers fail immediately;
this check does not execute default expressions or change the general oracle's
signature-parsing policy.

The baseline has no application output-token cap by default. With an explicit
positive starting allowance, a token-limited response triggers up to five
additional calls, doubling the allowance each time and recording usage from
every response. With no application cap, it does not repeat the same request
as a token-budget retry. Partial code is never executed after a truncated final
response; the outcome is `generated_code_truncated` with `exec_status="truncated"`,
a scorable failure. A genuine successful return of `None` remains distinct from
this missing result.

Runtime validation retries are separate. When validation triggers another attempt,
the baseline includes the previous generated response and validation error and
requests corrected Python code for the original specification. Syntax errors, runtime
errors and execution timeouts also use this retry allowance, even for return types
that permit `None`. Exhausted execution failures remain unsuccessful and scorable;
a successful execution returning `None` is validated normally. Benchmark expected
answers are not supplied as feedback. With the default one validation retry,
an explicit allowance and repeated token-budget exhaustion can use two attempts
of six backend calls each. Internal SDK HTTP retries are separate from these
counts. The generated-code process timeout
remains 10 seconds and does not limit model generation time. Aggregated summaries
append `exec_truncated` and include those failures in `exec_failure_rate`.

`CodeGenBaseline(timeout=...)` accepts positive, finite Python integers or floats
in seconds and normalizes them to a float. Invalid settings are rejected during
construction: this includes booleans, strings, `None`, zero, negative values,
non-finite numbers and integers too large to convert to a finite float.

Worker environment setup, process startup and result-transfer failures propagate
instead of triggering code-correction retries. The benchmark wrapper records
these as `harness_error` and excludes them from accuracy. An unexplained worker
exit is handled the same way. This includes results that cannot be serialized
for transfer; the runtime does not label transfer failures as program exceptions.
An explicit worker program-error report remains a scorable generated-code failure.


---

## Reproducible experiment manifests

Benchmark selection happens **once**, when a manifest is built. Runs consume the frozen manifest rather than sampling again.

The manifest records, among other fields:

- experiment and selection-algorithm versions;
- seed (`42` for the current protocol);
- benchmark item and invocation IDs;
- base/plus pool labels;
- available dataset revision metadata and selected-content fingerprints;
- validation mode;
- primary strategies;
- generation settings;
- explicit per-model native reasoning overrides (or no override);
- model IDs and resolved provider/backend/URL settings once the run is finalized.

Default manifest construction requires every selected item to resolve and receive a content fingerprint. At run time, missing fingerprints, missing or unexpected items, duplicate IDs and changed selected content are errors before model execution. The direct benchmark runner's explicit `--allow-content-drift` debug override permits continuation with warnings; those runs must not be reported as reproductions. The experiment orchestrator does not expose this override.

Pool fingerprints compare the collection of item IDs. They do not establish that the contents or ordering behind positional IDs are unchanged; selected-content verification provides that separate check. Dataset revision metadata describes what the loader exposes, not a verified source commit. HumanEval+ and MBPP+ currently record that metadata as unknown because their loaders return items without source metadata.

Rebuilding with matching selection inputs and rules reproduces the selection, not necessarily the complete manifest bytes: the creation timestamp changes, for example. Reusing a manifest also does not guarantee identical model outputs.

### Template manifests vs. frozen run manifests

Current code requires integer manifest schema version **2** and selection compatibility version **4**. Selection compatibility also covers content-fingerprint encoding, so schema version 2 alone is insufficient. Manifests with earlier selection versions must be rebuilt; reproduce historical runs with their corresponding code revision.

Historical `pilot-v2.json` and `main-v1.json` use the old schema. Current pilot/main presets use experiment IDs `pilot-v3` / `main-v2`; those names do not establish compatibility. Generate compatible manifests explicitly. Historical files are not automatically converted. Earlier code treated `max_new_tokens: null` as a model/strategy starting allowance; current code treats it as no application cap. Reproduce historical runs with their original code revision and recorded settings.

For a reportable run, create a model configuration file such as:

```json
{
  "model/id-a": {
    "provider": "groq",
    "max_new_tokens": null,
    "reasoning": {}
  },
  "model/id-b": {
    "provider": "cerebras",
    "max_new_tokens": null,
    "reasoning": {}
  }
}
```

Do **not** store API keys in this file.

Keep `max_new_tokens` null to avoid an application cap. Set a positive value
only when you want an explicit starting allowance; token-budget retries may
increase it. A per-model null overrides a numeric global manifest allowance.

Then build a frozen manifest from the workspace root:

```bash
python scripts/make_manifest.py \
  --preset pilot \
  --model-configs model_configs.json \
  --out manifests/pilot-v3-run.json
```

The runner refuses an unfrozen model configuration for a reportable run. Provider/backend/URL and native settings are taken from the manifest; contradictory CLI settings are rejected. `reasoning: {}` records no override. Experiment temperature is fixed to 0.0; starting token allowances and validation retry counts are validated separately. API keys remain CLI/environment settings and must not be frozen into the manifest.

---

## Running the experiment

The following commands assume the workspace layout containing `aicomputing/`, `scripts/`, `manifests/`, and `data/`.

### 1. Verify the manifest offline

```bash
python scripts/verify_experiment.py \
  --manifest manifests/pilot-v3-run.json
```

This checks synthetic plumbing and manifest metadata without dataset loads or provider calls. To also resolve real datasets, verify recorded content and rebuild selections, add `--check-datasets`; that mode may download data. Synthetic success alone does not verify the real dataset or model behavior.

### 2. Inspect the run plan

```bash
python scripts/run_experiment.py \
  --manifest manifests/pilot-v3-run.json \
  --dry-run
```

### 3. Run pilot-v3

```bash
python scripts/run_experiment.py \
  --manifest manifests/pilot-v3-run.json
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

The shared benchmark result schema also records available token and model-call accounting and Contract revision metadata, enabling paired Direct-vs-enhanced comparisons and reliability-vs-cost analysis.

---

## Result aggregation

Experiment results are written to the experiment-specific results directory. Summarize the pilot run from the workspace root:

```bash
python scripts/aggregate_results.py \
  --raw aicomputing/results/pilot-v3 \
  --out results/aggregated/pilot-v3.csv
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

The test suite covers the core runtime, parsing and validation, strategy behaviour, explicit native-setting propagation, cost accounting, benchmark selection, manifests, EvalPlus adaptation, failure classification, and integration paths.

The experiment also has an offline consistency verifier:

```bash
python scripts/verify_experiment.py
```

This default uses synthetic fixtures and no dataset/provider calls. `--manifest PATH` additionally inspects current-version metadata; `--manifest PATH --check-datasets` opts into real dataset checks, which may download data but still use fake model responses.

---

## Workspace layout

The research workspace is currently organized as:

```text
workspace/
├── aicomputing/                 # installable Python package
│   ├── src/aicomputing/         # runtime implementation
│   ├── benchmarks/              # adapters + unified benchmark runner
│   ├── pyproject.toml
│   └── README.md
├── tests/                       # package/experiment tests
├── scripts/                     # manifest, experiment, analysis scripts
├── manifests/                   # frozen benchmark selections
├── data/
    ├── mbpp_plus_ground_truth.json.gz
    ├── mbpp_plus_ground_truth.sha256
    └── mbpp_plus_ground_truth.json  # extracted locally

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
