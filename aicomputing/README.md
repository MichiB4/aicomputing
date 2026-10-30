# AI Computing (`aicomputing`)

> **Predictive execution with language models:** describe a function by its interface and behaviour, provide concrete inputs, and let a model-mediated runtime produce the return value — without generating or executing a target implementation first.

`aicomputing` is a research prototype for exploring **specification-driven execution**. In conventional programming, a specification is translated into an implementation and that implementation is executed. Here, the specification itself is the persistent program artifact:

```text
Conventional:  specification -> implementation -> interpreter/runtime -> output
AI Computing:  specification + input -> model-mediated runtime -> output
```

The distinction is not that the model “does no computation.” The distinction is architectural: no separately represented executable implementation of the target function is required at invocation time.

> **Standalone package:** `VirtualRuntime` is the oracle interface. Package use requires no benchmark datasets, manifests, or reference answers. The separate [evaluation guide](https://github.com/MichiB4/aicomputing/blob/master/docs/evaluation.md) covers model comparisons, benchmark setup, and research protocols.

---

## Quick start

```python
from aicomputing import ExecuteRequest, VirtualRuntime

with VirtualRuntime(
    backend="ollama",
    model="qwen3:4b",
    strategy="direct",
    validation_mode="strict",
) as runtime:
    request = ExecuteRequest(
        signature="repeat(s: str, n: int) -> str",
        docstring="Return the string s repeated n times.",
        inputs={"s": "ab", "n": 3},
        return_type=str,
        answer_format="framed_v1",
    )

    result = runtime.execute(request)

print(result.result)             # model-produced Python value, e.g. 'ababab'
print(result.validation_status)  # structural runtime status
print(result.raw_text)           # original model response
```

No Python implementation of `repeat` is supplied to the runtime. The model receives the function specification and concrete arguments and predicts the value the function should return.

Use a runtime as a context manager, as above, or call `runtime.close()` when
finished. The runtime closes a backend it created from a name and clears its
prediction cache. Repeated close calls are safe; executing or entering a closed
runtime raises `RuntimeError`.

A backend instance supplied by the caller remains caller-owned. This allows
multiple runtimes to share one backend: closing a runtime leaves that backend
open. The caller must close it after all runtimes have finished, or use
`with create_backend(...) as backend:` around their lifetimes.

Backend cleanup releases owned resources without loading a client or model.
OpenAI clients and explicit-host Ollama transports are closed; Ollama's shared
default SDK client remains open. Hugging Face drops its model and tokenizer
references without clearing process-wide GPU allocator caches. A later direct
backend call may recreate its client or reload its model. Close resources only
between generation calls.

This is a **probabilistic execution mechanism**, not a semantic guarantee. Passing structural validation means that the value fits the declared interface; it does not prove that the value is correct.

Use `answer_format="framed_v1"` for an explicit final-answer block. The same
prompt and parser rules apply to every adapter and prediction strategy:

```text
BEGIN FINAL ANSWER
{"result": "ababab"}
END FINAL ANSWER
```

Strategies that request an explanation put it before the block. The parser
reads the complete object inside the block, preserving Python types through
literal notation when JSON is insufficient. Missing or repeated blocks,
incomplete values, duplicate keys and trailing commentary fail parsing. It
does not fill missing digits or parentheses. The default remains `legacy`
for existing callers and manifests. The CLI offers `--answer-format framed_v1`.

Framed prompts repeat the required answer structure immediately after the
actual task. A procedure's `return` step or a `State:` record does not supply
the final answer. Models must still output the complete block; an omitted
answer remains a parsing failure.

An opt-in alternative is answer_format="framed_text_v1", with max_retries=0.
It uses the same prompts and boundaries, preserves decoded values without
output-type conversion or validation, and retains an isolated malformed result
field as an unchanged string. An unquoted 1234? becomes the string "1234?",
with parse_status="fallback_parsed" and a parse_error diagnostic. Missing or
ambiguous boundaries and additional outer fields still fail. No digits or
delimiters are repaired. Legacy and framed_v1 behavior is unchanged.
Reference-answer type comparisons belong to the separate evaluator.
The CLI also accepts `--answer-format framed_text_v1` and automatically sets
validation retries to zero; other answer formats keep one validation retry.

For OpenAI-compatible endpoints, transport settings can be supplied explicitly:

```python
runtime = VirtualRuntime(
    backend="openai", model="YOUR_MODEL", strategy="direct",
    timeout_seconds=100, sdk_max_retries=0, request_diagnostics=True,
)
```

This sets a 100-second HTTP phase/inactivity timeout, rather than a total
request deadline. Diagnostics include a client request ID, the provider's
request ID when available, and elapsed time. Successful call records appear
in `result.transport_diagnostics`; communication errors carry `.diagnostics`,
including timeout phases when available. Diagnostic records exclude prompts,
credentials and exception messages. SDK retries are separate from
`ExecuteRequest.max_retries` (validation retries); set that to zero too for
a single-attempt evaluation. Omitting the new transport settings retains the
existing SDK defaults, including two SDK retries.

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/MichiB4/aicomputing.git
cd aicomputing

# Core package (Ollama backend)
python -m pip install ./aicomputing

# Local Hugging Face inference
python -m pip install "./aicomputing[hf]"

# OpenAI-compatible APIs
python -m pip install "./aicomputing[openai]"
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

The `openai` backend works with services exposing a compatible chat-completions API, including hosted providers and local servers such as vLLM or LM Studio. Supply provider credentials through the environment or backend configuration. Keep credentials out of source control.


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
| `scorable` | Evaluation-oriented failure classification; does not establish answer correctness |
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

Seven execution strategies are available to package users. Strategy availability is independent of which strategies a particular evaluation compares. All strategies use the function signature, behavioral specification, and concrete inputs; they differ in how they ask the model to predict the return value.

| Strategy | Typical model calls before retries | Behavior |
|---|---:|---|
| `direct` | 1 | Predict the return value directly |
| `cot` | 1 | Trace the specified behavior before returning the value |
| `pseudocode` | 1 | Write a non-executable procedure, then simulate it |
| `confidence` | 1 | Let the model choose direct answering or explicit reasoning |
| `contract` | 2 | Predict, check the unverified candidate, and optionally revise |
| `consistency` | 5 by default | Sample multiple CoT answers and vote |
| `hybrid` | 1 | Route between Direct and CoT using task keywords |

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

### `consistency`

Makes repeated CoT predictions and chooses an answer by voting. The default is five samples, so one invocation normally consumes five model calls before retries.

### `hybrid`

Routes a request to Direct or CoT using fixed task-keyword heuristics. This routing is separate from backend-native reasoning settings.

---

## Prompt construction

Prompt wording is kept separate from both strategies and backends.

Each prediction request includes the same core information:

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

1. **validation retries** — output recovery or type/constraint validation failed;
2. **token-budget retries** — generation ended because the model hit its output limit.

Token usage and model-call counts accumulate across all calls and retries.

### Output-token allowances

`max_new_tokens=None` (the default) means no application output-token cap. OpenAI-compatible requests omit the output-limit
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

## Evaluation guide

The optional [evaluation guide](https://github.com/MichiB4/aicomputing/blob/master/docs/evaluation.md) documents benchmark adapters, dataset preparation, manifests, scoring, and experiment reproduction. These tools use the oracle package to measure model and strategy performance. They are not required to call `VirtualRuntime` from an application.

## License

MIT.
