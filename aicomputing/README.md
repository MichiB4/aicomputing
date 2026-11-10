# AI Computing (`aicomputing`)

> **Predictive execution with language models:** describe a function by its interface and behaviour, provide concrete inputs, and let a model-mediated runtime produce the return value — without generating or executing a target implementation first.

`aicomputing` is a research prototype for exploring **specification-driven execution**. In conventional programming, a specification is translated into an implementation and that implementation is executed. Here, the specification itself is the persistent program artifact:

```text
Conventional:  specification -> implementation -> interpreter/runtime -> output
AI Computing:  specification + input -> model-mediated runtime -> output
```

The distinction is not that the model “does no computation.” The distinction is architectural: no separately represented executable implementation of the target function is required at invocation time.

The optional `code_execute` strategy also supports the conventional path shown
above: generate a Python implementation and execute it locally. It is available
through the same package API and is used by the separate evaluator as a baseline.

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

If inference, runtime setup, or a context-manager body raises, cleanup errors are
reported as warnings and the original error propagates unchanged. A cleanup
failure on its own still raises. Failed runtime cleanup leaves the runtime open
so `close()` can be retried. Backend context managers use the same error policy.

A backend instance supplied by the caller remains caller-owned. This allows
multiple runtimes to share one backend: closing a runtime leaves that backend
open. The caller must close it after all runtimes have finished, or use
`with create_backend(...) as backend:` around their lifetimes.

Backend cleanup releases owned resources without loading a client or model.
OpenAI and Ollama backends each own a lazily created client and close its
transport. Hugging Face drops its model and tokenizer
references without clearing process-wide GPU allocator caches. A later direct
backend call may recreate its client or reload its model. Close resources only
between generation calls.

This is a **probabilistic execution mechanism**, not a semantic guarantee. Passing structural validation means that the value fits the declared interface; it does not prove that the value is correct.

Every prediction strategy requests the same final-answer block:

```text
BEGIN FINAL ANSWER
{"result": "ababab"}
END FINAL ANSWER
```

Strategies that request an explanation put it before the block. The parser
reads the complete object inside the block, preserving Python types through
literal notation when JSON is insufficient. Missing or repeated blocks, empty
or ambiguous fields, duplicate keys, and trailing commentary fail parsing.
The protocol is fixed; `result.answer_format` records its version as
`framed_answer_v1`. Code generation records its separate Python-source protocol.

Framed prompts repeat the required answer structure immediately after the
actual task. A procedure's `return` step or a `State:` record does not supply
the final answer. Models must still output the complete block; an omitted
answer remains a parsing failure.

When a nonempty result field is clearly isolated but is not a valid literal,
the parser preserves its text unchanged. For example, `{"result": 7?}` recovers
the string `"7?"`, with `parse_status="fallback_parsed"` and a diagnostic.
It never turns that text into the number 7 or repairs missing digits or delimiters.
`{"result": }` has no recoverable answer and fails parsing. The full model
response remains available in `raw_text` in every case. Leading thought blocks
and surrounding whitespace are retained there; parsing uses a separate cleaned
view of the response.

Output validation applies after parsing. To preserve parsed values without output conversion or checks, explicitly
set `validation_mode="none"`; use `max_retries=0` to avoid recovery retries.
Evaluation uses this combination and performs its own reference comparisons.
The CLI exposes the same output policy through `--validation-mode`.

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
a single-attempt evaluation. SDK retries default to zero; pass
`sdk_max_retries=N` to enable them explicitly. An omitted `timeout_seconds`
inherits the SDK timeout.

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/MichiB4/aicomputing.git
cd aicomputing

# Provider-independent core (no required third-party dependencies)
python -m pip install ./aicomputing

# Local Ollama server
python -m pip install "./aicomputing[ollama]"

# Local Hugging Face inference
python -m pip install "./aicomputing[hf]"

# OpenAI-compatible APIs
python -m pip install "./aicomputing[openai]"
```

### Ollama

Install the `ollama` extra above. The Ollama server must also be installed and
running locally:

```bash
ollama pull qwen3:4b
ollama serve
```

### Hugging Face

The `hf` backend runs `transformers` models locally. A CUDA-capable GPU is strongly recommended for non-trivial models.

### OpenAI-compatible APIs

Install the `openai` extra above; it requires OpenAI SDK 2.30.0 or newer.

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

A custom backend implementing only `generate(prompt)` inherits a chat adapter
that accepts exactly one `{"role": "user", "content": "..."}` message. It forwards
the string content unchanged, including an empty string. Other roles, multiple
messages and additional message fields require a `generate_chat` implementation;
the default adapter rejects them instead of discarding their meaning.

### Execution lifecycle

For a call `runtime.execute(request)`, the runtime roughly performs:

1. validate the request, function signature and supplied arguments;
2. resolve the selected strategy, model and output requirements;
3. check the result cache;
4. build the strategy-specific prompt;
5. call the backend/model;
6. parse the model response into a candidate Python value;
7. apply type and constraint validation;
8. retry with validation feedback if permitted;
9. return a structured `ExecuteResult` and optionally write a trace.

The model output is therefore a **candidate** until it crosses the parsing and validation boundary.

---

## Public API

`VirtualRuntime` requires an explicit backend name or `Backend` instance, for
example `VirtualRuntime(backend="openai", model="YOUR_MODEL")`.

### `ExecuteRequest`

The stable core of a request is:

| Field | Purpose |
|---|---|
| `signature` | Python-style function interface |
| `docstring` | Nonblank natural-language behavioural specification |
| `inputs` | Concrete arguments keyed by parameter name |
| `constraints` | Optional predicates checked on the returned value |
| `strategy` | Optional per-call strategy override |
| `model` | Optional per-call model override |
| `temperature`, `top_p` | Generation controls |
| `max_new_tokens` | Fixed output-token limit per call; `None` omits the application cap |
| `max_retries` | Maximum validation retries |
| `validation_mode` | `"strict"` (default), `"warn"`, or `"none"`; a request may inherit its runtime's setting |

`docstring` must be a string containing at least one non-whitespace character.
It is checked at construction and again before execution, including cache lookup,
in every output mode. Valid descriptions are passed through unchanged.

Inputs must match the signature in every output mode. The `validation_mode`
setting controls only the returned output; it is independent of answer framing.
The signature's return annotation is the only output-type declaration. For
example, `f(x: int) -> list[int]` requires a list of integers. Without a return
annotation, any decoded type is allowed; caller-supplied constraints still apply
in strict and warn modes. `request.return_type` exposes the derived type as a
read-only property and is not a constructor setting.

Only `None` inherits a model or strategy default. Overrides must otherwise be
nonempty strings; blank names, booleans and other types are rejected rather than
replaced with defaults. Names are passed literally. Runtime strategy configuration
also accepts `Strategy` instances; a request may select its runtime's custom
strategy by that instance's name.

`constraints` is a list of `Constraint` objects. Each has a nonempty string
`name`, a callable `predicate`, and an optional string `message` (default `""`).
Definitions are checked at construction and again before execution, including
after mutation and in `none` mode. These checks never call a predicate. Output
predicates run only in strict and warn modes; their exceptions propagate unchanged.
Predicates must return an actual `bool` (`True` or `False`). Other return values
raise `TypeError` as a caller error and do not trigger validation retries.

`temperature` and `top_p` must be finite built-in integers or floats; booleans
and numeric strings are rejected. Temperature must be nonnegative, with no shared
upper limit. `top_p` must lie between 0 and 1 inclusive. Requests are checked at
construction and again before execution; built-in backends also check direct
calls before creating a client or loading a model. Providers may impose stricter
limits.

### `ExecuteResult`

`ExecuteResult` contains the returned value and execution diagnostics. The
evaluator assigns benchmark correctness, accuracy eligibility, and
benchmark-specific outcome labels after receiving this result. Important fields include:

| Field | Meaning |
|---|---|
| `result` | Recovered or executed return value; may be withheld under strict validation |
| `raw_text` | Response text retained by the final strategy attempt; multi-stage strategies can combine replies |
| `model`, `strategy` | Effective model and strategy |
| `parse_status` | Prediction parsing: `parsed`, `fallback_parsed`, or `failed`; `None` for code execution or unrecorded parsing |
| `parse_error` | Prediction-parsing diagnostic, including successful fallback recovery |
| `provider_finish_reason` | Original provider finish reason for the response supplying the selected answer |
| `verdict_parse_status`, `verdict_parse_error` | Contract-check parsing outcome and diagnostic, independent of answer parsing; `None` for other strategies |
| `verdict_finish_reason` | Original provider finish reason for the contract checker response; `None` for other strategies |
| `execution_error` | Source-extraction or local-execution diagnostic; `None` for predictions |
| `validation_status` | `valid`, `not_checked`, `invalid_type`, `invalid_constraint`, `unparsed`, `execution_failed`, or infrastructure-only `unavailable` |
| `failure_type` | Structured reason for an execution failure, if any |
| `reasoning_requests` | Ordered native override records per returned response; not proof of model compliance |
| `reasoning_observations` | Ordered response measurements, recorded separately from requested settings |
| `input_tokens`, `output_tokens`, `total_tokens` | Provider-reported token usage when available |
| `model_calls` | Model responses recorded across stages and validation retries |
| `retries` | Additional strategy executions initiated by validation retries |
| `latency_ms` | Invocation time through checks and generation/execution, excluding result construction and tracing |
| `prompts` | Final-attempt prompts; repeated samples can share a single entry |
| `revised`, `pre_revision_value` | Contract-strategy revision metadata |
| `cached` | Whether the value was returned from the runtime cache |

A failed or truncated contract check can retain a valid candidate.
`provider_finish_reason` follows the response supplying the selected answer;
the `verdict_*` fields describe the checker separately. New evaluation CSV/JSON
rows include both sets of diagnostics without changing correctness or scoring
eligibility.

`ParseStatus` and `ValidationStatus` are shared string enums exported from
`aicomputing`. Constructors accept an enum member or an exact valid string,
normalize it to the enum, and reject unknown labels. JSON, CSV and string
formatting retain the values shown above. `ParseOutcome.status` and strategy
parsing statuses use the same `ParseStatus` definition. The result container
checks label validity; it does not enforce cross-field consistency or correctness.

The runtime treats an explicit `ParseStatus.FAILED` from a prediction strategy as failed
recovery, even when that strategy attaches a value. That value is withheld and
normal recovery retries remain available. An unrecorded parsing status (`None`)
still permits a non-`None` custom-strategy value, subject to the output policy.

Field scope matters when inspecting retries or cache hits:

| Scope | Fields |
|---|---|
| Invocation identity | `model`, `strategy` |
| Final attempt | `result`, `raw_text`, `prompts`, routing/revision fields, `provider_finish_reason`, warnings, statuses, error messages, `answer_format`, generated source and execution details |
| Entire invocation | `retries`, token counts, `model_calls`, `latency_ms`, and response-order reasoning/transport lists |
| Cache hit | `cached=True`; retains the original final-attempt evidence, records zero new model calls/tokens/retries and empty response-order lists; latency measures the current lookup |

The result is not a complete attempt or raw-response history. For example, a
successful retry retains the last reply while token counts include both replies.
Cached `execution_details` describes the earlier worker run, not a new execution.
The evaluator records the complete dispatched-call and attempt history separately.

Token fields remain `None` when reporting is missing or incomplete. Missing
usage is never converted to zero or estimated silently.
If a custom strategy omits `call_stats` on any attempt, invocation token totals
remain unknown even when other attempts report usage. `model_calls` and the
response-order diagnostic lists still contain only reported records; omitted
calls are not inferred.

---

## Execution strategies

Eight execution strategies are available to package users. Strategy availability is independent of which strategies a particular evaluation compares. All strategies use the function signature, behavioral specification, and concrete inputs. Seven predict the return value through the model; `code_execute` generates and runs Python.

Strategies and prompt builders accepting `include_fewshot` require a `bool`
(`True` or `False`), defaulting to `True`. `False` omits the strategy's worked
examples; shared value-notation examples remain.

| Strategy | Typical model calls before retries | Behavior |
|---|---:|---|
| `direct` | 1 | Predict the return value directly |
| `cot` | 1 | Trace the specified behavior before returning the value |
| `pseudocode` | 1 | Write a non-executable procedure, then simulate it |
| `confidence` | 1 | Let the model choose direct answering or explicit reasoning |
| `contract` | 2 | Predict, check the unverified candidate, and optionally revise |
| `consistency` | 5 by default | Sample multiple CoT answers and vote |
| `hybrid` | 1 | Route between Direct and CoT using task keywords |
| `code_execute` | 1 | Generate a Python function and execute it in a standard Python worker |

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

Both stages use the same configured backend-native reasoning settings and
requested output-token limit. `max_new_tokens=None` omits the application cap;
an explicit positive limit stays fixed for both calls and any validation retries.
The initial and final values are retained so beneficial and harmful revisions
can be measured separately.

Contract checking is **model-based semantic self-checking**, not formal verification.

### `consistency`

Makes repeated CoT predictions and chooses an answer by voting. The default is five samples, so one invocation normally consumes five model calls before retries. Samples use the request's `temperature`; with the default `0.0`, a deterministic provider returns identical samples, so set a positive temperature for meaningful voting.

### `hybrid`

Routes a request to Direct or CoT using fixed task-keyword heuristics. This routing is separate from backend-native reasoning settings.

### `code_execute`

Select `VirtualRuntime(..., strategy="code_execute")` to generate a Python
function and execute it on the supplied inputs. The strategy makes one model
call. It does not regenerate code after errors or increase an output-token cap.
Runtime validation retries do not apply; normal output type and constraint checks do.

The worker uses the application's Python interpreter, including its virtual
environment and installed packages. It inherits the working directory,
environment variables and operating-system permissions. Normal Python imports,
helper classes, introspection, `eval`, file access and subprocesses are available;
there is no import allowlist, AST policy, credential filtering or fixed memory
limit. Deployment permissions and isolation belong to the application.

`result.raw_text` retains the model response, `result.generated_code` retains
extracted Python, and `result.execution_details` records execution status,
returned value, timing and attempts. The response protocol is `python_code_v1`.
Failures include `error_type`, a single-line `error_message` capped at 500
characters, and `error_line` when a generated-code location is known. Lines are
one-based within `generated_code`; timeouts and failures outside generated code
have no source line. Each local attempt retains its own diagnostics. The
evaluator carries these details into its CSV and JSON.

Inputs remain a dictionary keyed by declared parameter name, including
positional-only (`/`) and keyword-only (`*`) parameters. The worker constructs
the corresponding Python call. Omitted arguments use the implementation's
defaults, including gaps before supplied positional-only arguments. Defaults of
generated Python functions, bound methods and partials are read without
evaluating annotations or copying default objects. Returned
values use the existing typed transport for supported built-in values and
containers, preserving tuples, sets, bytes, dictionary key types, infinity and NaN.

Nonempty replies that do not start with a code fence are treated as raw source.
A reply starting with a fence must contain one complete Python block and no
surrounding text. Fence markers inside Python strings are preserved. Python
compiles the extracted source in the worker: invalid
source, including invalid Unicode in the source text, is a `syntax_error`, while
empty or malformed fenced replies are a `format_error`. The raw response and
extracted source are retained for diagnosis.

Configure the worker timeout, local retries or task mode with the strategy factory:

```python
from aicomputing import create_strategy

strategy = create_strategy(
    "code_execute",
    execution_timeout=180,
    execution_retries=1,
    task_mode="function",  # or "word_problem" for arithmetic on one supplied question
)
# Pass strategy to VirtualRuntime(..., strategy=strategy).
```

Defaults are 120 seconds per execution attempt and two additional timeout
attempts, using the identical source and original input values. Other execution
failures are not retried. The timeout covers worker startup and execution.
The worker reports its actual process ID and waits until the parent finishes
cleanup setup before running any generated source. This also handles Windows
virtual environments whose Python launcher has a different process ID.
Worker processes and their children are cleaned up after execution or timeout.
Temporary request, startup and result files are removed afterward; printed output
is kept separate from the return-value channel.
Windows cleanup-group setup failures report the failed API and Windows error
code as a worker harness error, rather than continuing without group cleanup.
Cleanup attempts job/group termination, worker termination, a final wait of at
most five seconds, and temporary-file removal even if an earlier step fails.
An exception already raised by this execution, including cancellation, remains
primary; cleanup failures are logged by exception type. A cleanup-only failure
is reported as a harness error. Cancellation during cleanup propagates after
the remaining steps have been attempted.

Execution settings are validated at construction and before generation.
Invalid changes to the timeout, retry count, task mode or checkpoint callback
are rejected before a model call. `task_mode="function"` requests a general
implementation; `"word_problem"` requests arithmetic for the supplied question.
An optional `on_generated_code` callback receives extracted source before the
worker starts, for checkpointing or logging.

Generated-code failures return `validation_status="execution_failed"` and
`failure_type="execution_failure"`. Detailed failure reasons in
`execution_details` include syntax errors, runtime errors, serialization errors
and local timeouts. An internal worker harness failure raises
`CodeExecutionError` with source and execution evidence. `execution_error`
provides the concise diagnostic. Code execution leaves `parse_status` and
`parse_error` as `None`; those fields describe prediction-reply parsing.

Caching is disabled by default. If explicitly enabled for a named strategy,
cache hits reuse the prior result and diagnostics with `cached=True` and zero
new model or worker calls. Use uncached execution when environment state or
side effects matter.

The CLI supports the same worker settings:

```bash
aicomp "add(x: int, y: int) -> int" "Return x + y." --inputs '{"x": 19, "y": 23}' --backend ollama --strategy code_execute --execution-timeout 180 --execution-retries 1
```

---

## Prompt construction

Prompt wording is kept separate from both strategies and backends.

Each prediction request includes the same core information:

```text
Signature: <function signature>
Docstring: <behavioural specification>
Inputs: <concrete arguments>
```

Strategy-specific instructions determine whether the model should answer directly, expose a trace, construct pseudocode, or check a candidate. Predictions use the shared final-answer block. Clearly isolated malformed result text is preserved without guessing values from earlier reasoning or examples.

The prompt layer is intentionally provider-independent: changing from Ollama to Hugging Face or an OpenAI-compatible API does not redefine the function or execution strategy.

---

## Backends and model reasoning modes

All backends implement a common `Backend` interface and return a normalized `GenerateResponse`.

The optional constructor settings `base_url` and `api_key` (OpenAI), `host`
(Ollama), and `device` (Hugging Face) accept only `None` or strings containing at
least one non-whitespace character. `None` uses automatic defaults, including
SDK environment settings where supported. Supplied strings are passed unchanged;
blank values and incorrect types fail before client or model setup. The CLI
reports blank connection options as usage errors.

### Ollama

```python
runtime = VirtualRuntime(backend="ollama", model="qwen3:4b")
```

A custom Ollama host can be supplied with `host=...`. Otherwise, the client
uses `OLLAMA_HOST` or the SDK's local default. Each backend owns a separate
client, created on first use and closed by `backend.close()`.

The CLI's `--base-url` option supplies this host unchanged, including any path
and trailing slash. It does not remove `/v1`; supply the host expected by your
Ollama server.

Ollama replies containing tool-call indicators raise `ToolUseDetected`, as with
the OpenAI-compatible backend. Ordinary text mentioning tools is accepted.

### Hugging Face

```python
runtime = VirtualRuntime(
    backend="hf",
    model="Qwen/Qwen3-4B-Instruct-2507",
)
```

The model is loaded lazily and kept in memory across calls.
Chat calls preserve message roles through the tokenizer's chat template.
Without a template, only a single user message is accepted. Unsupported
conversations raise an error instead of being merged into a different prompt.

### OpenAI-compatible

```python
runtime = VirtualRuntime(
    backend="openai",
    model="<provider-model-id>",
    base_url="https://<provider>/v1",
    api_key="...",
)
```

The backend normalizes provider responses, token usage, and finish reasons.
Replies containing tool-call indicators raise `ToolUseDetected`; these indicators
can describe requested calls and do not prove that a tool ran. Ordinary text
mentioning tools is accepted. The `code_execute` strategy receives Python as
text and runs it in the local worker.

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

Output-token limits are separate from native reasoning: model names and prompting
strategies do not introduce a cap.

---

## Parsing, validation, and retries

Function signatures use Python's declaration parser and compiler checks without
executing the resulting code. Nested defaults and positional/keyword markers
are supported; defaults remain unevaluated source expressions. Duplicate
parameters, invalid names, function bodies and additional statements are rejected
before generation. Type annotations are resolved from the supported built-in
types and container aliases without evaluating annotation expressions.
Variadic parameters and generic function type parameters remain unsupported.

### Parsing

`aicomputing.parsing.parse_result(text)` reads one final-answer block and returns
a `ParseOutcome` with the recovered value, status, and optional diagnostic. It
preserves decoded types and performs no type conversion. It records:

- `parsed` — a complete JSON/Python literal was decoded;
- `fallback_parsed` — one isolated malformed result field was preserved as text;
- `failed` — the answer is missing, empty, or ambiguous.

### Structural validation

The runtime keeps three concepts separate:

| Property | Decided by |
|---|---|
| structural/type validity | mechanical check against the declared interface |
| constraint validity | caller-supplied executable predicates |
| semantic correctness | external ground truth or application-specific oracle |

Caller inputs are always checked before generation. Missing or unexpected
arguments, incorrect argument types, and malformed signatures raise immediately.

Output validation is configured on the runtime or overridden by a request:

| Mode | Output behavior |
|---|---|
| `strict` (default) | Apply type and constraint checks; withhold invalid values after retries |
| `warn` | Apply the same checks; return invalid values with diagnostics |
| `none` | Preserve parsed values without conversion, type checks, or constraint checks; report `not_checked` |

Strict and warn use the same limited conversions; neither truncates a fractional
float to an integer. Failed parsing and generated-code execution remain failures
in every mode. The runtime does **not** claim that a structurally valid value is
semantically correct.

### Retries

`ExecuteRequest.max_retries` controls additional strategy executions after failed
output recovery or type/constraint checks. Every retry uses the same requested
token limit. A provider-reported truncated response is retained as raw text and
is never decoded as a complete answer.

Backend transport retries and code-worker timeout retries are configured
separately. Worker retries run the identical generated program and inputs;
they do not ask the model to generate another implementation. Reported usage
and model-call counts accumulate across returned responses, including truncated
ones and validation retries.

### Output-token limits

`max_new_tokens=None` (the default) means no application output-token cap. OpenAI-compatible requests omit the output-limit
field; Ollama receives `num_predict=-1`. The Hugging Face backend must provide a
finite generation stopping bound: it uses the model's declared context capacity
minus the prompt length, falling back to tokenizer metadata only when the model
does not declare a capacity. An explicit limit is required if neither source
provides a finite capacity. Provider defaults, model output limits, and context windows
still apply; omitting an application cap does not guarantee unlimited generation.

A positive `max_new_tokens` or `--max-tokens` value stays fixed on every call.
Truncation never increases it. A configured validation retry can request a new
answer with feedback, using the same limit.
Hugging Face rejects a prompt plus explicit output budget that exceeds the
known context capacity before inference. The error reports the prompt length,
requested budget and capacity; the backend does not reduce the requested limit.

---

## Caching and tracing

Result caching is disabled by default (`cache_size=0`). Enable the in-memory
LRU cache explicitly for repeated identical requests, for example
`VirtualRuntime(backend="ollama", model="qwen3:4b", cache_size=128)`. The capacity
is a number of result entries. Disabled caching skips cache-state inspection,
lookups and storage.

The cache key includes the specification, inputs, model, strategy, relevant generation settings, backend class and declared backend configuration (including native reasoning overrides). A backend must explicitly supply a cache state; unknown custom backends bypass caching. Credentials are excluded from that state, so this is not an account- or environment-isolation guarantee.

Requests with constraints bypass caching in strict and warn modes, where the
predicates run. In `none` mode, skipped constraints do not affect cache eligibility
or the cache key; their definitions are still checked before a cached result can
be returned. Leave `cache_size=0` to request fresh model calls.

Optional JSONL tracing records request and result metadata, including the ordered `reasoning_requests`. Prompt/raw-text fields describe the strategy result; they are not a complete history of every internal call or retry.

`trace_file=None` disables tracing. Otherwise, supply a nonblank string or a
`pathlib.Path`; blank strings and incorrect types are rejected before backend
creation. The CLI reports a blank `--trace` destination as a usage error. Valid
paths retain their supplied names. Filesystem write failures still warn without
preventing a prediction from being returned.

Trace records retain the declared `ExecuteResult` fields, including `result`,
`model_calls`, `failure_type`, `parse_status`, `parse_error`, and `execution_error`. The separate
`parsed_result` and `parsed_type` fields preserve the recovered candidate even
when strict validation withholds it. Direct logging uses
`Tracer(path).log(request, result, parsed_result=...)`; omit `parsed_result` to
use the returned value. Parent directories are created when writing a record.
Malformed Unicode characters are escaped in JSONL, preserving the original text
when decoded. Serialization or write failures warn without stopping predictions.

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

The package installs the `aicomp` command. Choose its backend explicitly with
the required `--backend` option:

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

The command returns exit code 0 and prints the returned value on stdout when
prediction succeeds. A valid `None` prints `null`. With `--validation-mode warn`,
invalid output is still returned with exit code 0 and a warning on stderr.

Parsing failures, failed code execution, and outputs withheld by strict validation
return exit code 1, print a short error on stderr, and leave stdout empty. Known
provider and worker errors also return exit code 1 with a short diagnostic.
Missing backend dependencies return exit code 1 with an installation hint.
Argument usage errors return exit code 2. Use `--verbose` to include response and
validation details on stderr. The same exit behavior applies to
`python -m aicomputing.cli`.
Malformed signatures, wrongly typed or missing inputs, and invalid generation
settings are usage errors checked before a backend is created.

---

## Evaluation guide

The optional [evaluation guide](https://github.com/MichiB4/aicomputing/blob/master/docs/evaluation.md) documents benchmark data, dataset preparation, task selection, scoring, and experiment reproduction. These tools use the oracle package to measure model and strategy performance. They are not required to call `VirtualRuntime` from an application.

## License

Licensed under the [MIT License](LICENSE).
