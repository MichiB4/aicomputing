# aicomputing

> **Predictive execution with LLMs** — declare a function by its signature and docstring, let the model predict the output.

Instead of writing an implementation, you write a specification. The `VirtualRuntime` sends that specification to a language model and returns the predicted result as a typed Python value.

```python
from aicomputing import VirtualRuntime, ExecuteRequest

vr = VirtualRuntime(backend="ollama", model="qwen3:4b")

result = vr.execute(ExecuteRequest(
    signature="dijkstra(graph: dict, start: str) -> dict[str, int]",
    docstring="Return shortest distances from start to all other nodes. "
              "Graph format: {node: [(neighbour, weight), ...]}",
    inputs={
        "graph": {"A": [("B", 4), ("C", 2)], "B": [("D", 10)], "C": [("D", 3)], "D": []},
        "start": "A",
    },
    return_type=dict,
))

print(result.result)   # {'A': 0, 'B': 4, 'C': 2, 'D': 5}
print(result.model)    # qwen3:4b
```

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Prompting Strategies](#prompting-strategies)
- [Backends](#backends)
- [CLI](#cli)
- [Benchmarks](#benchmarks)
- [Open Points](#open-points)

---

## Installation

**Prerequisites:** Python 3.10+

```bash
# Clone and install core library (Ollama backend only)
git clone <repo-url>
cd aicomputing
pip install -e .

# With HuggingFace Transformers backend
pip install -e ".[hf]"

# With benchmark tooling
pip install -e ".[benchmarks]"

# Everything
pip install -e ".[hf,benchmarks]"
```

For the **Ollama backend**, Ollama must be running locally:

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3:4b
ollama serve          # runs on localhost:11434 by default
```

For the **HuggingFace backend**, a CUDA-capable GPU is recommended. The default model is `Qwen/Qwen3-4B-Instruct-2507`; it is downloaded automatically on first use.

For the **OpenAI-compatible backend** (Groq, Cerebras, OpenRouter, SambaNova, vLLM, LM Studio, or any server exposing the OpenAI chat completions API), install with `pip install -e ".[openai]"` and pass `base_url` / `api_key`.

---

## Quick Start

### Basic prediction

```python
from aicomputing import VirtualRuntime, ExecuteRequest

vr = VirtualRuntime(backend="ollama", model="qwen3:4b", strategy="direct")

# Predict the output of a string function
req = ExecuteRequest(
    signature="repeat(s: str, n: int) -> str",
    docstring="Return the string s repeated n times.",
    inputs={"s": "ab", "n": 3},
    return_type=str,
)

res = vr.execute(req)
print(res.result)    # 'ababab'
print(res.raw_text)  # full model output for debugging
```

### Hybrid classical + AI pipeline

```python
import json
from aicomputing import VirtualRuntime, ExecuteRequest

vr = VirtualRuntime(backend="ollama", model="qwen3:4b", strategy="cot")

# Load reviews with regular Python
with open("data/reviews.json") as f:
    reviews = json.load(f)

# Score each review with the virtual runtime
scores = []
for review in reviews:
    res = vr.execute(ExecuteRequest(
        signature="score_review(text: str) -> int",
        docstring="Return a sentiment score 0–5 for the given product review. "
                  "0 = terrible, 5 = excellent. Output a single integer.",
        inputs={"text": review["text"]},
        return_type=int,
    ))
    scores.append(max(0, min(5, int(res.result))))

# Aggregate with regular Python
print(f"Average score: {sum(scores) / len(scores):.2f}")
```

---

## Architecture

The library is organized into five layers that are strictly separated so each can evolve independently.

```
┌─────────────────────────────────────────────────────┐
│                     Public API                      │
│        VirtualRuntime · ExecuteRequest · ExecuteResult │
└────────────────────────┬────────────────────────────┘
                         │ dispatches to
┌────────────────────────▼────────────────────────────┐
│                  Strategy Layer                     │
│  DirectStrategy · CoTStrategy · HybridStrategy       │
│  ConfidenceStrategy · + generate_with_retry()        │
└──────────┬─────────────────────────┬────────────────┘
           │ builds prompts via      │ calls
┌──────────▼──────────────┐ ┌────────▼──────────────────┐
│    Prompting Layer      │ │      Backend Layer        │
│  build_direct_prompt()  │ │  OllamaBackend             │
│  build_cot_prompt()     │ │  HFBackend                 │
│  build_confidence_      │ │  OpenAICompatBackend       │
│    prompt()             │ │  (common Backend ABC)      │
└──────────────────────────┘ └──────────┬─────────────────┘
                                        │ raw text
┌───────────────────────────────────────▼──────────────┐
│                   Parsing Layer                      │
│     JSON extraction → scalar fallback → coerce       │
└────────────────────────────────────────────────────┘
```

`HybridStrategy` has no dedicated prompt builder of its own — it delegates
to `build_direct_prompt()`/`build_cot_prompt()` after picking one via its
internal keyword classifier.

### `ExecuteRequest` and `ExecuteResult`

`ExecuteRequest` encodes everything the runtime needs to call the model:

| Field | Type | Description |
|---|---|---|
| `signature` | `str` | Python-style function signature, e.g. `"add(a: int, b: int) -> int"` |
| `docstring` | `str` | Natural-language description of what the function does |
| `inputs` | `dict[str, Any]` | Concrete argument values keyed by parameter name |
| `return_type` | `Optional[Any]` | Expected Python type used to coerce the parsed output |
| `constraints` | `list[Constraint]` | Predicates checked against the output (see `Constraint`) |
| `temperature` | `float` | Sampling temperature (default `0.0` for reproducibility) |
| `max_new_tokens` | `int \| None` | Override the strategy's token budget |
| `strategy` | `str \| None` | Per-call strategy override (default: the runtime's strategy) |
| `model` | `str \| None` | Per-call model override (default: the runtime's model) |
| `strict_types` | `bool \| None` | Per-call override of the runtime's `strict_types` setting |
| `max_retries` | `int` | Max validation retries if output fails type/constraint checks (default `1`) |
| `think` | `bool \| None` | Per-call override of the strategy's default thinking mode |

`ExecuteResult` is what `.execute()` returns:

| Field | Description |
|---|---|
| `result` | Parsed, type-coerced Python value |
| `raw_text` | Full model output string (useful for debugging) |
| `model` | Model identifier that produced the result |
| `strategy` | Name of the strategy that actually ran (relevant after a `hybrid`/`confidence` call) |
| `cached` | Whether this result came from the in-memory cache |
| `tokens_used` | Token count reported by the backend (if available) |
| `retries` | Number of validation retries (output failed parsing/validation) |
| `token_retries` | Number of token-budget doublings (model hit the length limit) |
| `type_warnings` / `constraint_warnings` | Non-fatal validation issues (populated when `strict_types=False`) |
| `latency_ms` | Wall-clock time for the call |

### `VirtualRuntime`

The main entry point. It wires a strategy to a backend and exposes a single `.execute(request)` method. Construction parameters:

```python
VirtualRuntime(
    backend="ollama",        # "ollama" | "hf" | "openai"
    model="qwen3:4b",        # any model identifier valid for the backend
    strategy="direct",       # "direct" | "cot" | "hybrid" | "confidence"
    strict_types=True,       # raise on type/constraint mismatch vs. return warnings
    cache_size=2048,         # in-memory LRU result cache size (keyed on request hash)
    trace_file=None,         # path to write a JSONL trace of every call
)
```

### Strategy Layer

Each strategy is a class implementing the `Strategy` ABC with a single method `execute(request, backend) -> StrategyResult`. Strategies own prompt construction and decide how many backend calls to make.

**`generate_with_retry(call, parse, initial_budget)`** is a shared utility used by all strategies. If the model's output is truncated (hits the token limit), it doubles the budget and retries — up to five times or until the cap of 32 768 tokens, whichever comes first. This is especially important for thinking-capable models where the internal reasoning trace can be long.

### Prompting Layer

Prompt builder functions are pure functions `(ExecuteRequest) -> str`. They are separated from strategy classes so that prompt wording can be changed without touching retry or parsing logic. Each builder injects a few-shot preamble with worked examples that anchor the output format (`{"result": ...}`) before appending the user's function specification.

### Backend Layer

Backends implement a common `Backend` ABC with a single method `generate(prompt, *, model=None, temperature=0.0, top_p=1.0, max_new_tokens=512, stop=None, **kwargs) -> GenerateResponse`. This makes it trivial to add new backends (vLLM, LM Studio, etc.) without touching anything else.

- **`OllamaBackend`** — calls the local Ollama HTTP API. Supports any model in the Ollama registry. Stateless; each call is a fresh HTTP request.
- **`HFBackend`** — loads a model with `transformers`, applies the model's native chat template, and runs inference locally. Keeps the model in memory across calls.
- **`OpenAICompatBackend`** — talks to any server exposing the OpenAI chat completions API (Groq, Cerebras, OpenRouter, SambaNova, vLLM, LM Studio, ...). Thinking-mode handling (native toggle vs. `/no_think` suffix vs. no-op) is resolved per model family via `model_config.py`.

### Parsing Layer

Raw model output is processed in two steps:

1. **JSON extraction** — searches for a `{"result": ...}` object in the output text using a regex, then `json.loads` the match.
2. **Scalar fallback** — if no JSON object is found, attempts to parse bare scalars (`int`, `float`, `bool`, `null`, quoted strings).
3. **Type coercion** — the parsed value is cast to `return_type` if specified (e.g. `int("5")` → `5`).

If parsing fails, the parse error is fed back into the context and generation is retried (within the strategy's retry budget).

---

## Prompting Strategies

### `direct` — Direct Prediction

The simplest strategy. Builds a few-shot prompt with three worked examples, appends the target function specification and inputs, and asks the model for a single JSON answer. Uses greedy decoding (`temperature=0`) for reproducibility.

**Best for:** Simple, single-step functions; fast sweeps; establishing a baseline.

**Token budget:** 512 by default, doubling on truncation up to a 32,768-token cap. Pass `think=True` on the request to start at 8,192 instead.

### `cot` — Structured Chain-of-Thought

Instructs the model to trace execution symbolically before answering: maintain an explicit variable-state log through each sequential step, conditional branch, and loop iteration, then emit the final `{"result": ...}` JSON. Reasoning steps are unconstrained; only the final answer is format-enforced.

**Best for:** Multi-step functions, conditional logic, loops; understanding *why* the model gives a particular answer.

**Token budget:** 8,192 by default (this strategy assumes reasoning is needed), doubling on truncation up to a 32,768-token cap. Pass `think=False` on the request to start at 512 instead.

### `hybrid` — Keyword-Routed direct/cot

Classifies each call with a rule-based keyword/structure heuristic and delegates to either `direct` or `cot` — no self-consistency or sampling involved. Multimodal inputs/outputs and NLP-ish keywords (`classify`, `summarize`, `sentiment`, ...) route to `direct`; algorithmic/mathematical keywords (`sort`, `dijkstra`, `solve`, ...), deeply nested inputs, or a `float` return type route to `cot`. See `strategies/hybrid.py` for the exact rules.

**Best for:** A cheap, deterministic way to avoid paying for `cot`'s larger token budget on tasks that don't need it.

**Caveat:** this is a hand-written, unvalidated heuristic — see the `PROVISIONAL` note in `hybrid.py` and [Known open questions](#open-points) below. It has no learning component and doesn't improve as models get better at judging their own task difficulty; see `confidence` for an alternative that does.

### `confidence` — Self-Assessed Reasoning

A single prompt, a single call. Rather than the runtime pre-classifying the
task (as `hybrid` does with keywords), the model is instructed to decide
for itself, per call, whether to answer immediately or trace its reasoning
first — using the same state-tracking trace format as `cot` when it does.
The few-shot examples show both output shapes without indicating which one
to prefer, so the choice is left entirely to the model's own confidence.

This exists as a candidate answer to the "universal vs. task-specific
strategy" open question that doesn't rely on hand-maintained keyword
lists: as models get better at judging their own uncertainty, they should
skip the trace more often on their own, with no code change required. It
has not been benchmarked against `direct`, `cot`, or `hybrid` — treat it
as a research probe, not a validated default.

**Best for:** Exploring whether models can self-regulate reasoning depth;
not for squeezing out the best benchmark score today.

---

## Backends

### Ollama

```python
vr = VirtualRuntime(backend="ollama", model="qwen3:4b")
# or with a custom host
vr = VirtualRuntime(backend="ollama", model="qwen3:4b", host="http://localhost:11434")
```

Pull any model with `ollama pull <name>` before use. See [ollama.com/library](https://ollama.com/library) for available models.

### HuggingFace (`hf`)

```python
vr = VirtualRuntime(backend="hf", model="Qwen/Qwen3-4B-Instruct-2507")
```

The model is downloaded to `~/.cache/huggingface` on first use.

### OpenAI-compatible (`openai`)

```python
vr = VirtualRuntime(
    backend="openai",
    model="qwen/qwen3-32b",
    base_url="https://api.groq.com/openai/v1",
    api_key="...",
)
```

Works with any server exposing the OpenAI chat completions API — Groq, Cerebras, OpenRouter, SambaNova, vLLM, LM Studio. `benchmarks/bench_args.py` wraps this with a `--provider` shortcut that fills in `base_url` for known providers.

---

## CLI

The `aicomp` command lets you call the runtime from the terminal without writing Python. `signature` and `docstring` are positional arguments; everything else is a flag.

```bash
aicomp \
  "max_of_two(a: int, b: int) -> int" \
  "Return the larger of the two integers." \
  --inputs '{"a": 7, "b": 3}' \
  --backend ollama \
  --model qwen3:4b \
  --strategy direct
```

```
7
```

The result is printed to stdout as JSON; pass `--verbose` to also print strategy, model, latency, token count, retries, and raw model output to stderr:

```bash
aicomp ... --verbose 2>&1 1>/dev/null   # see only the diagnostic lines
```

Full options:

```
signature       Python-style function signature, e.g. "add(x: int, y: int) -> int" (positional, required)
docstring       Natural-language description (positional, required)
--inputs        JSON object of input values (required)
--return-type   int | float | str | bool | none — coerce the result (default: none)
--backend       ollama | hf | openai (default: ollama)
--model         Model identifier (default depends on backend)
--base-url      API base URL (for the openai backend)
--api-key       API key (for the openai backend)
--strategy      direct | cot | hybrid | confidence (default: direct)
--temp          Sampling temperature (default: 0.0)
--max-tokens    Override the starting token budget (default: auto per strategy/model)
--no-strict     Disable strict type checking (return warnings instead of raising)
--trace         Path to write a JSONL trace of the call
--think / --no-think   Force thinking mode on/off (default: strategy decides)
--verbose, -v   Print diagnostics to stderr
```

---

## Benchmarks

Benchmark runners live in `benchmarks/`. Each runner adapts a standard dataset into `ExecuteRequest` objects and reports exact-match accuracy alongside raw LLM-call counts and optional CSV output.

### HumanEval (Oracle mode)

```bash
python benchmarks/humaneval_oracle.py \
  --backend ollama \
  --model qwen3:4b \
  --strategy cot \
  --out-csv results/humaneval_cot.csv
```

This runs all 164 HumanEval problems in oracle mode: rather than generating code, the model is asked to predict the return value for each test-case input. Accuracy is measured as exact-match against the known ground-truth outputs.

### Other benchmarks

BIG-Bench Hard (date understanding, general reasoning), HotpotQA, and GSM8K runners follow the same interface:

```bash
python benchmarks/bigbench_oracle.py  --strategy cot  --out-csv results/bbh_cot.csv
python benchmarks/hotpotqa_oracle.py  --strategy cot  --out-csv results/hotpotqa_cot.csv
python benchmarks/gsm8k_oracle.py     --strategy cot  --out-csv results/gsm8k_cot.csv
```

Results CSVs contain columns: `task_id, passed, status, llm_calls, model, strategy, backend`.

---

## Open Points

These are the known open questions and pending work items, roughly in priority order.

### Which prompting strategies to use across benchmarks?

The four strategies (direct, cot, hybrid, confidence) were designed for general use, but the evaluation spans four very different task types — algorithmic reasoning, symbolic reasoning, multi-hop QA, and arithmetic. A prompt that generalizes across all of them may not be optimal for any single one. `hybrid` and `confidence` are two competing, unvalidated answers to this question — see the `hybrid` strategy description above for the tradeoff between them.

**The core tension:** heavy prompt engineering (task-specific few-shot examples, tailored instructions) could significantly boost per-benchmark accuracy, but it would conflate the effect of the prompting strategy with the effect of prompt wording. Since the goal is to characterize model capability across task types, not to maximize benchmark scores, a single generalizable prompt is arguably more scientifically honest. This needs to be decided and justified in the thesis.

**Options to consider:**
- One universal prompt across all benchmarks (current approach) — cleanest comparison.
- One prompt per task *category* (algorithmic, QA, arithmetic) — reasonable middle ground.
- Task-specific prompts — highest accuracy ceiling but harder to interpret.

The likely finding is that prompt engineering effects are real but smaller than model-size or strategy effects — worth measuring explicitly as an ablation.

### What other benchmarks to use for LLM-specific tasks?

The four current benchmarks (HumanEval, BIG-Bench Hard, HotpotQA, GSM8K) all originate from the classical NLP/code-generation world and were not designed with the oracle framing in mind. For tasks where LLMs have a natural advantage — perception, natural language understanding, judgment — additional benchmarks would be more discriminating.

**Candidates worth investigating:**

| Benchmark | Task type | Why relevant |
|---|---|---|
| MMLU | World knowledge QA | Broad coverage; many tasks map well to function specs |
| HellaSwag | Commonsense reasoning | Tests intuitive judgment, hard to hand-code |
| TruthfulQA | Factual accuracy | Tests reliability of model-as-oracle |
| MBPP | Code / algorithmic | Complements HumanEval with different problem styles |
| HumanEval+ | Code (augmented) | Harder test cases for the same HumanEval problems |
| NaturalQuestions | Open-domain QA | More naturalistic than HotpotQA |
| MATH | Competition math | Harder arithmetic regime than GSM8K |

### Code cleanup

The codebase grew incrementally and several things need tidying before the evaluation runs:

- **Consistent error handling** — parsing failures currently raise in some paths and return `None` in others; this should be unified.
- **Type annotations** — several internal functions are missing return type hints.
- **Dead code** — periodic sweep for unused imports/commented-out code in experiment scripts (`scripts/`, `benchmarks/`).
- **Test coverage** — `tests/test_core.py` now covers signature parsing, type validation, output parsing, caching, and strategy-level behavior (including `hybrid`'s classifier and `confidence`'s dual output shapes) against fake backends. `smoke_test.py` remains the only test that hits a live model — still worth automating against a fixed local model if CI ever gets one.
- **Logging** — `print` statements in experiment scripts should go through the `logging` module so verbosity is controllable.
- **Config management** — model name, backend, strategy are currently passed as CLI flags; a YAML/TOML config file would make large experiment sweeps more reproducible.

### Ollama is not usable with thinking models on current hardware

Thinking-capable models (e.g. `qwen3:4b` with thinking mode, DeepSeek-R1 variants) generate a long internal reasoning trace before the final answer. This blows up both memory usage and inference time on consumer hardware, making Ollama impractical for `cot`, and for `hybrid`/`confidence` calls that end up routed or self-routed into reasoning, on these models.

**Options, roughly in order of effort:**

1. **Use non-thinking instruct variants locally.** Most model families release a separate `-Instruct` checkpoint without thinking. These run comfortably with Ollama and still benefit from the CoT prompt — the model reasons in the visible output rather than a hidden trace. Accuracy is lower than the thinking variant but experiments are feasible.

2. **Use API-based thinking models.** `claude-3-7-sonnet` and `o4-mini` both expose thinking/extended-reasoning modes via API. This adds cost but no hardware constraint, and allows comparing the same model family with and without thinking enabled.

3. **Move local inference to a cloud GPU.** Running Ollama or a vLLM server on a rented A100 (e.g. via RunPod, Lambda Labs, or Google Colab Pro+) removes the hardware bottleneck entirely. The Ollama backend already supports a custom `--host` flag so no code changes are needed.

4. **Use HuggingFace backend with quantization.** The HuggingFace backend can load 4-bit quantized models via `bitsandbytes` (add `load_in_4bit=True` to the backend config). A 4-bit 7B thinking model needs ~6 GB VRAM and runs on most modern laptops with a discrete GPU, though slowly.