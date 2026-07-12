# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`aicomputing` is a master's thesis project investigating "AI computing" as a
third computational paradigm alongside classical and quantum computing. LLMs
act as a **virtual runtime** that predicts a Python function's output from
its specification alone — signature + docstring + concrete inputs — without
ever seeing or executing a hand-written implementation.

This is **spec-only execution reasoning**. It is the property that
distinguishes this work from related efforts (CRUXEval, CodeI/O, SURGE,
"LLMs as Code Executors," Modelizer): those measure code *execution*
prediction; this measures whether a model can act as the implementation
itself, given only the contract. Never conflate the two when writing docs,
comments, or benchmark descriptions — the distinction is the paper's central
claim.

Two thesis contributions live in this repo:
1. The `aicomputing` package architecture (five layers, below)
2. A multi-model evaluation of where current LLMs succeed/fail as virtual
   runtimes, across HumanEval, GSM8K, BBH, and HotpotQA

## Architecture — five layers

Requests flow top to bottom; keep changes scoped to one layer where possible.

1. **API** (`request.py`, `runtime.py`, `__init__.py`) — `ExecuteRequest` /
   `ExecuteResult` data contract, and `VirtualRuntime`, the orchestrator
   implementing the E-AI evaluation rule: parse signature → cache lookup →
   run strategy → validate/coerce output → retry on failure → cache + trace.
2. **Strategy** (`strategies/`) — *how* to turn a spec into a prediction.
   `direct` (one-shot JSON), `cot` (Structured CoT with explicit
   sequential/branch/loop state tracking), `hybrid` (keyword/structure
   heuristics routing each task to direct or cot), `confidence` (single
   prompt, model self-assesses per call whether to trace first — no
   external routing logic at all).
3. **Prompting** (`prompting.py`) — builds the actual prompt strings.
   Calm declarative system prompts (no ALL-CAPS/"YOU MUST"), few-shot
   examples used for *format alignment only*, never to model reasoning.
4. **Backend** (`backends/`) — talks to the actual model: `ollama` (local),
   `hf` (HuggingFace Transformers, local), `openai` (any OpenAI-compatible
   API — Groq, Cerebras, OpenRouter, SambaNova).
5. **Parsing/Validation** (`parsing.py`, `types.py`, `validation.py`) —
   extracts a typed Python value from raw model text and checks it against
   the declared signature (the T-AI typing rule).

Cross-cutting: `model_config.py` (thinking-mode registry + token budgets +
provider presets), `cache.py` (LRU result cache), `trace.py` (JSONL call
logging).

## Environment & constraints

- Local dev: Windows + PowerShell, Ollama with `qwen3:4b`, RTX 2070 (8GB
  VRAM). **Thinking-capable models are not locally feasible** — anything
  beyond ~4-8B non-thinking is a remote-provider job.
- Remote providers: Groq, Cerebras, OpenRouter, SambaNova via `--provider`
  in `benchmarks/bench_args.py`.
- Full evaluation matrix targets: Qwen3 at 4B/32B/235B, DeepSeek-R1,
  Llama 3.3 70B, Gemini 3.1 Pro.

## Commands

```bash
# Install (editable, with dev + benchmark extras)
pip install -e ".[dev,benchmarks]"

# Run the full test suite — offline, no LLM calls
PYTHONPATH=src python3 -m pytest tests/ -q

# Single AI function call via CLI
aicomp "add(x: int, y: int) -> int" "Return the sum." --inputs '{"x":2,"y":3}'

# Run a benchmark locally
python benchmarks/gsm8k_oracle.py --model qwen3:4b --strategy direct --limit 50

# Run a benchmark against a remote provider
python benchmarks/bbh_oracle.py --provider groq --model qwen/qwen3-32b \
    --api-key $GROQ_API_KEY --limit 100

# End-to-end smoke test against a live Ollama server (not part of pytest)
python smoke_test.py --model qwen3:4b
```

Always run pytest after any change to `src/aicomputing/` and iterate until
green before considering a task done — don't just eyeball the diff.
`scripts/debug_raw.py` and `scripts/ollama_diagnostic.py` are standalone,
live-Ollama debugging aids (not tests) for inspecting raw model/backend
output when a strategy misbehaves.

## Conventions to follow

- **Every bug fix gets a regression test** in `tests/test_core.py`,
  reproducing the failure before the fix and asserting the fix after.
- **Strategies predict results, never generate/execute code.** The only
  exception is `benchmarks/baselines/code_gen_baseline.py`, which exists
  solely as a comparison baseline and is explicitly *not* AI computing —
  don't let this pattern leak into `strategies/`.
- **Token budgets are auto-managed** via `model_config.get_token_budget()`
  (512 non-thinking / 8192 thinking, doubling up to a 32768 cap on
  truncation). Don't hardcode `max_new_tokens` in new benchmark code unless
  intentionally overriding.
- **Model thinking behavior is centralized** in `model_config.py`'s
  `_MODEL_PATTERNS`. New model families go there, not as special-casing
  inside a backend.
- **Results/traces stay out of the diff.** CSVs and JSONL traces live in
  `results/` and `traces/`, gitignored — don't commit experiment output
  alongside code changes.
- Keep the five layers decoupled: a strategy should only import from
  `backends.base`, `request`, and its own package — never reach into a
  specific backend implementation.

## Known open questions (don't silently resolve these — flag them)

1. Universal vs. task-specific prompting strategy across benchmarks.
   `hybrid` and `confidence` are two competing candidate answers, neither
   validated: `hybrid` routes direct/cot via a hand-written keyword
   classifier (fixed human judgment, needs upkeep, doesn't improve with
   model capability); `confidence` instead has the model decide per call
   whether to reason, with no external routing logic to maintain — the
   strategy that's meant to keep making sense as models get smarter,
   since a stronger model should self-report confidence and skip tracing
   more often on its own. Neither is benchmarked against plain `direct`
   or `cot` yet.
2. What additional LLM-native benchmarks belong in the evaluation.
3. Gemini's thinking mode is not yet correctly modeled in
   `model_config.py` (currently falls under `NO_THINKING`, which is wrong
   for Gemini 3.1 Pro reasoning) — fix before running it as the
   proprietary upper bound.