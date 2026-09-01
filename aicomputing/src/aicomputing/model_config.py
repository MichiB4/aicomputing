"""
Model configuration registry for AI Computing.

Maps model name patterns to their thinking / reasoning behaviour so the
runtime can handle the `think` flag correctly for each provider and model
family without requiring manual configuration.

Thinking behaviour classes
--------------------------
ALWAYS_THINKS
    The model always generates a reasoning trace before answering.
    Disabling thinking is not supported — `think=False` is ignored with
    a warning. Example: DeepSeek-R1.

THINKING_TOGGLE
    The model supports both thinking and non-thinking modes via a soft
    instruction appended to the prompt (`/no_think` for Qwen3 family,
    or a system message for others). Example: Qwen3-32B, GPT-OSS-120B.

NO_THINKING
    The model has no thinking capability — it answers directly.
    The `think` parameter is irrelevant. Example: Llama 3.3, Phi-4-mini,
    all instruct variants without CoT fine-tuning.

Provider presets
----------------
PROVIDER_PRESETS maps a short provider name (e.g. "groq", "cerebras") to
the base URL and the models available on that provider. Used by benchmark
scripts to avoid repeating long URLs in every command.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class ThinkingMode(Enum):
    ALWAYS_THINKS = auto()    # DeepSeek-R1 family
    THINKING_TOGGLE = auto()  # Qwen3, GPT-OSS-120B — toggle via /no_think
    NO_THINKING = auto()      # Llama, Phi, instruct-only models


# ---------------------------------------------------------------------------
# Model registry
#
# Keys are lowercase substrings matched against the model name.
# Order matters — first match wins, so more specific patterns come first.
# ---------------------------------------------------------------------------

_MODEL_PATTERNS: list[tuple[str, ThinkingMode]] = [
    # DeepSeek-R1 family — always reasons, no toggle
    ("deepseek-r1",          ThinkingMode.ALWAYS_THINKS),
    ("deepseek/r1",          ThinkingMode.ALWAYS_THINKS),
    ("r1-distill",           ThinkingMode.ALWAYS_THINKS),

    # Instruct-only variants of thinking model families were fine-tuned
    # WITHOUT the thinking capability (e.g. "qwen3:4b-instruct"). This
    # must come BEFORE the family patterns below, because those names
    # also contain "qwen3".
    ("-instruct",            ThinkingMode.NO_THINKING),

    # Qwen3 thinking variants — have the /no_think toggle
    ("qwen3",                ThinkingMode.THINKING_TOGGLE),
    ("qwq",                  ThinkingMode.THINKING_TOGGLE),

    # GPT-OSS — OpenAI's open-source model, supports /no_think
    ("gpt-oss",              ThinkingMode.THINKING_TOGGLE),

    # Non-thinking models
    ("llama",                ThinkingMode.NO_THINKING),
    ("phi",                  ThinkingMode.NO_THINKING),
    ("mistral",              ThinkingMode.NO_THINKING),
    ("gemma",                ThinkingMode.NO_THINKING),
    ("gemini",               ThinkingMode.NO_THINKING),
    ("claude",               ThinkingMode.NO_THINKING),
    ("gpt-4",                ThinkingMode.NO_THINKING),
    ("gpt-5",                ThinkingMode.NO_THINKING),
    ("qwen2",                ThinkingMode.NO_THINKING),
]


def get_thinking_mode(model: str) -> ThinkingMode:
    """Return the thinking behaviour for a given model name."""
    lower = model.lower()
    for pattern, mode in _MODEL_PATTERNS:
        if pattern in lower:
            return mode
    # Unknown model — assume no thinking to avoid appending /no_think incorrectly
    return ThinkingMode.NO_THINKING


@dataclass(frozen=True)
class ThinkingResolution:
    """
    What a strategy asked for, and what the model will actually do.

    These are not the same thing, and the difference is an experimental
    confound rather than a detail. `direct` is the minimal execution
    condition -- specification + input -> model -> output -- so it requests
    think=False. On an ALWAYS_THINKS model that request cannot be honoured:
    the model reasons regardless, and `direct` silently stops being the
    minimal condition while still being reported as it.

    Nothing here changes what the model does; the point is that the run
    records what actually happened, so a Direct cell measured on a model that
    cannot disable reasoning is identifiable rather than pooled with cells
    where the condition held.

        requested   what the strategy asked for
        effective   what the model will actually do
        honoured    whether the two agree
        reason      why not, when they disagree
    """

    model: str
    mode: "ThinkingMode"
    requested: bool
    effective: bool
    reason: Optional[str] = None

    @property
    def honoured(self) -> bool:
        return self.requested == self.effective


def resolve_thinking(model: str, requested: bool) -> ThinkingResolution:
    """
    Resolve a requested thinking mode against a model family's capability.

    Single place where model-specific thinking behaviour is interpreted, so
    strategies never branch on model name. Backends call this and report the
    result; they do not decide it.
    """
    mode = get_thinking_mode(model)

    if mode == ThinkingMode.ALWAYS_THINKS:
        if not requested:
            return ThinkingResolution(
                model, mode, requested=False, effective=True,
                reason=("model always generates reasoning traces; "
                        "think=False cannot be honoured"),
            )
        return ThinkingResolution(model, mode, True, True)

    if mode == ThinkingMode.THINKING_TOGGLE:
        # /no_think is a soft instruction, not an API guarantee, but it is the
        # documented mechanism for these families and is the best available.
        return ThinkingResolution(model, mode, requested, requested)

    # NO_THINKING: the model has no reasoning trace to enable. A request for
    # thinking cannot be honoured either -- worth recording, because a CoT
    # cell on such a model is testing the visible SCoT trace alone.
    if requested:
        return ThinkingResolution(
            model, mode, requested=True, effective=False,
            reason="model has no native reasoning mode; think=True has no effect",
        )
    return ThinkingResolution(model, mode, False, False)


def supports_no_think_toggle(model: str) -> bool:
    """True if the model supports the /no_think instruction to disable thinking."""
    return get_thinking_mode(model) == ThinkingMode.THINKING_TOGGLE


def always_thinks(model: str) -> bool:
    """True if the model always generates reasoning traces (e.g. DeepSeek-R1)."""
    return get_thinking_mode(model) == ThinkingMode.ALWAYS_THINKS


# Default for the `think` flag, shared by every backend.
#
# Backends must not each pick their own default: the same call issued against
# Ollama and against an OpenAI-compatible provider has to mean the same thing,
# or a caller that omits `think` silently gets reasoning on one provider and
# not on the other. Strategies always pass `think` explicitly, so this governs
# direct backend use (e.g. benchmarks/baselines/code_gen_baseline.py).
#
# False is the default because reasoning is the expensive, opt-in behaviour and
# it lines up with _BUDGET_NO_THINK below.
DEFAULT_THINK = False


# Token budgets — simple and fixed.
#   NO_THINKING  →  512   (JSON answer is short; SCoT trace also fits)
#   thinking     →  8192  (reasoning trace consumes invisible tokens)
_BUDGET_NO_THINK = 512
_BUDGET_THINK    = 8192

# Default starting token budget when a caller omits max_new_tokens entirely.
# Matches the non-thinking budget; every in-package call path passes an
# explicit value computed by get_token_budget().
DEFAULT_MAX_NEW_TOKENS = _BUDGET_NO_THINK


def get_token_budget(model: str, think: bool) -> int:
    """
    Return the starting token budget for a model + think combination.

    Rules:
      - think=True  → 8192 (reasoning trace needs space)
      - think=False → 512  (short JSON answer; SCoT trace fits too)
      - ALWAYS_THINKS models ignore think=False and always get 8192
    """
    if not think:
        mode = get_thinking_mode(model)
        if mode == ThinkingMode.ALWAYS_THINKS:
            return _BUDGET_THINK   # can't disable thinking, give it space
        return _BUDGET_NO_THINK
    return _BUDGET_THINK


# ---------------------------------------------------------------------------
# Provider presets
#
# Maps a short --provider name to base_url and a note about available models.
# Used by benchmark scripts so users only need to pass --provider groq
# instead of the full URL.
# ---------------------------------------------------------------------------

# Provider model catalogs drift over time (models get renamed/delisted).
# The groq/cerebras lists below were last verified against the live
# GET /v1/models endpoint of each provider on 2026-07-12 — qwen/qwen3-32b
# no longer exists on Groq (superseded by qwen/qwen3.6-27b), and Cerebras's
# free tier no longer serves qwen3-235b-a22b, llama3.3-70b, or the
# "openai/"-prefixed gpt-oss-120b naming at all (it's unprefixed there).
# If a benchmark run 404s with "model_not_found", re-verify with:
#   from openai import OpenAI
#   OpenAI(base_url=PROVIDER_PRESETS[p]["base_url"], api_key=...).models.list()
PROVIDER_PRESETS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            # Re-verified against Groq's live /v1/models on 2026-08-29. Both
            # Llama entries that were here (llama-3.3-70b-versatile,
            # llama-3.1-8b-instant) have been withdrawn; qwen3.8-27b is new.
            "openai/gpt-oss-120b",
            "qwen/qwen3.6-27b",
            "qwen/qwen3.8-27b",
        ],
        "notes": "30 RPM, 1K req/day for 70B+. Fastest inference.",
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        "models": [
            "gpt-oss-120b",
            "gemma-4-31b",
            "zai-glm-4.7",
        ],
        "notes": "30 RPM, 1M tokens/day. Best free throughput. No Qwen3 "
                 "or Llama 3.3 currently offered on the free tier (as of "
                 "the 2026-07-12 verification above).",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "deepseek/deepseek-r1:free",
            "qwen/qwen3-235b-a22b:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ],
        "notes": "50 req/day free, 1K/day with $10 balance.",
    },
    "sambanova": {
        "base_url": "https://api.sambanova.ai/v1",
        "models": [
            "DeepSeek-R1",
            "Llama-3.3-70B-Instruct",
        ],
        "notes": "Persistent free tier, 10-30 RPM.",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "models": [
            "qwen3:8b-instruct",
            "qwen3:4b-instruct",
            "qwen2.5-coder:7b",
            "llama3.3:8b",
            "phi4-mini",
        ],
        "notes": "Local inference, no rate limits.",
    },
}


def get_provider_preset(provider: str) -> Optional[dict]:
    """Return preset config for a provider name, or None if unknown."""
    return PROVIDER_PRESETS.get(provider.lower())


def list_providers() -> list[str]:
    """Return all known provider names."""
    return list(PROVIDER_PRESETS.keys())
