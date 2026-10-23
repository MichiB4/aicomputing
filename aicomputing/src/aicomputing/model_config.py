"""Shared starting-token allowances and static provider presets.

The ordered model-name rules preserve existing budget choices. They do not
classify native reasoning capabilities or configure backend thinking controls.
Explicit backend settings determine which native controls are sent.

Provider presets contain static connection settings and example model IDs;
they do not verify current model availability or account limits.
"""

from __future__ import annotations

from typing import Optional


_BUDGET_DEFAULT = 512
_BUDGET_EXTENDED = 8192

# Default for direct backend calls that omit max_new_tokens.
DEFAULT_MAX_NEW_TOKENS = _BUDGET_DEFAULT

# First matching lowercase substring wins. Preserve this order, including
# smaller-budget entries that can precede other patterns in combined names.
_MODEL_BUDGETS: list[tuple[str, int]] = [
    ('deepseek-r1', _BUDGET_EXTENDED),
    ('deepseek/r1', _BUDGET_EXTENDED),
    ('r1-distill', _BUDGET_EXTENDED),
    ('qwen3-4b-instruct-2507', _BUDGET_DEFAULT),
    ('qwen3-30b-a3b-instruct-2507', _BUDGET_DEFAULT),
    ('qwen3-235b-a22b-instruct-2507', _BUDGET_DEFAULT),
    ('qwen3:4b-instruct', _BUDGET_DEFAULT),
    ('qwen3:30b-a3b-instruct-2507', _BUDGET_DEFAULT),
    ('qwen3:30b-instruct', _BUDGET_DEFAULT),
    ('qwen3:235b-a22b-instruct-2507', _BUDGET_DEFAULT),
    ('qwen3:235b-instruct', _BUDGET_DEFAULT),
    ('qwen3-4b-thinking-2507', _BUDGET_EXTENDED),
    ('qwen3-30b-a3b-thinking-2507', _BUDGET_EXTENDED),
    ('qwen3-235b-a22b-thinking-2507', _BUDGET_EXTENDED),
    ('qwen3:4b-thinking', _BUDGET_EXTENDED),
    ('qwen3:30b-a3b-thinking-2507', _BUDGET_EXTENDED),
    ('qwen3:30b-thinking', _BUDGET_EXTENDED),
    ('qwen3:235b-a22b-thinking-2507', _BUDGET_EXTENDED),
    ('qwen3:235b-thinking', _BUDGET_EXTENDED),
    ('qwen3', _BUDGET_DEFAULT),
    ('qwq', _BUDGET_EXTENDED),
    ('gpt-oss', _BUDGET_EXTENDED),
]


def get_token_budget(model: str, extended: bool) -> int:
    """Select a starting allowance without configuring native reasoning.

    Extended calls receive 8192 tokens. Other calls use the ordered model
    rules, falling back to 512. Automatic truncation retries may grow either.
    """
    if extended:
        return _BUDGET_EXTENDED
    lower = model.lower()
    for pattern, budget in _MODEL_BUDGETS:
        if pattern in lower:
            return budget
    return _BUDGET_DEFAULT


# ---------------------------------------------------------------------------
# Provider presets
#
# Static endpoint settings and example model IDs for benchmark convenience.
# Model availability and account limits must be checked with each provider.
# ---------------------------------------------------------------------------

PROVIDER_PRESETS: dict[str, dict] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        # Examples from https://console.groq.com/docs/models.
        "models": [
            "openai/gpt-oss-120b",
            "qwen/qwen3.8-27b",  # Preview model.
        ],
        "notes": (
            "Model availability and rate limits depend on the account "
            "and model. Check the provider console before a benchmark run."
        ),
    },
    "cerebras": {
        "base_url": "https://api.cerebras.ai/v1",
        # Examples from https://inference-docs.cerebras.ai/models/overview.
        "models": [
            "gpt-oss-120b",
            "qwen-3.8-27b",
        ],
        "notes": (
            "Model availability and rate limits depend on the account, "
            "model, and service tier. Check the provider console "
            "before a benchmark run."
        ),
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "deepseek/deepseek-r1:free",
            "meta-llama/llama-3.3-70b-instruct:free",
        ],
        "notes": (
            "Free model variants have usage limits and may be unavailable. "
            "Check current model routes and account limits before a run: "
            "https://openrouter.ai/docs/api_reference/limits"
        ),
    },
    "sambanova": {
        "base_url": "https://api.sambanova.ai/v1",
        # Examples from https://docs.sambanova.ai/docs/en/models/sambacloud-models.
        "models": [
            "gpt-oss-120b",
            "Meta-Llama-3.3-70B-Instruct",
        ],
        "notes": (
            "Model availability and rate limits depend on the account, "
            "model, and service tier. Check the provider console "
            "before a benchmark run."
        ),
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "models": [
            "qwen3:4b-instruct",
            "qwen2.5-coder:7b",
            "phi4-mini",
        ],
        "notes": (
            "Local inference. Pull the selected model before use. "
            "Throughput depends on hardware and server configuration."
        ),
    },
}


def get_provider_preset(provider: str) -> Optional[dict]:
    """Look up a registered provider name case-insensitively.

    Return None if the name is unregistered. The returned dictionary is
    shared with PROVIDER_PRESETS; modifying it changes the registry.
    """
    return PROVIDER_PRESETS.get(provider.lower())


def list_providers() -> list[str]:
    """Return registered provider names in registry insertion order."""
    return list(PROVIDER_PRESETS.keys())
