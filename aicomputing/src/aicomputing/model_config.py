"""Uncapped generation defaults and static provider presets.

Explicit token limits and provider capacities still apply. Presets contain
example model IDs, not verified availability or account limits.
"""

from __future__ import annotations

from typing import Optional


DEFAULT_MAX_NEW_TOKENS: Optional[int] = None


def get_token_budget(model: str, extended: bool) -> Optional[int]:
    """Return the uncapped default; arguments are retained for compatibility."""
    return DEFAULT_MAX_NEW_TOKENS


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
    """Look up a provider case-insensitively, returning None if unknown.

    The returned dictionary is shared with PROVIDER_PRESETS.
    """
    return PROVIDER_PRESETS.get(provider.lower())


def list_providers() -> list[str]:
    """Return registered provider names in registry insertion order."""
    return list(PROVIDER_PRESETS)
