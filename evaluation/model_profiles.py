"""Model identities, connection defaults and token-price assumptions."""
from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal


@dataclass(frozen=True)
class ModelProfile:
    model: str
    input_rate: Decimal
    cached_input_rate: Decimal
    cache_write_rate: Decimal
    output_rate: Decimal
    price_checked_date: str
    output_reservation: int = 128_000
    max_input_reservation: int = 272_000
    framing_reserve: int = 4096
    provider: str = "openai"
    api_base: str = "https://api.openai.com/v1"
    key_env: str | None = "OPENAI_API_KEY"
    price_source: str | None = None
    service_tier: str | None = "default"
    reasoning_effort: str = "none"
    api_format: str = "chat_completions"

    def pricing_snapshot(self):
        """Record per-million rates while accounting uses per-token Decimals."""
        return {
            "input_usd_per_million": format((self.input_rate * 1_000_000).normalize(), "f"),
            "cached_input_usd_per_million": format((self.cached_input_rate * 1_000_000).normalize(), "f"),
            "cache_write_usd_per_million": format((self.cache_write_rate * 1_000_000).normalize(), "f"),
            "output_usd_per_million": format((self.output_rate * 1_000_000).normalize(), "f"),
            "provider_output_reservation_tokens": self.output_reservation,
            "max_input_reservation_tokens": self.max_input_reservation,
            "framing_reserve": self.framing_reserve,
            "price_source": self.price_source or "https://developers.openai.com/api/docs/models/" + self.model,
            "price_checked_date": self.price_checked_date,
        }


_PROFILES = {
    "gpt-6-luna": ModelProfile(
        model="gpt-6-luna",
        input_rate=Decimal("0.10") / 1_000_000,
        cached_input_rate=Decimal("0.01") / 1_000_000,
        cache_write_rate=Decimal("0.125") / 1_000_000,
        output_rate=Decimal("0.50") / 1_000_000,
        price_checked_date="2026-11-16",
    ),
    "gpt-6.1-terra": ModelProfile(
        model="gpt-6.1-terra",
        input_rate=Decimal("1.00") / 1_000_000,
        cached_input_rate=Decimal("0.05") / 1_000_000,
        cache_write_rate=Decimal("1.25") / 1_000_000,
        output_rate=Decimal("5.00") / 1_000_000,
        price_checked_date="2026-11-16",
    ),
    "gpt-6.1-sol": ModelProfile(
        model="gpt-6.1-sol",
        input_rate=Decimal("2.00") / 1_000_000,
        cached_input_rate=Decimal("0.10") / 1_000_000,
        cache_write_rate=Decimal("2.50") / 1_000_000,
        output_rate=Decimal("10.00") / 1_000_000,
        price_checked_date="2026-11-16",
    ),
    "qwen-3.8-27b": ModelProfile(
        model="qwen-3.8-27b",
        input_rate=Decimal("0.99") / 1_000_000,
        cached_input_rate=Decimal("0.99") / 1_000_000,
        cache_write_rate=Decimal("0.99") / 1_000_000,
        output_rate=Decimal("1.49") / 1_000_000,
        price_checked_date="2026-11-16",
        output_reservation=32_768, max_input_reservation=65_536,
        provider="cerebras", api_base="https://api.cerebras.ai/v1",
        key_env="CEREBRAS_API_KEY", price_source="https://api.cerebras.ai/public/v1/models",
        service_tier=None,
    ),
    "qwen3:0.6b": ModelProfile(
        model="qwen3:0.6b",
        input_rate=Decimal(0), cached_input_rate=Decimal(0),
        cache_write_rate=Decimal(0), output_rate=Decimal(0),
        price_checked_date="2026-10-30",
        output_reservation=32_768, max_input_reservation=32_768,
        provider="ollama", api_base="http://127.0.0.1:11434/v1", key_env=None,
        price_source="https://docs.ollama.com/api/usage", service_tier=None,
    ),
}
SUPPORTED_MODELS = tuple(_PROFILES)


def get_model_profile(model, *, reasoning_effort="none"):
    """Look up a known model; custom models supply their own rates in the CLI."""
    if not isinstance(model, str) or model not in _PROFILES:
        raise ValueError("Unsupported evaluation model")
    profile = _PROFILES[model]
    if reasoning_effort not in ("none", "low", "medium", "high", "xhigh", "max"):
        raise ValueError("Unsupported reasoning effort")
    if profile.provider != "openai" and reasoning_effort != "none":
        raise ValueError("Native reasoning levels require the OpenAI provider")
    return replace(profile, reasoning_effort=reasoning_effort,
                   api_format="chat_completions" if reasoning_effort == "none" else "responses")
