"""Research entity extraction through the public aicomputing API."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import os
from pathlib import Path
import time

from aicomputing import Constraint, ExecuteRequest, ToolUseDetected, ValidationStatus, VirtualRuntime
from aicomputing.backends.base import BackendCommunicationError, BackendTimeoutError

from .contract import EntityRecord, Extraction, SIGNATURE, SPECIFICATION, Sentence


@dataclass(frozen=True)
class Settings:
    model: str = "gpt-6.1-terra"
    timeout: float = 120
    max_output_tokens: int | None = None
    key_file: Path | None = field(default=None, repr=False)
    input_price: float | None = None
    output_price: float | None = None

    def __post_init__(self):
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("Model must be nonblank text")
        if type(self.timeout) not in (int, float) or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Timeout must be finite and positive")
        if self.max_output_tokens is not None and (type(self.max_output_tokens) is not int or self.max_output_tokens < 1):
            raise ValueError("Output token limit must be a positive integer or None")
        if self.key_file is not None and not isinstance(self.key_file, Path):
            raise ValueError("key_file must be a Path")
        if (self.input_price is None) != (self.output_price is None):
            raise ValueError("Specify both input and output prices or neither")
        for value in (self.input_price, self.output_price):
            if value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0):
                raise ValueError("Prices must be finite and nonnegative")

    @classmethod
    def from_env(cls):
        def price(name):
            value = os.environ.get(name)
            return float(value) if value else None
        key_file = os.environ.get("PAPER_PARSER_KEY_FILE")
        return cls(model=os.environ.get("PAPER_PARSER_MODEL", "gpt-6.1-terra"),
                   timeout=float(os.environ.get("PAPER_PARSER_TIMEOUT", "120")),
                   max_output_tokens=(int(os.environ["PAPER_PARSER_MAX_OUTPUT_TOKENS"])
                                      if os.environ.get("PAPER_PARSER_MAX_OUTPUT_TOKENS") else None),
                   key_file=Path(key_file) if key_file else None,
                   input_price=price("PAPER_PARSER_INPUT_PRICE"), output_price=price("PAPER_PARSER_OUTPUT_PRICE"))

    def has_credentials(self):
        return self.key_file is not None or bool(os.environ.get("OPENAI_API_KEY", "").strip())

    def public(self):
        return {"model": self.model, "strategy": "direct", "reasoning_effort": "none",
                "http_timeout_seconds": self.timeout, "max_output_tokens": self.max_output_tokens,
                "validation_mode": "strict", "validation_retries": 0, "sdk_retries": 0, "cache_size": 0,
                "input_price_per_million": self.input_price, "output_price_per_million": self.output_price}

    def runtime(self):
        if not self.has_credentials():
            raise ValueError("Configure OPENAI_API_KEY or PAPER_PARSER_KEY_FILE")
        key = None
        if self.key_file is not None:
            values = []
            for line in self.key_file.read_text(encoding="utf-8-sig").splitlines():
                line = line.strip()
                if line.startswith("export "):
                    line = line[7:].lstrip()
                name, separator, value = line.partition("=")
                if separator and name.strip() == "OPENAI_API_KEY":
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                        value = value[1:-1]
                    if value and not any(char.isspace() for char in value):
                        values.append(value)
            if len(values) != 1:
                raise ValueError("Credential file must contain one nonempty OPENAI_API_KEY assignment")
            key = values[0]
        return VirtualRuntime(backend="openai", model=self.model, strategy="direct", api_key=key,
                              validation_mode="strict", cache_size=0, reasoning_effort="none",
                              timeout_seconds=self.timeout, sdk_max_retries=0, request_diagnostics=True)


def extract(sentence: Sentence, runtime, settings: Settings) -> Extraction:
    issues = []

    def schema(value):
        try:
            EntityRecord.from_dict(value, sentence)
            return True
        except ValueError as exc:
            issues.append(str(exc))
            return False

    request = ExecuteRequest(signature=SIGNATURE, docstring=SPECIFICATION,
                             inputs={"tokens": list(sentence.tokens)},
                             constraints=[Constraint("valid_token_spans", schema,
                                                     "Return valid, nonoverlapping spans over the supplied tokens.")],
                             validation_mode="strict", max_retries=0, max_new_tokens=settings.max_output_tokens)
    started = time.perf_counter()
    try:
        prediction = runtime.execute(request)
    except (BackendCommunicationError, ToolUseDetected, ModuleNotFoundError) as exc:
        status = ("timeout" if isinstance(exc, BackendTimeoutError) else
                  "provider_error" if isinstance(exc, BackendCommunicationError) else
                  "provider_unavailable" if isinstance(exc, ModuleNotFoundError) else "tool_use_rejected")
        # Exception messages can contain credentials or request headers.
        return Extraction("model", status, None, time.perf_counter() - started,
                          error_type=type(exc).__name__,
                          diagnostics=getattr(exc, "diagnostics", None) or {})
    evidence = asdict(prediction)
    warnings = list(prediction.type_warnings) + list(prediction.constraint_warnings)
    warnings += issues[-1:]
    if prediction.parse_error:
        warnings.append(prediction.parse_error)
    valid = prediction.failure_type is None and prediction.validation_status == ValidationStatus.VALID
    return Extraction("model", "completed" if valid else "invalid_output",
                      EntityRecord.from_dict(prediction.result, sentence) if valid else None,
                      time.perf_counter() - started, warnings=warnings, prediction=evidence)


def estimated_cost(outcome, settings):
    if outcome.engine == "rules":
        return 0.0
    if settings.input_price is None or outcome.prediction is None:
        return None
    incoming, outgoing = (outcome.prediction.get(key) for key in ("input_tokens", "output_tokens"))
    if type(incoming) is not int or type(outgoing) is not int:
        return None
    return (incoming * settings.input_price + outgoing * settings.output_price) / 1_000_000
