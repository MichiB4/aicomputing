"""Local Transformers inference for one configured model; stop sequences are unsupported.

enable_thinking is passed to the chat template; support depends on that template.
Recorded settings do not establish the model's reasoning mode.

None permits generation until EOS or the remaining model context. Unknown
context capacity requires an explicit token allowance. Leading <think> blocks
are stripped, including openers supplied by the template; raw output is retained.

Install: pip install aicomputing[hf]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..model_config import DEFAULT_MAX_NEW_TOKENS
from ..request import validate_generation_limits
from ._text_utils import strip_thinking
from .base import Backend, GenerateResponse, ReasoningCapabilities


def _declared_context_length(model: Any, tokenizer: Any) -> Optional[int]:
    """Find finite context capacity, ignoring output defaults and sentinels >=1e20.

    Prefer text configuration, then model configuration, then tokenizer fallback.
    Within configurations, prefer max_position_embeddings over legacy aliases.
    Sliding windows and RoPE factors do not define total capacity.
    """
    config = getattr(model, "config", None)
    configurations = (getattr(config, "text_config", None), config)
    for configuration in configurations:
        for field in (
            "max_position_embeddings", "n_positions", "n_ctx",
            "max_seq_len", "max_sequence_length", "seq_length",
        ):
            value = getattr(configuration, field, None)
            if type(value) is int and 0 < value < 10**20:
                return value
    value = getattr(tokenizer, "model_max_length", None)
    if type(value) is int and 0 < value < 10**20:
        return value
    return None


class HFBackend(Backend):
    """Backend using HuggingFace Transformers (local inference)."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: Optional[str] = None,
        *,
        enable_thinking: Optional[bool] = None,
    ) -> None:
        if enable_thinking is not None and not isinstance(enable_thinking, bool):
            raise TypeError("enable_thinking must be a bool or None")

        self._enable_thinking = enable_thinking
        self._model_name = model
        # Preserve the selection while _load resolves the execution device.
        self._configured_device = device
        self._device = device
        self._model = None
        self._tokenizer = None

    def close(self) -> None:
        """Drop loaded model/tokenizer references; later generation reloads.

        This does not import inference dependencies or clear process-wide
        GPU allocator caches. Do not close during generation.
        """
        self._model = None
        self._tokenizer = None
        self._device = self._configured_device

    reasoning_parameter = "enable_thinking"
    _reasoning_value_types = (bool,)

    def reasoning_capabilities(
        self, model: Optional[str] = None, *, discover: bool = False,
    ) -> ReasoningCapabilities:
        local = super().reasoning_capabilities(model, discover=discover)
        if local.model == self._model_name and self._tokenizer is not None:
            if getattr(self._tokenizer, "chat_template", None) is None:
                return ReasoningCapabilities(
                    backend=local.backend, model=local.model, parameter=local.parameter,
                    support="unsupported", source="loaded_tokenizer_without_chat_template",
                )
        return local

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if tokenizer.pad_token is None and tokenizer.eos_token:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self._model_name, torch_dtype="auto",
        )
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Publish initialized state only after device placement succeeds.
        self._tokenizer = tokenizer
        self._device = device
        self._model = model

    @property
    def name(self) -> str:
        return "hf"

    @property
    def default_model(self) -> str:
        return self._model_name

    def cache_state(self) -> Optional[Dict[str, Any]]:
        """Describe configured model, device selection and template control."""
        if type(self) is not HFBackend:
            # Subclasses must explicitly account for their additional behavior.
            return None
        return {
            "default_model": self._model_name,
            "device": self._configured_device,
            "enable_thinking": self._enable_thinking,
        }

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: Optional[int] = DEFAULT_MAX_NEW_TOKENS,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> GenerateResponse:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported generation options: {names}")
        validate_generation_limits(max_new_tokens, 0)
        if model is not None and model != self._model_name:
            raise ValueError(
                f"HFBackend cannot use model {model!r}; it is configured for "
                f"{self._model_name!r}. Create another HFBackend for a different model."
            )
        if stop:
            raise ValueError("HFBackend does not support nonempty stop sequences")

        import torch
        self._load()

        messages = [{"role": "user", "content": prompt}]
        template_kwargs: Dict[str, Any] = {}
        if self._enable_thinking is not None:
            template_kwargs["enable_thinking"] = self._enable_thinking

        has_chat_template = self._tokenizer.chat_template is not None
        if not has_chat_template:
            if template_kwargs:
                raise ValueError(
                    "enable_thinking requires a tokenizer chat template"
                )
            text = prompt
        else:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                **template_kwargs,
            )

        # Rendered chat templates already supply their required special tokens.
        inputs = self._tokenizer(
            [text], return_tensors="pt", add_special_tokens=not has_chat_template,
        ).to(self._device)
        input_length = len(inputs.input_ids[0])
        context_length = _declared_context_length(self._model, self._tokenizer)
        effective_max_new_tokens = max_new_tokens
        if effective_max_new_tokens is None:
            # Avoid Transformers' finite default by using the remaining context.
            if context_length is None:
                raise ValueError(
                    "HFBackend cannot infer a finite model context length; "
                    "set an explicit positive max_new_tokens for this model."
                )
            effective_max_new_tokens = context_length - input_length
            if effective_max_new_tokens <= 0:
                raise ValueError(
                    f"HFBackend input uses {input_length} tokens and leaves no "
                    f"generation space in the {context_length}-token context."
                )

        eos_token_id = self._model.generation_config.eos_token_id
        if eos_token_id is None:
            eos_token_id = self._tokenizer.eos_token_id
        eos_token_ids = (
            [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id or []
        )

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": effective_max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs["do_sample"] = False  # greedy decoding

        with torch.inference_mode():
            gen_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = gen_ids[0][input_length:]
        generated_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        raw_text = generated_text.strip()
        # A template may supply the opener before newly generated tokens begin.
        if has_chat_template and text.rstrip().endswith("<think>"):
            raw_text = "<think>" + raw_text
        raw_text = strip_thinking(raw_text)

        # Reaching the allowance without EOS signals truncation.
        n_new = len(new_tokens)
        ended_with_eos = (
            n_new > 0
            and new_tokens[-1].item() in eos_token_ids
        )
        finish_reason = (
            "length" if (n_new >= effective_max_new_tokens and not ended_with_eos)
            else "stop"
        )

        return GenerateResponse(
            text=raw_text,
            model=self._model_name,
            tokens_used=n_new,
            input_tokens=input_length,
            output_tokens=n_new,
            requested_reasoning=dict(template_kwargs),
            finish_reason=finish_reason,
            raw={
                "generated_text": generated_text,
                "generation_limits": {
                    "policy": "model_context" if max_new_tokens is None else "explicit",
                    "requested_max_new_tokens": max_new_tokens,
                    "effective_max_new_tokens": effective_max_new_tokens,
                    "context_length": context_length,
                },
            },
        )

    def is_available(self) -> bool:
        """Check dependency importability without loading a model."""
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False
