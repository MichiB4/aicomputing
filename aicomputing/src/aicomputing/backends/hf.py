"""
HuggingFace Transformers backend for AI Computing.

Loads a model locally via the transformers library.
Install: pip install aicomputing[hf]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..model_config import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_THINK,
    ThinkingMode,
    get_thinking_mode,
)
from ._text_utils import append_no_think
from .base import Backend, GenerateResponse

logger = logging.getLogger("aicomputing")


class HFBackend(Backend):
    """Backend using HuggingFace Transformers (local inference)."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: Optional[str] = None,
    ) -> None:
        self._model_name = model
        self._device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        if self._tokenizer.pad_token is None and self._tokenizer.eos_token:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name, torch_dtype="auto",
        )
        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._device = device

    @property
    def name(self) -> str:
        return "hf"

    @property
    def default_model(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        stop: Optional[List[str]] = None,
        think: bool = DEFAULT_THINK,
        **kwargs,
    ) -> GenerateResponse:
        import torch
        self._load()

        mdl = model or self._model_name

        # Honour `think` the same way the networked backends do: the toggle is
        # a prompt-level convention (/no_think), not an API parameter, so it
        # works identically for local inference. Previously this backend had
        # no `think` parameter at all and silently swallowed it via **kwargs,
        # so thinking-mode ablations were a no-op here.
        messages = [{"role": "user", "content": prompt}]
        thinking_mode = get_thinking_mode(mdl)
        if thinking_mode == ThinkingMode.THINKING_TOGGLE and not think:
            messages = append_no_think(messages)
        elif thinking_mode == ThinkingMode.ALWAYS_THINKS and not think:
            logger.warning(
                "Model %r always generates reasoning traces — `think=False` "
                "has no effect on the hf backend.", mdl,
            )

        # Try chat template, fall back to raw prompt
        try:
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            text = messages[-1]["content"]

        inputs = self._tokenizer([text], return_tensors="pt").to(self._device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
        }
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs["do_sample"] = False  # greedy decoding

        with torch.inference_mode():
            gen_ids = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = gen_ids[0][len(inputs.input_ids[0]):]
        raw_text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # If the model used the full budget without emitting EOS, it was
        # cut off. This drives the strategy layer's budget-doubling retry.
        n_new = len(new_tokens)
        ended_with_eos = (
            n_new > 0
            and self._tokenizer.eos_token_id is not None
            and new_tokens[-1].item() == self._tokenizer.eos_token_id
        )
        finish_reason = "length" if (n_new >= max_new_tokens and not ended_with_eos) else "stop"

        return GenerateResponse(
            text=raw_text,
            model=model or self._model_name,
            tokens_used=n_new,
            finish_reason=finish_reason,
            provider_finish_reason=finish_reason,
        )

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            import torch  # noqa: F401
            return True
        except ImportError:
            return False
