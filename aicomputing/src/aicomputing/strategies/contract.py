"""
Contract-check strategy (predict, then test the prediction against the spec).

Two calls. The first is the `direct` prompt verbatim, so the difference between
this strategy's score and `direct`'s is attributable to the check alone rather
than to a different way of asking the original question. The second presents the
resulting value back as an unverified proposal and asks whether the
specification requires it, accepting or replacing it.

Why the specification and not something else. A virtual runtime has no
implementation to diff against and no interpreter to appeal to; the docstring is
the only statement of intended behaviour available at inference time. Testing a
candidate against it is the one form of verification the paradigm admits without
reintroducing the runtime.

Relation to the formal account. `VirtualRuntime` already discharges the E-AI
validation premise at the *type* level — parse, coerce, `validate_output`,
`validate_constraints`, retry. Nothing in that path asks whether a well-typed
value is the *right* value, so a plausible wrong answer passes unremarked. This
strategy discharges the premise at the contract level instead, and it does so
inside the strategy rather than in the runtime, which is why the premise holds
for some arms and not others.

PROVISIONAL, and instrumented accordingly. The known failure mode is a model
that rubber-stamps a candidate rather than testing it, which yields a strategy
identical to `direct` at twice the cost. The prompt mitigates by never
attributing the candidate to the model (see CONTRACT_CHECK_SYSTEM), but wording
does not settle the question. Every result therefore carries:

  revised             — did the check actually change the value?
  pre_revision_value  — what the value was before the check

from which the revision rate and the flip rate (correct -> incorrect versus
incorrect -> correct) are computable per cell. A revision rate at or near zero
is a reportable finding about self-verification, not a failed run.

Fail-safe direction. A check whose verdict cannot be parsed keeps the candidate.
A garbled second call must never be able to corrupt a good first answer, so
replacement requires an explicit "revise" verdict *and* an extractable value.

Token budget: resolves think=True unless the request says otherwise. Both calls
are sized from the same budget; `tokens_used` and `retries` are summed across
them, matching how `consistency` reports its samples.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Optional, Tuple

from ..backends.base import Backend
from ..model_config import get_token_budget
from ..parsing import coerce_type, parse_result
from ..prompting import (
    build_contract_check_prompt,
    build_direct_prompt,
    build_retry_prompt,
)
from ..request import ExecuteRequest
from .base import (CallStats, Strategy, StrategyResult,
                   generate_with_retry, make_parser, _is_strict)

logger = logging.getLogger("aicomputing")

_VERDICT_RE = re.compile(r"""["']?verdict["']?\s*:\s*["'](accept|revise)["']""", re.I)


class ContractStrategy(Strategy):
    """Predict with `direct`, then test the candidate against the docstring."""

    def __init__(self, include_fewshot: bool = True) -> None:
        self._include_fewshot = include_fewshot

    @property
    def name(self) -> str:
        return "contract"

    def execute(
        self,
        request: ExecuteRequest,
        backend: Backend,
        feedback: Optional[dict] = None,
    ) -> StrategyResult:
        # Stage-specific thinking, so that Direct vs Contract isolates the
        # VERIFICATION step rather than confounding it with reasoning.
        #
        # The prediction stage reuses the direct prompt verbatim, so it must
        # also reuse the direct CONDITION: think=False. If it reasoned, the
        # comparison would read "added reasoning + verification", and the
        # contribution of the check -- the thing this arm exists to measure --
        # could not be separated from it.
        #
        # The check stage deliberately reasons: deciding whether a value
        # satisfies a specification is the deliberative half of the arm.
        #
        # An explicit request.think from the caller overrides both, which is
        # what the --think/--no-think ablation needs.
        explicit = request.think
        predict_think = explicit if explicit is not None else False
        check_think = explicit if explicit is not None else True

        model_name = request.model or backend.default_model
        predict_budget_start = request.max_new_tokens or get_token_budget(
            model_name, predict_think)
        check_budget_start = request.max_new_tokens or get_token_budget(
            model_name, check_think)

        def make_call(prompt: str, think: bool):
            def call(max_new_tokens: int):
                return backend.generate(
                    prompt,
                    model=request.model,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_new_tokens=max_new_tokens,
                    think=think,
                )
            return call

        # Two sinks. Both stages parse, but only the PREDICTION produces the
        # value this invocation returns; the check produces a verdict object.
        # Sharing one sink let the verifier's JSON overwrite the answer's
        # parse status, so a clean answer followed by a messy verdict was
        # reported as a fallback parse of the answer.
        psink: dict = {}
        vsink: dict = {}
        strict = _is_strict(request)
        parse = make_parser(request.return_type, strict, psink)
        parse_verdict = make_parser(None, strict, vsink)

        # --- Stage 1: predict, using the direct prompt unchanged -------------
        predict_prompt = build_direct_prompt(
            signature=request.signature,
            docstring=request.docstring,
            inputs=request.inputs,
            include_fewshot=self._include_fewshot,
        )
        if feedback:
            predict_prompt = build_retry_prompt(
                original_prompt=predict_prompt,
                raw_output=feedback.get("raw_output", ""),
                error_message=feedback.get("error", ""),
            )

        # One accumulator for both stages: contract's whole cost claim is that
        # it spends a second call, so that call must appear in the accounting.
        stats = CallStats()
        candidate, predict_response, predict_retries, predict_budget = generate_with_retry(
            call=make_call(predict_prompt, predict_think),
            parse=parse,
            initial_budget=predict_budget_start,
            model_name=model_name,
            stats=stats,
        )

        # --- Stage 2: test the candidate against the specification -----------
        check_prompt = build_contract_check_prompt(
            signature=request.signature,
            docstring=request.docstring,
            inputs=request.inputs,
            candidate=candidate,
            include_fewshot=self._include_fewshot,
        )

        _, check_response, check_retries, check_budget = generate_with_retry(
            call=make_call(check_prompt, check_think),
            parse=parse_verdict,
            initial_budget=check_budget_start,
            model_name=model_name,
            stats=stats,
        )

        final_value, revised = _apply_verdict(
            candidate=candidate,
            check_text=check_response.text,
            return_type=request.return_type,
        )

        tokens = _sum_optional(predict_response.tokens_used, check_response.tokens_used)

        return StrategyResult(
            value=final_value,
            # Both stages, in order: the prediction prompt then the
            # check prompt. Logging only the first would hide the half
            # of this strategy that distinguishes it from `direct`.
            prompts=[predict_prompt, check_prompt],
            raw_text=(
                f"[contract {'revise' if revised else 'accept'}] "
                f"{predict_response.text}\n--- check ---\n{check_response.text}"
            ),
            tokens_used=tokens,
            provider_finish_reason=check_response.provider_finish_reason,
            # Both calls start from the same budget but may double
            # independently. Report the larger ceiling, the figure that matters
            # against a provider's per-request cap — consistent with how
            # `consistency` aggregates across its samples.
            token_budget_used=max(predict_budget, check_budget),
            retries=predict_retries + check_retries,
            call_stats=stats,
            # The ANSWER's parse status, never the verifier's.
            parse_status=psink.get("parse_status"),
            parse_error=psink.get("parse_error"),
            verdict_parse_status=vsink.get("parse_status"),
            revised=revised,
            pre_revision_value=candidate,
        )


def _extract_verdict(check_text: str) -> Tuple[Optional[str], Any, bool]:
    """
    Pull the verdict object out of a check response.

    Returns (verdict, result, has_result). `has_result` distinguishes a
    verdict object that carried a result key from one that did not — including
    the case where the result really is None.

    The value is read out of the *verdict object itself* rather than via
    parse_result over the whole response. parse_result is total: when nothing
    parses it hands back the input string unchanged (parsing.py), so a rambling
    check with no JSON in it would otherwise yield a str that looks like a
    successful revision and silently replace a good answer with prose.
    """
    if not check_text:
        return None, None, False

    candidates = [m.group(0) for m in re.finditer(r"\{[^{}]*\}", check_text)]
    candidates += [
        m.group(0) for m in re.finditer(r"\{[\s\S]*\}", check_text)
        if m.group(0) not in candidates
    ]
    # Shortest first: the tightest object containing the verdict is the one the
    # prompt asked for, not an outer blob that happens to enclose it.
    candidates.sort(key=len)

    for cand in candidates:
        obj = None
        for attempt in (cand, cand.replace("'", '"')):
            try:
                obj = json.loads(attempt)
                break
            except (json.JSONDecodeError, ValueError):
                continue
        if obj is None:
            try:
                obj = ast.literal_eval(cand)
            except (ValueError, SyntaxError):
                continue
        if isinstance(obj, dict):
            verdict = obj.get("verdict")
            if isinstance(verdict, str) and verdict.lower() in ("accept", "revise"):
                return verdict.lower(), obj.get("result"), "result" in obj

    # No verdict object. Fall back to a bare verdict mention so that a model
    # which wrote the word without the JSON wrapper is still understood as
    # accepting — it supplied no replacement value either way.
    match = _VERDICT_RE.search(check_text)
    if match:
        return match.group(1).lower(), None, False
    return None, None, False


def _apply_verdict(
    candidate: Any,
    check_text: str,
    return_type: Any,
) -> Tuple[Any, bool]:
    """
    Decide the final value from the check response.

    Returns (final_value, revised). `revised` is True only when the check asked
    for a revision AND supplied a usable value that actually differs from the
    candidate. A "revise" verdict that hands back the same value is not a
    revision, and counting it as one would inflate the very rate this strategy
    exists to measure.

    Every failure path keeps the candidate. The check is allowed to be useless;
    it is never allowed to be destructive.

    A `null` replacement counts as a failure path, not a revision. This is a
    deliberate trade and it is reversible, so the reasoning is recorded here.
    A function may legitimately return None, so an explicit `"result": null`
    could in principle be a correct correction — but a model that cannot decide
    punting with `null` is far more common than a genuine null-valued one, and
    the costs are asymmetric: a wrongly accepted `null` destroys a correct
    answer, is never cached (runtime.py's write gate excludes None), and reads
    downstream as a parse failure rather than as a revision. Refusing it costs
    only the rare genuine case, which the retry path can still reach. If a
    future evaluation needs Optional-returning functions to be revisable, gate
    this on the declared return type rather than removing the check.
    """
    verdict, checked, has_result = _extract_verdict(check_text)

    if verdict is None:
        logger.info("[contract] no verdict in check response; keeping candidate")
        return candidate, False

    if verdict == "accept":
        return candidate, False

    if not has_result:
        logger.info("[contract] 'revise' without a replacement value; keeping candidate")
        return candidate, False

    if checked is None:
        logger.info(
            "[contract] 'revise' proposing null; keeping candidate — see "
            "_apply_verdict's docstring for why null is not a usable revision"
        )
        return candidate, False

    # No coercion here. This call used coerce_type(checked, return_type) with
    # the default strict=False, so a revised answer was repaired under WARN
    # rules even in a STRICT run -- Contract could return a value that the
    # policy would have rejected from any other arm, and the arm with a
    # self-check got a laxer type contract than the one it is compared against.
    #
    # The runtime coerces and validates the returned value once, under the
    # resolved policy, exactly as it does for every other strategy.
    if _same_value(checked, candidate):
        return candidate, False

    return checked, True


def _same_value(a: Any, b: Any) -> bool:
    """Compare two parsed values structurally, tolerating unhashable types."""
    try:
        return json.dumps(a, sort_keys=True, default=str) == json.dumps(
            b, sort_keys=True, default=str
        )
    except TypeError:
        return repr(a) == repr(b)


def _sum_optional(*values: Optional[int]) -> Optional[int]:
    """Sum token counts, returning None when no call reported one."""
    present = [v for v in values if v is not None]
    return sum(present) if present else None
