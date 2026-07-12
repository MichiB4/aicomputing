"""
Prompt construction for AI Computing strategies.

Each strategy may use a different prompt template. This module provides
builders for the main templates: direct prediction, chain-of-thought (SCoT),
and code generation.

2026 Design Principles Applied
-------------------------------
1. Context over coaching: describe *what* the model should do precisely,
   not *how* to think. Avoid generic "think step by step" instructions —
   they are redundant or harmful for modern reasoning models that handle
   internal reasoning automatically.

2. Calm, declarative system prompts: aggressive ALL-CAPS, "NEVER EVER",
   "YOU MUST" formatting overtriggers models and degrades output quality.
   Use plain, precise language instead.

3. Few-shot examples for format alignment only: few-shot CoT no longer
   improves reasoning in modern models. Examples should demonstrate the
   *output format*, not model the reasoning process.

4. Structured CoT (SCoT) for code tracing: rather than asking the model
   to "reason briefly", prompt it to trace execution through concrete
   programming structures (sequential / branch / loop) while maintaining
   an explicit variable-state log. This is the 2026 best practice for
   symbolic execution tasks.

5. Scratchpad / explicit state tracking: the primary failure mode in
   code tracing is the model losing track of variable values mid-trace.
   Prompting for a state snapshot after each meaningful step prevents
   this.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Direct prediction prompt
# ---------------------------------------------------------------------------

DIRECT_SYSTEM = """\
You are a virtual runtime. Given a Python function's signature, docstring, \
and concrete input values, predict the output the function would return. \
Respond with a JSON object containing a single key "result". \
Do not include prose, code, or explanations — only the JSON object."""

# Few-shot examples serve as *format alignment* only.
# They anchor the model to the exact output structure and JSON key expected.
# They do not attempt to demonstrate reasoning chains.
DIRECT_FEWSHOT = """\
# Output format examples
Signature: add(a: int, b: int) -> int
Docstring: Return the sum of a and b.
Inputs: {"a": 2, "b": 3}
Output: {"result": 5}

Signature: repeat(s: str, n: int) -> str
Docstring: Return s repeated n times.
Inputs: {"s": "ha", "n": 3}
Output: {"result": "hahaha"}

Signature: max_of_two(x: float, y: float) -> float
Docstring: Return the larger of x and y.
Inputs: {"x": 1.5, "y": -4.2}
Output: {"result": 1.5}"""


def build_direct_prompt(
    signature: str,
    docstring: str,
    inputs: Dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    inputs_json = json.dumps(inputs, ensure_ascii=False, default=str)
    user_block = (
        f"Signature: {signature}\n"
        f"Docstring: {docstring}\n"
        f"Inputs: {inputs_json}\n"
        f"Output:"
    )
    parts = [DIRECT_SYSTEM]
    if include_fewshot:
        parts.append(DIRECT_FEWSHOT)
    parts.append(user_block)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Structured Chain-of-Thought (SCoT) prompt
#
# This implements the 2026 best practice for symbolic code tracing:
# Structured CoT (SCoT) with explicit state tracking.
#
# Instead of asking the model to "think step by step" (vague and
# counterproductive for reasoning models), the prompt provides a concrete
# execution trace structure that mirrors actual programming constructs:
#
#   - Sequential steps: evaluate each expression, update state
#   - Branches: evaluate the condition, follow the matching path
#   - Loops: unroll each iteration, track state after each pass
#
# An explicit state snapshot after every meaningful step functions as a
# scratchpad, preventing the model from losing variable values mid-trace.
# ---------------------------------------------------------------------------

COT_SYSTEM = """\
You are a virtual runtime that traces Python function execution symbolically.

Given a function's signature, docstring, and concrete inputs, trace the \
execution as follows:

- Sequential steps: evaluate each expression in order and record the result.
- Branch (if/elif/else): evaluate the condition, state which branch is taken, \
then continue inside that branch.
- Loop (for/while): unroll each iteration one at a time, updating state \
after each pass.

After every meaningful step, record the current variable state:
  State: {var1: value, var2: value, ...}

On the final line, output only the predicted return value as JSON:
  {"result": <value>}

Do not write or execute code. Reason through the execution symbolically."""

# The few-shot example demonstrates the *trace format and state-tracking
# convention* — not the reasoning itself.
COT_FEWSHOT = """\
# Trace format example
Signature: gcd(a: int, b: int) -> int
Docstring: Return the greatest common divisor of a and b using the Euclidean \
algorithm.
Inputs: {"a": 12, "b": 8}

Sequential: enter gcd(12, 8)
  State: {a: 12, b: 8}
Loop iteration 1: b=8, not 0 — compute 12 mod 8 = 4, recurse with (8, 4)
  State: {a: 8, b: 4}
Loop iteration 2: b=4, not 0 — compute 8 mod 4 = 0, recurse with (4, 0)
  State: {a: 4, b: 0}
Branch: b == 0 → True — return a
{"result": 4}"""


def build_cot_prompt(
    signature: str,
    docstring: str,
    inputs: Dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    inputs_json = json.dumps(inputs, ensure_ascii=False, default=str)
    user_block = (
        f"Signature: {signature}\n"
        f"Docstring: {docstring}\n"
        f"Inputs: {inputs_json}\n"
        f"\nTrace:"
    )
    parts = [COT_SYSTEM]
    if include_fewshot:
        parts.append(COT_FEWSHOT)
    parts.append(user_block)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Confidence-gated prompt (self-assessed reasoning)
#
# Unlike `direct` (never reasons) and `cot` (always reasons), this prompt
# does not pre-decide how much reasoning the task needs. The model itself
# chooses, per call, whether to answer immediately or trace first — using
# the same state-tracking trace format as `cot` when it does.
#
# Rationale: `hybrid` makes this direct/cot choice with a hand-written
# keyword classifier, which encodes a fixed, human-authored opinion about
# task difficulty that has to be maintained and never improves with model
# capability. Handing the choice to the model instead means the strategy
# has nothing to update as models get better at judging their own
# confidence — a stronger model should simply reason less often on its own,
# with no code change required.
# ---------------------------------------------------------------------------

CONFIDENCE_SYSTEM = """\
You are a virtual runtime. Given a Python function's signature, docstring, \
and concrete input values, predict the output the function would return.

Decide for yourself, per call, whether you need to reason before answering:
- If you are confident of the result, answer immediately.
- If not, trace the execution first: evaluate sequential steps, branches, \
and loop iterations one at a time, recording the variable state after each \
meaningful step as `State: {var1: value, var2: value, ...}`.

Either way, end your response with the predicted return value as JSON on \
its own line: {"result": <value>}. Do not write or execute code."""

# Two examples anchor both valid output shapes — answer-only and traced —
# without telling the model which one to prefer for a given task. Format
# alignment only, per the module's few-shot design principle above.
CONFIDENCE_FEWSHOT = """\
# Output format examples

## No trace needed — answered immediately
Signature: add(a: int, b: int) -> int
Docstring: Return the sum of a and b.
Inputs: {"a": 2, "b": 3}
Output: {"result": 5}

## Trace used first, then the answer
Signature: gcd(a: int, b: int) -> int
Docstring: Return the greatest common divisor of a and b using the \
Euclidean algorithm.
Inputs: {"a": 12, "b": 8}
Output:
Loop iteration 1: b=8, not 0 — compute 12 mod 8 = 4, recurse with (8, 4)
  State: {a: 8, b: 4}
Loop iteration 2: b=4, not 0 — compute 8 mod 4 = 0, recurse with (4, 0)
  State: {a: 4, b: 0}
Branch: b == 0 → True — return a
{"result": 4}"""


def build_confidence_prompt(
    signature: str,
    docstring: str,
    inputs: Dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    inputs_json = json.dumps(inputs, ensure_ascii=False, default=str)
    user_block = (
        f"Signature: {signature}\n"
        f"Docstring: {docstring}\n"
        f"Inputs: {inputs_json}\n"
        f"Output:"
    )
    parts = [CONFIDENCE_SYSTEM]
    if include_fewshot:
        parts.append(CONFIDENCE_FEWSHOT)
    parts.append(user_block)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Code generation prompt (for comparison baseline only — NOT AI computing)
#
# Used by benchmarks/baselines/code_gen_baseline.py to compare the
# traditional "generate code + execute" approach against spec-only
# prediction. This prompt is NOT used by any core strategy.
# ---------------------------------------------------------------------------

CODEGEN_SYSTEM = """\
You are a Python code generator. Given a function signature and docstring, \
write a complete Python function implementation. Output only the function \
body (do not repeat the def line). No explanations, no tests, no markdown \
fences. The code must be valid Python that can be indented under the def line."""


def build_codegen_prompt(
    signature: str,
    docstring: str,
) -> str:
    user_block = (
        f"def {signature}:\n"
        f'    """{docstring}"""\n'
        f"    # Implementation:"
    )
    return f"{CODEGEN_SYSTEM}\n\n{user_block}"


# ---------------------------------------------------------------------------
# Retry prompt (when output fails validation)
#
# Calm, precise error feedback. Avoids aggressive imperatives ("try again!",
# "you must") which cause the model to over-correct or loop on errors.
# ---------------------------------------------------------------------------

def build_retry_prompt(
    original_prompt: str,
    raw_output: str,
    error_message: str,
) -> str:
    return (
        f"{original_prompt}\n\n"
        f"Previous output:\n{raw_output}\n\n"
        f"Validation error: {error_message}\n"
        f'Return corrected output as valid JSON only: {{"result": <value>}}'
    )
