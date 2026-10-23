"""Build prompts for AI Computing strategies and the code-generation baseline.

Prediction prompt builders:
- build_direct_prompt: request a predicted result without an explanation.
- build_cot_prompt: request an execution trace with variable-state tracking.
- build_pseudocode_prompt: request a procedure followed by its simulation.
- build_confidence_prompt: let the model choose whether to provide a trace.
- build_contract_check_prompt: request a check of a proposed result.

Hybrid delegates to the direct or CoT strategy. Consistency repeatedly
uses the CoT prompt. Contract uses the direct prompt for its initial
prediction, followed by the contract-check prompt.

build_codegen_prompt requests a Python function body for the comparison
baseline. This module constructs that prompt; it does not execute code.

build_retry_prompt appends a previous response and validation feedback
to an existing prompt.

Prompt instructions and examples request particular response formats.
They do not guarantee model compliance or prediction correctness.
"""

from __future__ import annotations

from typing import Any


def _format_prompt_value(value: Any) -> str:
    """Describe supported built-in values using Python notation.

    Reject unsupported types and cycles rather than silently losing data.
    The returned text is for the model to read; it is never evaluated.
    """
    active: set[int] = set()

    def render(item: Any) -> str:
        kind = type(item)
        if item is None or kind in (bool, int, str, bytes):
            return repr(item)
        if kind is float:
            text = repr(item)
            return f"float({text!r})" if text in {"nan", "inf", "-inf"} else text
        if kind not in (list, tuple, dict, set, frozenset):
            raise TypeError(
                "Unsupported prompt value type: "
                f"{kind.__module__}.{kind.__qualname__}"
            )
        identity = id(item)
        if identity in active:
            raise ValueError("Cyclic values cannot be represented in prompts")
        active.add(identity)
        try:
            if kind is dict:
                entries = [f"{render(key)}: {render(val)}" for key, val in item.items()]
                return "{" + ", ".join(entries) + "}"
            entries = [render(val) for val in item]
            body = ", ".join(entries)
            if kind is list:
                return "[" + body + "]"
            if kind is tuple:
                return "(" + body + ("," if len(item) == 1 else "") + ")"
            if kind is set:
                return "{" + body + "}" if item else "set()"
            return "frozenset({" + body + "})" if item else "frozenset()"
        finally:
            active.remove(identity)

    return render(value)


RESULT_VALUE_FORMAT = """\
For the final response object, use JSON when it preserves the required Python \
value and dictionary keys. Otherwise use a Python literal dictionary for the \
whole object, preserving types and keys. Examples: {"result": {1: "a"}}, \
{"result": {1, 2}}, {"result": set()}, {"result": b"abc"}.
In Python literals use None, True, and False. Do not stringify sets, bytes, \
or dictionary keys. Use only literal data; set() is allowed for an empty set. \
Do not include executable statements or other function calls."""


# ---------------------------------------------------------------------------
# Direct prediction prompt
# ---------------------------------------------------------------------------

DIRECT_SYSTEM = """\
You are a virtual runtime. Given a Python function's signature, docstring, \
and concrete input values, predict the output the function would return. \
Respond with an object containing a single key "result". \
Do not include prose, code, or explanations — only the result object."""

# Examples demonstrate the result-object format without explanatory traces.
DIRECT_FEWSHOT = """\
# Output format examples
Signature: add(a: int, b: int) -> int
Docstring: Return the sum of a and b.
Inputs (Python values): {"a": 2, "b": 3}
Output: {"result": 5}

Signature: repeat(s: str, n: int) -> str
Docstring: Return s repeated n times.
Inputs (Python values): {"s": "ha", "n": 3}
Output: {"result": "hahaha"}

Signature: max_of_two(x: float, y: float) -> float
Docstring: Return the larger of x and y.
Inputs (Python values): {"x": 1.5, "y": -4.2}
Output: {"result": 1.5}"""


def build_direct_prompt(
    signature: str,
    docstring: str,
    inputs: dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    """Build a direct-prediction prompt, optionally including examples."""
    inputs_text = _format_prompt_value(inputs)
    user_block = (
        f"Signature: {signature}\n"
        f"Docstring: {docstring}\n"
        f"Inputs (Python values): {inputs_text}\n"
        f"Output:"
    )
    parts = [DIRECT_SYSTEM, RESULT_VALUE_FORMAT]
    if include_fewshot:
        parts.append(DIRECT_FEWSHOT)
    parts.append(user_block)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Structured chain-of-thought prompt
#
# Request a predicted execution trace organized into sequential steps,
# branch decisions, and loop iterations, with variable-state snapshots.
#
# These instructions specify the requested response format; they do not
# guarantee a correct trace or predicted result.
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

On the final line, output only the predicted return value in a result object:
  {"result": <value>}

Do not write or execute code. Reason through the execution symbolically."""

# The worked example shows a trace, variable-state updates, and a final result.
COT_FEWSHOT = """\
# Trace format example
Signature: gcd(a: int, b: int) -> int
Docstring: Return the greatest common divisor of a and b using the Euclidean \
algorithm.
Inputs (Python values): {"a": 12, "b": 8}

Sequential: start with the supplied inputs
  State: {a: 12, b: 8}
Loop condition: b != 0 → True
Loop iteration 1: compute 12 mod 8 = 4; update (a, b) to (8, 4)
  State: {a: 8, b: 4}
Loop condition: b != 0 → True
Loop iteration 2: compute 8 mod 4 = 0; update (a, b) to (4, 0)
  State: {a: 4, b: 0}
Loop condition: b != 0 → False — exit the loop
Sequential: return a, which is 4
{"result": 4}"""


def build_cot_prompt(
    signature: str,
    docstring: str,
    inputs: dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    """Build a structured-trace prompt, optionally including an example."""
    inputs_text = _format_prompt_value(inputs)
    user_block = (
        f"Signature: {signature}\n"
        f"Docstring: {docstring}\n"
        f"Inputs (Python values): {inputs_text}\n"
        "\nTrace:"
    )
    parts = [COT_SYSTEM, RESULT_VALUE_FORMAT]
    if include_fewshot:
        parts.append(COT_FEWSHOT)
    parts.append(user_block)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pseudocode-then-simulate prompt
#
# Request a numbered plain-English procedure, followed by a simulation
# on the supplied inputs with state updates and a final predicted result.
#
# Both stages are requested in the same response. The strategy does not
# execute the procedure, freeze it separately, or independently verify
# that the simulation follows it.
# ---------------------------------------------------------------------------

PSEUDOCODE_SYSTEM = """\
You are a virtual runtime. Given a Python function's signature, docstring, \
and concrete input values, work in two stages.

Stage 1 — Procedure. Write the procedure the docstring describes as numbered \
steps in plain English. Use ordinary words for control flow ("for each", \
"if", "otherwise", "return"). Do not write Python or any other executable \
language, and do not use language-specific syntax or library calls.

Stage 2 — Simulation. Execute your own numbered procedure on the given \
inputs, one step at a time. After every step that changes a value, record \
the current state:
  State: {var1: value, var2: value, ...}

On the final line, output only the predicted return value in a result object:
  {"result": <value>}

Commit to the procedure in stage 1 before beginning stage 2, and do not \
revise it once stage 2 has started. The answer is whatever simulating the \
procedure produces."""

# The worked example shows a procedure, simulated state updates, and a final result.
PSEUDOCODE_FEWSHOT = """\
# Output format example
Signature: count_above(xs: list, t: int) -> int
Docstring: Return how many items in xs are strictly greater than t.
Inputs (Python values): {"xs": [3, 9, 4], "t": 4}

Procedure:
  1. Set a counter to zero.
  2. For each item in xs, do step 3.
  3. If the item is strictly greater than t, add one to the counter.
  4. Return the counter.

Simulation:
Step 1: counter set to 0
  State: {counter: 0, t: 4}
Step 3 with item 3: 3 greater than 4 is false — counter unchanged
  State: {counter: 0, item: 3}
Step 3 with item 9: 9 greater than 4 is true — counter becomes 1
  State: {counter: 1, item: 9}
Step 3 with item 4: 4 greater than 4 is false — counter unchanged
  State: {counter: 1, item: 4}
Step 4: return counter
{"result": 1}"""


def build_pseudocode_prompt(
    signature: str,
    docstring: str,
    inputs: dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    """Build a procedure-and-simulation prompt with optional examples."""
    inputs_text = _format_prompt_value(inputs)
    user_block = (
        f"Signature: {signature}\n"
        f"Docstring: {docstring}\n"
        f"Inputs (Python values): {inputs_text}\n"
        "\nProcedure:"
    )
    parts = [PSEUDOCODE_SYSTEM, RESULT_VALUE_FORMAT]
    if include_fewshot:
        parts.append(PSEUDOCODE_FEWSHOT)
    parts.append(user_block)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Contract-check prompt (second stage of the contract strategy)
#
# Ask the model to assess a proposed candidate against the function's
# signature, docstring, and concrete inputs, then accept or revise it.
#
# This is a model judgment, not a comparison with a reference implementation.
# Neither acceptance nor revision establishes that the result is correct.
# ---------------------------------------------------------------------------

CONTRACT_CHECK_SYSTEM = """\
You are checking a proposed result against a function's specification.

You are given a Python function's signature, docstring, concrete input \
values, and a proposed return value. The proposed value has not been \
verified. It may be correct or incorrect.

Decide whether it is what the specification requires for these inputs:
- Does the value have the type and shape the signature declares?
- Does it satisfy every condition the docstring states?
- Would the procedure the docstring describes, applied to these inputs, \
produce this value?

End your response with a single verdict object on its own line, in one of \
these two forms:
  {"verdict": "accept", "result": <the proposed value, unchanged>}
  {"verdict": "revise", "result": <the value the specification requires>}

Use "accept" when the proposed value is what the specification requires. \
Use "revise" only when you can state what the correct value is. Do not \
write or execute code."""

# Examples show one accepted candidate and one revised candidate.
CONTRACT_CHECK_FEWSHOT = """\
# Output format examples

Signature: add(a: int, b: int) -> int
Docstring: Return the sum of a and b.
Inputs (Python values): {"a": 2, "b": 3}
Proposed result (Python value): 5
Check: the docstring requires the sum; 2 + 3 is 5; the declared type is int.
{"verdict": "accept", "result": 5}

Signature: initials(name: str) -> str
Docstring: Return the uppercase first letter of each word, separated by dots.
Inputs (Python values): {"name": "ada lovelace"}
Proposed result (Python value): "AL"
Check: the docstring requires dot separation, which the proposed value omits.
{"verdict": "revise", "result": "A.L"}"""


def build_contract_check_prompt(
    signature: str,
    docstring: str,
    inputs: dict[str, Any],
    candidate: Any,
    include_fewshot: bool = True,
) -> str:
    """Build a candidate-checking prompt, optionally including examples."""
    inputs_text = _format_prompt_value(inputs)
    candidate_text = _format_prompt_value(candidate)
    user_block = (
        f"Signature: {signature}\n"
        f"Docstring: {docstring}\n"
        f"Inputs (Python values): {inputs_text}\n"
        f"Proposed result (Python value): {candidate_text}\n"
        "Check:"
    )
    parts = [CONTRACT_CHECK_SYSTEM, RESULT_VALUE_FORMAT]
    if include_fewshot:
        parts.append(CONTRACT_CHECK_FEWSHOT)
    parts.append(user_block)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Confidence prompt (model-selected visible tracing)
#
# Ask the model to choose between answering directly and providing a
# visible execution trace before its final predicted result.
#
# The strategy does not measure confidence or apply a routing threshold.
# This response-format choice is separate from the backend thinking setting.
# ---------------------------------------------------------------------------

CONFIDENCE_SYSTEM = """\
You are a virtual runtime. Given a Python function's signature, docstring, \
and concrete input values, predict the output the function would return.

Decide for yourself, per call, whether you need to reason before answering:
- If you are confident of the result, answer immediately.
- If not, trace the execution first: evaluate sequential steps, branches, \
and loop iterations one at a time, recording the variable state after each \
meaningful step as `State: {var1: value, var2: value, ...}`.

Either way, end your response with the predicted return value in an object on \
its own line: {"result": <value>}. Do not write or execute code."""

# Examples show answer-only and trace-first responses, each ending
# with a result-key object.
CONFIDENCE_FEWSHOT = """\
# Output format examples

## Answer-only response
Signature: add(a: int, b: int) -> int
Docstring: Return the sum of a and b.
Inputs (Python values): {"a": 2, "b": 3}
Output: {"result": 5}

## Trace-first response
Signature: gcd(a: int, b: int) -> int
Docstring: Return the greatest common divisor of a and b using the \
Euclidean algorithm.
Inputs (Python values): {"a": 12, "b": 8}
Output:
Sequential: start with the supplied inputs
  State: {a: 12, b: 8}
Loop condition: b != 0 → True
Loop iteration 1: compute 12 mod 8 = 4; update (a, b) to (8, 4)
  State: {a: 8, b: 4}
Loop condition: b != 0 → True
Loop iteration 2: compute 8 mod 4 = 0; update (a, b) to (4, 0)
  State: {a: 4, b: 0}
Loop condition: b != 0 → False — exit the loop
Sequential: return a, which is 4
{"result": 4}"""


def build_confidence_prompt(
    signature: str,
    docstring: str,
    inputs: dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    """Build a model-selected tracing prompt with optional examples."""
    inputs_text = _format_prompt_value(inputs)
    user_block = (
        f"Signature: {signature}\n"
        f"Docstring: {docstring}\n"
        f"Inputs (Python values): {inputs_text}\n"
        "Output:"
    )
    parts = [CONFIDENCE_SYSTEM, RESULT_VALUE_FORMAT]
    if include_fewshot:
        parts.append(CONFIDENCE_FEWSHOT)
    parts.append(user_block)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Code-generation prompt for the comparison baseline
#
# Request a Python function body from the signature and docstring.
# The benchmark baseline assembles and executes the generated code;
# this module only constructs the prompt.
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
    """Build a function-body prompt with the specification as a Python string literal."""
    # Keep this list aligned with code_gen_baseline._ALLOWED_MODULES.
    environment = (
        "The following modules are already available as globals: "
        "math, json, re, itertools, functools, collections, string, statistics. "
        "Use them directly, or import from these modules. Imports of other "
        "module roots are not allowed."
    )
    user_block = (
        f"def {signature}:\n"
        f"    {docstring!r}\n"
        "    # Implementation:"
    )
    return f"{CODEGEN_SYSTEM}\n\n{environment}\n\n{user_block}"


# ---------------------------------------------------------------------------
# Validation-retry prompt
#
# Append the previous response and validation feedback to the original
# prompt, followed by a request for a corrected result-key object.
# ---------------------------------------------------------------------------


def build_retry_prompt(
    original_prompt: str,
    raw_output: str,
    error_message: str,
) -> str:
    """Append a previous response and validation feedback to a prompt."""
    return (
        f"{original_prompt}\n\n"
        f"Previous output:\n{raw_output}\n\n"
        f"Validation error: {error_message}\n"
        'Return corrected output only as {"result": <value>}.\n'
        f"{RESULT_VALUE_FORMAT}"
    )
