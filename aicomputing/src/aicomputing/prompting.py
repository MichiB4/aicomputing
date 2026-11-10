"""Build prediction, contract-check, and validation-retry prompts.

Hybrid delegates to direct/CoT; consistency repeats CoT; contract predicts with
direct before checking. Instructions and examples do not guarantee compliance
or correctness.
"""

from __future__ import annotations

from typing import Any

from .answer_protocol import FINAL_END, frame_answer
from .validation import validate_bool


def format_prompt_value(value: Any) -> str:
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


TASK_INSTRUCTIONS = """\
Predict the Python return value required by the supplied signature,
docstring, and concrete inputs.

Implementation code is intentionally omitted. Apply the behavior
specified in the docstring. The absence of implementation code alone
is not a reason to decline. Do not invent unspecified behavior.

Worked examples demonstrate the method and response format.
Answer only the ACTUAL TASK."""


FRAMED_VALUE_RULES = f"""\
Put exactly one final-answer block at the end of your response:
{frame_answer('<one complete result object, or verdict object for a check>')}
The marker lines are boundaries, not part of the returned value. Put any
permitted reasoning before the block. Write nothing after {FINAL_END}.
Inside the block, output one complete literal object and nothing else.
Use fully evaluated values, not arithmetic expressions or function calls.
Never use question marks, ellipses, placeholders, or omitted digits in a
numeric answer. Provide sufficient floating-point digits; do not round unless
the specification requests rounding. Close every bracket and parenthesis,
preserve nesting and dictionary key types, and include the comma in a
singleton tuple, for example (20,). Preserve punctuation that is actually
part of a required string value. Escape newlines in string literals so a
line of data cannot be mistaken for a boundary marker."""


FRAMED_PROMPT_REVISION = "final_answer_reminder_v1"


def _final_answer_reminder(*, verdict: bool) -> str:
    """Keep the required answer next to the task, after any worked examples."""
    schema = (
        '{"verdict": <"accept" or "revise">, "result": <complete concrete value>}'
        if verdict else '{"result": <complete concrete value>}'
    )
    return (
        "FINAL RESPONSE REQUIREMENT FOR THE ACTUAL TASK\n"
        "Complete every requested part of your response, including the final answer. "
        "After any requested procedure, trace, simulation, or check, write the "
        "concrete value explicitly in the final-answer block. A return instruction "
        "or a State: record does not replace the final answer. "
        "Do not stop after the explanation.\n"
        "Finish your response in this exact structure, replacing the placeholders "
        "with complete literal data for the actual task:\n"
        f"{frame_answer(schema)}\n"
        f"Write nothing after {FINAL_END}."
    )





def _format_prediction_task(
    signature: str,
    docstring: str,
    inputs: dict[str, Any],
) -> str:
    """Format a signature, specification and concrete inputs for prediction."""
    inputs_text = format_prompt_value(inputs)
    return (
        f"Signature: {signature}\n"
        f"Specification (docstring):\n{docstring}\n"
        f"Inputs (Python values): {inputs_text}"
    )


def _build_prediction_prompt(
    strategy_instructions: str,
    task: str,
    response_cue: str,
    worked_examples: str | None,
    *, verdict: bool = False,
) -> str:
    """Separate instructions, illustrative examples, and the actual task."""
    parts = [
        f"TASK INSTRUCTIONS\n{TASK_INSTRUCTIONS}",
        "STRATEGY AND RESPONSE FORMAT\n"
        f"{strategy_instructions}\n\n"
        "Angle-bracketed text in a response schema is a placeholder. "
        "Replace it with the required value; do not output the placeholder text.",
        "VALUE NOTATION — FORMATTING EXAMPLES ONLY\n"
        f"{RESULT_VALUE_FORMAT}\n"
        "These examples illustrate notation; they do not answer the actual task. "
        "Use the object keys required by the strategy above.",
    ]
    parts.append(FRAMED_VALUE_RULES)
    if worked_examples is not None:
        parts.append(f"BEGIN WORKED EXAMPLES\n\n{worked_examples}\n\nEND WORKED EXAMPLES")
    parts.append(f"BEGIN ACTUAL TASK\n{task}\nEND ACTUAL TASK")
    parts.append(_final_answer_reminder(verdict=verdict))
    parts.append(f"YOUR RESPONSE TO THE ACTUAL TASK:\n{response_cue}")
    return "\n\n".join(parts)


# Direct prediction prompt

DIRECT_INSTRUCTIONS = """\
You are a virtual runtime. Given a Python function's signature, docstring, and concrete input values, predict the output the function would return. Respond with an object containing a single key "result". Do not include prose, code, or explanations — only the final-answer block."""

# Examples demonstrate the result-object format without explanatory traces.
DIRECT_FEWSHOT = f"""\
BEGIN EXAMPLE 1
Example task:
Signature: add(a: int, b: int) -> int
Docstring: Return the sum of a and b.
Inputs (Python values): {{"a": 2, "b": 3}}
Example response:
{frame_answer('{"result": 5}')}
END EXAMPLE 1

BEGIN EXAMPLE 2
Example task:
Signature: repeat(s: str, n: int) -> str
Docstring: Return s repeated n times.
Inputs (Python values): {{"s": "ha", "n": 3}}
Example response:
{frame_answer('{"result": "hahaha"}')}
END EXAMPLE 2

BEGIN EXAMPLE 3
Example task:
Signature: max_of_two(x: float, y: float) -> float
Docstring: Return the larger of x and y.
Inputs (Python values): {{"x": 1.5, "y": -4.2}}
Example response:
{frame_answer('{"result": 1.5}')}
END EXAMPLE 3"""


def build_direct_prompt(
    signature: str,
    docstring: str,
    inputs: dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    """Build a direct-prediction prompt, optionally including examples."""
    validate_bool(include_fewshot, "include_fewshot")
    user_block = _format_prediction_task(signature, docstring, inputs)
    return _build_prediction_prompt(
        DIRECT_INSTRUCTIONS, user_block, "Final-answer block:",
        DIRECT_FEWSHOT if include_fewshot else None,
    )


# Structured chain-of-thought prompt
#
# Request a trace of the specified behavior, with conditional steps and
# repeated operations where needed, and intermediate-state snapshots.
#
# These instructions specify the requested response format; they do not
# guarantee a correct trace or predicted result.

COT_INSTRUCTIONS = """\
You are a virtual runtime that predicts a Python function's return value
from its signature, docstring, and concrete inputs.

The docstring describes the intended behavior. Implementation code is
intentionally omitted. Apply the specified behavior to the supplied inputs.
The absence of implementation code alone is not a reason to decline the task.
Do not invent behavior that the specification does not state.

Reason through the task step by step:
- Apply the operations implied by the specified behavior in order.
- Where a condition is required, evaluate it and explain its consequence.
- Where repeated operations are required, work through the iterations,
  updating the relevant intermediate values.

After every meaningful step, record the current state:
  State: {var1: value, var2: value, ...}

Inside the final-answer block, output the predicted return value in a result object:
  {"result": <value>}

Do not write or execute code. Reason through the specified behavior symbolically."""

# The worked example shows a trace, variable-state updates, and a final result.
COT_FEWSHOT = f"""\
BEGIN EXAMPLE 1
Example task:
Signature: gcd(a: int, b: int) -> int
Docstring: Return the greatest common divisor of a and b using the Euclidean algorithm.
Inputs (Python values): {{"a": 12, "b": 8}}

Example response:
Sequential: start with the supplied inputs
  State: {{a: 12, b: 8}}
Loop condition: b != 0 → True
Loop iteration 1: compute 12 mod 8 = 4; update (a, b) to (8, 4)
  State: {{a: 8, b: 4}}
Loop condition: b != 0 → True
Loop iteration 2: compute 8 mod 4 = 0; update (a, b) to (4, 0)
  State: {{a: 4, b: 0}}
Loop condition: b != 0 → False — exit the loop
Sequential: return a, which is 4
{frame_answer('{"result": 4}')}
END EXAMPLE 1"""


def build_cot_prompt(
    signature: str,
    docstring: str,
    inputs: dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    """Build a specification-trace prompt, optionally including an example."""
    validate_bool(include_fewshot, "include_fewshot")
    user_block = _format_prediction_task(signature, docstring, inputs)
    return _build_prediction_prompt(
        COT_INSTRUCTIONS, user_block, "Trace, then final-answer block:",
        COT_FEWSHOT if include_fewshot else None,
    )


# Pseudocode-then-simulate prompt
#
# Request a numbered plain-English procedure, followed by a simulation
# on the supplied inputs with state updates and a final predicted result.
#
# Both stages are requested in the same response. The strategy does not
# execute the procedure, freeze it separately, or independently verify
# that the simulation follows it.

PSEUDOCODE_INSTRUCTIONS = """\
You are a virtual runtime. Given a Python function's signature, docstring, and concrete input values, work in two stages.

Stage 1 — Procedure. Write the procedure the docstring describes as numbered steps in plain English. Use ordinary words for control flow ("for each", "if", "otherwise", "return"). Do not write Python or any other executable language, and do not use language-specific syntax or library calls.

Stage 2 — Simulation. Execute your own numbered procedure on the given inputs, one step at a time. After every step that changes a value, record the current state:
  State: {var1: value, var2: value, ...}

Inside the final-answer block, output only the predicted return value in a result object:
  {"result": <value>}

Commit to the procedure in stage 1 before beginning stage 2, and do not revise it once stage 2 has started. The answer is whatever simulating the procedure produces."""

# The worked example shows a procedure, simulated state updates, and a final result.
PSEUDOCODE_FEWSHOT = f"""\
BEGIN EXAMPLE 1
Example task:
Signature: count_above(xs: list, t: int) -> int
Docstring: Return how many items in xs are strictly greater than t.
Inputs (Python values): {{"xs": [3, 9, 4], "t": 4}}

Example response:
Procedure:
  1. Set a counter to zero.
  2. For each item in xs, do step 3.
  3. If the item is strictly greater than t, add one to the counter.
  4. Return the counter.

Simulation:
Step 1: counter set to 0
  State: {{counter: 0, t: 4}}
Step 3 with item 3: 3 greater than 4 is false — counter unchanged
  State: {{counter: 0, item: 3}}
Step 3 with item 9: 9 greater than 4 is true — counter becomes 1
  State: {{counter: 1, item: 9}}
Step 3 with item 4: 4 greater than 4 is false — counter unchanged
  State: {{counter: 1, item: 4}}
Step 4: return counter
{frame_answer('{"result": 1}')}
END EXAMPLE 1"""


def build_pseudocode_prompt(
    signature: str,
    docstring: str,
    inputs: dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    """Build a procedure-and-simulation prompt with optional examples."""
    validate_bool(include_fewshot, "include_fewshot")
    user_block = _format_prediction_task(signature, docstring, inputs)
    return _build_prediction_prompt(
        PSEUDOCODE_INSTRUCTIONS, user_block, "Procedure, then simulation, then final-answer block:",
        PSEUDOCODE_FEWSHOT if include_fewshot else None,
    )


# Contract-check prompt (second stage of the contract strategy)
#
# Ask the model to assess a proposed candidate against the function's
# signature, docstring, and concrete inputs, then accept or revise it.
#
# This is a model judgment, not a comparison with a reference implementation.
# Neither acceptance nor revision establishes that the result is correct.

CONTRACT_CHECK_INSTRUCTIONS = """\
You are checking a proposed result against a function's specification.

You are given a Python function's signature, docstring, concrete input values, and a proposed return value. The proposed value has not been verified. It may be correct or incorrect.

Decide whether it is what the specification requires for these inputs:
- Does the value have the type and shape the signature declares?
- Does it satisfy every condition the docstring states?
- Would the procedure the docstring describes, applied to these inputs, produce this value?

Put a single verdict object in the final-answer block, in one of these two forms:
  {"verdict": "accept", "result": <the proposed value, unchanged>}
  {"verdict": "revise", "result": <the value the specification requires>}

Use "accept" when the proposed value is what the specification requires. Use "revise" only when you can state what the correct value is. Do not write or execute code."""

# Examples show one accepted candidate and one revised candidate.
CONTRACT_CHECK_FEWSHOT = f"""\
BEGIN EXAMPLE 1
Example task:
Signature: add(a: int, b: int) -> int
Docstring: Return the sum of a and b.
Inputs (Python values): {{"a": 2, "b": 3}}
Proposed result (Python value): 5
Example response:
Check: the docstring requires the sum; 2 + 3 is 5; the declared type is int.
{frame_answer('{"verdict": "accept", "result": 5}')}
END EXAMPLE 1

BEGIN EXAMPLE 2
Example task:
Signature: initials(name: str) -> str
Docstring: Return the uppercase first letter of each word, separated by dots.
Inputs (Python values): {{"name": "ada lovelace"}}
Proposed result (Python value): "AL"
Example response:
Check: the docstring requires dot separation, which the proposed value omits.
{frame_answer('{"verdict": "revise", "result": "A.L"}')}
END EXAMPLE 2"""


def build_contract_check_prompt(
    signature: str,
    docstring: str,
    inputs: dict[str, Any],
    candidate: Any,
    include_fewshot: bool = True,
) -> str:
    """Build a candidate-checking prompt, optionally including examples."""
    validate_bool(include_fewshot, "include_fewshot")
    task_text = _format_prediction_task(signature, docstring, inputs)
    candidate_text = format_prompt_value(candidate)
    user_block = (
        f"{task_text}\n"
        f"Unverified candidate return value (Python value): {candidate_text}\n"
        "This is a proposed answer to check, not a verified reference answer."
    )
    return _build_prediction_prompt(
        CONTRACT_CHECK_INSTRUCTIONS, user_block, "Check, then final-answer block:",
        CONTRACT_CHECK_FEWSHOT if include_fewshot else None,
        verdict=True,
    )


# Confidence prompt (model-selected visible tracing)
#
# Ask the model to choose between answering directly and providing a
# visible execution trace before its final predicted result.
#
# The strategy does not measure confidence or apply a routing threshold.
# This response-format choice is separate from the backend thinking setting.

CONFIDENCE_INSTRUCTIONS = """\
You are a virtual runtime. Given a Python function's signature, docstring, and concrete input values, predict the output the function would return.

Decide for yourself, per call, whether you need to reason before answering:
- If you are confident of the result, answer immediately.
- If not, trace the execution first: evaluate sequential steps, branches, and loop iterations one at a time, recording the variable state after each meaningful step as `State: {var1: value, var2: value, ...}`.

Either way, end your response with the predicted return value in an object inside the final-answer block: {"result": <value>}. Do not write or execute code."""

# Examples show answer-only and trace-first responses, each ending
# with a result-key object.
CONFIDENCE_FEWSHOT = f"""\
# Output format examples

## Answer-only response
Signature: add(a: int, b: int) -> int
Docstring: Return the sum of a and b.
Inputs (Python values): {{"a": 2, "b": 3}}
Output:
{frame_answer('{"result": 5}')}

## Trace-first response
Signature: gcd(a: int, b: int) -> int
Docstring: Return the greatest common divisor of a and b using the Euclidean algorithm.
Inputs (Python values): {{"a": 12, "b": 8}}
Output:
Sequential: start with the supplied inputs
  State: {{a: 12, b: 8}}
Loop condition: b != 0 → True
Loop iteration 1: compute 12 mod 8 = 4; update (a, b) to (8, 4)
  State: {{a: 8, b: 4}}
Loop condition: b != 0 → True
Loop iteration 2: compute 8 mod 4 = 0; update (a, b) to (4, 0)
  State: {{a: 4, b: 0}}
Loop condition: b != 0 → False — exit the loop
Sequential: return a, which is 4
{frame_answer('{"result": 4}')}"""


def build_confidence_prompt(
    signature: str,
    docstring: str,
    inputs: dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    """Build a model-selected tracing prompt with optional examples."""
    validate_bool(include_fewshot, "include_fewshot")
    user_block = _format_prediction_task(signature, docstring, inputs)
    return _build_prediction_prompt(
        CONFIDENCE_INSTRUCTIONS, user_block,
        "Answer, with any needed trace before the final-answer block:",
        CONFIDENCE_FEWSHOT if include_fewshot else None,
    )


# Validation-retry prompt
#
# Append the previous response and validation feedback to the original
# prompt, followed by a request for a corrected result-key object.


def build_retry_prompt(
    original_prompt: str,
    raw_output: str,
    error_message: str,
) -> str:
    """Append a previous response and validation feedback to a prompt."""
    return (
        f"{original_prompt}\n\nPrevious output:\n{raw_output}\n\n"
        f"Validation error: {error_message}\n"
        "Return the corrected complete result object in one final-answer block.\n"
        f"{RESULT_VALUE_FORMAT}\n{FRAMED_VALUE_RULES}"
    )
