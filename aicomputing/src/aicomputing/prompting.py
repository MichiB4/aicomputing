"""
Prompt construction for AI Computing strategies.

Each strategy may use a different prompt template. This module provides
builders for: direct prediction (build_direct_prompt), structured
chain-of-thought (build_cot_prompt), pseudocode-then-simulate
(build_pseudocode_prompt), confidence-gated self-assessed reasoning
(build_confidence_prompt), contract checking of a proposed result
(build_contract_check_prompt), code generation for the comparison baseline
(build_codegen_prompt), and validation-retry wrapping (build_retry_prompt).

The `hybrid`, `consistency` and `contract` strategies add no first-stage
template of their own: hybrid delegates to the direct or cot builder per its
routing decision, consistency samples the cot prompt repeatedly, and contract
predicts with the direct prompt before applying its own check prompt.

The ladder of committed structure
---------------------------------
The builders differ in how much intermediate representation the model must
produce before it is allowed to answer:

    direct       nothing
    cot          a state-annotated trace, no committed procedure
    pseudocode   a committed non-executable procedure, then a trace of it
    contract     an answer, then a test of that answer against the spec
    confidence   the model chooses between the first two shapes
    codegen      an executable procedure — NOT AI computing, baseline only

Only the last one crosses out of the paradigm, because only there does a
real interpreter, rather than the model, produce the value.

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
# Pseudocode-then-simulate prompt
#
# The model first commits to a procedure, then executes that procedure. This
# is the distinction from `cot`: SCoT traces execution straight from the
# specification, so nothing stops the model from working backwards from an
# answer that looks plausible and narrating a trace that agrees with it. Here
# the procedure is fixed in stage 1 and the value is whatever simulating it in
# stage 2 produces.
#
# Adapted from the pseudocode-simulation path of DuET (Han et al., 2026),
# which generates pseudocode from a problem description and has the model
# simulate it. DuET's second path compiles real code and runs it in an actual
# interpreter; that path is deliberately NOT taken here — it would reintroduce
# the runtime a virtual runtime exists to replace, and it is what the
# code-generation baseline measures instead.
#
# Hence the "plain English, no executable syntax" constraint below. It is not
# a style preference: pseudocode the model could have executed mentally as
# real Python collapses this strategy into the baseline.
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

On the final line, output only the predicted return value as JSON:
  {"result": <value>}

Commit to the procedure in stage 1 before beginning stage 2, and do not \
revise it once stage 2 has started. The answer is whatever simulating the \
procedure produces."""

# Format alignment only: this demonstrates the two-stage output shape and the
# state-log convention, not how to reason about the task.
PSEUDOCODE_FEWSHOT = """\
# Output format example
Signature: count_above(xs: list, t: int) -> int
Docstring: Return how many items in xs are strictly greater than t.
Inputs: {"xs": [3, 9, 4], "t": 4}

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
    inputs: Dict[str, Any],
    include_fewshot: bool = True,
) -> str:
    inputs_json = json.dumps(inputs, ensure_ascii=False, default=str)
    user_block = (
        f"Signature: {signature}\n"
        f"Docstring: {docstring}\n"
        f"Inputs: {inputs_json}\n"
        f"\nProcedure:"
    )
    parts = [PSEUDOCODE_SYSTEM]
    if include_fewshot:
        parts.append(PSEUDOCODE_FEWSHOT)
    parts.append(user_block)
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Contract-check prompt (second stage of the `contract` strategy)
#
# The specification is the only ground truth a virtual runtime has: there is
# no implementation to compare against and no interpreter to appeal to. This
# prompt asks the model to test a proposed value against that specification.
#
# Note the framing. The candidate is introduced as "proposed" and as having
# come from another source, never as "your answer". Asking a model to review
# its own output invites it to defend the output rather than test it, which
# would make the check a no-op that still costs a call. The wording will not
# eliminate that failure mode — measure it, via the revision and flip rates
# the strategy records, rather than assuming it away.
# ---------------------------------------------------------------------------

CONTRACT_CHECK_SYSTEM = """\
You are checking a proposed result against a function's specification.

You are given a Python function's signature, docstring, concrete input \
values, and a proposed return value. The proposed value came from another \
source and has not been verified. It may be correct or incorrect.

Decide whether it is what the specification requires for these inputs:
- Does the value have the type and shape the signature declares?
- Does it satisfy every condition the docstring states?
- Would the procedure the docstring describes, applied to these inputs, \
produce this value?

End your response with a single JSON object on its own line, in one of \
these two forms:
  {"verdict": "accept", "result": <the proposed value, unchanged>}
  {"verdict": "revise", "result": <the value the specification requires>}

Use "accept" when the proposed value is what the specification requires. \
Use "revise" only when you can state what the correct value is. Do not \
write or execute code."""

# Two examples, one of each verdict, so neither outcome is the primed
# default. Format alignment only.
CONTRACT_CHECK_FEWSHOT = """\
# Output format examples

Signature: add(a: int, b: int) -> int
Docstring: Return the sum of a and b.
Inputs: {"a": 2, "b": 3}
Proposed result: 5
Check: the docstring requires the sum; 2 + 3 is 5; the declared type is int.
{"verdict": "accept", "result": 5}

Signature: initials(name: str) -> str
Docstring: Return the uppercase first letter of each word, separated by dots.
Inputs: {"name": "ada lovelace"}
Proposed result: "AL"
Check: the docstring requires dot separation, which the proposed value omits.
{"verdict": "revise", "result": "A.L"}"""


def build_contract_check_prompt(
    signature: str,
    docstring: str,
    inputs: Dict[str, Any],
    candidate: Any,
    include_fewshot: bool = True,
) -> str:
    inputs_json = json.dumps(inputs, ensure_ascii=False, default=str)
    candidate_json = json.dumps(candidate, ensure_ascii=False, default=str)
    user_block = (
        f"Signature: {signature}\n"
        f"Docstring: {docstring}\n"
        f"Inputs: {inputs_json}\n"
        f"Proposed result: {candidate_json}\n"
        f"Check:"
    )
    parts = [CONTRACT_CHECK_SYSTEM]
    if include_fewshot:
        parts.append(CONTRACT_CHECK_FEWSHOT)
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
    # The sandbox that runs this code provides these modules as globals and
    # blocks the import statement outright (there is no __import__ in its
    # builtins). Saying so is not a hint about the task: without it the model
    # writes the idiomatic `import re` and the generated function dies on its
    # first line, scoring a code-generation failure that is really a harness
    # restriction. On the first pilot this was 15 of 30 baseline failures.
    # Keep the list in sync with code_gen_baseline._build_safe_globals().
    environment = (
        "The following modules are already available as globals: "
        "math, json, re, itertools, functools, collections, string, statistics. "
        "Use them directly. Do not write import statements - imports are "
        "disabled in the execution environment and will fail."
    )
    user_block = (
        f"def {signature}:\n"
        f'    """{docstring}"""\n'
        f"    # Implementation:"
    )
    return f"{CODEGEN_SYSTEM}\n\n{environment}\n\n{user_block}"


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
