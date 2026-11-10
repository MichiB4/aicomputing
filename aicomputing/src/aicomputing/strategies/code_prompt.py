"""Provider-independent source-generation instructions for concrete tasks."""
from ..prompting import format_prompt_value

ANSWER_FORMAT = "python_code_v1"

PROMPT_REVISION = "code_execute_v8_standard_python"


def build_prompt(signature, docstring, inputs, *, task_mode="function"):
    """Build source instructions without benchmark IDs or grading targets.

    ``word_problem`` requests arithmetic for the supplied question. The default
    ``function`` mode requests an implementation for general function inputs.
    """
    if task_mode not in ("function", "word_problem"):
        raise ValueError("task_mode must be 'function' or 'word_problem'")
    task_instructions = (
        "Write arithmetic code for this specific supplied word problem. Translate "
        "the quantities and relationships in the question into clearly named "
        "variables. Interpret numbers written in words, fractions, percentages, "
        "units, groups, and time periods according to their meaning in the text. "
        "Constants stated in this question may appear directly in the program. "
        "Compute the result with arithmetic; do not return a literal final answer. "
        "Do not build a generic natural-language parser or use regex number "
        "extraction to guess the roles of quantities. The question parameter "
        "identifies this supplied problem; you need not solve unrelated questions."
        if task_mode == "word_problem" else
        "Implement the specified behavior using the function parameters, for "
        "general inputs of the declared function. Do not substitute the actual "
        "input or hardcode public-example answers. Public examples clarify the "
        "behavior; the actual invocation is given separately below."
    )
    return (
        "Implement the Python function described below. Return only Python source, "
        "optionally inside one ```python fence. Include the complete function "
        "definition with the exact function name and argument names. Helpers are "
        "allowed. The Python worker will call the function with the actual inputs. "
        "Do not print or call the target function at module scope.\n\n"
        "CODE REVIEW\n"
        "Return complete, syntactically valid Python code that implements the "
        "requested function. Before submitting, review the entire program for "
        "syntax errors, valid identifiers, indentation, balanced delimiters, "
        "imports, and return statements.\n\n"
        "SEMANTIC REVIEW\n"
        "Check the implementation against the specification and supplied public "
        "examples. Verify intermediate calculations and that the returned value "
        "answers the requested question. Use full-precision standard-library "
        "constants and functions unless an approximation is explicitly required.\n\n"
        f"IMPLEMENTATION REQUIREMENTS\n{task_instructions}\n\n"
        "EXECUTION ENVIRONMENT\n"
        "Standard Python using the application's interpreter, installed packages, "
        "working directory and environment. Include every import used by your "
        "implementation, for example `import math` before `math.sqrt(...)` or "
        "`import re` before `re.split(...)`. Normal Python features, imports and "
        "helper classes are available.\n\n"
        "Annotations may be omitted. "
        "Return the Python value, preserving its container types.\n\n"
        "BEGIN ACTUAL TASK\n"
        f"Signature: {signature}\n"
        f"Docstring: {docstring}\n"
        f"Inputs (Python values): {format_prompt_value(inputs)}\n"
        "END ACTUAL TASK\n\n"
        "YOUR PYTHON IMPLEMENTATION:\n"
    )
