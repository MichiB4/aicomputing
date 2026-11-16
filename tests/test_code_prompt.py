"""Prompt contracts for source generation, with no model or dataset access."""
from types import SimpleNamespace
import unittest

from aicomputing.strategies.code_prompt import build_prompt
from evaluation.benchmarks.prepared_data import code_task_specification


PREFIX = (
    "The following is the original public task specification. Any calls, "
    "input/output pairs, or assertions inside it are public examples, not the "
    "actual input to solve.\n\n"
)
DIRECTIVE = (
    "Predict only the return value for the concrete Inputs (Python values) "
    "in the actual task. Do not answer a public example."
)


def item(docstring, *, item_id="HumanEval/example", expected="PRIVATE_GRADING_TARGET"):
    return SimpleNamespace(
        item_id=item_id, signature="split_words(text)", docstring=docstring,
        inputs={"text": "new,actual input"}, expected=expected,
    )


class CodeBaselinePromptTests(unittest.TestCase):
    def test_review_wrapper_removed_without_changing_original_spec_or_examples(self):
        original = "Split words on commas and spaces.\nKeep punctuation inside words."
        specification = (
            "BEGIN ORIGINAL PUBLIC TASK SPECIFICATION\n" + original
            + "\nEND ORIGINAL PUBLIC TASK SPECIFICATION"
        )
        examples = (
            "BEGIN ORIGINAL PUBLIC TASK EXAMPLES\n"
            "assert split_words('one, two') == ['one', 'two']\n"
            "END ORIGINAL PUBLIC TASK EXAMPLES"
        )
        wrapped = PREFIX + specification + "\n\n" + examples + "\n\n" + DIRECTIVE
        task = item(wrapped)
        prompt = build_prompt(task.signature, code_task_specification(task.docstring), task.inputs)
        self.assertIn(specification, prompt)
        self.assertIn(examples, prompt)
        self.assertNotIn(DIRECTIVE, prompt)
        self.assertEqual(task.docstring, wrapped)
        self.assertIn("new,actual input", prompt)

    def test_prediction_like_text_inside_public_spec_is_preserved(self):
        original = "Return whether the supplied text equals this literal:\n" + DIRECTIVE
        block = (
            "BEGIN ORIGINAL PUBLIC TASK SPECIFICATION\n" + original
            + "\nEND ORIGINAL PUBLIC TASK SPECIFICATION"
        )
        prompt = build_prompt("split_words(text)", code_task_specification(PREFIX + block + "\n\n" + DIRECTIVE),
                              {"text": "input"})
        self.assertIn(block, prompt)
        self.assertEqual(prompt.count(DIRECTIVE), 1)

    def test_unwrapped_specification_is_not_rewritten(self):
        original = "Preserve this public text exactly.\n" + DIRECTIVE
        prompt = build_prompt("split_words(text)", code_task_specification(original), {"text": "input"})
        self.assertIn(original, prompt)

    def test_import_guidance_describes_standard_python(self):
        prompt = build_prompt("split_words(text)", "Split words.", {"text": "input"})
        self.assertIn("import math", prompt)
        self.assertIn("import re", prompt)
        self.assertIn("installed packages", prompt)
        self.assertIn("helper classes are available", prompt)
        self.assertNotIn("Available computational modules", prompt)
        self.assertRegex(prompt.lower(), r"(all|required|every|each).{0,45}imports?")

    def test_gsm8k_prompt_targets_question_arithmetic_and_excludes_gold(self):
        question = "A boat takes on two liters per ten feet and rows twenty feet in sixteen seconds."
        task = SimpleNamespace(
            item_id="gsm8k/example", signature="solve_math(question: str) -> float",
            docstring="Solve the word problem.", inputs={"question": question},
            expected="PRIVATE_GSM8K_TARGET_924857",
        )
        prompt = build_prompt(task.signature, task.docstring, task.inputs, task_mode="word_problem")
        self.assertIn(question, prompt)
        self.assertNotIn(task.expected, prompt)
        self.assertRegex(prompt.lower(), r"(specific|supplied|actual).{0,45}(problem|question)")
        self.assertRegex(prompt.lower(), r"(spelled|written.{0,30}words|word.{0,20}numbers)")
        self.assertIn("fraction", prompt.lower())
        self.assertIn("unit", prompt.lower())
        self.assertIn("quantit", prompt.lower())
        self.assertIn("constant", prompt.lower())
        self.assertRegex(prompt.lower(), r"(regex|regular.expression)")

    def test_task_mode_is_explicit_and_independent_of_benchmarks(self):
        signature = "solve_math(question: str) -> float"
        inputs = {"question": "Half of twenty students are female. How many are female?"}
        arithmetic = build_prompt(signature, "Solve the supplied problem.", inputs, task_mode="word_problem")
        general = build_prompt(signature, "Solve the supplied problem.", inputs)
        self.assertIn("Write arithmetic code", arithmetic)
        self.assertIn("general inputs", general)
        with self.assertRaises(ValueError):
            build_prompt(signature, "Solve the supplied problem.", inputs, task_mode="gsm8k")


if __name__ == "__main__":
    unittest.main()
