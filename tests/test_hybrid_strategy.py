"""Hybrid routing follows its documented fixed rules and preserves delegation."""

import json
import unittest

from aicomputing import Backend, ExecuteRequest, VirtualRuntime
from aicomputing.answer_protocol import frame_answer
from aicomputing.backends.base import GenerateResponse
from aicomputing.strategies.hybrid import HybridStrategy


class RoutingBackend(Backend):
    name = "offline"
    default_model = "offline"

    def __init__(self, *values):
        self.values = values or (5,)
        self.calls = []

    def generate(self, prompt, **settings):
        value = self.values[min(len(self.calls), len(self.values) - 1)]
        self.calls.append((prompt, settings))
        return GenerateResponse(text=frame_answer(json.dumps({"result": value})), model="offline",
                                input_tokens=2, output_tokens=3,
                                finish_reason="stop", provider_finish_reason="end_turn")


def request(docstring="Return the supplied value.", *, signature="f(x: Any) -> int", value=5, **settings):
    defaults = dict(signature=signature, docstring=docstring, inputs={"x": value}, max_retries=0)
    return ExecuteRequest(**(defaults | settings))


class HybridRoutingTests(unittest.TestCase):
    def test_keywords_match_whole_words(self):
        cases = (("Compute the stage value.", "cot"),
                 ("Return an assortment.", "direct"),
                 ("Return the transformation name.", "direct"))
        strategy = HybridStrategy()
        for description, expected in cases:
            with self.subTest(description=description):
                self.assertEqual(strategy.classify_task(request(description)), expected)

    def test_case_and_snake_case_normalization_apply_to_signature_and_description(self):
        cases = (
            ("COMPUTE(x: Any) -> int", "Return the supplied value.", "cot"),
            ("f(x: Any) -> int", "CALCULATE the value.", "cot"),
            ("f(x: Any) -> int", "CLASSIFY a graph.", "direct"),
            ("parse_text(x: Any) -> int", "Compute the value.", "direct"),
            ("score_review(x: Any) -> int", "Return a numerical value.", "direct"),
            ("binary_search(x: Any) -> int", "Return the supplied value.", "cot"),
        )
        strategy = HybridStrategy()
        for signature, description, expected in cases:
            with self.subTest(signature=signature, description=description):
                self.assertEqual(strategy.classify_task(request(description, signature=signature)), expected)

    def test_nlp_keywords_take_priority_over_math_algorithms_and_input_shapes(self):
        cases = (
            request("Classify the graph.", signature="compute(x: Any) -> int", value={"items": [1]}),
            request("Compute the graph path.", signature="classify(x: Any) -> int", value=list(range(21))),
        )
        strategy = HybridStrategy()
        for task in cases:
            with self.subTest(signature=task.signature, description=task.docstring):
                self.assertEqual(strategy.classify_task(task), "direct")

    def test_input_shape_boundaries_and_direct_fallback(self):
        cases = (
            (5, "direct"),
            ("compute a graph", "direct"),
            ([], "direct"),
            (list(range(20)), "direct"),
            (list(range(21)), "cot"),
            ({"inner": [1]}, "cot"),
            ({"inner": {}}, "cot"),
            ({"inner": ()}, "cot"),
            ({"inner": 1}, "direct"),
            ([{"inner": 1}], "direct"),
            (tuple(range(21)), "direct"),
        )
        strategy = HybridStrategy()
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(strategy.classify_task(request(value=value)), expected)

    def test_selected_delegate_supplies_prompt_usage_and_result_metadata(self):
        cases = (("Return the supplied value.", "direct", "only the final-answer block"),
                 ("Compute the supplied value.", "cot", "Reason through the task step by step:"))
        for description, route, marker in cases:
            with self.subTest(route=route):
                backend = RoutingBackend()
                task = request(description, temperature=0.3, top_p=0.8, max_new_tokens=128)
                with VirtualRuntime(backend=backend, strategy="hybrid") as runtime:
                    result = runtime.execute(task)
                self.assertEqual((result.result, result.strategy, result.substrategy), (5, "hybrid", route))
                self.assertEqual((result.model_calls, result.total_tokens, len(backend.calls)), (1, 5, 1))
                self.assertEqual(result.provider_finish_reason, "end_turn")
                self.assertTrue(result.raw_text.startswith(f"[hybrid -> {route}] "))
                self.assertEqual(result.prompts, [backend.calls[0][0]])
                prompt, settings = backend.calls[0]
                self.assertIn(marker, prompt)
                self.assertIn(description, prompt)
                self.assertEqual((settings["temperature"], settings["top_p"], settings["max_new_tokens"]),
                                 (0.3, 0.8, 128))

    def test_retry_feedback_is_forwarded_without_changing_the_route(self):
        cases = (("Return the supplied value.", "direct", "compute", "only the final-answer block"),
                 ("Compute the supplied value.", "cot", "classify", "Reason through the task step by step:"))
        for description, route, invalid_value, marker in cases:
            with self.subTest(route=route):
                backend = RoutingBackend(invalid_value, 5)
                with VirtualRuntime(backend=backend, strategy="hybrid") as runtime:
                    result = runtime.execute(request(description, max_retries=1))
                self.assertEqual((result.result, result.substrategy, result.retries, result.model_calls),
                                 (5, route, 1, 2))
                self.assertEqual(len(backend.calls), 2)
                first, second = (call[0] for call in backend.calls)
                self.assertIn(marker, first)
                self.assertIn(marker, second)
                self.assertNotIn("Previous output:", first)
                self.assertIn("Previous output:", second)
                self.assertIn("Validation error:", second)
                self.assertIn(invalid_value, second)


if __name__ == "__main__":
    unittest.main()
