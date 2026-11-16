"""Worked-example settings require Booleans at every public entry point."""

import json
import unittest
from unittest.mock import patch

from aicomputing import Backend, ExecuteRequest, VirtualRuntime, create_strategy
from aicomputing import prompting
from aicomputing.answer_protocol import frame_answer
from aicomputing.backends.base import GenerateResponse


STRATEGIES = ("direct", "cot", "pseudocode", "confidence", "contract", "consistency")
BUILDERS = (
    prompting.build_direct_prompt,
    prompting.build_cot_prompt,
    prompting.build_pseudocode_prompt,
    prompting.build_confidence_prompt,
    prompting.build_contract_check_prompt,
)
INVALID_FLAGS = (None, 0, 1, -1, 0.0, 1.0, "", "false", "true", [], [False], {}, {"enabled": False}, b"false")


class FewshotBackend(Backend):
    name = "offline"
    default_model = "offline"

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, **settings):
        self.prompts.append(prompt)
        reply = {"result": 5}
        if "You are checking a proposed result" in prompt:
            reply["verdict"] = "accept"
        return GenerateResponse(text=frame_answer(json.dumps(reply)), model="offline")


class FewshotSettingTests(unittest.TestCase):
    def test_strategy_constructors_reject_non_boolean_settings(self):
        for name in STRATEGIES:
            for flag in INVALID_FLAGS:
                with self.subTest(strategy=name, flag=flag):
                    with self.assertRaisesRegex(TypeError, "^include_fewshot must be a bool$"):
                        create_strategy(name, include_fewshot=flag)

    def test_prompt_builders_reject_non_boolean_settings_before_formatting_values(self):
        with patch.object(prompting, "format_prompt_value", side_effect=AssertionError("Values were formatted")) as formatter:
            for builder in BUILDERS:
                for flag in INVALID_FLAGS:
                    with self.subTest(builder=builder.__name__, flag=flag):
                        settings = dict(signature="f(x: int) -> int", docstring="Return x.",
                                        inputs={"x": 5}, include_fewshot=flag)
                        if builder is prompting.build_contract_check_prompt:
                            settings["candidate"] = 5
                        with self.assertRaisesRegex(TypeError, "^include_fewshot must be a bool$"):
                            builder(**settings)
            formatter.assert_not_called()

    def test_boolean_settings_control_worked_examples_through_strategy_execution(self):
        for name in STRATEGIES:
            for enabled in (True, False):
                with self.subTest(strategy=name, enabled=enabled):
                    backend = FewshotBackend()
                    strategy = create_strategy(name, include_fewshot=enabled)
                    request = ExecuteRequest(signature="f(x: int) -> int", docstring="Return x.",
                                             inputs={"x": 5}, max_retries=0)
                    with VirtualRuntime(backend=backend, strategy=strategy) as runtime:
                        result = runtime.execute(request)
                    self.assertEqual((result.result, result.validation_status), (5, "valid"))
                    self.assertTrue(backend.prompts)
                    for prompt in backend.prompts:
                        self.assertEqual("BEGIN WORKED EXAMPLES" in prompt, enabled)
                        self.assertIn("VALUE NOTATION", prompt)


if __name__ == "__main__":
    unittest.main()
