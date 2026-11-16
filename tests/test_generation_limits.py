"""Generation bounds and explicit model-call counts, without inference services."""
import unittest
from unittest.mock import patch

from aicomputing import ExecuteRequest, VirtualRuntime
from aicomputing.answer_protocol import frame_answer
from aicomputing.backends.base import Backend, GenerateResponse
from aicomputing.backends.hf import HFBackend
from aicomputing.backends.ollama import OllamaBackend
from aicomputing.backends.openai_compat import OpenAICompatBackend


INVALID_SETTINGS = (
    [(name, value, TypeError) for name in ("temperature", "top_p")
     for value in (None, True, False, "0.5", [], {}, 1j)]
    + [(name, value, ValueError) for name in ("temperature", "top_p")
       for value in (float("nan"), float("inf"), float("-inf"), -1)]
    + [("top_p", 1.01, ValueError)]
)


class FinishingBackend(Backend):
    name = "offline"
    default_model = "offline"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def generate(self, prompt, **options):
        self.calls.append((prompt, options))
        payload, finish = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return GenerateResponse(text=frame_answer(payload), model=self.default_model,
                                finish_reason=finish, provider_finish_reason=finish,
                                input_tokens=5, output_tokens=10)


def request(**settings):
    defaults = dict(signature="f(x: int) -> int", docstring="Return x.", inputs={"x": 5},
                    max_new_tokens=100, max_retries=0)
    return ExecuteRequest(**(defaults | settings))


class GenerationSettingTests(unittest.TestCase):
    def test_requests_reject_invalid_generation_settings(self):
        for name, value, error in INVALID_SETTINGS:
            with self.subTest(name=name, value=value), self.assertRaisesRegex(error, name):
                request(**{name: value})

    def test_mutated_requests_are_rechecked_before_any_backend_call_in_every_output_mode(self):
        for mode in ("strict", "warn", "none"):
            for name, value in (("temperature", float("nan")), ("temperature", True), ("top_p", 2)):
                with self.subTest(mode=mode, name=name, value=value):
                    backend = FinishingBackend(('{"result": 5}', "stop"))
                    changed = request()
                    setattr(changed, name, value)
                    with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                        with self.assertRaisesRegex((TypeError, ValueError), name):
                            runtime.execute(changed)
                    self.assertFalse(backend.calls)

    def test_boundaries_and_temperature_above_two_are_forwarded_unchanged(self):
        for temperature, top_p in ((0, 0), (-0.0, 1), (0.7, 0.9), (3.5, 0.5), (10**400, 1)):
            with self.subTest(temperature=temperature, top_p=top_p):
                backend = FinishingBackend(('{"result": 5}', "stop"))
                with VirtualRuntime(backend=backend) as runtime:
                    result = runtime.execute(request(temperature=temperature, top_p=top_p))
                self.assertEqual(result.result, 5)
                self.assertEqual(backend.calls[0][1]["temperature"], temperature)
                self.assertEqual(backend.calls[0][1]["top_p"], top_p)

    def test_builtin_backends_reject_settings_before_client_or_model_loading(self):
        for backend_type, loader in ((OpenAICompatBackend, "_get_client"),
                                     (OllamaBackend, "_get_client"), (HFBackend, "_load")):
            backend = backend_type(model="offline")
            with patch.object(backend, loader) as load:
                for method, argument in (("generate", "prompt"),
                                         ("generate_chat", [{"role": "user", "content": "prompt"}])):
                    for name, value, error in INVALID_SETTINGS:
                        with self.subTest(backend=backend_type.__name__, method=method, name=name, value=value):
                            with self.assertRaisesRegex(error, name):
                                getattr(backend, method)(argument, **{name: value})
                load.assert_not_called()

    def test_default_chat_adapter_rejects_settings_before_delegating(self):
        backend = FinishingBackend(('{"result": 5}', "stop"))
        for name, value, error in INVALID_SETTINGS:
            with self.subTest(name=name, value=value), self.assertRaisesRegex(error, name):
                backend.generate_chat([{"role": "user", "content": "prompt"}], **{name: value})
        self.assertFalse(backend.calls)


class FixedTokenLimitTests(unittest.TestCase):
    def test_single_call_strategies_do_not_expand_or_repeat_truncated_responses(self):
        for strategy in ("direct", "cot", "pseudocode", "confidence", "hybrid"):
            for limit in (100, None):
                with self.subTest(strategy=strategy, limit=limit):
                    backend = FinishingBackend(('{"result": 999}', "length"))
                    with VirtualRuntime(backend=backend, strategy=strategy) as runtime:
                        result = runtime.execute(request(max_new_tokens=limit))
                    self.assertEqual([call[1]["max_new_tokens"] for call in backend.calls], [limit])
                    self.assertIsNone(result.result)
                    self.assertEqual((result.model_calls, result.retries), (1, 0))
                    self.assertEqual((result.input_tokens, result.output_tokens, result.total_tokens), (5, 10, 15))
                    self.assertEqual(result.parse_status, "failed")
                    self.assertIn("truncated", result.parse_error)
                    self.assertIn('999', result.raw_text)

    def test_validation_retry_keeps_the_same_limit_and_accounts_for_both_calls(self):
        backend = FinishingBackend(('{"result": 999}', "length"), ('{"result": 5}', "stop"))
        with VirtualRuntime(backend=backend) as runtime:
            result = runtime.execute(request(max_retries=1))
        self.assertEqual([call[1]["max_new_tokens"] for call in backend.calls], [100, 100])
        self.assertEqual((result.result, result.retries, result.model_calls), (5, 1, 2))
        self.assertEqual((result.input_tokens, result.output_tokens, result.total_tokens), (10, 20, 30))
        self.assertIn("Validation error: response truncated", backend.calls[1][0])

    def test_only_the_configured_validation_retries_repeat_generation(self):
        backend = FinishingBackend(('{"result": 999}', "length"))
        with VirtualRuntime(backend=backend) as runtime:
            result = runtime.execute(request(max_retries=3))
        self.assertEqual([call[1]["max_new_tokens"] for call in backend.calls], [100] * 4)
        self.assertEqual((result.result, result.retries, result.model_calls), (None, 3, 4))

    def test_contract_checks_an_unparsed_candidate_without_expanding_either_call(self):
        for verdict, expected in (("accept", None), ("revise", 5)):
            with self.subTest(verdict=verdict):
                check = '{"verdict": "' + verdict + '", "result": 5}'
                backend = FinishingBackend(('{"result": 999}', "length"), (check, "stop"))
                with VirtualRuntime(backend=backend, strategy="contract") as runtime:
                    result = runtime.execute(request())
                self.assertEqual([call[1]["max_new_tokens"] for call in backend.calls], [100, 100])
                self.assertIn("Unverified candidate return value (Python value): None", backend.calls[1][0])
                self.assertEqual((result.result, result.model_calls), (expected, 2))
                self.assertEqual(result.parse_status, "failed" if verdict == "accept" else "parsed")

    def test_contract_never_uses_a_replacement_from_a_truncated_check(self):
        backend = FinishingBackend(('{"result": 5}', "stop"),
                                   ('{"verdict": "revise", "result": 999}', "length"))
        with VirtualRuntime(backend=backend, strategy="contract") as runtime:
            result = runtime.execute(request())
        self.assertEqual([call[1]["max_new_tokens"] for call in backend.calls], [100, 100])
        self.assertEqual((result.result, result.model_calls, result.revised), (5, 2, False))

    def test_consistency_takes_only_its_planned_samples_and_excludes_truncated_votes(self):
        backend = FinishingBackend(('{"result": 999}', "length"), ('{"result": 5}', "stop"),
                                   ('{"result": 5}', "stop"), ('{"result": 999}', "length"),
                                   ('{"result": 6}', "stop"))
        with VirtualRuntime(backend=backend, strategy="consistency") as runtime:
            result = runtime.execute(request(temperature=0.7))
        self.assertEqual([call[1]["max_new_tokens"] for call in backend.calls], [100] * 5)
        self.assertEqual((result.result, result.model_calls), (5, 5))
        self.assertEqual((result.input_tokens, result.output_tokens, result.total_tokens), (25, 50, 75))
        self.assertTrue(result.raw_text.startswith("[consistency 2/5]"))

    def test_no_output_policy_accepts_a_provider_reported_truncated_value(self):
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                backend = FinishingBackend(('{"result": 999}', "length"))
                with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                    result = runtime.execute(request())
                self.assertEqual((result.result, result.validation_status, result.model_calls), (None, "unparsed", 1))


if __name__ == "__main__":
    unittest.main()
