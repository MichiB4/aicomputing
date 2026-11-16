"""Only None inherits selection defaults; invalid names never reach inference."""

import unittest
from unittest.mock import patch

from aicomputing import Backend, ExecuteRequest, Strategy, VirtualRuntime, create_backend, create_strategy
from aicomputing.answer_protocol import frame_answer
from aicomputing.backends.base import GenerateResponse
from aicomputing.backends.hf import HFBackend
from aicomputing.backends.ollama import OllamaBackend
from aicomputing.backends.openai_compat import OpenAICompatBackend
from aicomputing.strategies.base import StrategyResult


INVALID_NAMES = ([(value, TypeError) for value in (False, True, 0, 0.0, [], ["name"], {}, {"name": "value"})]
                 + [(value, ValueError) for value in ("", "  ", "\t\n")])


class SelectionBackend(Backend):
    name = "offline"
    default_model = "backend-default"

    def __init__(self):
        self.calls = []

    def cache_state(self):
        return {}

    def generate(self, prompt, **settings):
        self.calls.append(settings)
        model = self.default_model if settings.get("model") is None else settings["model"]
        return GenerateResponse(text=frame_answer('{"result": 5}'), model=model)


class CustomStrategy(Strategy):
    name = "custom-route"

    def execute(self, request, backend, feedback=None):
        return StrategyResult(value=request.inputs["x"], raw_text="Custom strategy value", parse_status="parsed")


def request(**settings):
    return ExecuteRequest(signature="f(x: int) -> int", docstring="Return x.", inputs={"x": 5}, **settings)


class NameOverrideTests(unittest.TestCase):
    def test_optional_request_names_reject_blank_and_wrong_types(self):
        for name in ("model", "strategy"):
            for value, error in INVALID_NAMES:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(error, name):
                    request(**{name: value})

    def test_invalid_runtime_selections_fail_before_backend_creation(self):
        for name in ("model", "strategy"):
            cases = INVALID_NAMES + ([(None, TypeError)] if name == "strategy" else [])
            for value, error in cases:
                with self.subTest(name=name, value=value), patch("aicomputing.runtime.create_backend") as factory:
                    with self.assertRaisesRegex(error, name):
                        VirtualRuntime(backend="openai", **{name: value})
                    factory.assert_not_called()

    def test_malformed_mutated_overrides_cannot_fall_back_or_return_a_cached_prediction(self):
        for mode in ("strict", "warn", "none"):
            for name in ("model", "strategy"):
                for value, error in ((False, TypeError), ([], TypeError), ("", ValueError)):
                    with self.subTest(mode=mode, name=name, value=value):
                        backend = SelectionBackend()
                        call = request()
                        with VirtualRuntime(cache_size=8, backend=backend, validation_mode=mode) as runtime:
                            runtime.execute(call)
                            self.assertEqual(len(runtime.cache), 1)
                            setattr(call, name, value)
                            with self.assertRaisesRegex(error, name):
                                runtime.execute(call)
                            self.assertEqual(runtime.cache.hits, 0)
                        self.assertEqual(len(backend.calls), 1)

    def test_none_and_literal_overrides_resolve_without_rewriting_the_request(self):
        cases = ((None, None, "direct", None, "backend-default", "direct"),
                 ("runtime-model", None, "cot", None, "runtime-model", "cot"),
                 ("runtime-model", " Org/Custom:Tag ", "direct", "cot", " Org/Custom:Tag ", "cot"))
        for configured_model, override_model, configured_strategy, override_strategy, expected_model, expected_strategy in cases:
            with self.subTest(model=override_model, strategy=override_strategy):
                backend = SelectionBackend()
                call = request(model=override_model, strategy=override_strategy)
                with VirtualRuntime(backend=backend, model=configured_model, strategy=configured_strategy) as runtime:
                    result = runtime.execute(call)
                self.assertEqual((result.model, result.strategy), (expected_model, expected_strategy))
                self.assertEqual(backend.calls[0]["model"], expected_model)
                self.assertEqual((call.model, call.strategy), (override_model, override_strategy))

    def test_custom_strategy_instances_can_be_selected_without_registry_membership(self):
        for override in (None, "custom-route"):
            with self.subTest(override=override):
                backend = SelectionBackend()
                strategy = CustomStrategy()
                with patch("aicomputing.runtime.create_strategy") as factory:
                    with VirtualRuntime(backend=backend, strategy=strategy) as runtime:
                        result = runtime.execute(request(strategy=override))
                    factory.assert_not_called()
                self.assertEqual((result.result, result.strategy), (5, "custom-route"))
                self.assertFalse(backend.calls)

    def test_invalid_custom_names_are_rejected_before_inference(self):
        for attribute in ("backend_default", "strategy_name"):
            for value, error in INVALID_NAMES + [(None, TypeError)]:
                with self.subTest(attribute=attribute, value=value):
                    backend = SelectionBackend()
                    strategy = CustomStrategy()
                    if attribute == "backend_default":
                        backend.default_model = value
                    else:
                        strategy.name = value
                    with self.assertRaises(error):
                        VirtualRuntime(backend=backend, strategy=strategy)
                    self.assertFalse(backend.calls)

    def test_backend_model_names_are_required_and_direct_overrides_are_optional(self):
        for backend_type, loader in ((OpenAICompatBackend, "_get_client"),
                                     (OllamaBackend, "_get_client"), (HFBackend, "_load")):
            for value, error in INVALID_NAMES + [(None, TypeError)]:
                with self.subTest(backend=backend_type.__name__, configured_model=value):
                    with self.assertRaisesRegex(error, "model"):
                        backend_type(model=value)
            backend = backend_type(model="provider/Custom:Tag")
            with patch.object(backend, loader) as load:
                for method, argument in (("generate", "prompt"),
                                         ("generate_chat", [{"role": "user", "content": "prompt"}])):
                    for value, error in INVALID_NAMES:
                        with self.subTest(backend=backend_type.__name__, method=method, model=value):
                            with self.assertRaisesRegex(error, "model"):
                                getattr(backend, method)(argument, model=value)
                load.assert_not_called()

    def test_default_chat_adapter_rejects_invalid_names_before_delegating(self):
        backend = SelectionBackend()
        for value, error in INVALID_NAMES:
            with self.subTest(model=value), self.assertRaisesRegex(error, "model"):
                backend.generate_chat([{"role": "user", "content": "prompt"}], model=value)
        self.assertFalse(backend.calls)

    def test_factories_reject_invalid_names_and_keep_unknown_name_errors(self):
        for factory, label in ((create_backend, "backend"), (create_strategy, "strategy")):
            for value, error in INVALID_NAMES + [(None, TypeError)]:
                with self.subTest(factory=label, name=value), self.assertRaisesRegex(error, label):
                    factory(value)
            with self.assertRaisesRegex(ValueError, "Unknown " + label):
                factory("not-registered")


if __name__ == "__main__":
    unittest.main()
