"""Missing attempt accounting must not turn partial usage into complete totals."""

import json
import unittest

from aicomputing import Backend, ExecuteRequest, Strategy, VirtualRuntime
from aicomputing.answer_protocol import frame_answer
from aicomputing.backends.base import GenerateResponse
from aicomputing.parsing import parse_result
from aicomputing.strategies.base import CallStats, StrategyResult


class AccountingBackend(Backend):
    name = "offline"
    default_model = "offline"

    def __init__(self, attempts, input_tokens=3, output_tokens=5):
        self.attempts = attempts
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls = 0

    def generate(self, prompt, **settings):
        self.calls += 1
        payload = {"result": 5} if self.calls == self.attempts else {"other": 0}
        return GenerateResponse(text=frame_answer(json.dumps(payload)), model="offline",
                                input_tokens=self.input_tokens, output_tokens=self.output_tokens)


class SelectiveAccountingStrategy(Strategy):
    name = "selective-accounting"

    def __init__(self, recorded):
        self.recorded = iter(recorded)

    def execute(self, request, backend, feedback=None):
        response = backend.generate("Offline accounting test")
        outcome = parse_result(response.parsing_text)
        stats = CallStats().record(response) if next(self.recorded) else None
        return StrategyResult(value=outcome.value, raw_text=response.text,
                              parse_status=outcome.status, call_stats=stats)


class CallAccountingTests(unittest.TestCase):
    def test_missing_attempt_accounting_keeps_invocation_token_totals_unknown(self):
        plans = ((False, True), (True, False), (False, False),
                 (False, True, True), (True, False, True))
        for mode in ("strict", "warn", "none"):
            for recorded in plans:
                with self.subTest(mode=mode, recorded=recorded):
                    backend = AccountingBackend(len(recorded))
                    request = ExecuteRequest(signature="f(x: int) -> int", docstring="Return x.",
                                             inputs={"x": 5}, max_retries=len(recorded) - 1)
                    with VirtualRuntime(backend=backend, strategy=SelectiveAccountingStrategy(recorded),
                                        validation_mode=mode) as runtime:
                        result = runtime.execute(request)
                    self.assertEqual(result.result, 5)
                    self.assertEqual(result.validation_status, "not_checked" if mode == "none" else "valid")
                    self.assertEqual((backend.calls, result.retries), (len(recorded), len(recorded) - 1))
                    self.assertEqual(result.model_calls, sum(recorded))
                    self.assertEqual(len(result.reasoning_requests), sum(recorded))
                    self.assertIsNone(result.input_tokens)
                    self.assertIsNone(result.output_tokens)
                    self.assertIsNone(result.total_tokens)

    def test_fully_reported_usage_keeps_exact_totals_including_reported_zero(self):
        for mode in ("strict", "warn", "none"):
            for input_tokens, output_tokens in ((0, 0), (3, 5)):
                with self.subTest(mode=mode, usage=(input_tokens, output_tokens)):
                    backend = AccountingBackend(3, input_tokens, output_tokens)
                    request = ExecuteRequest(signature="f(x: int) -> int", docstring="Return x.",
                                             inputs={"x": 5}, max_retries=2)
                    with VirtualRuntime(backend=backend, strategy=SelectiveAccountingStrategy((True,) * 3),
                                        validation_mode=mode) as runtime:
                        result = runtime.execute(request)
                    self.assertEqual((result.result, result.model_calls, result.retries), (5, 3, 2))
                    self.assertEqual((result.input_tokens, result.output_tokens, result.total_tokens),
                                     (3 * input_tokens, 3 * output_tokens, 3 * (input_tokens + output_tokens)))


if __name__ == "__main__":
    unittest.main()
