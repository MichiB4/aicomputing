"""Offline coverage for signatures, validation, parsing, prediction strategies and the CLI."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from aicomputing import AITypeError, Constraint, ExecuteRequest, Strategy, ValidationMode, VirtualRuntime, parse_signature
from aicomputing.backends.base import Backend, BackendCommunicationError, BackendTimeoutError, GenerateResponse, ToolUseDetected
from aicomputing.cli import main as cli_main
from aicomputing.parsing import coerce_type, parse_result
from aicomputing.status import ParseStatus
from aicomputing.strategies.base import StrategyResult
from aicomputing.answer_protocol import ANSWER_FORMAT, frame_answer
from aicomputing.validation import check_value_type, validate_inputs


class ScriptedBackend(Backend):
    name = "scripted"
    default_model = "offline-model"

    def __init__(self, *responses, raw=False):
        self.responses = list(responses) if raw else [frame_answer(text) for text in responses]
        self.calls = []

    def cache_state(self):
        return {"responses": tuple(self.responses)}

    def generate(self, prompt, **options):
        self.calls.append((prompt, options))
        text = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return GenerateResponse(text=text, model=options.get("model") or self.default_model,
                                input_tokens=3, output_tokens=5, finish_reason="stop")


class ScriptedStrategy(Strategy):
    name = "custom"

    def __init__(self, *responses):
        self.responses = list(responses)
        self.feedback = []

    def execute(self, request, backend, feedback=None):
        self.feedback.append(feedback)
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


def add_request(**options):
    return ExecuteRequest(signature="add(x: int, y: int) -> int", docstring="Return x + y.",
                          inputs={"x": 2, "y": 3}, **options)


class SignatureAndValidationTests(unittest.TestCase):
    def test_parameters_defaults_and_generic_types(self):
        sig = parse_signature("f(xs: list[int], m: dict[str, tuple[int, ...]] = None) -> set[str]")
        self.assertEqual([p.name for p in sig.params], ["xs", "m"])
        self.assertEqual(sig.params[1].default, "None")
        self.assertEqual(sig.return_type, set[str])

    def test_markers_are_skipped_and_malformed_names_raise(self):
        self.assertEqual([p.name for p in parse_signature("f(a, /, b, *, c)").params], ["a", "b", "c"])
        for signature in ("f(x, y z)", "f(1w)", "f(: int)"):
            with self.subTest(signature=signature), self.assertRaises(ValueError):
                parse_signature(signature)

    def test_value_types_reject_bool_for_int_and_wrong_tuple_length(self):
        check_value_type([1, 2], list[int])
        check_value_type(3, float)
        for value, expected in ((True, int), ((1, 2), tuple[int]), (["a"], list[int])):
            with self.subTest(value=value), self.assertRaises(AITypeError):
                check_value_type(value, expected)

    def test_inputs_reject_missing_extra_and_wrong_typed_arguments(self):
        sig = parse_signature("add(x: int, y: int) -> int")
        for inputs in ({"x": 1}, {"x": 1, "y": 2, "z": 3}, {"x": "1", "y": 2}, None, [1, 2]):
            with self.subTest(inputs=inputs), self.assertRaises(AITypeError):
                validate_inputs(sig, inputs)

    def test_request_return_type_is_read_only_and_tracks_its_signature(self):
        request = add_request()
        self.assertIs(request.return_type, int)
        request.signature = "add(x: int, y: int) -> str"
        self.assertIs(request.return_type, str)
        with self.assertRaises(AttributeError):
            request.return_type = int
        with self.assertRaises(TypeError):
            add_request(return_type=str)


class RequestDescriptionTests(unittest.TestCase):
    def test_descriptions_must_be_nonblank_strings_in_every_output_mode(self):
        cases = [(value, TypeError) for value in (None, False, True, 0, 1.5, [], {}, b"Return x.")]
        cases += [(value, ValueError) for value in ("", " ", "\t\r\n", "\u00a0\u2003")]
        for mode in ("strict", "warn", "none"):
            for description, error in cases:
                with self.subTest(mode=mode, description=description), self.assertRaisesRegex(error, "docstring"):
                    ExecuteRequest(signature="f(x: int) -> int", docstring=description,
                                   inputs={"x": 5}, validation_mode=mode)

    def test_mutated_descriptions_are_rejected_before_strategy_calls(self):
        for strategy in ("direct", "hybrid", "code_execute"):
            for mode in ("strict", "warn", "none"):
                for description, error in ((None, TypeError), ([], TypeError), (" \t\n", ValueError)):
                    with self.subTest(strategy=strategy, mode=mode, description=description):
                        backend = ScriptedBackend('{"result": 5}')
                        request = add_request()
                        request.docstring = description
                        with VirtualRuntime(backend=backend, strategy=strategy, validation_mode=mode) as runtime:
                            with patch.object(runtime.strategy, "execute") as execute:
                                with self.assertRaisesRegex(error, "docstring"):
                                    runtime.execute(request)
                                execute.assert_not_called()
                        self.assertFalse(backend.calls)

    def test_invalid_description_cannot_reach_a_populated_cache(self):
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                backend = ScriptedBackend('{"result": 5}')
                request = add_request()
                original = request.docstring
                with VirtualRuntime(cache_size=8, backend=backend, validation_mode=mode) as runtime:
                    runtime.execute(request)
                    self.assertEqual(len(runtime.cache), 1)
                    with patch.object(runtime.cache, "get", wraps=runtime.cache.get) as get:
                        for description, error in ((None, TypeError), ([], TypeError), ("", ValueError)):
                            request.docstring = description
                            with self.assertRaisesRegex(error, "docstring"):
                                runtime.execute(request)
                        get.assert_not_called()
                    request.docstring = original
                    self.assertTrue(runtime.execute(request).cached)
                self.assertEqual(len(backend.calls), 1)

    def test_nonblank_multiline_descriptions_are_not_normalized(self):
        description = " \tReturn x + y.\n\nPreserve every newline.\r\n "
        for strategy in ("direct", "hybrid"):
            with self.subTest(strategy=strategy):
                backend = ScriptedBackend('{"result": 5}')
                request = add_request()
                request.docstring = description
                with VirtualRuntime(backend=backend, strategy=strategy) as runtime:
                    self.assertEqual(runtime.execute(request).result, 5)
                self.assertEqual(request.docstring, description)
                self.assertIn(description, backend.calls[0][0])


class ParsingTests(unittest.TestCase):
    def test_unframed_answers_and_reasoning_are_not_guessed(self):
        for text in ('7', '{"result": 7}', 'State: {a: 1}\nso {"result": 7}', 'hello there'):
            with self.subTest(text=text):
                self.assertEqual(parse_result(text).status, ParseStatus.FAILED)

    def test_malformed_numeric_answer_is_preserved_as_exact_text(self):
        for text in ('7?', '1234...', '1 + 2'):
            with self.subTest(text=text):
                outcome = parse_result(frame_answer('{"result": ' + text + '}'))
                self.assertEqual((outcome.value, outcome.status), (text, ParseStatus.FALLBACK))
                self.assertTrue(outcome.error)

    def test_decoding_does_not_convert_numeric_strings(self):
        outcome = parse_result(frame_answer('{"result": "7"}'))
        self.assertEqual((outcome.value, outcome.status), ("7", ParseStatus.PARSED))

    def test_framed_requires_one_final_block(self):
        good = parse_result('why\nBEGIN FINAL ANSWER\n{"result": [1, 2]}\nEND FINAL ANSWER')
        self.assertEqual(good.value, [1, 2])
        bad = parse_result('{"result": 1}')
        self.assertEqual(bad.status, ParseStatus.FAILED)

    def test_coercion_keeps_unconvertible_values(self):
        self.assertEqual(coerce_type("12", int), 12)
        self.assertEqual(coerce_type("x", int), "x")
        self.assertEqual(coerce_type([1, 2], tuple[int, int]), (1, 2))
        self.assertIs(coerce_type(True, int), True)

    def test_python_container_types_and_dictionary_keys_are_preserved(self):
        cases = [('(1,)', (1,)), ('{1, 2}', {1, 2}), ('set()', set()),
                 ('b"abc"', b'abc'), ('{1: "a"}', {1: 'a'}), ('None', None),
                 ('[True, 1, 1.5, "x"]', [True, 1, 1.5, 'x'])]
        for literal, expected in cases:
            with self.subTest(literal=literal):
                outcome = parse_result(frame_answer('{"result": ' + literal + '}'))
                self.assertEqual(outcome.status, ParseStatus.PARSED)
                self.assertEqual(outcome.value, expected)
                self.assertIs(type(outcome.value), type(expected))

    def test_missing_empty_and_ambiguous_result_fields_fail(self):
        for payload in ('{}', '{"result": }', '{"other": 7}',
                        '{"result": 7, "extra": 8}', '{"result": 7?, "extra": 8}',
                        '{"result": 7, "result": 8}', '{"result": {"x": 1, "x": 2}}',
                        "{'result': {'x': 1, 'x': 2}}", '{"result": "unfinished}',
                        '{"result": (1]}'):
            with self.subTest(payload=payload):
                self.assertEqual(parse_result(frame_answer(payload)).status, ParseStatus.FAILED)

    def test_only_the_final_block_is_read_and_extra_blocks_or_trailing_text_fail(self):
        final = frame_answer('{"result": 7}')
        outcome = parse_result('Example: {"result": 999}\nState: {x: 42}\n' + final)
        self.assertEqual(outcome.value, 7)
        for reply in (final + '\n' + final, final + '\nActually, 8.',
                      'END FINAL ANSWER\n' + final, 'BEGIN FINAL ANSWER\n{"result": 7}'):
            with self.subTest(reply=reply):
                self.assertEqual(parse_result(reply).status, ParseStatus.FAILED)

    def test_malformed_container_text_is_not_replaced_by_a_valid_inner_fragment(self):
        literal = '[1, {"x": 2}BROKEN]'
        outcome = parse_result(frame_answer('{"result": ' + literal + '}'))
        self.assertEqual((outcome.value, outcome.status), (literal, ParseStatus.FALLBACK))

    def test_contract_recovery_preserves_result_text_and_requires_a_valid_verdict(self):
        good = frame_answer('{"verdict": "revise", "result": 7?}')
        outcome = parse_result(good, unwrap_result=False)
        self.assertEqual(outcome.value, {"verdict": "revise", "result": "7?"})
        self.assertEqual(outcome.status, ParseStatus.FALLBACK)
        for payload in ('{"verdict": "maybe", "result": 7}',
                        '{"verdict": "revise", "result": }',
                        '{"verdict": "revise", "result": 7?, "extra": 8}'):
            with self.subTest(payload=payload):
                self.assertEqual(parse_result(frame_answer(payload), unwrap_result=False).status, ParseStatus.FAILED)

    def test_answer_format_is_not_a_request_setting(self):
        with self.assertRaises(TypeError):
            add_request(answer_format="legacy")

    def test_contract_recovery_accepts_both_field_orders_without_changing_result_text(self):
        answers = ('7?', '1234...', '1 + 2', '[1, {"verdict": "accept"}]BROKEN',
                   '"a, b" + suffix', '"""a, b""" + suffix')
        for verdict in ("accept", "revise"):
            for quote in ('"', "'"):
                for answer in answers:
                    fields = [f'{quote}verdict{quote}: {quote}{verdict}{quote}',
                              f'{quote}result{quote}: {answer}']
                    for ordered in (fields, fields[::-1]):
                        payload = "{\n  " + ",\n  ".join(ordered) + "\n}"
                        with self.subTest(payload=payload):
                            outcome = parse_result(frame_answer(payload), unwrap_result=False)
                            self.assertEqual(outcome.value, {"verdict": verdict, "result": answer})
                            self.assertEqual(outcome.status, ParseStatus.FALLBACK)
                            self.assertTrue(outcome.error)

    def test_contract_recovery_rejects_ambiguous_fields_in_both_orders(self):
        cases = (
            ('"verdict": "revise"', '"result": '),
            ('"verdict": "maybe"', '"result": 7?'),
            ('"verdict": 1', '"result": 7?'),
            ('"verdict": "revise"', '"result": "unfinished'),
            ('"verdict": "revise"', '"result": """unfinished'),
            ('"verdict": "revise"', '"result": (1]'),
            ('"verdict": "revise"', '"result": {"x": 1'),
            ('"verdict": "revise"', '"result": 7?, 8?'),
            ('"verdict": "revise"', '"result": 7?', '"extra": 8'),
            ('"verdict": "revise"', '"result": 7?', '"result": 8?'),
            ('"verdict": "revise"', '"result": 7?', '"verdict": "accept"'),
            ('"verdict": "revise"', '"result": {"x": 1, "x": 2}'),
        )
        for fields in cases:
            for ordered in (fields, fields[::-1]):
                payload = "{" + ", ".join(ordered) + "}"
                with self.subTest(payload=payload):
                    outcome = parse_result(frame_answer(payload), unwrap_result=False)
                    self.assertEqual(outcome.status, ParseStatus.FAILED)
                    self.assertIsNone(outcome.value)

    def test_unclosed_containers_cannot_hide_a_trailing_contract_verdict(self):
        for answer in ('[1, 2', '(1, 2', '[{"x": 1}', '("x", [2'):
            with self.subTest(answer=answer):
                leading = '{"verdict": "revise", "result": ' + answer + '}'
                outcome = parse_result(frame_answer(leading), unwrap_result=False)
                self.assertEqual(outcome.status, ParseStatus.FALLBACK)
                self.assertEqual(outcome.value, {"verdict": "revise", "result": answer})
                trailing = '{"result": ' + answer + ', "verdict": "revise"}'
                outcome = parse_result(frame_answer(trailing), unwrap_result=False)
                self.assertEqual(outcome.status, ParseStatus.FAILED)
                self.assertIsNone(outcome.value)


class PredictionStrategyTests(unittest.TestCase):
    def test_explicit_parse_failure_overrides_attached_values_in_every_output_mode(self):
        for mode in ("strict", "warn", "none"):
            for value in (7, 0, False, "", [], None):
                with self.subTest(mode=mode, value=value):
                    predicate = unittest.mock.Mock(side_effect=AssertionError("Failed recovery reached a predicate"))
                    strategy = ScriptedStrategy(StrategyResult(
                        value=value, raw_text="custom response", parse_status=ParseStatus.FAILED,
                        parse_error="Custom parsing failed.",
                    ))
                    backend = ScriptedBackend('{"result": 7}')
                    request = ExecuteRequest(signature="f() -> Any", docstring="Return a value.", inputs={},
                                             constraints=[Constraint("valid", predicate)], max_retries=0)
                    with VirtualRuntime(backend=backend, strategy=strategy, validation_mode=mode) as runtime:
                        result = runtime.execute(request)
                    self.assertIsNone(result.result)
                    self.assertEqual((result.validation_status, result.failure_type, result.parse_status),
                                     ("unparsed", "parse_failure", ParseStatus.FAILED))
                    self.assertEqual(result.parse_error, "Custom parsing failed.")
                    self.assertEqual(result.raw_text, "custom response")
                    predicate.assert_not_called()
                    self.assertFalse(backend.calls)

    def test_explicit_parse_failure_with_a_value_can_retry_and_recover(self):
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                strategy = ScriptedStrategy(
                    StrategyResult(7, "failed reply", parse_status=ParseStatus.FAILED, parse_error="Custom parsing failed."),
                    StrategyResult(5, "recovered reply", parse_status=ParseStatus.PARSED),
                )
                with VirtualRuntime(backend=ScriptedBackend("unused"), strategy=strategy, validation_mode=mode) as runtime:
                    result = runtime.execute(add_request(max_retries=1))
                self.assertEqual((result.result, result.retries), (5, 1))
                self.assertIsNone(result.failure_type)
                self.assertEqual(result.validation_status, "not_checked" if mode == "none" else "valid")
                self.assertEqual(result.parse_status, ParseStatus.PARSED)
                self.assertEqual(len(strategy.feedback), 2)
                self.assertIsNone(strategy.feedback[0])
                self.assertEqual(strategy.feedback[1], {"raw_output": "failed reply", "error": "Custom parsing failed."})

    def test_explicit_parse_failure_with_a_value_is_not_cached(self):
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                response = StrategyResult(7, "failed reply", parse_status=ParseStatus.FAILED)
                with patch("aicomputing.strategies.direct.DirectStrategy.execute", return_value=response) as execute:
                    with VirtualRuntime(backend=ScriptedBackend("unused"), cache_size=8, validation_mode=mode) as runtime:
                        first = runtime.execute(add_request(max_retries=0))
                        second = runtime.execute(add_request(max_retries=0))
                        self.assertEqual(len(runtime.cache), 0)
                self.assertEqual(execute.call_count, 2)
                self.assertFalse(first.cached or second.cached)
                self.assertEqual((first.failure_type, second.failure_type), ("parse_failure", "parse_failure"))

    def test_custom_values_with_unrecorded_parsing_status_remain_usable(self):
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                strategy = ScriptedStrategy(StrategyResult(7, "custom reply"))
                with VirtualRuntime(backend=ScriptedBackend("unused"), strategy=strategy, validation_mode=mode) as runtime:
                    result = runtime.execute(add_request(max_retries=0))
                self.assertEqual(result.result, 7)
                self.assertIsNone(result.failure_type)
                self.assertIsNone(result.parse_status)
                self.assertEqual(result.validation_status, "not_checked" if mode == "none" else "valid")

    def test_prompt_strategies_use_their_own_prompt(self):
        markers = {"direct": "only the final-answer block", "cot": "step by step",
                   "pseudocode": "Stage 1", "confidence": "Decide for yourself", "hybrid": "only the final-answer block"}
        for strategy, marker in markers.items():
            with self.subTest(strategy=strategy):
                backend = ScriptedBackend('{"result": 5}')
                with VirtualRuntime(backend=backend, strategy=strategy) as runtime:
                    result = runtime.execute(add_request())
                self.assertEqual((result.result, result.validation_status), (5, "valid"))
                self.assertIn(marker, backend.calls[0][0])
                self.assertEqual(result.model_calls, 1)
                self.assertEqual(result.answer_format, ANSWER_FORMAT)

    def test_validation_retry_sends_feedback_then_accepts(self):
        backend = ScriptedBackend('{"result": "five"}', '{"result": 5}')
        with VirtualRuntime(backend=backend) as runtime:
            result = runtime.execute(add_request())
        self.assertEqual((result.result, result.retries, result.model_calls), (5, 1, 2))
        self.assertIn("Validation error", backend.calls[1][0])

    def test_strict_mode_withholds_invalid_output(self):
        backend = ScriptedBackend('{"result": "five"}')
        with VirtualRuntime(backend=backend, validation_mode="strict") as runtime:
            result = runtime.execute(add_request(max_retries=0))
        self.assertEqual((result.result, result.validation_status), (None, "invalid_type"))

    def test_default_runtime_requests_fresh_predictions_without_cache_inspection(self):
        backend = ScriptedBackend('{"result": 1}', '{"result": 2}')
        with patch.object(backend, "cache_state", side_effect=AssertionError("Disabled cache inspected backend state")) as state:
            with VirtualRuntime(backend=backend) as runtime:
                first = runtime.execute(add_request())
                second = runtime.execute(add_request())
                self.assertEqual(runtime.cache.maxsize, 0)
                self.assertEqual(len(runtime.cache), 0)
            state.assert_not_called()
        self.assertEqual((first.result, second.result), (1, 2))
        self.assertFalse(first.cached or second.cached)
        self.assertEqual((first.model_calls, second.model_calls, len(backend.calls)), (1, 1, 2))

    def test_valid_predictions_are_cached(self):
        backend = ScriptedBackend('{"result": 5}')
        with VirtualRuntime(cache_size=8, backend=backend) as runtime:
            runtime.execute(add_request())
            hit = runtime.execute(add_request())
        self.assertTrue(hit.cached)
        self.assertEqual((hit.result, len(backend.calls)), (5, 1))

    def test_contract_accepts_or_revises_the_candidate(self):
        for check, value, revised in (('{"verdict": "accept", "result": 6}', 6, False),
                                      ('{"verdict": "revise", "result": 5}', 5, True)):
            with self.subTest(check=check):
                backend = ScriptedBackend('{"result": 6}', check)
                with VirtualRuntime(backend=backend, strategy="contract") as runtime:
                    result = runtime.execute(add_request())
                self.assertEqual((result.result, result.revised, result.pre_revision_value), (value, revised, 6))
                self.assertIn("checking a proposed result", backend.calls[1][0])

    def test_consistency_returns_the_majority_value(self):
        backend = ScriptedBackend('{"result": 4}', '{"result": 5}', '{"result": 5}')
        with VirtualRuntime(backend=backend, strategy="consistency") as runtime:
            result = runtime.execute(add_request(temperature=0.7))
        self.assertEqual((result.result, result.model_calls), (5, 5))
        self.assertTrue(result.raw_text.startswith("[consistency 4/5]"))
        self.assertEqual({options["temperature"] for _, options in backend.calls}, {0.7})


class ValidationPolicyTests(unittest.TestCase):
    def test_output_type_comes_only_from_the_signature(self):
        for annotation, expected, status in ((" -> int", None, "invalid_type"),
                                              (" -> str", "wrong", "valid"),
                                              ("", "wrong", "valid")):
            with self.subTest(annotation=annotation):
                backend = ScriptedBackend('{"result": "wrong"}')
                with VirtualRuntime(backend=backend) as runtime:
                    request = ExecuteRequest(signature="f(x: int)" + annotation,
                                             docstring="Return x.", inputs={"x": 2}, max_retries=0)
                    result = runtime.execute(request)
                self.assertEqual((result.result, result.validation_status), (expected, status))

    def test_unknown_signature_return_types_fail_before_generation_in_every_mode(self):
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                backend = ScriptedBackend('{"result": 5}')
                with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                    request = ExecuteRequest(signature="f(x: int) -> UnknownType",
                                             docstring="Return x.", inputs={"x": 2})
                    with self.assertRaises(ValueError):
                        runtime.execute(request)
                self.assertFalse(backend.calls)

    def test_contract_and_consistency_keep_numeric_strings_in_none_mode(self):
        for strategy in ("contract", "consistency"):
            with self.subTest(strategy=strategy):
                responses = ('{"result": "7"}', '{"verdict": "revise", "result": "8"}') if strategy == "contract" else ('{"result": "7"}',)
                backend = ScriptedBackend(*responses)
                with VirtualRuntime(backend=backend, strategy=strategy, validation_mode="none") as runtime:
                    result = runtime.execute(add_request(max_retries=0))
                self.assertEqual(result.result, "8" if strategy == "contract" else "7")
                self.assertIs(type(result.result), str)
                self.assertEqual(result.validation_status, "not_checked")
                if strategy == "contract":
                    self.assertEqual(result.pre_revision_value, "7")
                    self.assertIn("Unverified candidate return value (Python value): '7'", backend.calls[1][0])

    def test_cache_entries_follow_changed_signature_annotations(self):
        backend = ScriptedBackend('{"result": "7"}')
        with VirtualRuntime(cache_size=8, backend=backend) as runtime:
            request = add_request(max_retries=0)
            integer = runtime.execute(request)
            request.signature = "add(x: int, y: int) -> str"
            text = runtime.execute(request)
            request.signature = "add(x: int, y: int) -> int"
            cached = runtime.execute(request)
        self.assertEqual((integer.result, text.result, cached.result), (7, "7", 7))
        self.assertFalse(text.cached)
        self.assertTrue(cached.cached)
        self.assertEqual(len(backend.calls), 2)

    def test_strict_is_the_default_for_invalid_outputs(self):
        backend = ScriptedBackend('{"result": "wrong"}')
        with VirtualRuntime(backend=backend) as runtime:
            self.assertIs(runtime.validation_mode, ValidationMode.STRICT)
            result = runtime.execute(add_request(max_retries=0))
        self.assertEqual((result.result, result.validation_status), (None, "invalid_type"))
        self.assertTrue(result.type_warnings)

    def test_each_output_policy_applies_to_literals_and_recovered_text(self):
        for payload, value in (('{"result": "wrong"}', "wrong"), ('{"result": 7?}', "7?")):
            for mode, expected, status in (("strict", None, "invalid_type"),
                                           ("warn", value, "invalid_type"),
                                           ("none", value, "not_checked")):
                with self.subTest(payload=payload, mode=mode):
                    backend = ScriptedBackend(payload)
                    with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                        result = runtime.execute(add_request(max_retries=0))
                    self.assertEqual((result.result, result.validation_status), (expected, status))
                    self.assertEqual(bool(result.type_warnings), mode != "none")
                    self.assertEqual(result.failure_type, None if mode == "none" else "structural_validation_failure")

    def test_none_does_not_convert_a_parsed_numeric_string(self):
        backend = ScriptedBackend('{"result": "7"}')
        with VirtualRuntime(backend=backend, validation_mode="none") as runtime:
            result = runtime.execute(add_request())
        self.assertIs(type(result.result), str)
        self.assertEqual((result.result, result.validation_status, result.model_calls), ("7", "not_checked", 1))

    def test_input_errors_always_raise_before_generation(self):
        for mode in ("strict", "warn", "none"):
            for inputs in ({"x": "2", "y": 3}, {"x": 2}, {"x": 2, "y": 3, "extra": 1}, None):
                with self.subTest(mode=mode, inputs=inputs):
                    backend = ScriptedBackend('{"result": 5}')
                    with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                        req = add_request()
                        req.inputs = inputs
                        with self.assertRaises(AITypeError):
                            runtime.execute(req)
                    self.assertFalse(backend.calls)

    def test_malformed_signature_always_raises_before_generation(self):
        for mode in ("strict", "warn", "none"):
            for signature in ("not a function signature", "1f(x: int)", "f(x: int, x: int)", "f(x: int = 1, y: int)"):
                with self.subTest(mode=mode, signature=signature):
                    backend = ScriptedBackend('{"result": 5}')
                    with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                        req = add_request()
                        req.signature = signature
                        with self.assertRaises(ValueError):
                            runtime.execute(req)
                    self.assertFalse(backend.calls)

    def test_request_policy_overrides_the_runtime_default(self):
        for runtime_mode, request_mode, expected in (("warn", "strict", None),
                                                     ("strict", "warn", "wrong"),
                                                     ("strict", "none", "wrong")):
            with self.subTest(runtime_mode=runtime_mode, request_mode=request_mode):
                backend = ScriptedBackend('{"result": "wrong"}')
                with VirtualRuntime(backend=backend, validation_mode=runtime_mode) as runtime:
                    result = runtime.execute(add_request(validation_mode=request_mode, max_retries=0))
                self.assertEqual(result.result, expected)

    def test_none_skips_output_constraints_while_checked_modes_report_them(self):
        for mode, expected, status in (("strict", None, "invalid_constraint"),
                                       ("warn", -1, "invalid_constraint"),
                                       ("none", -1, "not_checked")):
            with self.subTest(mode=mode):
                predicate = unittest.mock.Mock(return_value=False)
                backend = ScriptedBackend('{"result": -1}')
                with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                    result = runtime.execute(add_request(constraints=[Constraint("positive", predicate)], max_retries=0))
                self.assertEqual((result.result, result.validation_status), (expected, status))
                self.assertEqual(predicate.call_count, 0 if mode == "none" else 1)

    def test_none_still_reports_parse_failure(self):
        backend = ScriptedBackend('No final-answer block.', raw=True)
        with VirtualRuntime(backend=backend, validation_mode="none") as runtime:
            result = runtime.execute(add_request(max_retries=0))
        self.assertEqual((result.result, result.validation_status, result.failure_type), (None, "unparsed", "parse_failure"))

    def test_none_can_return_a_successfully_parsed_none(self):
        backend = ScriptedBackend('{"result": null}')
        with VirtualRuntime(backend=backend, validation_mode="none") as runtime:
            result = runtime.execute(add_request(max_retries=0))
        self.assertEqual((result.result, result.validation_status, result.parse_status), (None, "not_checked", "parsed"))

    def test_cache_entries_do_not_cross_validation_policies(self):
        backend = ScriptedBackend('{"result": "7"}')
        with VirtualRuntime(cache_size=8, backend=backend) as runtime:
            unchecked = runtime.execute(add_request(validation_mode="none", max_retries=0))
            checked = runtime.execute(add_request(max_retries=0))
            cached = runtime.execute(add_request(validation_mode="none", max_retries=0))
        self.assertEqual((unchecked.result, checked.result, cached.result), ("7", 7, "7"))
        self.assertTrue(cached.cached)
        self.assertEqual(cached.validation_status, "not_checked")
        self.assertEqual(len(backend.calls), 2)

    def test_boolean_mode_aliases_and_removed_strict_types_are_rejected(self):
        for mode in (True, False, 1):
            with self.subTest(mode=mode):
                with patch("aicomputing.runtime.create_backend") as factory:
                    with self.assertRaises(TypeError):
                        VirtualRuntime(backend="ollama", validation_mode=mode)
                factory.assert_not_called()
                with self.assertRaises(TypeError):
                    add_request(validation_mode=mode)
        with self.assertRaises(TypeError):
            add_request(strict_types=True)
        with self.assertRaises(TypeError):
            VirtualRuntime(backend=ScriptedBackend('5'), strict_types=True)


class BackendSelectionTests(unittest.TestCase):
    def test_missing_or_invalid_backend_is_rejected_before_construction(self):
        with patch("aicomputing.runtime.create_backend") as factory:
            with self.assertRaisesRegex(TypeError, "backend"):
                VirtualRuntime()
            for backend in (None, False, 0, [], {}):
                with self.subTest(backend=backend), self.assertRaisesRegex(TypeError, "backend"):
                    VirtualRuntime(backend=backend)
            factory.assert_not_called()

    def test_unknown_backend_name_is_rejected_before_generation(self):
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            VirtualRuntime(backend="unknown")


class CliTests(unittest.TestCase):
    def run_cli(self, *argv, backend):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("aicomputing.runtime.create_backend", return_value=backend), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = cli_main(list(argv))
        return code, stdout.getvalue(), stderr.getvalue()

    def test_prints_the_prediction_as_json(self):
        code, out, _ = self.run_cli("add(x: int, y: int) -> int", "Return x + y.", "--backend", "ollama", "--inputs", '{"x": 2, "y": 3}',
                                    backend=ScriptedBackend('{"result": 5}'))
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out), 5)

    def test_values_json_cannot_preserve_print_as_python(self):
        _, out, _ = self.run_cli("f(x: int) -> set[int]", "Return {x}.", "--backend", "ollama", "--inputs", '{"x": 2}',
                                 backend=ScriptedBackend('{"result": {2}}'))
        self.assertEqual(out.strip(), "{2}")
        _, out, _ = self.run_cli("f(x: int) -> dict", "Map x.", "--backend", "ollama", "--inputs", '{"x": 2}',
                                 backend=ScriptedBackend('{"result": {2: "a"}}'))
        self.assertEqual(out.strip(), "{2: 'a'}")

    def test_invalid_inputs_json_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as raised:
            self.run_cli("f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", "{x: 2}", backend=ScriptedBackend("1"))
        self.assertEqual(raised.exception.code, 2)

    def test_invalid_generation_settings_are_usage_errors_before_backend_creation(self):
        for options in (("--temp", "nan"), ("--temp", "inf"), ("--temp", "-1"), ("--max-tokens", "0")):
            with self.subTest(options=options):
                error = io.StringIO()
                with patch("aicomputing.runtime.create_backend") as factory, contextlib.redirect_stderr(error):
                    with self.assertRaises(SystemExit) as raised:
                        cli_main(["f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", '{"x": 2}', *options])
                self.assertEqual(raised.exception.code, 2)
                self.assertNotIn("Traceback", error.getvalue())
                factory.assert_not_called()

    def test_blank_model_names_are_usage_errors_before_backend_creation(self):
        for model in ("", "  ", "\t\n"):
            with self.subTest(model=model):
                error = io.StringIO()
                with patch("aicomputing.runtime.create_backend") as factory, contextlib.redirect_stderr(error):
                    with self.assertRaises(SystemExit) as raised:
                        cli_main(["f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", '{"x": 2}',
                                  "--model", model])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("model must be a nonempty string", error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())
                factory.assert_not_called()

    def test_blank_connection_options_are_usage_errors_before_backend_creation(self):
        for backend in ("openai", "ollama", "hf"):
            for option in ("--base-url", "--api-key"):
                for value in ("", " \t "):
                    with self.subTest(backend=backend, option=option, value=value):
                        error = io.StringIO()
                        with patch("aicomputing.runtime.create_backend") as factory, contextlib.redirect_stderr(error):
                            with self.assertRaises(SystemExit) as raised:
                                cli_main(["f(x: int) -> int", "Return x.", "--backend", backend,
                                          "--inputs", '{"x": 2}', option, value])
                        self.assertEqual(raised.exception.code, 2)
                        self.assertIn("error:", error.getvalue())
                        self.assertNotIn("Traceback", error.getvalue())
                        factory.assert_not_called()

    def test_blank_trace_destinations_are_usage_errors_before_backend_creation(self):
        for path in ("", " \t "):
            with self.subTest(path=path):
                error = io.StringIO()
                with patch("aicomputing.runtime.create_backend") as factory, contextlib.redirect_stderr(error):
                    with self.assertRaises(SystemExit) as raised:
                        cli_main(["f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", '{"x": 2}',
                                  "--trace", path])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--trace must be a nonempty string", error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())
                factory.assert_not_called()

    def test_ollama_host_reaches_the_backend_with_its_supplied_path_and_trailing_slash(self):
        for host in ("https://literal.invalid/v1", "https://literal.invalid/custom/v1/",
                     "http://localhost:11434/proxy/", "localhost:11434"):
            with self.subTest(host=host):
                output = io.StringIO()
                with patch("aicomputing.runtime.create_backend", return_value=ScriptedBackend('{"result": 7}')) as factory:
                    with contextlib.redirect_stdout(output):
                        code = cli_main(["f(x: int) -> int", "Return seven.", "--backend", "ollama", "--inputs", '{"x": 2}',
                                         "--base-url", host])
                self.assertEqual((code, output.getvalue()), (0, "7\n"))
                self.assertEqual(factory.call_args.kwargs["host"], host)
                self.assertNotIn("base_url", factory.call_args.kwargs)

    def test_blank_native_reasoning_options_are_usage_errors_before_backend_creation(self):
        for backend, option in (("openai", "--reasoning-effort"), ("ollama", "--ollama-think")):
            for value in ("", " \t "):
                with self.subTest(backend=backend, option=option, value=value):
                    error = io.StringIO()
                    with patch("aicomputing.runtime.create_backend") as factory, contextlib.redirect_stderr(error):
                        with self.assertRaises(SystemExit) as raised:
                            cli_main(["f(x: int) -> int", "Return x.", "--backend", backend,
                                      "--inputs", '{"x": 2}', option, value])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn(option + " must be a nonempty string", error.getvalue())
                    self.assertNotIn("Traceback", error.getvalue())
                    factory.assert_not_called()

    def test_openai_connection_values_reach_the_backend_unchanged(self):
        base_url, api_key = "https://literal.invalid/Custom/v1/", "  offline-key  "
        output = io.StringIO()
        with patch("aicomputing.runtime.create_backend", return_value=ScriptedBackend('{"result": 7}')) as factory:
            with contextlib.redirect_stdout(output):
                code = cli_main(["f(x: int) -> int", "Return seven.", "--backend", "openai", "--inputs", '{"x": 2}',
                                 "--base-url", base_url, "--api-key", api_key])
        self.assertEqual((code, output.getvalue()), (0, "7\n"))
        self.assertEqual(factory.call_args.kwargs["base_url"], base_url)
        self.assertEqual(factory.call_args.kwargs["api_key"], api_key)

    def test_blank_descriptions_are_usage_errors_before_backend_creation(self):
        for description in ("", "  ", "\t\n"):
            with self.subTest(description=description):
                error = io.StringIO()
                with patch("aicomputing.runtime.create_backend") as factory, contextlib.redirect_stderr(error):
                    with self.assertRaises(SystemExit) as raised:
                        cli_main(["f(x: int) -> int", description, "--backend", "ollama", "--inputs", '{"x": 2}'])
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("docstring must be a nonempty string", error.getvalue())
                self.assertNotIn("Traceback", error.getvalue())
                factory.assert_not_called()

    def test_invalid_signatures_and_inputs_are_usage_errors_before_backend_creation(self):
        cases = (("not a signature", '{"x": 2}'), ("f(x: unknown) -> int", '{"x": 2}'),
                 ("f(x: int) -> int", "[]"), ("f(x: int) -> int", "{}"),
                 ("f(x: int) -> int", '{"x": "wrong"}'), ("f(x: int) -> int", '{"x": 2, "extra": 3}'))
        for mode in ("strict", "warn", "none"):
            for signature, inputs in cases:
                with self.subTest(mode=mode, signature=signature, inputs=inputs):
                    error = io.StringIO()
                    with patch("aicomputing.runtime.create_backend") as factory, contextlib.redirect_stderr(error):
                        with self.assertRaises(SystemExit) as raised:
                            cli_main([signature, "Return x.", "--backend", "ollama", "--inputs", inputs,
                                      "--validation-mode", mode])
                    self.assertEqual(raised.exception.code, 2)
                    self.assertIn("error:", error.getvalue())
                    self.assertNotIn("Traceback", error.getvalue())
                    factory.assert_not_called()

    def test_cli_rejects_the_removed_return_type_override(self):
        backend = ScriptedBackend('{"result": 5}')
        with self.assertRaises(SystemExit) as raised:
            self.run_cli("f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", '{"x": 2}',
                         "--return-type", "str", backend=backend)
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(backend.calls)

    def test_cli_exposes_explicit_output_validation_modes(self):
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                code, output, diagnostic = self.run_cli(
                    "f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", '{"x": 2}',
                    "--validation-mode", mode, "--verbose", backend=ScriptedBackend('{"result": "wrong"}'))
                self.assertEqual(code, 1 if mode == "strict" else 0)
                if mode == "strict":
                    self.assertEqual(output, "")
                    self.assertIn("aicomp: error:", diagnostic)
                else:
                    self.assertEqual(json.loads(output), "wrong")
                    self.assertEqual("aicomp: warning:" in diagnostic, mode == "warn")
                self.assertIn("not_checked" if mode == "none" else "invalid_type", diagnostic)

    def test_valid_none_is_successful_in_every_output_mode(self):
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                code, output, diagnostic = self.run_cli(
                    "f(x: int) -> None", "Return None.", "--backend", "ollama", "--inputs", '{"x": 2}',
                    "--validation-mode", mode, backend=ScriptedBackend('{"result": null}'))
                self.assertEqual((code, output, diagnostic), (0, "null\n", ""))

    def test_parse_failure_is_an_error_in_every_output_mode(self):
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                code, output, diagnostic = self.run_cli(
                    "f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", '{"x": 2}',
                    "--validation-mode", mode, backend=ScriptedBackend("Unframed\nresponse", raw=True))
                self.assertEqual((code, output), (1, ""))
                self.assertIn("unparsed", diagnostic)
                self.assertNotIn("Unframed", diagnostic)
                self.assertEqual(len(diagnostic.splitlines()), 1)

    def test_warn_mode_reports_rejected_types_without_verbose_output(self):
        code, output, diagnostic = self.run_cli(
            "f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", '{"x": 2}',
            "--validation-mode", "warn", backend=ScriptedBackend('{"result": "wrong"}'))
        self.assertEqual((code, json.loads(output)), (0, "wrong"))
        self.assertIn("aicomp: warning: invalid_type", diagnostic)
        self.assertEqual(len(diagnostic.splitlines()), 1)

    def test_generated_code_runtime_failure_reports_its_location_even_in_warn_mode(self):
        code, output, diagnostic = self.run_cli(
            "f(x: int) -> int", "Divide by zero.", "--backend", "ollama", "--inputs", '{"x": 2}',
            "--strategy", "code_execute", "--validation-mode", "warn",
            backend=ScriptedBackend("def f(x):\n    return x / 0", raw=True))
        self.assertEqual((code, output), (1, ""))
        self.assertIn("execution_failed", diagnostic)
        self.assertIn("ZeroDivisionError", diagnostic)
        self.assertIn("line 2", diagnostic)
        self.assertNotIn("Traceback", diagnostic)

    def test_worker_harness_failure_is_reported_without_a_traceback(self):
        details = dict(status="harness_error", value=None, error_type="OSError",
                       error_message="Worker launch failed", elapsed_seconds=0)
        with patch("aicomputing.strategies.code_execute.execute_code_with_retries", return_value=details):
            code, output, diagnostic = self.run_cli(
                "f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", '{"x": 2}',
                "--strategy", "code_execute", backend=ScriptedBackend("def f(x): return x", raw=True))
        self.assertEqual((code, output), (1, ""))
        self.assertIn("Python worker failed: OSError: Worker launch failed", diagnostic)
        self.assertNotIn("Traceback", diagnostic)

    def test_known_provider_failures_are_concise_and_still_close_the_backend(self):
        for exception in (BackendTimeoutError, BackendCommunicationError, ToolUseDetected):
            with self.subTest(exception=exception.__name__):
                backend = ScriptedBackend("unused")
                with patch.object(backend, "generate", side_effect=exception("First line\n" + "x" * 1000)), \
                        patch.object(backend, "close") as close:
                    code, output, diagnostic = self.run_cli(
                        "f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", '{"x": 2}', backend=backend)
                self.assertEqual((code, output), (1, ""))
                self.assertTrue(diagnostic.startswith("aicomp: error: First line "))
                self.assertEqual(len(diagnostic.splitlines()), 1)
                self.assertLessEqual(len(diagnostic.strip().removeprefix("aicomp: error: ")), 500)
                close.assert_called_once()

    def test_unexpected_programming_errors_are_not_suppressed(self):
        backend = ScriptedBackend("unused")
        with patch.object(backend, "generate", side_effect=AssertionError("Unexpected bug")):
            with self.assertRaisesRegex(AssertionError, "Unexpected bug"):
                self.run_cli("f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", '{"x": 2}', backend=backend)

    def test_provider_error_is_reported_when_backend_cleanup_also_fails(self):
        backend = ScriptedBackend("unused")
        with patch.object(backend, "generate", side_effect=BackendCommunicationError("Provider request failed")), \
                patch.object(backend, "close", side_effect=OSError("Cleanup failure")), \
                self.assertLogs("aicomputing", level="WARNING") as logs:
            code, output, diagnostic = self.run_cli(
                "f(x: int) -> int", "Return x.", "--backend", "ollama", "--inputs", '{"x": 2}', backend=backend)
        self.assertEqual((code, output), (1, ""))
        self.assertIn("aicomp: error: Provider request failed", diagnostic)
        self.assertNotIn("Traceback", diagnostic)
        self.assertIn("OSError", logs.output[0])

    def test_backend_is_required_and_invalid_names_are_rejected_before_runtime_creation(self):
        arguments = ["f(x: int) -> int", "Return x.", "--inputs", '{"x": 2}']
        for options in ([], ["--backend", "unknown"]):
            with self.subTest(options=options):
                error = io.StringIO()
                with patch("aicomputing.cli.VirtualRuntime") as factory, contextlib.redirect_stderr(error):
                    with self.assertRaises(SystemExit) as raised:
                        cli_main(arguments + options)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("--backend", error.getvalue())
                factory.assert_not_called()
