"""Check caller declarations before inference without running output predicates."""

import json
import unittest
from unittest.mock import Mock

from aicomputing import Backend, Constraint, ExecuteRequest, VirtualRuntime
from aicomputing.answer_protocol import frame_answer
from aicomputing.backends.base import GenerateResponse
from aicomputing.validation import validate_constraints


class OfflineBackend(Backend):
    name = "offline"
    default_model = "offline"

    def __init__(self, value=5):
        self.value = value
        self.calls = []

    def generate(self, prompt, **options):
        self.calls.append(prompt)
        return GenerateResponse(text=frame_answer(json.dumps({"result": self.value})), model="offline",
                                input_tokens=2, output_tokens=3)


class CacheableBackend(OfflineBackend):
    def cache_state(self):
        return {"value": self.value}


def request(**settings):
    return ExecuteRequest(signature="f(x: int) -> int", docstring="Return x.", inputs={"x": 5}, **settings)


class ConstraintDefinitionTests(unittest.TestCase):
    def test_malformed_constraint_fields_are_rejected_at_construction(self):
        cases = [("name", value, TypeError) for value in (None, False, 1, [])]
        cases += [("name", value, ValueError) for value in ("", "  ", "\t\n")]
        cases += [("predicate", value, TypeError) for value in (None, False, 1, "check", [], {})]
        cases += [("message", value, TypeError) for value in (None, False, 1, [])]
        for field, value, error in cases:
            with self.subTest(field=field, value=value):
                settings = dict(name="positive", predicate=lambda value: value > 0)
                settings[field] = value
                with self.assertRaisesRegex(error, field):
                    Constraint(**settings)

    def test_definition_checks_never_run_predicates(self):
        predicate = Mock(side_effect=AssertionError("Predicate executed during declaration checking"))
        constraint = Constraint("positive", predicate, message="Must be positive")
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                request(constraints=[constraint], validation_mode=mode)
        predicate.assert_not_called()

    def test_boolean_predicate_results_are_returned_unchanged(self):
        for returned in (True, False):
            with self.subTest(returned=returned):
                predicate = Mock(return_value=returned)
                constraint = Constraint("positive", predicate)
                self.assertIs(constraint.check(5), returned)
                predicate.assert_called_once_with(5)

    def test_non_boolean_results_are_rejected_by_direct_and_standalone_checks(self):
        for returned in ("False", "", 1, 0, 1.0, 0.0, None, [], [1], {}, {"ok": False}):
            for check in ("direct", "strict", "permissive"):
                with self.subTest(returned=returned, check=check):
                    predicate = Mock(return_value=returned)
                    constraint = Constraint("positive", predicate, "Must be positive")
                    expected = f"Constraint 'positive' predicate must return bool, got {type(returned).__name__}"
                    with self.assertRaises(TypeError) as caught:
                        if check == "direct":
                            constraint.check(5)
                        else:
                            validate_constraints(5, [constraint], strict=check == "strict")
                    self.assertEqual(str(caught.exception), expected)
                    predicate.assert_called_once_with(5)

    def test_non_boolean_results_raise_without_retries_and_are_skipped_in_none_mode(self):
        for mode in ("strict", "warn", "none"):
            for returned in ("False", "", 1, 0, None, [], [1]):
                with self.subTest(mode=mode, returned=returned):
                    predicate = Mock(return_value=returned)
                    backend = OfflineBackend()
                    call = request(constraints=[Constraint("positive", predicate)], max_retries=3)
                    with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                        if mode == "none":
                            result = runtime.execute(call)
                            self.assertEqual((result.result, result.validation_status), (5, "not_checked"))
                            predicate.assert_not_called()
                        else:
                            with self.assertRaisesRegex(TypeError, "predicate must return bool"):
                                runtime.execute(call)
                            predicate.assert_called_once_with(5)
                    self.assertEqual(len(backend.calls), 1)

    def test_invalid_constraint_collections_and_entries_are_rejected_in_every_mode(self):
        cases = (None, False, "", {}, (), (Constraint("positive", lambda value: value > 0),),
                 [None], ["positive"], [lambda value: value > 0])
        for mode in ("strict", "warn", "none"):
            for constraints in cases:
                with self.subTest(mode=mode, constraints=constraints), self.assertRaisesRegex(TypeError, "constraints"):
                    request(constraints=constraints, validation_mode=mode)

    def test_mutated_definitions_fail_before_any_backend_call_in_every_mode(self):
        for mode in ("strict", "warn", "none"):
            for field, value, error in (("name", "", ValueError), ("predicate", False, TypeError),
                                        ("message", None, TypeError)):
                with self.subTest(mode=mode, field=field):
                    backend = OfflineBackend()
                    predicate = Mock(return_value=True)
                    constraint = Constraint("positive", predicate)
                    changed = request(constraints=[constraint])
                    setattr(constraint, field, value)
                    with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                        with self.assertRaisesRegex(error, field):
                            runtime.execute(changed)
                    self.assertFalse(backend.calls)
                    predicate.assert_not_called()

    def test_mutated_constraint_list_fails_before_any_backend_call(self):
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                backend = OfflineBackend()
                changed = request()
                changed.constraints.append(None)
                with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                    with self.assertRaisesRegex(TypeError, r"constraints\[0\]"):
                        runtime.execute(changed)
                self.assertFalse(backend.calls)

    def test_standalone_validation_checks_all_definitions_before_running_a_predicate(self):
        predicate = Mock(return_value=True)
        first = Constraint("first", predicate)
        invalid = Constraint("second", lambda value: True)
        invalid.predicate = False
        for constraints in ([first, None], [first, invalid]):
            with self.subTest(constraints=constraints), self.assertRaises(TypeError):
                validate_constraints(5, constraints)
        predicate.assert_not_called()

    def test_predicate_exceptions_propagate_unchanged_and_are_skipped_in_none_mode(self):
        for mode in ("strict", "warn", "none"):
            with self.subTest(mode=mode):
                error = RuntimeError("Predicate bug")
                predicate = Mock(side_effect=error)
                backend = OfflineBackend()
                with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                    call = request(constraints=[Constraint("positive", predicate)], max_retries=3)
                    if mode == "none":
                        result = runtime.execute(call)
                        self.assertEqual((result.result, result.validation_status), (5, "not_checked"))
                        predicate.assert_not_called()
                    else:
                        with self.assertRaises(RuntimeError) as caught:
                            runtime.execute(call)
                        self.assertIs(caught.exception, error)
                        predicate.assert_called_once_with(5)
                self.assertEqual(len(backend.calls), 1)

    def test_valid_constraint_failures_keep_existing_retries_and_output_policy(self):
        for mode, expected, status in (("strict", None, "invalid_constraint"),
                                       ("warn", -1, "invalid_constraint"), ("none", -1, "not_checked")):
            with self.subTest(mode=mode):
                predicate = Mock(return_value=False)
                backend = OfflineBackend(-1)
                with VirtualRuntime(backend=backend, validation_mode=mode) as runtime:
                    result = runtime.execute(request(
                        constraints=[Constraint("positive", predicate, "Must be positive")], max_retries=1))
                self.assertEqual((result.result, result.validation_status), (expected, status))
                self.assertEqual(len(backend.calls), 1 if mode == "none" else 2)
                self.assertEqual(predicate.call_count, 0 if mode == "none" else 2)
                self.assertEqual(result.retries, 0 if mode == "none" else 1)
                if mode != "none":
                    self.assertIn("Must be positive", backend.calls[-1])


class ConstraintCacheTests(unittest.TestCase):
    def test_none_mode_caches_skipped_constraints_and_reports_zero_new_model_work(self):
        for runtime_mode, request_mode in (("none", None), ("strict", "none"), ("warn", "none")):
            with self.subTest(runtime_mode=runtime_mode, request_mode=request_mode):
                predicate = Mock(side_effect=AssertionError("Skipped predicate was executed"))
                backend = CacheableBackend()
                call = request(constraints=[Constraint("positive", predicate)], validation_mode=request_mode)
                with VirtualRuntime(cache_size=8, backend=backend, validation_mode=runtime_mode) as runtime:
                    first = runtime.execute(call)
                    second = runtime.execute(call)
                self.assertFalse(first.cached)
                self.assertTrue(second.cached)
                self.assertEqual((first.result, second.result, second.validation_status), (5, 5, "not_checked"))
                self.assertEqual(first.total_tokens, 5)
                self.assertEqual((second.model_calls, second.input_tokens, second.output_tokens, second.total_tokens),
                                 (0, 0, 0, 0))
                self.assertEqual(second.raw_text, first.raw_text)
                self.assertEqual(len(backend.calls), 1)
                predicate.assert_not_called()

    def test_checked_constraints_still_bypass_cache_even_when_they_pass(self):
        for runtime_mode, request_mode in (("strict", None), ("warn", None), ("none", "strict"), ("none", "warn")):
            with self.subTest(runtime_mode=runtime_mode, request_mode=request_mode):
                predicate = Mock(return_value=True)
                backend = CacheableBackend()
                call = request(constraints=[Constraint("positive", predicate)], validation_mode=request_mode)
                with VirtualRuntime(cache_size=8, backend=backend, validation_mode=runtime_mode) as runtime:
                    first = runtime.execute(call)
                    second = runtime.execute(call)
                self.assertFalse(first.cached or second.cached)
                self.assertEqual(len(backend.calls), 2)
                self.assertEqual(predicate.call_count, 2)

    def test_none_mode_ignores_constraint_identity_when_reusing_a_prediction(self):
        first_predicate = Mock(return_value=True)
        second_predicate = Mock(return_value=False)
        backend = CacheableBackend()
        with VirtualRuntime(cache_size=8, backend=backend, validation_mode="none") as runtime:
            first = runtime.execute(request(constraints=[Constraint("first", first_predicate)]))
            second = runtime.execute(request(constraints=[Constraint("second", second_predicate)]))
            third = runtime.execute(request())
        self.assertEqual((first.result, second.result, third.result), (5, 5, 5))
        self.assertTrue(second.cached and third.cached)
        self.assertEqual(len(backend.calls), 1)
        first_predicate.assert_not_called()
        second_predicate.assert_not_called()

    def test_malformed_mutations_are_rejected_before_a_cache_hit(self):
        for field, value, error in (("name", "", ValueError), ("predicate", False, TypeError),
                                    ("message", None, TypeError), ("list_entry", None, TypeError)):
            with self.subTest(field=field):
                predicate = Mock(return_value=True)
                constraint = Constraint("positive", predicate)
                call = request(constraints=[constraint])
                backend = CacheableBackend()
                with VirtualRuntime(cache_size=8, backend=backend, validation_mode="none") as runtime:
                    runtime.execute(call)
                    self.assertEqual(len(runtime.cache), 1)
                    if field == "list_entry":
                        call.constraints.append(value)
                    else:
                        setattr(constraint, field, value)
                    with self.assertRaises(error):
                        runtime.execute(call)
                    self.assertEqual(runtime.cache.hits, 0)
                self.assertEqual(len(backend.calls), 1)
                predicate.assert_not_called()

    def test_disabled_cache_and_unknown_backend_state_still_make_fresh_calls(self):
        for backend, size in ((CacheableBackend(), 0), (OfflineBackend(), 2048)):
            with self.subTest(backend=type(backend).__name__, size=size):
                predicate = Mock(return_value=True)
                call = request(constraints=[Constraint("positive", predicate)])
                with VirtualRuntime(backend=backend, validation_mode="none", cache_size=size) as runtime:
                    first = runtime.execute(call)
                    second = runtime.execute(call)
                self.assertFalse(first.cached or second.cached)
                self.assertEqual(len(backend.calls), 2)
                predicate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
