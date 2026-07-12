"""
Tests for the aicomputing package.

These tests cover signature parsing, type validation, output parsing,
and caching — all without requiring an LLM backend.
"""

import pytest
from typing import Any

# --- Signature parsing ---

from aicomputing.types import parse_signature, resolve_type_str, Image, Audio


class TestSignatureParsing:
    def test_simple(self):
        sig = parse_signature("add(x: int, y: int) -> int")
        assert sig.name == "add"
        assert len(sig.params) == 2
        assert sig.params[0].name == "x"
        assert sig.params[0].resolved_type is int
        assert sig.params[1].resolved_type is int
        assert sig.return_type is int

    def test_generic_list(self):
        sig = parse_signature("sort(arr: list[int]) -> list[int]")
        assert sig.params[0].annotation == "list[int]"
        assert sig.return_type == list[int]

    def test_generic_dict(self):
        sig = parse_signature("dijkstra(g: dict[str, list], s: str) -> dict[str, int]")
        assert sig.name == "dijkstra"
        assert sig.return_type == dict[str, int]

    def test_no_return_type(self):
        sig = parse_signature("greet(name: str)")
        assert sig.return_type is Any

    def test_no_type_annotations(self):
        sig = parse_signature("f(x, y)")
        assert sig.params[0].resolved_type is Any
        assert sig.params[1].resolved_type is Any

    def test_media_types(self):
        sig = parse_signature("describe(img: Image) -> str")
        assert sig.params[0].resolved_type is Image
        assert sig.return_type is str

    def test_float_return(self):
        sig = parse_signature("average(a: int, b: int) -> float")
        assert sig.return_type is float

    def test_invalid_signature(self):
        with pytest.raises(ValueError):
            parse_signature("not a signature at all")


class TestResolveType:
    def test_primitives(self):
        assert resolve_type_str("int") is int
        assert resolve_type_str("str") is str
        assert resolve_type_str("float") is float
        assert resolve_type_str("bool") is bool

    def test_generic(self):
        assert resolve_type_str("list[int]") == list[int]
        assert resolve_type_str("dict[str, int]") == dict[str, int]

    def test_media(self):
        assert resolve_type_str("Image") is Image
        assert resolve_type_str("Audio") is Audio

    def test_unknown(self):
        assert resolve_type_str("SomeUnknownType") is Any


# --- Validation ---

from aicomputing.validation import (
    check_value_type, validate_inputs, validate_output,
    validate_constraints, Constraint, AITypeError, AIConstraintError,
)


class TestTypeChecking:
    def test_int(self):
        check_value_type(42, int)

    def test_int_fail(self):
        with pytest.raises(AITypeError):
            check_value_type("hello", int)

    def test_float_widening(self):
        check_value_type(42, float)  # int -> float is OK

    def test_list_int(self):
        check_value_type([1, 2, 3], list[int])

    def test_list_int_fail(self):
        with pytest.raises(AITypeError):
            check_value_type([1, "two", 3], list[int])

    def test_dict(self):
        check_value_type({"a": 1, "b": 2}, dict[str, int])

    def test_dict_fail(self):
        with pytest.raises(AITypeError):
            check_value_type({"a": "one"}, dict[str, int])

    def test_none(self):
        check_value_type(None, type(None))

    def test_any(self):
        check_value_type("anything", Any)
        check_value_type(42, Any)
        check_value_type(None, Any)


class TestInputValidation:
    def test_valid(self):
        sig = parse_signature("add(x: int, y: int) -> int")
        warnings = validate_inputs(sig, {"x": 1, "y": 2}, strict=True)
        assert warnings == []

    def test_missing_param(self):
        sig = parse_signature("add(x: int, y: int) -> int")
        with pytest.raises(AITypeError, match="Missing"):
            validate_inputs(sig, {"x": 1}, strict=True)

    def test_wrong_type(self):
        sig = parse_signature("add(x: int, y: int) -> int")
        with pytest.raises(AITypeError):
            validate_inputs(sig, {"x": 1, "y": "two"}, strict=True)

    def test_non_strict(self):
        sig = parse_signature("add(x: int, y: int) -> int")
        warnings = validate_inputs(sig, {"x": 1, "y": "two"}, strict=False)
        assert len(warnings) > 0


class TestOutputValidation:
    def test_valid(self):
        warnings = validate_output(42, int, strict=True)
        assert warnings == []

    def test_invalid_strict(self):
        with pytest.raises(AITypeError):
            validate_output("hello", int, strict=True)

    def test_invalid_non_strict(self):
        warnings = validate_output("hello", int, strict=False)
        assert len(warnings) > 0


class TestConstraints:
    def test_pass(self):
        c = Constraint(name="positive", predicate=lambda x: x > 0)
        failures = validate_constraints(5, [c])
        assert failures == []

    def test_fail_strict(self):
        c = Constraint(name="positive", predicate=lambda x: x > 0, message="must be positive")
        with pytest.raises(AIConstraintError, match="must be positive"):
            validate_constraints(-1, [c], strict=True)

    def test_fail_non_strict(self):
        c = Constraint(name="positive", predicate=lambda x: x > 0, message="must be positive")
        failures = validate_constraints(-1, [c], strict=False)
        assert "must be positive" in failures[0]


# --- Parsing ---

from aicomputing.parsing import parse_result, extract_json_result, coerce_type, _MISSING


class TestParsing:
    def test_json_result(self):
        assert extract_json_result('{"result": 42}') == 42

    def test_json_with_noise(self):
        text = 'Here is the answer: {"result": [1, 2, 3]} done.'
        assert extract_json_result(text) == [1, 2, 3]

    def test_json_dict_result(self):
        text = '{"result": {"A": 0, "B": 4}}'
        assert extract_json_result(text) == {"A": 0, "B": 4}

    def test_bare_int(self):
        assert parse_result("42") == 42

    def test_bare_float(self):
        assert parse_result("3.14") == 3.14

    def test_bare_string(self):
        assert parse_result('"hello"') == "hello"

    def test_bare_bool(self):
        assert parse_result("True") is True
        assert parse_result("false") is False

    def test_bare_none(self):
        assert parse_result("None") is None
        assert parse_result("null") is None

    def test_coerce_int_to_float(self):
        assert coerce_type(42, float) == 42.0

    def test_coerce_list(self):
        result = coerce_type([1, 2, 3], list[int])
        assert result == [1, 2, 3]

    def test_parse_with_return_type(self):
        result = parse_result('{"result": "42"}', return_type=int)
        assert result == 42

    # --- Fix: fallback to non-"result" JSON keys ---

    def test_json_fallback_key(self):
        """Model used 'groups' instead of 'result'."""
        text = '{"groups": ["()", "(())", "(()())"]}'
        assert extract_json_result(text) == ["()", "(())", "(()())"]

    def test_json_fallback_answer_key(self):
        text = '{"answer": 42}'
        assert extract_json_result(text) == 42

    def test_json_result_preferred_over_fallback(self):
        """'result' key takes priority over other keys."""
        text = '{"result": 5, "explanation": "2+3"}'
        assert extract_json_result(text) == 5

    # --- Fix: list → tuple coercion ---

    def test_coerce_list_to_tuple(self):
        from typing import Tuple
        result = coerce_type([0, 1], Tuple[int, int])
        assert result == (0, 1)
        assert isinstance(result, tuple)

    def test_coerce_list_to_plain_tuple(self):
        result = coerce_type([0, 1], tuple)
        assert result == (0, 1)
        assert isinstance(result, tuple)

    def test_coerce_bool_case_insensitive(self):
        assert coerce_type("true", bool) is True
        assert coerce_type("True", bool) is True
        assert coerce_type("false", bool) is False
        assert coerce_type("FALSE", bool) is False

    def test_parse_json_true_false(self):
        """JSON uses lowercase true/false."""
        assert parse_result('{"result": true}') is True
        assert parse_result('{"result": false}') is False


# --- Cache ---

from aicomputing.cache import ResultCache


class TestCache:
    def test_put_get(self):
        cache = ResultCache()
        cache.put("sig", "doc", {"x": 1}, "model", "direct", "result_value")
        assert cache.get("sig", "doc", {"x": 1}, "model", "direct") == "result_value"

    def test_miss(self):
        cache = ResultCache()
        assert cache.get("sig", "doc", {"x": 1}, "model", "direct") is None

    def test_lru_eviction(self):
        cache = ResultCache(maxsize=2)
        cache.put("a", "", {}, "m", "s", "val_a")
        cache.put("b", "", {}, "m", "s", "val_b")
        cache.put("c", "", {}, "m", "s", "val_c")  # evicts "a"
        assert cache.get("a", "", {}, "m", "s") is None
        assert cache.get("b", "", {}, "m", "s") == "val_b"

    def test_hit_miss_counters(self):
        cache = ResultCache()
        cache.put("sig", "doc", {}, "m", "s", "val")
        cache.get("sig", "doc", {}, "m", "s")  # hit
        cache.get("other", "doc", {}, "m", "s")  # miss
        assert cache.hits == 1
        assert cache.misses == 1


# --- Regression tests for fixed bugs ---

class TestGenericParamParsing:
    """Params with commas inside brackets used to produce phantom params."""

    def test_dict_param(self):
        sig = parse_signature("dijkstra(g: dict[str, list], s: str) -> dict[str, int]")
        assert [p.name for p in sig.params] == ["g", "s"]
        assert sig.params[0].annotation == "dict[str, list]"
        assert sig.params[1].resolved_type is str

    def test_nested_generic_param(self):
        sig = parse_signature("f(pairs: list[tuple[int, int]], k: int) -> int")
        assert [p.name for p in sig.params] == ["pairs", "k"]
        assert sig.params[0].annotation == "list[tuple[int, int]]"

    def test_strict_validation_with_generic_params(self):
        sig = parse_signature("dijkstra(g: dict[str, list], s: str) -> dict[str, int]")
        # Previously raised "Missing required input: 'list'"
        warnings = validate_inputs(sig, {"g": {"A": []}, "s": "A"}, strict=True)
        assert warnings == []

    def test_default_value(self):
        sig = parse_signature("f(x: int, y: int = 3) -> int")
        assert sig.params[1].default == "3"
        # y has a default, so it may be omitted
        assert validate_inputs(sig, {"x": 1}, strict=True) == []


class TestRetryFeedback:
    """Validation retries must change the prompt via build_retry_prompt."""

    def test_direct_strategy_uses_feedback(self):
        from aicomputing.strategies.direct import DirectStrategy
        from aicomputing.backends.base import Backend, GenerateResponse
        from aicomputing.request import ExecuteRequest

        prompts = []

        class FakeBackend(Backend):
            @property
            def name(self): return "fake"
            @property
            def default_model(self): return "fake-model"
            def generate(self, prompt, **kw):
                prompts.append(prompt)
                return GenerateResponse(text='{"result": 5}', model="fake-model",
                                        finish_reason="stop")

        req = ExecuteRequest(signature="add(x: int, y: int) -> int",
                             docstring="sum", inputs={"x": 2, "y": 3})
        s = DirectStrategy()
        s.execute(req, FakeBackend())
        s.execute(req, FakeBackend(), feedback={"raw_output": "oops", "error": "bad type"})
        assert "Validation error: bad type" in prompts[1]
        assert prompts[0] != prompts[1]


class TestRuntimeCaching:
    def test_failed_results_not_cached(self):
        from aicomputing.runtime import VirtualRuntime
        from aicomputing.request import ExecuteRequest
        from aicomputing.backends.base import Backend, GenerateResponse

        calls = {"n": 0}

        class FlakyBackend(Backend):
            @property
            def name(self): return "fake"
            @property
            def default_model(self): return "fake-model"
            def generate(self, prompt, **kw):
                calls["n"] += 1
                # First call: garbage that fails type validation; then valid.
                text = '{"result": "not a number"}' if calls["n"] == 1 else '{"result": 7}'
                return GenerateResponse(text=text, model="fake-model", finish_reason="stop")

        vr = VirtualRuntime(backend=FlakyBackend(), model="fake-model")
        req = ExecuteRequest(signature="f(x: int) -> int", docstring="d",
                             inputs={"x": 1}, return_type=int, max_retries=0)
        r1 = vr.execute(req)
        assert r1.type_warnings  # first result is invalid
        r2 = vr.execute(req)
        assert r2.result == 7
        assert r2.cached is False  # the bad result must not have been cached

    def test_cached_result_is_a_copy(self):
        from aicomputing.runtime import VirtualRuntime
        from aicomputing.request import ExecuteRequest
        from aicomputing.backends.base import Backend, GenerateResponse

        class OkBackend(Backend):
            @property
            def name(self): return "fake"
            @property
            def default_model(self): return "fake-model"
            def generate(self, prompt, **kw):
                return GenerateResponse(text='{"result": 1}', model="fake-model",
                                        finish_reason="stop")

        vr = VirtualRuntime(backend=OkBackend(), model="fake-model")
        req = ExecuteRequest(signature="f(x: int) -> int", docstring="d",
                             inputs={"x": 1}, return_type=int)
        vr.execute(req)
        hit = vr.execute(req)
        assert hit.cached is True
        hit.result = "mutated"
        hit2 = vr.execute(req)
        assert hit2.result == 1  # cache entry untouched


class TestRuntimeStrictDefault:
    def test_runtime_strict_false_applies(self):
        """VirtualRuntime(strict_types=False) used to be silently ignored."""
        from aicomputing.runtime import VirtualRuntime
        from aicomputing.request import ExecuteRequest
        from aicomputing.backends.base import Backend, GenerateResponse

        class OkBackend(Backend):
            @property
            def name(self): return "fake"
            @property
            def default_model(self): return "fake-model"
            def generate(self, prompt, **kw):
                return GenerateResponse(text='{"result": 1}', model="fake-model",
                                        finish_reason="stop")

        vr = VirtualRuntime(backend=OkBackend(), model="fake-model", strict_types=False)
        # Wrong input type — must NOT raise because runtime strict is off
        req = ExecuteRequest(signature="f(x: int) -> int", docstring="d",
                             inputs={"x": "not-an-int"}, return_type=int)
        r = vr.execute(req)
        assert r.result == 1


class TestHybridClassifyTask:
    """classify_task() must match whole words only — see hybrid.py docstring."""

    def _classify(self, signature, docstring, inputs=None):
        from aicomputing.strategies.hybrid import HybridStrategy
        from aicomputing.request import ExecuteRequest

        req = ExecuteRequest(signature=signature, docstring=docstring, inputs=inputs or {})
        return HybridStrategy().classify_task(req)

    def test_nlp_keyword_routes_direct(self):
        assert self._classify(
            "score_review(text: str) -> int", "Rate the review sentiment.",
        ) == "direct"

    def test_algo_keyword_routes_cot(self):
        assert self._classify(
            "sort(arr: list[int]) -> list[int]", "Return a sorted copy of the array.",
        ) == "cot"

    def test_math_keyword_routes_cot(self):
        assert self._classify(
            "solve(a: float, b: float) -> float", "Solve the equation for x.",
        ) == "cot"

    def test_default_routes_direct(self):
        assert self._classify("add(x: int, y: int) -> int", "Return the sum.") == "direct"

    def test_snake_case_signature_keyword_matches(self):
        """'binary_search' must be recognised via the signature name alone,
        with no keyword hits in the docstring."""
        assert self._classify(
            "binary_search(arr: list, target: int) -> int",
            "Return the index of target in arr.",
        ) == "cot"

    def test_path_does_not_match_inside_empathy(self):
        """Regression: naive substring matching used to route this to 'cot'
        because "path" is a substring of "empathy"."""
        assert self._classify(
            "measure(x: str) -> str", "Return the empathy level of the given text.",
        ) == "direct"

    def test_tag_does_not_match_inside_postage(self):
        """Regression: naive substring matching used to route this to
        'direct' because "tag" is a substring of "postage", silently
        overriding the correct algorithmic ('sort') signal."""
        assert self._classify(
            "sort_items(arr: list) -> list", "Sort items by postage class.",
        ) == "cot"


class TestConfidenceStrategy:
    """The model decides per-call whether to answer directly or trace first;
    the strategy must accept and parse both output shapes unchanged."""

    def test_prompt_shows_both_output_shapes(self):
        from aicomputing.prompting import build_confidence_prompt

        prompt = build_confidence_prompt(
            signature="add(a: int, b: int) -> int",
            docstring="Return the sum.",
            inputs={"a": 2, "b": 3},
        )
        assert '{"result": 5}' in prompt   # answer-only example
        assert "State: {a: 8, b: 4}" in prompt  # traced example

    def test_accepts_immediate_answer(self):
        from aicomputing.strategies.confidence import ConfidenceStrategy
        from aicomputing.backends.base import Backend, GenerateResponse
        from aicomputing.request import ExecuteRequest

        class FakeBackend(Backend):
            @property
            def name(self): return "fake"
            @property
            def default_model(self): return "fake-model"
            def generate(self, prompt, **kw):
                return GenerateResponse(text='{"result": 5}', model="fake-model",
                                        finish_reason="stop")

        req = ExecuteRequest(signature="add(x: int, y: int) -> int",
                             docstring="sum", inputs={"x": 2, "y": 3})
        result = ConfidenceStrategy().execute(req, FakeBackend())
        assert result.value == 5

    def test_accepts_traced_answer(self):
        from aicomputing.strategies.confidence import ConfidenceStrategy
        from aicomputing.backends.base import Backend, GenerateResponse
        from aicomputing.request import ExecuteRequest

        traced_text = (
            "Sequential: enter gcd(12, 8)\n"
            "  State: {a: 12, b: 8}\n"
            'Branch: b == 0 -> False\n'
            '{"result": 4}'
        )

        class FakeBackend(Backend):
            @property
            def name(self): return "fake"
            @property
            def default_model(self): return "fake-model"
            def generate(self, prompt, **kw):
                return GenerateResponse(text=traced_text, model="fake-model",
                                        finish_reason="stop")

        req = ExecuteRequest(signature="gcd(a: int, b: int) -> int",
                             docstring="gcd", inputs={"a": 12, "b": 8})
        result = ConfidenceStrategy().execute(req, FakeBackend())
        assert result.value == 4

    def test_registered_in_factory(self):
        from aicomputing.strategies import create_strategy
        from aicomputing.strategies.confidence import ConfidenceStrategy

        assert isinstance(create_strategy("confidence"), ConfidenceStrategy)


class TestTokenBudgetRetry:
    def test_retries_at_cap_before_giving_up(self):
        from aicomputing.strategies.base import generate_with_retry, _MAX_TOKEN_CAP
        from aicomputing.backends.base import GenerateResponse

        budgets = []

        def call(budget):
            budgets.append(budget)
            return GenerateResponse(text="", model="m", finish_reason="length")

        value, resp, retries = generate_with_retry(
            call=call, parse=lambda t: t, initial_budget=8192, model_name="m")
        assert value is None
        # 8192 -> 16384 -> 32768 (cap) must all actually be attempted
        assert budgets == [8192, 16384, _MAX_TOKEN_CAP]
