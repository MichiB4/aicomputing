"""Published-data inspection accepts literals without running source programs."""
import pytest

from evaluation.benchmarks import mbpp_published as published


def program(inputs="[[1], [2]]", outputs="[2, 3]"):
    return ("import unavailable_module\n"
            "def unused(): raise RuntimeError('must never run')\n"
            f"inputs = {inputs}\nresults = {outputs}\n"
            "for i, (inp, exp) in enumerate(zip(inputs, results)):\n"
            "    assertion(increment(*inp), exp, 0)\n")


def test_published_literals_are_read_without_executing_imports_or_functions():
    result = published.extract_pairs(program(), "increment", 1)
    assert result.pairs == [(0, [1], 2), (1, [2], 3)]
    assert result.excluded == {}


def test_nonliteral_targets_are_excluded_without_execution_or_renumbering():
    result = published.extract_pairs(program(outputs="[side_effect(), 3]"), "increment", 1)
    assert result.pairs == [(1, [2], 3)]
    assert set(result.excluded) == {0}


def test_unreviewed_module_statements_and_missing_targets_are_rejected():
    with pytest.raises(ValueError, match="layout"):
        published.extract_pairs("side_effect()\n" + program(), "increment", 1)
    with pytest.raises(published.NoPublishedTargets):
        published.extract_pairs("inputs = [[1]]", "increment", 1)


def test_scoring_profile_requires_a_reviewed_loop():
    helpers = published._FLOAT_HELPER + published._ASSERTION
    with pytest.raises(ValueError, match="scoring loop"):
        published.scoring_profile(helpers)
    with pytest.raises(ValueError, match="scoring loop"):
        published.scoring_profile(helpers + "\nside_effect()\n")
    source = helpers + "\nfor i, (inp, exp) in enumerate(zip(inputs, results)):\n    assertion(increment(*inp), exp, 0)\n"
    assert published.scoring_profile(source) == "mbpp_exact_or_float_v1"
