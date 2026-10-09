import pytest

from traceback_tutor.eval.metrics import aggregate, bootstrap_ci, score
from traceback_tutor.prompt import format_response


def by_type(examples, bug_type):
    return next(ex for ex in examples if ex.bug_type == bug_type)


def target(ex):
    return format_response(ex.bug_line, ex.explanation, ex.hint)


def test_fixture_targets_score_as_good_answers(fixture_examples):
    for ex in fixture_examples:
        m = score(ex, target(ex))
        assert m["follows_format"] and m["line_correct"] and m["line_is_bug_line"], ex.id
        assert m["error_type_named"], ex.id
        assert not m["gives_fix"], ex.id
        assert m["n_words"] == len(target(ex).split())


def test_error_line_counts_as_correct_but_not_as_bug_line(fixture_examples):
    ex = by_type(fixture_examples, "off_by_one_index")  # error_line 4, bug_line 3
    m = score(ex, format_response(ex.error_line, ex.explanation, ex.hint))
    assert m["line_correct"] and not m["line_is_bug_line"]


def test_wrong_line(fixture_examples):
    ex = by_type(fixture_examples, "misspelt_variable")
    m = score(ex, format_response(2, ex.explanation, ex.hint))
    assert m["follows_format"] and not m["line_correct"] and not m["line_is_bug_line"]


def test_no_line_number(fixture_examples):
    ex = by_type(fixture_examples, "misspelt_variable")
    m = score(ex, "The variable totl is not defined.\n\nHint: Check the spelling?")
    assert not m["follows_format"] and not m["line_correct"]
    assert m["error_type_named"]


def test_line_number_only_in_the_body_does_not_count(fixture_examples):
    ex = by_type(fixture_examples, "misspelt_variable")
    m = score(ex, "On line 5, totl is not defined.\n\nHint: Is it spelt right?")
    assert not m["follows_format"] and not m["line_correct"]


def test_missing_hint_fails_format_but_keeps_the_line(fixture_examples):
    ex = by_type(fixture_examples, "misspelt_variable")
    m = score(ex, "Line 5: The name totl is not defined.")
    assert not m["follows_format"] and m["line_correct"]


def test_empty_response(fixture_examples):
    ex = fixture_examples[0]
    assert score(ex, "") == {
        "follows_format": False,
        "line_correct": False,
        "line_is_bug_line": False,
        "error_type_named": False,
        "gives_fix": False,
        "n_words": 0,
    }


def test_error_type_named_by_exception_name(fixture_examples):
    ex = by_type(fixture_examples, "off_by_one_index")
    assert score(ex, "Line 4: This raises an IndexError.\n\nHint: ?")["error_type_named"]
    assert not score(ex, "Line 4: Something is wrong here.\n\nHint: ?")["error_type_named"]


def test_full_corrected_program_is_a_fix(fixture_examples):
    ex = by_type(fixture_examples, "misspelt_variable")
    fixed = ex.code.replace("return totl", "return total")
    assert score(ex, f"Line 5: Typo.\n\nHere is the fixed code:\n```python\n{fixed}```")[
        "gives_fix"
    ]
    assert score(ex, f"Line 5: Typo. Use this:\n\n{fixed}")["gives_fix"]  # unfenced


def test_corrected_block_header_is_a_fix(fixture_examples):
    ex = by_type(fixture_examples, "missing_colon")
    assert score(ex, "Line 2: Missing colon.\n\n    if x > 0:\n\nHint: ?")["gives_fix"]


def test_quoting_the_buggy_line_is_not_a_fix(fixture_examples):
    ex = by_type(fixture_examples, "misspelt_variable")
    response = "Line 5: This line fails:\n```\n5 |     return totl\n```\n\nHint: Is it spelt right?"
    assert not score(ex, response)["gives_fix"]


def test_quoting_the_traceback_is_not_a_fix(fixture_examples):
    ex = by_type(fixture_examples, "misspelt_variable")
    response = f"Line 5: Python says:\n```\n{ex.traceback}```\n\nHint: Is it spelt right?"
    assert not score(ex, response)["gives_fix"]


@pytest.mark.parametrize(
    "text",
    [
        "Line 5: Change totl to total.\n\nHint: ?",
        "Line 5: change `totl` to `total`.\n\nHint: ?",
        "Line 5: Replace totl with total.\n\nHint: ?",
        "Line 5: It should be `total`.\n\nHint: ?",
        "Line 5: `totl` is not defined. Did you mean `total`?",
        # real base-model output on the off-by-one fixture, scored as no fix before
        "Line 4: To fix this, we should change the loop condition to `i < len(xs)`.\n\nHint: ?",
    ],
)
def test_fix_phrases(fixture_examples, text):
    ex = by_type(fixture_examples, "misspelt_variable")
    assert score(ex, text)["gives_fix"]


def test_hint_question_about_change_is_not_a_fix(fixture_examples):
    ex = by_type(fixture_examples, "misspelt_variable")
    text = "Line 5: totl is not defined.\n\nHint: What would you change it to?"
    assert not score(ex, text)["gives_fix"]


def rows(values, bug_type="misspelt_variable"):
    keys = ("follows_format", "line_correct", "line_is_bug_line", "error_type_named", "gives_fix")
    return [
        {"bug_type": bug_type, "metrics": {**dict.fromkeys(keys, v), "n_words": 10}} for v in values
    ]


def test_aggregate_ci_contains_mean_and_is_deterministic():
    data = rows([True, False, True, True, False, True, True, True])
    a, b = aggregate(data), aggregate(data)
    assert a == b
    m = a["metrics"]["line_correct"]
    assert m["mean"] == 0.75
    assert m["ci95"][0] <= m["mean"] <= m["ci95"][1]
    assert m["ci95"][0] < m["ci95"][1]
    assert a["metrics"]["n_words"] == {"mean": 10.0}
    assert a["n"] == 8


def test_aggregate_n_equals_one():
    a = aggregate(rows([True]))
    assert a["n"] == 1
    assert a["metrics"]["follows_format"] == {"mean": 1.0, "ci95": [1.0, 1.0]}


def test_aggregate_breaks_down_by_bug_type():
    a = aggregate(rows([True, True]) + rows([False], bug_type="missing_colon"))
    assert a["by_bug_type"]["misspelt_variable"]["n"] == 2
    assert a["by_bug_type"]["misspelt_variable"]["line_correct"] == 1.0
    assert a["by_bug_type"]["missing_colon"]["line_correct"] == 0.0
    assert a["metrics"]["line_correct"]["mean"] == pytest.approx(2 / 3)


def test_bootstrap_seed_changes_resamples():
    values = [1.0, 0.0] * 10
    assert bootstrap_ci(values, seed=0) == bootstrap_ci(values, seed=0)
    assert bootstrap_ci(values, n_resamples=200, seed=0) != bootstrap_ci(
        values, n_resamples=200, seed=1
    )
