from dataclasses import replace
from types import SimpleNamespace

import pytest

from traceback_tutor import teacher
from traceback_tutor.prompt import SYSTEM_PROMPT, build_user_message, format_response
from traceback_tutor.schema import read_jsonl, write_jsonl
from traceback_tutor.teacher import Usage, build_request, explain, load_prefix, passes_filters


class FakeClient:
    """Stands in for openai.OpenAI; returns a canned response and records requests."""

    def __init__(self, text: str = "Line 1: Something broke.\n\nHint: Where?"):
        self.text = text
        self.requests = []
        self.responses = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        usage = SimpleNamespace(
            input_tokens=1500,
            input_tokens_details=SimpleNamespace(cached_tokens=1200, cache_write_tokens=0),
            output_tokens=120,
            output_tokens_details=SimpleNamespace(reasoning_tokens=40),
        )
        return SimpleNamespace(output_text=self.text, status="completed", usage=usage)


def by_type(examples, bug_type):
    return next(ex for ex in examples if ex.bug_type == bug_type)


# ---------------------------------------------------------------- request builder


def test_prefix_is_identical_across_examples_and_comes_first(fixture_examples):
    prefix = load_prefix()
    requests = [build_request(prefix, ex) for ex in fixture_examples]
    fixed = [r["input"][0] for r in requests]
    assert all(f == fixed[0] for f in fixed)
    text = fixed[0]["content"][0]["text"]
    assert text.startswith(SYSTEM_PROMPT)
    assert "# Style guide" in text
    assert fixed[0]["content"][0]["prompt_cache_breakpoint"] == {"mode": "explicit"}
    # nothing from the example being labelled leaks into the fixed part
    assert fixture_examples[1].code not in text


def test_variable_part_is_last_and_carries_the_known_labels(fixture_examples):
    prefix = load_prefix()
    ex = by_type(fixture_examples, "off_by_one_index")
    request = build_request(prefix, ex)
    last = request["input"][-1]
    assert last["role"] == "user"
    assert last["content"].startswith(build_user_message(ex.code, ex.traceback))
    assert f"line {ex.bug_line}" in last["content"] and f"line {ex.error_line}" in last["content"]
    assert ex.error_type in last["content"]
    assert request["model"] == "gpt-6.1-sol"
    assert request["reasoning"] == {"effort": "low"}
    assert request["max_output_tokens"] == 300


def test_explain_parses_the_response_into_the_example(fixture_examples):
    ex = replace(fixture_examples[0], explanation=None, hint=None, teacher_model=None)
    client = FakeClient("Line 5: `totl` was never created.\n\nHint: What name did you use?")
    result = explain(client, "PREFIX", ex)
    assert result.example.explanation == "`totl` was never created."
    assert result.example.hint == "What name did you use?"
    assert result.example.teacher_model == "gpt-6.1-sol"
    assert result.usage == Usage(input=1500, cached_input=1200, output=120, reasoning=40)
    assert len(client.requests) == 1


# ---------------------------------------------------------------- cost


def test_cost_uses_separate_prices_for_cached_and_uncached_input():
    assert Usage(input=1_000_000).cost() == pytest.approx(2.00)
    assert Usage(input=1_000_000, cached_input=1_000_000).cost() == pytest.approx(0.10)
    assert Usage(input=1_000_000, cache_write=1_000_000).cost() == pytest.approx(2.50)
    assert Usage(output=1_000_000, reasoning=500_000).cost() == pytest.approx(10.00)
    usage = Usage(input=1500, cached_input=1200, output=120)
    assert usage.cost() == pytest.approx((300 * 2.00 + 1200 * 0.10 + 120 * 10.00) / 1e6)


def test_usage_adds_up():
    total = Usage(1, 2, 3, 4, 5) + Usage(10, 20, 30, 40, 50)
    assert total == Usage(11, 22, 33, 44, 55)


# ---------------------------------------------------------------- filters


def test_filters_pass_all_hand_written_fixtures(fixture_examples):
    for ex in fixture_examples:
        response = format_response(ex.bug_line, ex.explanation, ex.hint)
        assert passes_filters(ex, response) == (True, "ok"), ex.id


def test_filters_accept_citing_the_error_line(fixture_examples):
    ex = by_type(fixture_examples, "off_by_one_index")
    response = format_response(ex.error_line, ex.explanation, ex.hint)
    assert passes_filters(ex, response) == (True, "ok")


BAD_RESPONSES = {
    "wrong_line": "Line 3: Python doesn't know `totl`.\n\nHint: Is it spelt the same?",
    "bad_format": "Python doesn't know `totl`.\n\nHint: Is it spelt the same?",
    "code_block": "Line 5: The name is wrong.\n\n```python\nreturn total\n```\n\nHint: See?",
    "corrected_code": "Line 5: The name is misspelt.\n\nHint: Did you mean `return total`?",
    "gives_fix": "Line 5: The name is misspelt. Change `totl` to `total`.\n\nHint: Same name?",
    "no_hint": "Line 5: Python doesn't know any variable called `totl`.",
    "hint_not_question": "Line 5: `totl` was never created.\n\nHint: Look at line 2.",
    "multiple_hints": "Line 5: `totl` doesn't exist.\n\nHint: Same name?\n\nHint: Line 2?",
    "too_long": "Line 5: " + "word " * 81 + "\n\nHint: Same name?",
}


@pytest.mark.parametrize("reason", BAD_RESPONSES)
def test_filters_reject_bad_responses(fixture_examples, reason):
    ex = by_type(fixture_examples, "misspelt_variable")
    assert passes_filters(ex, BAD_RESPONSES[reason]) == (False, reason)


@pytest.mark.parametrize(
    "response",
    [
        "Line 2: Text and a number can't be joined.\n\nHint: Have you tried `str(age)`?",
        "Line 2: The line is missing something. Add a colon at the end.\n\nHint: What else?",
        "Line 2: It should be `if x > 0:` here.\n\nHint: What is missing?",
        "Line 2: Text and numbers don't mix.\n\nHint: What about `+ str(age)` instead?",
    ],
)
def test_filters_reject_giving_the_fix(fixture_examples, response):
    ex = by_type(fixture_examples, "str_plus_number")
    if "colon" in response or "if x" in response:
        ex = by_type(fixture_examples, "missing_colon")
    ok, reason = passes_filters(ex, response)
    assert not ok and reason in ("corrected_code", "gives_fix")


# ---------------------------------------------------------------- CLI


def test_cli_skips_examples_already_done(tmp_path, monkeypatch, fixture_examples):
    client = FakeClient("Line 5: `totl` was never created.\n\nHint: Same name?")
    monkeypatch.setattr(teacher, "OpenAI", lambda **kwargs: client)
    in_path, out_path = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    write_jsonl(in_path, fixture_examples)

    teacher.main(["--in", str(in_path), "--out", str(out_path), "--limit", "2"])
    assert len(client.requests) == 2
    teacher.main(["--in", str(in_path), "--out", str(out_path)])
    assert len(client.requests) == 5  # only the 3 not yet done
    teacher.main(["--in", str(in_path), "--out", str(out_path)])
    assert len(client.requests) == 5

    # the canned answer only fits the misspelt_variable fixture; the rest are rejected
    kept = list(read_jsonl(out_path))
    assert [ex.bug_type for ex in kept] == ["misspelt_variable"]
    assert len((tmp_path / "out.log.jsonl").read_text().splitlines()) == 5
