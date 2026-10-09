from traceback_tutor.prompt import (
    build_messages,
    format_response,
    number_lines,
    parse_response,
)


def test_number_lines_pads_to_widest_number():
    code = "\n".join(f"x{i}" for i in range(10))
    lines = number_lines(code).splitlines()
    assert lines[0] == " 1 | x0" and lines[9] == "10 | x9"


def test_format_then_parse_round_trips():
    text = format_response(3, "The loop goes one step too far.", "What does range(4) give?")
    parsed = parse_response(text)
    assert parsed.line == 3
    assert parsed.explanation == "The loop goes one step too far."
    assert parsed.hint == "What does range(4) give?"


def test_parse_is_lenient_about_bad_output():
    parsed = parse_response("Something went wrong somewhere.")
    assert parsed.line is None and parsed.hint is None
    assert parsed.explanation == "Something went wrong somewhere."


def test_messages_contain_numbered_code_and_traceback(fixture_examples):
    ex = fixture_examples[0]
    system, user = build_messages(ex.code, ex.traceback)
    assert system["role"] == "system" and user["role"] == "user"
    assert "5 | " in user["content"] and "NameError" in user["content"]


def test_fixture_targets_cite_the_bug_line(fixture_examples):
    for ex in fixture_examples:
        target = format_response(ex.bug_line, ex.explanation, ex.hint)
        assert parse_response(target).line == ex.bug_line
