import pytest

from traceback_tutor.schema import BUG_TYPES, Example, read_jsonl, write_jsonl


def test_fixtures_cover_every_v0_bug_type(fixture_examples):
    v0 = {name for name, b in BUG_TYPES.items() if b.in_v0}
    assert {ex.bug_type for ex in fixture_examples} == v0


def test_fixtures_raise_the_expected_error(fixture_examples):
    for ex in fixture_examples:
        assert ex.error_type == BUG_TYPES[ex.bug_type].expected_error, ex.id


def test_fixture_lines_are_inside_the_code(fixture_examples):
    for ex in fixture_examples:
        n = len(ex.code.splitlines())
        assert 1 <= ex.bug_line <= n and 1 <= ex.error_line <= n, ex.id
        assert 'File "main.py"' in ex.traceback


def test_jsonl_round_trip(tmp_path, fixture_examples):
    path = tmp_path / "x.jsonl"
    assert write_jsonl(path, fixture_examples) == len(fixture_examples)
    assert list(read_jsonl(path)) == fixture_examples


def test_unknown_field_is_rejected():
    with pytest.raises(ValueError):
        Example.from_dict({"id": "x", "not_a_field": 1})
