import json
from dataclasses import replace

import pytest

from traceback_tutor.prompt import SYSTEM_PROMPT, build_user_message, parse_response
from traceback_tutor.to_chat import export


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def split_examples(fixture_examples):
    splits = ["train", "train", "val", "test", "test_ood"]
    return [replace(ex, split=s) for ex, s in zip(fixture_examples, splits)]


def test_export_writes_mlx_chat_format(tmp_path, split_examples):
    counts = export(split_examples, tmp_path)
    assert counts == {"train.jsonl": 2, "valid.jsonl": 1}

    row = read_rows(tmp_path / "train.jsonl")[0]
    assert list(row) == ["messages"]
    system, user, assistant = row["messages"]
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant"]
    ex = split_examples[0]
    assert system["content"] == SYSTEM_PROMPT
    assert user["content"] == build_user_message(ex.code, ex.traceback)
    parsed = parse_response(assistant["content"])
    assert (parsed.line, parsed.explanation, parsed.hint) == (ex.bug_line, ex.explanation, ex.hint)


def test_test_splits_never_reach_training_files(tmp_path, split_examples):
    export(split_examples, tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["train.jsonl", "valid.jsonl"]
    written = {
        row["messages"][1]["content"]
        for name in ("train.jsonl", "valid.jsonl")
        for row in read_rows(tmp_path / name)
    }
    for ex in split_examples:
        user = build_user_message(ex.code, ex.traceback)
        assert (user in written) == (ex.split in ("train", "val")), ex.id


def test_export_refuses_unlabelled_examples(tmp_path, split_examples):
    unlabelled = replace(split_examples[0], explanation=None)
    with pytest.raises(ValueError, match="no teacher explanation"):
        export([unlabelled], tmp_path)
