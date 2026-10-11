import json

import pytest
from conftest import FIXTURES

from traceback_tutor.eval import compare, rescore, run_eval
from traceback_tutor.generate import Generation
from traceback_tutor.prompt import build_messages, format_response


def oracle(examples):
    """Fake model that answers every fixture with its hand-written target."""
    answers = {
        build_messages(ex.code, ex.traceback)[1]["content"]: format_response(
            ex.bug_line, ex.explanation, ex.hint
        )
        for ex in examples
    }
    calls = []

    def generate_fn(messages):
        calls.append(messages)
        return Generation(
            answers[messages[1]["content"]], n_tokens=20, seconds=0.5, finish_reason="stop"
        )

    return generate_fn, calls


def cli_args(tmp_path, *extra):
    base = ["--model", "m", "--data", str(FIXTURES), "--name", "t", "--runs-dir", str(tmp_path)]
    return [*base, *extra]  # later flags win, so extra can override --model


def test_run_eval_end_to_end(tmp_path, fixture_examples):
    generate_fn, calls = oracle(fixture_examples)
    argv = cli_args(tmp_path, "--model", "fake/model")
    result = run_eval.main(argv, generate_fn=generate_fn)

    out = tmp_path / "t"
    rows = [json.loads(line) for line in (out / "responses.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == [ex.id for ex in fixture_examples]
    assert set(rows[0]) == {
        "id",
        "bug_type",
        "bug_line",
        "error_line",
        "response",
        "metrics",
        "seconds",
        "n_tokens",
        "finish_reason",
    }
    assert all(r["metrics"]["follows_format"] for r in rows)

    saved = json.loads((out / "metrics.json").read_text())
    assert saved == result
    config = saved["config"]
    assert config["model"] == "fake/model" and config["adapter"] is None
    assert config["n"] == len(fixture_examples) == saved["n"] == len(calls)
    assert config["generation"] == {"sampler": "greedy", "temperature": 0.0, "max_tokens": 300}
    assert config["data_sha256"] and config["timestamp"] and "git_commit" in config
    assert saved["metrics"]["line_correct"]["mean"] == 1.0
    assert saved["metrics"]["gives_fix"]["mean"] == 0.0
    assert saved["speed"]["tokens_per_second"] == 40.0
    assert saved["speed"]["truncated_rate"] == 0.0
    assert set(saved["by_bug_type"]) == {ex.bug_type for ex in fixture_examples}


def test_limit_and_split(tmp_path, fixture_examples):
    assert len(run_eval.select_examples(FIXTURES, limit=2)) == 2
    assert run_eval.select_examples(FIXTURES, split="test") == []  # fixtures have no split
    generate_fn, _ = oracle(fixture_examples)
    argv = cli_args(tmp_path)
    with pytest.raises(SystemExit):
        run_eval.main([*argv, "--split", "test"], generate_fn=generate_fn)


def test_refuses_to_overwrite_a_run(tmp_path, fixture_examples):
    generate_fn, _ = oracle(fixture_examples)
    argv = cli_args(tmp_path, "--limit", "1")
    run_eval.main(argv, generate_fn=generate_fn)
    with pytest.raises(SystemExit):
        run_eval.main(argv, generate_fn=generate_fn)
    run_eval.main([*argv, "--overwrite"], generate_fn=generate_fn)


def test_rescore_reproduces_scores_and_records_commit(tmp_path, fixture_examples):
    generate_fn, _ = oracle(fixture_examples)
    first = run_eval.main(cli_args(tmp_path), generate_fn=generate_fn)
    again = rescore.rescore(tmp_path / "t")
    assert again["metrics"] == first["metrics"]
    assert "git_commit" in again["config"]["rescored"]
    assert again["config"]["data_sha256"] == first["config"]["data_sha256"]


def test_compare_pairs_examples(tmp_path, fixture_examples):
    good, _ = oracle(fixture_examples)

    def no_hint(messages):
        return Generation("Line 1: Something broke.", n_tokens=5, seconds=0.1, finish_reason="stop")

    run_eval.main([*cli_args(tmp_path), "--name", "bad"], generate_fn=no_hint)
    run_eval.main([*cli_args(tmp_path), "--name", "good"], generate_fn=good)
    result = compare.compare(tmp_path / "bad", tmp_path / "good")
    fmt = result["metrics"]["follows_format"]
    assert (fmt["baseline"], fmt["candidate"], fmt["n"]) == (0, 5, 5)
    assert fmt["diff"] == 1.0 and fmt["only_candidate"] == 5
    assert fmt["mcnemar_p"] == pytest.approx(2 / 32)
    # only the off-by-one fixture has its bug and crash on different lines
    assert result["metrics"]["line_is_bug_line_when_bug_differs_from_crash"]["n"] == 1


def test_compare_refuses_different_examples(tmp_path, fixture_examples):
    good, _ = oracle(fixture_examples)
    run_eval.main([*cli_args(tmp_path), "--name", "a"], generate_fn=good)
    run_eval.main([*cli_args(tmp_path), "--name", "b", "--limit", "2"], generate_fn=good)
    with pytest.raises(SystemExit):
        compare.compare(tmp_path / "a", tmp_path / "b")
