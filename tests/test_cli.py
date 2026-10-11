from traceback_tutor import cli
from traceback_tutor.generate import Generation


def fake_generate(response):
    calls = []

    def generate_fn(messages):
        calls.append(messages)
        return Generation(response, n_tokens=10, seconds=0.1, finish_reason="stop")

    return generate_fn, calls


def test_program_without_errors(tmp_path, capsys):
    path = tmp_path / "ok.py"
    path.write_text("print(1 + 1)\n")
    generate_fn, calls = fake_generate("unused")
    assert cli.main([str(path)], generate_fn=generate_fn) == 0
    assert capsys.readouterr().out.strip() == "Your program ran without errors."
    assert calls == []


def test_crashing_program_is_explained(tmp_path, capsys, fixture_examples):
    ex = fixture_examples[0]  # misspelt_variable, crashes on line 5
    path = tmp_path / "buggy.py"
    path.write_text(ex.code)
    generate_fn, calls = fake_generate(
        "Line 5: `totl` was never created.\n\nHint: Check the spelling?"
    )
    assert cli.main([str(path)], generate_fn=generate_fn) == 1

    user = calls[0][1]["content"]
    assert 'File "main.py", line 5' in user and "NameError" in user
    assert str(tmp_path) not in user  # temp path normalised away
    out = capsys.readouterr().out
    assert "crashed with NameError" in out
    assert "Line 5:  return totl" in out
    assert "Hint: Check the spelling?" in out


def test_unparseable_response_is_printed_raw(tmp_path, capsys):
    path = tmp_path / "buggy.py"
    path.write_text("x = [1]\nprint(x[3])\n")
    generate_fn, _ = fake_generate("I think the index is too big.")
    assert cli.main([str(path)], generate_fn=generate_fn) == 1
    assert capsys.readouterr().out.rstrip().endswith("I think the index is too big.")


def test_endless_loop_times_out(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "TIMEOUT", 0.5)
    path = tmp_path / "loop.py"
    path.write_text("while True:\n    pass\n")
    generate_fn, calls = fake_generate("unused")
    assert cli.main([str(path)], generate_fn=generate_fn) == 2
    assert "didn't finish" in capsys.readouterr().out
    assert calls == []
