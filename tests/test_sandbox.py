import sys

import pytest

from traceback_tutor.sandbox import STDLIB_DIR, parse_error_type, run_sandboxed

on_macos = pytest.mark.skipif(sys.platform != "darwin", reason="sandbox-exec is macOS only")


def test_clean_run_reports_no_crash():
    r = run_sandboxed("print('hi')\n")
    assert not r.crashed and not r.timed_out and not r.out_of_memory
    assert r.stdout == "hi\n" and r.traceback == "" and r.error_type is None


def test_runtime_error_gives_normalised_traceback_and_innermost_line():
    code = "def f(x):\n    return totl\n\nf(1)\n"
    r = run_sandboxed(code)
    assert r.crashed and r.error_type == "NameError"
    assert r.error_line == 2  # inside f, not the call on line 4
    assert 'File "main.py", line 4' in r.traceback and 'File "main.py", line 2' in r.traceback
    assert "tmp" not in r.traceback and "/" not in r.traceback.split("\n")[1]


def test_syntax_error_is_reported_with_its_line():
    r = run_sandboxed("x = 1\nif x > 0\n    pass\n")
    assert r.crashed and r.error_type == "SyntaxError" and r.error_line == 2
    assert r.traceback.strip().endswith("SyntaxError: expected ':'")


def test_infinite_loop_times_out():
    r = run_sandboxed("while True:\n    pass\n", timeout=1)
    assert r.timed_out and not r.crashed


def test_memory_hog_is_killed():
    code = "xs = []\nwhile True:\n    xs.append(list(range(10000)))\n"
    r = run_sandboxed(code, timeout=20, memory_mb=100)
    assert r.out_of_memory and not r.timed_out and not r.crashed


def test_stdin_is_closed():
    r = run_sandboxed("input()\n")
    assert r.error_type == "EOFError"


def test_stdlib_paths_are_normalised():
    r = run_sandboxed('import re\nre.compile("(")\n')
    assert r.crashed and r.error_line == 2
    assert STDLIB_DIR not in r.traceback and "/usr/lib/python3.12/re/" in r.traceback


def test_error_type_parsing():
    assert parse_error_type("Traceback ...\nValueError: bad\n") == "ValueError"
    assert parse_error_type("...\njson.decoder.JSONDecodeError: x") == "JSONDecodeError"
    assert parse_error_type("...\nAssertionError\n") == "AssertionError"
    assert parse_error_type("") is None


@on_macos
def test_network_is_blocked():
    code = "import socket\nsocket.create_connection(('1.1.1.1', 80), timeout=2)\n"
    r = run_sandboxed(code)
    assert r.crashed and r.error_type == "PermissionError"


@on_macos
def test_writes_outside_the_temp_dir_are_blocked(tmp_path):
    target = tmp_path / "escaped.txt"
    r = run_sandboxed(f"open({str(target)!r}, 'w').write('x')\n")
    assert r.crashed and r.error_type == "PermissionError"
    assert not target.exists()
