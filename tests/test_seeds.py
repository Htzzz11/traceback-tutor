import ast

from traceback_tutor.seeds import Seed, clean_code, fingerprint, stdlib_only


def test_clean_code_normalises_mbpp_formatting():
    raw = "def f(x): \r\n\treturn x  \r\n"
    assert clean_code(raw) == "def f(x):\n    return x"


def test_stdlib_only():
    assert stdlib_only(ast.parse("import re\nfrom collections import Counter"))
    assert not stdlib_only(ast.parse("import numpy as np"))
    assert not stdlib_only(ast.parse("from pandas import DataFrame"))


def test_fingerprint_matches_renamed_copies_only():
    a = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s"
    b = "def add_up(nums):\n    acc = 0\n    for n in nums:\n        acc += n\n    return acc"
    c = "def total(xs):\n    s = 1\n    for x in xs:\n        s *= x\n    return s"
    fa, fb, fc = (fingerprint(ast.parse(x)) for x in (a, b, c))
    assert fa == fb and fa != fc


def test_with_tests_appends_the_test_calls_after_a_blank_line():
    seed = Seed("t", "def f(x):\n    return x", ["assert f(1) == 1", "assert f(2) == 2"])
    assert seed.with_tests() == "def f(x):\n    return x\n\nassert f(1) == 1\nassert f(2) == 2\n"
    assert seed.program_lines == 2
