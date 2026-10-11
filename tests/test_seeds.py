import ast

from traceback_tutor.seeds import (
    Seed,
    clean_code,
    fingerprint,
    near_duplicate_groups,
    shape,
    stdlib_only,
    uses_semicolons,
)


def test_clean_code_normalises_mbpp_formatting():
    raw = "def f(x): \r\n\treturn x  \r\n"
    assert clean_code(raw) == "def f(x):\n    return x"


def test_uses_semicolons_ignores_semicolons_in_strings():
    assert uses_semicolons("a = 1; b = 2")
    assert uses_semicolons("count = 0;")
    assert not uses_semicolons("s = 'a;b'\nt = s.split(';')")


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


# Near-copies found in MBPP by the leakage audit (mbpp-10/232, 768/855, 208/759, 461/818).
SMALLEST = "import heapq\ndef small(xs, n):\n    return heapq.nsmallest(n, xs)"
LARGEST = "import heapq\ndef large(ys, k):\n    return heapq.nlargest(k, ys)"
IS_ODD = "def odd(x):\n    return x % 2 == 1"
IS_EVEN = "def even(x):\n    return x % 2 != 0"
IMPORT_TOP = "import re\ndef f(s):\n    return re.search(r'[0-9]+', s)"
IMPORT_INSIDE = "def g(t):\n    import re\n    return re.search(r'[a-z]', t)"
UPPER = (
    "def upper(s):\n    n = 0\n    for i in range(len(s)):\n"
    "        if s[i] >= 'A' and s[i] <= 'Z': n += 1\n        return n"
)
LOWER = (
    "def lower(s):\n    n = 0\n    for i in range(len(s)):\n"
    "        if s[i] >= 'a' and s[i] <= 'z': n += 1\n    return n"
)
PAIRS_SUM = (  # mbpp-527 vs 296: same nested pair loop, similarity only 0.91
    "def pairs_sum(arr, n, s):\n    count = 0\n    for i in range(0, n):\n"
    "        for j in range(i + 1, n):\n            if arr[i] + arr[j] == s:\n"
    "                count += 1\n    return count"
)
INVERSIONS = (
    "def inversions(arr, n):\n    c = 0\n    for i in range(n):\n"
    "        for j in range(i + 1, n):\n            if (arr[i] > arr[j]):\n"
    "                c += 1\n    return c"
)
DATE = "import re\ndef fmt(dt):\n    return re.sub(r'(\\d+)-(\\d+)', '\\2-\\1', dt)"
DATE_DEAD_LINE = DATE + "\n    return fmt(dt)"  # mbpp-427 vs 928
UNRELATED = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s"


def test_shape_ignores_attributes_constants_operators_and_imports():
    for a, b in [(SMALLEST, LARGEST), (IS_ODD, IS_EVEN), (IMPORT_TOP, IMPORT_INSIDE)]:
        assert shape(ast.parse(a)) == shape(ast.parse(b))
    assert shape(ast.parse(IS_ODD)) != shape(ast.parse(UNRELATED))


def test_near_duplicate_groups_join_near_copies_and_keep_others_apart():
    codes = [SMALLEST, LARGEST, UPPER, LOWER, UNRELATED, IS_ODD, IS_EVEN]
    codes += [PAIRS_SUM, INVERSIONS, DATE, DATE_DEAD_LINE]
    seeds = [Seed(f"s{i}", code, []) for i, code in enumerate(codes)]
    groups = near_duplicate_groups(seeds)
    assert groups["s0"] == groups["s1"] == "s0"  # same shape
    assert groups["s2"] == groups["s3"] == "s2"  # `return` indented differently: similarity
    assert groups["s5"] == groups["s6"] == "s5"
    assert groups["s7"] == groups["s8"] == "s7"  # containment
    assert groups["s9"] == groups["s10"] == "s9"  # every line of one appears in the other
    assert len(set(groups.values())) == 6 and groups["s4"] == "s4"


def test_near_duplicate_groups_join_programs_defining_the_same_function():
    # mbpp-420/837: two different `cube_Sum` formulas, AST similarity only 0.75
    odd = (
        "def cube_Sum(n):\n    s = 0\n    for i in range(n):\n        s += (2*i+1)**3\n    return s"
    )
    even = "def cube_Sum(n):\n    total = 0\n    for k in range(1, n + 1):\n        total += 8*k*k*k\n    return total"
    groups = near_duplicate_groups(
        [Seed("a", odd, []), Seed("b", even, []), Seed("c", UNRELATED, [])]
    )
    assert groups == {"a": "a", "b": "a", "c": "c"}
