"""Each mutation is checked by running its output: the bug must raise the error it claims."""

import random

import pytest

from traceback_tutor.build_dataset import has_expected_error
from traceback_tutor.mutate import MUTATIONS
from traceback_tutor.mutate.misspelt_variable import misspell
from traceback_tutor.sandbox import run_sandboxed
from traceback_tutor.schema import BUG_TYPES
from traceback_tutor.seeds import Seed

SUM_LIST = Seed(
    "t-sum",
    "def sum_list(nums):\n"
    "    total = 0\n"
    "    for n in nums:\n"
    "        total = total + n\n"
    "    return total",
    ["assert sum_list([1, 2, 3]) == 6"],
)
MAX_ITEM = Seed(
    "t-max",
    "def max_item(xs):\n"
    "    best = xs[0]\n"
    "    for i in range(len(xs)):\n"
    "        if xs[i] > best:\n"
    "            best = xs[i]\n"
    "    return best",
    ["assert max_item([3, 9, 4]) == 9"],
)
SIGN = Seed(
    "t-sign",
    'def sign(x):\n    if x > 0:\n        return "positive"\n    return "not positive"',
    ['assert sign(5) == "positive"'],
)


def mutate(bug_type, seed, rng_seed=0):
    return MUTATIONS[bug_type](seed.with_tests(), seed.program_lines, random.Random(rng_seed))


def changed_lines(before, after):
    a, b = before.split("\n"), after.split("\n")
    assert len(a) == len(b), "a mutation must not add or remove lines"
    return [i + 1 for i, (x, y) in enumerate(zip(a, b)) if x != y]


def assert_crashes_as(bug_type, code, error_line=None):
    run = run_sandboxed(code)
    assert has_expected_error(bug_type, run), run.traceback
    if error_line is not None:
        assert run.error_line == error_line
    return run


def test_every_v0_bug_type_has_a_mutation():
    assert set(MUTATIONS) == {n for n, b in BUG_TYPES.items() if b.in_v0}


@pytest.mark.parametrize("bug_type", sorted(MUTATIONS))
@pytest.mark.parametrize("seed", [SUM_LIST, MAX_ITEM, SIGN], ids=lambda s: s.seed_id)
def test_mutation_changes_exactly_the_bug_line_and_is_deterministic(bug_type, seed):
    for rng_seed in range(10):
        m = mutate(bug_type, seed, rng_seed)
        if m is None:
            continue
        assert changed_lines(seed.with_tests(), m.code) == [m.bug_line]
        assert mutate(bug_type, seed, rng_seed) == m


# misspelt_variable


def test_misspelt_variable_raises_name_error_where_the_typo_is():
    for rng_seed in range(10):
        m = mutate("misspelt_variable", SUM_LIST, rng_seed)
        assert m.bug_line in (3, 4, 5)  # never the test call on line 7
        assert_crashes_as("misspelt_variable", m.code, error_line=m.bug_line)


def test_misspell_makes_a_new_one_letter_typo():
    taken = {"total", "tota", "totl"}
    for rng_seed in range(20):
        typo = misspell("total", random.Random(rng_seed), taken)
        assert typo not in taken and typo[0] == "t" and abs(len(typo) - 5) <= 1


def test_misspell_leaves_underscores_and_digits_alone():
    for rng_seed in range(30):
        typo = misspell("left_el2", random.Random(rng_seed), set())
        assert "__" not in typo and typo.count("_") == 1 and typo.endswith("2")
        assert typo.index("_") in (3, 4, 5)  # only the letters around it moved


def test_misspelt_variable_skips_short_names():
    seed = Seed("t", "def f(x):\n    return x", ["assert f(1) == 1"])
    assert mutate("misspelt_variable", seed) is None


# str_plus_number


def test_str_plus_number_drops_a_str_conversion():
    seed = Seed("t", 'def label(n):\n    return "n=" + str(n)', ['assert label(3) == "n=3"'])
    m = mutate("str_plus_number", seed)
    assert m.code.split("\n")[1] == '    return "n=" + n'
    run = assert_crashes_as("str_plus_number", m.code, error_line=2)
    assert "can only concatenate str" in run.traceback


def test_str_plus_number_quotes_a_number():
    seed = Seed("t", "def inc(x):\n    return x + 1", ["assert inc(1) == 2"])
    m = mutate("str_plus_number", seed)
    assert m.code.split("\n")[1] == '    return x + "1"'
    assert_crashes_as("str_plus_number", m.code, error_line=2)


def test_str_plus_number_can_store_a_number_as_text_and_crash_later():
    seed = Seed(
        "t",
        "def count_evens(xs):\n"
        "    count = 0\n"
        "    for x in xs:\n"
        "        if x % 2 == 0:\n"
        "            count += 1\n"
        "    return count",
        ["assert count_evens([1, 2, 4]) == 2"],
    )
    bug_lines = set()
    for rng_seed in range(10):
        m = mutate("str_plus_number", seed, rng_seed)
        assert_crashes_as("str_plus_number", m.code, error_line=5)
        bug_lines.add(m.bug_line)
    assert bug_lines == {2, 5}  # `count = "0"` crashes on line 5 too


def test_str_plus_number_skips_numbers_inside_bigger_expressions():
    # quotes inside range() or a larger formula look nothing like a learner's mistake
    for line in ["return sum(range(n + 1))", "y = (n + 1) * 2", "return n + 1 + n"]:
        seed = Seed("t", f"def f(n):\n    {line}", ["f(2)"])
        assert mutate("str_plus_number", seed) is None, line


def test_str_plus_number_handles_non_ascii_lines():
    seed = Seed("t", 'def f(n):\n    return "café: " + str(n)', ['assert f(1) == "café: 1"'])
    assert mutate("str_plus_number", seed).code.split("\n")[1] == '    return "café: " + n'


# off_by_one_index


def test_off_by_one_always_uses_a_loop_bound_when_there_is_one():
    # MAX_ITEM also has xs[0] and xs[i] sites, but the loop bound is the classic bug
    mutants = [mutate("off_by_one_index", MAX_ITEM, s) for s in range(20)]
    [(code, bug_line)] = {(m.code, m.bug_line) for m in mutants}
    assert "range(len(xs) + 1)" in code and bug_line == 3
    assert_crashes_as("off_by_one_index", code, error_line=4)  # crash is a line later


def test_off_by_one_while_condition():
    seed = Seed(
        "t",
        "def total(xs):\n    i, s = 0, 0\n    while i < len(xs):\n        s += xs[i]\n"
        "        i += 1\n    return s",
        ["assert total([1, 2]) == 3"],
    )
    m = mutate("off_by_one_index", seed)
    assert m.bug_line == 3 and "while i <= len(xs):" in m.code
    assert_crashes_as("off_by_one_index", m.code, error_line=4)


@pytest.mark.parametrize(
    "program, test, new_line",
    [
        ("def last(xs):\n    return xs[len(xs) - 1]", "assert last([1, 2]) == 2", "xs[len(xs)]"),
        ("def second(pair):\n    return pair[1]", "assert second((1, 2)) == 2", "pair[2]"),
        (
            "def head(xs, i):\n    return xs[i]",
            "assert head([1, 2], 1) == 2",
            "xs[i + 1]",
        ),
    ],
)
def test_off_by_one_index_sites(program, test, new_line):
    m = mutate("off_by_one_index", Seed("t", program, [test]))
    assert m.bug_line == 2 and new_line in m.code
    assert_crashes_as("off_by_one_index", m.code, error_line=2)


def test_off_by_one_needs_an_index():
    seed = Seed("t", "def add(a, b):\n    return a + b", ["assert add(1, 2) == 3"])
    assert mutate("off_by_one_index", seed) is None


# wrong_arg_count


def test_wrong_arg_count_only_sometimes_uses_a_test_call():
    # The program has no call of its own: no automatic fallback to the test call.
    seed = Seed("t", "def area(w, h):\n    return w * h", ["assert area(4, 3) == 12"])
    mutants = [mutate("wrong_arg_count", seed, s) for s in range(200)]
    used = [m for m in mutants if m is not None]
    assert 0 < len(used) < 15
    for m in used:  # a test call only ever loses an argument
        assert m.bug_line == 4
        assert m.code.split("\n")[3] in {"assert area(4) == 12", "assert area(3) == 12"}
        assert_crashes_as("wrong_arg_count", m.code, error_line=4)


def test_wrong_arg_count_adds_an_argument_only_to_the_students_own_functions():
    seed = Seed(
        "t",
        "def helper(a, b):\n    return a * b\n\ndef area(w, h):\n    return helper(w, h) + len([w])",
        ["assert area(4, 3) == 13"],
    )
    lines = {mutate("wrong_arg_count", seed, s).code.split("\n")[4] for s in range(40)}
    assert "    return helper(w, h, h) + len([w])" in lines
    assert "    return helper(w) + len([w])" in lines
    assert not any("len([w], [w])" in line for line in lines)


def test_wrong_arg_count_prefers_calls_in_the_program():
    seed = Seed("t", "def fix(s):\n    return s.replace('a', 'b')", ["assert fix('a') == 'b'"])
    in_tests = sum(mutate("wrong_arg_count", seed, s).bug_line == 4 for s in range(200))
    assert 0 < in_tests < 15


def test_wrong_arg_count_never_empties_a_one_argument_builtin():
    # len() or range() with nothing inside isn't a learner's mistake
    seed = Seed("t", "def f(xs):\n    return list(range(len(xs)))", ["assert f([1]) == [0]"])
    for s in range(50):
        m = mutate("wrong_arg_count", seed, s)
        assert m is None or m.bug_line == 4  # only ever the test call


def test_wrong_arg_count_never_drops_a_long_argument():
    seed = Seed(
        "t",
        "def pos(xs, k):\n    return pow(sum(x for x in xs if x > 0), k)",
        ["assert pos([1, -2, 3], 2) == 16"],
    )
    lines = {mutate("wrong_arg_count", seed, s).code.split("\n")[1] for s in range(30)}
    assert lines == {"    return pow(sum(x for x in xs if x > 0))"}


def test_wrong_arg_count_keeps_keyword_arguments():
    seed = Seed(
        "t",
        "def scale(x, k, offset=0):\n    return x * k + offset\n\n"
        "def run(x):\n    return scale(x, 2, offset=1)",
        ["assert run(3) == 7"],
    )
    lines = {mutate("wrong_arg_count", seed, s).code.split("\n")[4] for s in range(40)}
    assert "    return scale(x, offset=1)" in lines
    assert_crashes_as("wrong_arg_count", seed.with_tests().replace("x, 2, offset", "x, offset"))


# missing_colon


def test_missing_colon_raises_syntax_error_on_the_bug_line():
    bug_lines = set()
    for rng_seed in range(10):
        m = mutate("missing_colon", SIGN, rng_seed)
        assert_crashes_as("missing_colon", m.code, error_line=m.bug_line)
        bug_lines.add(m.bug_line)
    assert bug_lines == {1, 2}


def test_missing_colon_skips_one_liners_and_trailing_comments():
    seed = Seed("t", "def f(x):  # note\n    if x: return 1\n    return 0", ["assert f(1) == 1"])
    assert mutate("missing_colon", seed) is None
