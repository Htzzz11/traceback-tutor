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


def test_str_plus_number_handles_non_ascii_lines():
    seed = Seed("t", 'def f(n):\n    return "café: " + str(n)', ['assert f(1) == "café: 1"'])
    assert mutate("str_plus_number", seed).code.split("\n")[1] == '    return "café: " + n'


# off_by_one_index


def test_off_by_one_loop_bound_crashes_on_the_next_line():
    mutants = {mutate("off_by_one_index", MAX_ITEM, s).code for s in range(20)}
    [code] = [c for c in mutants if "range(len(xs) + 1)" in c]
    assert_crashes_as("off_by_one_index", code, error_line=4)  # bug_line is 3


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


def test_wrong_arg_count_on_a_test_call_drops_or_adds_an_argument():
    seed = Seed("t", "def area(w, h):\n    return w * h", ["assert area(4, 3) == 12"])
    seen = set()
    for rng_seed in range(30):
        m = mutate("wrong_arg_count", seed, rng_seed)
        assert m.bug_line == 4  # the program has no call, so the test call is used
        seen.add(m.code.split("\n")[3])
        assert_crashes_as("wrong_arg_count", m.code, error_line=4)
    assert seen == {"assert area(4) == 12", "assert area(3) == 12", "assert area(4, 3, 3) == 12"}


def test_wrong_arg_count_prefers_calls_in_the_program():
    seed = Seed("t", "def size(xs):\n    return len(xs)", ["assert size([1]) == 1"])
    in_tests = sum(mutate("wrong_arg_count", seed, s).bug_line == 4 for s in range(100))
    assert 0 < in_tests < 25


def test_wrong_arg_count_keeps_keyword_arguments():
    seed = Seed(
        "t", "def desc(xs):\n    return sorted(xs, reverse=True)", ["assert desc([1, 2]) == [2, 1]"]
    )
    codes = {mutate("wrong_arg_count", seed, s).code.split("\n")[1] for s in range(20)}
    assert "    return sorted(reverse=True)" in codes
    assert_crashes_as("wrong_arg_count", seed.with_tests().replace("xs, reverse", "reverse"))


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
