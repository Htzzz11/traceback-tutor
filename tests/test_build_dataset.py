from collections import Counter, defaultdict

from traceback_tutor.build_dataset import (
    V0_BUG_TYPES,
    assign_splits,
    balance,
    has_expected_error,
    make_examples,
)
from traceback_tutor.sandbox import RunResult
from traceback_tutor.schema import BUG_TYPES, Example
from traceback_tutor.seeds import Seed

SEEDS = [
    Seed(
        "mbpp-1",
        "def max_item(xs):\n"
        "    best = xs[0]\n"
        "    for i in range(len(xs)):\n"
        "        if xs[i] > best:\n"
        "            best = xs[i]\n"
        "    return best",
        ["assert max_item([3, 9, 4]) == 9", "assert max_item([1]) == 1"],
    ),
    Seed(
        "mbpp-2",
        "def count_evens(xs):\n"
        "    count = 0\n"
        "    for x in xs:\n"
        "        if x % 2 == 0:\n"
        "            count += 1\n"
        "    return count",
        ["assert count_evens([1, 2, 4]) == 2"],
    ),
    Seed(
        "mbpp-3",
        "def area(w, h):\n"
        "    return w * h\n"
        "\n"
        "def total_area(rects):\n"
        "    return sum(area(w, h) for w, h in rects)",
        ["assert total_area([(1, 2), (3, 4)]) == 14"],
    ),
]


def fake_examples(n_seeds=50, bug_types=V0_BUG_TYPES):
    return [
        Example(f"s{i}-{b}-0", f"s{i}", b, "code", "tb", "E", 1, 1)
        for i in range(n_seeds)
        for b in bug_types
    ]


def test_make_examples_keeps_only_expected_crashes():
    examples = make_examples(SEEDS, workers=4)
    assert len({ex.id for ex in examples}) == len(examples)
    assert {ex.bug_type for ex in examples} == set(V0_BUG_TYPES)
    files = {s.seed_id: s.with_tests() for s in SEEDS}
    for ex in examples:
        assert ex.error_type == BUG_TYPES[ex.bug_type].expected_error, ex.id
        assert 'File "main.py"' in ex.traceback
        before, after = files[ex.seed_id].split("\n"), ex.code.split("\n")
        assert [i + 1 for i, (a, b) in enumerate(zip(before, after)) if a != b] == [ex.bug_line]
        assert 1 <= ex.error_line <= len(after)


def test_make_examples_is_reproducible():
    assert make_examples(SEEDS, workers=4) == make_examples(SEEDS, workers=2)


def test_message_check_rejects_a_type_error_from_another_bug():
    run = RunResult(
        crashed=True,
        timed_out=False,
        error_type="TypeError",
        error_line=2,
        traceback="Traceback ...\nTypeError: 'int' object is not iterable\n",
        stdout="",
    )
    assert not has_expected_error("wrong_arg_count", run)
    run.traceback = "Traceback ...\nTypeError: f() missing 1 required positional argument: 'b'\n"
    assert has_expected_error("wrong_arg_count", run)
    assert not has_expected_error("str_plus_number", run)


def test_splits_never_share_a_seed():
    examples = assign_splits(fake_examples())
    splits_of = defaultdict(set)
    for ex in examples:
        splits_of[ex.seed_id].add(ex.split)
    assert all(len(s) == 1 for s in splits_of.values())
    assert Counter(ex.split for ex in examples) == {"train": 200, "val": 25, "test": 25}


def test_near_duplicate_groups_never_straddle_splits():
    groups = {f"s{i}": f"s{i - i % 5}" for i in range(50)}  # 10 groups of 5 seeds
    examples = assign_splits(fake_examples(), groups=groups)
    splits_of = defaultdict(set)
    for ex in examples:
        splits_of[groups[ex.seed_id]].add(ex.split)
    assert all(len(s) == 1 for s in splits_of.values())
    assert Counter(splits_of[g].pop() for g in splits_of) == {"train": 8, "val": 1, "test": 1}


def test_held_out_bug_types_only_go_to_test_ood():
    examples = assign_splits(fake_examples(), holdout_bug_types=("missing_colon",))
    for ex in examples:
        assert (ex.split == "test_ood") == (ex.bug_type == "missing_colon")
    # the held-out seeds' other examples stay in normal splits
    in_distribution = [ex for ex in examples if ex.split != "test_ood"]
    assert {ex.seed_id for ex in in_distribution} == {f"s{i}" for i in range(50)}


def test_splits_are_reproducible_and_depend_on_the_seed():
    a = assign_splits(fake_examples(), seed=0)
    assert a == assign_splits(fake_examples(), seed=0)
    assert a != assign_splits(fake_examples(), seed=1)


def test_balance_caps_each_bug_type():
    examples = fake_examples(n_seeds=10) + fake_examples(n_seeds=3, bug_types=["extra"])
    kept = balance(examples, per_type=4)
    counts = Counter(ex.bug_type for ex in kept)
    assert counts == {**{b: 4 for b in V0_BUG_TYPES}, "extra": 3}
    assert kept == balance(examples, per_type=4)
