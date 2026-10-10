"""Stream A: seeds -> mutants -> sandbox -> Examples -> split JSONL files.

Run: uv run python -m traceback_tutor.build_dataset --out data/v0
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from .mutate import MUTATIONS
from .sandbox import RunResult, run_sandboxed
from .schema import BUG_TYPES, Example, write_jsonl
from .seeds import Seed, load_seeds, near_duplicate_groups

V0_BUG_TYPES = [name for name, b in BUG_TYPES.items() if b.in_v0]
SPLITS = ("train", "val", "test", "test_ood")

# TypeError covers far more than these two bugs (e.g. max(a) -> "'int' object is not
# iterable"), so for them the message must match too, or the label would be wrong.
MESSAGE_CHECKS = {
    "str_plus_number": re.compile(
        r"can only concatenate str \(not \"(int|float)\"\) to str"
        r"|unsupported operand type\(s\) for \+=?: '(int|float)' and 'str'"
    ),
    "wrong_arg_count": re.compile(
        r"\d+ given\)|(was|were) given|missing \d+ required|missing required argument"
        r"|expected (at least |at most )?\d+ arguments?, got \d+"
    ),
}


def make_examples(
    seeds: list[Seed],
    bug_types: list[str] = V0_BUG_TYPES,
    per_seed: int = 1,
    attempts: int = 4,
    seed: int = 0,
    workers: int = 8,
) -> list[Example]:
    """For each seed and bug type, mutate the file (program + test calls), run, keep crashes.

    Up to `attempts` distinct mutants are tried per (seed, bug type), and the first
    `per_seed` that crash with the expected error are kept. Mutants that don't crash,
    time out or crash with another error are dropped (relabelling is v1).
    """
    candidates = []  # (seed, bug_type, mutation result), in a fixed order
    for s in seeds:
        file = s.with_tests()
        for bug_type in bug_types:
            # One rng per (seed, bug type): results don't depend on which other seeds ran.
            rng = random.Random(f"{seed}:{s.seed_id}:{bug_type}")
            seen = set()
            for _ in range(attempts):
                m = MUTATIONS[bug_type](file, s.program_lines, rng)
                if m is not None and m.code not in seen:
                    seen.add(m.code)
                    candidates.append((s, bug_type, m))

    with ThreadPoolExecutor(workers) as pool:
        runs = list(pool.map(lambda c: run_sandboxed(c[2].code), candidates))

    examples, kept = [], Counter()
    for (s, bug_type, m), run in zip(candidates, runs):
        key = (s.seed_id, bug_type)
        if kept[key] >= per_seed or not has_expected_error(bug_type, run):
            continue
        examples.append(
            Example(
                id=f"{s.seed_id}-{bug_type}-{kept[key]}",
                seed_id=s.seed_id,
                bug_type=bug_type,
                code=m.code,
                traceback=run.traceback,
                error_type=run.error_type,
                error_line=run.error_line,
                bug_line=m.bug_line,
            )
        )
        kept[key] += 1
    return examples


def has_expected_error(bug_type: str, run: RunResult) -> bool:
    if not run.crashed or run.error_line is None:
        return False
    if run.error_type != BUG_TYPES[bug_type].expected_error:
        return False
    check = MESSAGE_CHECKS.get(bug_type)
    return check is None or bool(check.search(run.traceback.strip().splitlines()[-1]))


def balance(examples: list[Example], per_type: int, seed: int = 0) -> list[Example]:
    """Keep a random `per_type` examples of each bug type (all of them if fewer).

    Sampling across all seeds, rather than taking fewer seeds, keeps the bug types
    balanced and the programs as varied as possible.
    """
    rng = random.Random(seed)
    kept = set()
    for bug_type in sorted({ex.bug_type for ex in examples}):
        ids = [ex.id for ex in examples if ex.bug_type == bug_type]
        kept.update(rng.sample(ids, min(per_type, len(ids))))
    return [ex for ex in examples if ex.id in kept]


def assign_splits(
    examples: list[Example],
    holdout_bug_types: tuple[str, ...] = (),
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
    groups: dict[str, str] | None = None,
) -> list[Example]:
    """Split by group of seeds (never by example) into train/val/test.

    `groups` maps seed_id -> group id (see seeds.near_duplicate_groups), so near-copies
    of a program land in the same split; without it, each seed is its own group.
    Examples of a held-out bug type go to "test_ood" regardless of seed, and
    those seeds' other examples stay in their normal split.
    """
    group_of = {ex.seed_id: (groups or {}).get(ex.seed_id, ex.seed_id) for ex in examples}
    group_ids = sorted(set(group_of.values()))
    random.Random(seed).shuffle(group_ids)
    n_train = round(ratios[0] * len(group_ids))
    n_val = round(ratios[1] * len(group_ids))
    split_of = {}
    for i, group_id in enumerate(group_ids):
        split_of[group_id] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    out = []
    for ex in examples:
        held_out = ex.bug_type in holdout_bug_types
        out.append(replace(ex, split="test_ood" if held_out else split_of[group_of[ex.seed_id]]))
    return out


def counts(examples: list[Example]) -> dict:
    by_split = Counter(ex.split for ex in examples)
    by_type = Counter(ex.bug_type for ex in examples)
    table = Counter((ex.split, ex.bug_type) for ex in examples)
    return {
        "total": len(examples),
        "seeds": len({ex.seed_id for ex in examples}),
        "per_split": {s: by_split[s] for s in SPLITS if by_split[s]},
        "per_bug_type": dict(sorted(by_type.items())),
        "per_split_and_bug_type": {
            s: {b: table[s, b] for b in sorted(by_type) if table[s, b]}
            for s in SPLITS
            if by_split[s]
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-type", type=int, default=None, help="max examples per bug type")
    p.add_argument("--per-seed", type=int, default=1, help="examples per seed and bug type")
    p.add_argument("--holdout", nargs="*", default=[], help="bug types for test_ood")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    unknown = set(args.holdout) - set(V0_BUG_TYPES)
    if unknown:
        p.error(f"unknown holdout bug types: {sorted(unknown)}")

    seeds = load_seeds()
    examples = make_examples(seeds, per_seed=args.per_seed, seed=args.seed)
    found = Counter(ex.bug_type for ex in examples)
    if args.per_type is not None:
        examples = balance(examples, args.per_type, seed=args.seed)
    groups = near_duplicate_groups(seeds)
    examples = assign_splits(examples, tuple(args.holdout), seed=args.seed, groups=groups)

    for split in SPLITS:
        path = args.out / f"{split}.jsonl"
        path.unlink(missing_ok=True)  # don't leave a stale file from an earlier build
        rows = [ex for ex in examples if ex.split == split]
        if rows:
            write_jsonl(path, rows)
    stats = {"args": {k: str(v) for k, v in vars(args).items()}, **counts(examples)}
    stats["seeds_available"] = len(seeds)
    stats["seed_groups"] = len({groups[ex.seed_id] for ex in examples})
    stats["found_before_balancing"] = dict(sorted(found.items()))
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
