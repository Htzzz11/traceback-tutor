"""Stream A: seeds -> mutants -> sandbox -> Examples -> split JSONL files.

Run: uv run python -m traceback_tutor.build_dataset --out data/v0
"""

from __future__ import annotations

from .schema import Example


def make_examples(per_seed: int = 3, seed: int = 0) -> list[Example]:
    """For each seed and v0 bug type, mutate, append test calls, run, keep crashes.

    Drop mutants that don't crash or time out. If the crash's error type differs
    from BUG_TYPES[bug_type].expected_error, drop it for now (relabelling is v1).
    """
    raise NotImplementedError


def assign_splits(
    examples: list[Example],
    holdout_bug_types: tuple[str, ...] = (),
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> list[Example]:
    """Split by seed_id (never by example) into train/val/test.

    Examples of a held-out bug type go to "test_ood" regardless of seed, and
    those seeds' other examples stay in their normal split.
    """
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit("not implemented yet")
