"""Stream A: load correct seed programs and their test cases (MBPP, CC-BY-4.0)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Seed:
    seed_id: str  # e.g. "mbpp-11"
    code: str  # correct reference solution
    test_calls: list[str]  # statements that exercise the code, e.g. MBPP's assert lines


def load_seeds(limit: int | None = None) -> list[Seed]:
    """Load MBPP seed programs whose reference solution passes its own tests.

    Drop seeds that fail or use imports beyond the standard library.
    """
    raise NotImplementedError
