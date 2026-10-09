"""Stream A: bug injection. One Mutation per bug type, registered in MUTATIONS.

AST-level mutations for runtime errors; text-level edits for syntax errors,
because a program with a syntax error can't be parsed into an AST.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol


@dataclass
class MutationResult:
    code: str  # the mutated program (without test calls)
    bug_line: int  # 1-based line the mutation changed


class Mutation(Protocol):
    bug_type: str  # key of schema.BUG_TYPES

    def apply(self, code: str, rng: random.Random) -> MutationResult | None:
        """Return a mutated copy of `code`, or None if this bug can't be injected."""
        ...


MUTATIONS: dict[str, Mutation] = {}  # bug_type -> Mutation; filled as mutations are added
