"""SyntaxError: delete the colon at the end of a block-opening line, e.g. `if x > 0:`.

A text edit, not an AST one: the result can't be parsed, and the line is the bug.
Only lines that end in the colon qualify (not `if x: return y`, not a trailing comment).
"""

from __future__ import annotations

import random
import re

from .base import MutationResult

BLOCK_START = re.compile(r"^\s*(if|elif|else|for|while|def|class|try|except|finally|with)\b")


def apply(code: str, program_lines: int, rng: random.Random) -> MutationResult | None:
    lines = code.split("\n")
    program = lines[:program_lines]
    sites = [i for i, line in enumerate(program) if BLOCK_START.match(line) and line.endswith(":")]
    if not sites:
        return None
    i = rng.choice(sites)
    lines[i] = lines[i][:-1]
    return MutationResult("\n".join(lines), i + 1)
