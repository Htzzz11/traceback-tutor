"""Types and helpers shared by the mutations."""

from __future__ import annotations

import ast
import random
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class MutationResult:
    code: str  # the mutated file
    bug_line: int  # 1-based line the mutation changed


# A mutation takes the whole file and returns a mutated copy, or None if this bug
# can't be injected into it. Lines 1..program_lines are the program; the lines after
# them are the test calls. All randomness comes from `rng`.
Mutation = Callable[[str, int, random.Random], MutationResult | None]


def on_one_line(node: ast.AST) -> bool:
    return node.lineno == node.end_lineno


def source(code: str, node: ast.AST) -> str:
    return ast.get_source_segment(code, node)


def replace_node(code: str, node: ast.AST, new_text: str) -> MutationResult:
    """Replace the source text of a one-line `node`, leaving the rest of the file as is.

    The AST picks the site; editing the text (instead of ast.unparse) keeps the
    student's formatting and comments, and every other line keeps its number.
    """
    assert on_one_line(node)
    lines = code.split("\n")
    i = node.lineno - 1
    raw = lines[i].encode()  # col offsets are UTF-8 byte offsets
    lines[i] = (raw[: node.col_offset] + new_text.encode() + raw[node.end_col_offset :]).decode()
    return MutationResult("\n".join(lines), node.lineno)


def pick(sites_by_kind: dict[str, list], rng: random.Random):
    """Pick a kind of edit uniformly among kinds that have a site, then one of its sites.

    Without this, the most common kind of site (e.g. `xs[i]` for off-by-one) would
    crowd out rarer, more instructive ones. Returns None if there are no sites.
    """
    kinds = [kind for kind, sites in sites_by_kind.items() if sites]
    if not kinds:
        return None
    return rng.choice(sites_by_kind[rng.choice(kinds)])
