"""NameError: misspell one use of a name in the program, e.g. `return total` -> `return totl`.

Not in the test calls: misspelling the function name there is a different bug.
"""

from __future__ import annotations

import ast
import builtins
import keyword
import random

from .base import MutationResult, on_one_line, replace_node

MIN_LENGTH = 3  # typos in `i` or `n` don't look like typos
BUILTINS = set(dir(builtins))


def apply(code: str, program_lines: int, rng: random.Random) -> MutationResult | None:
    tree = ast.parse(code)
    taken = defined_names(tree) | BUILTINS
    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and len(node.id) >= MIN_LENGTH
        and node.id not in BUILTINS
        and on_one_line(node)
        and node.lineno <= program_lines
    ]
    if not sites:
        return None
    node = rng.choice(sites)
    typo = misspell(node.id, rng, taken)
    return replace_node(code, node, typo) if typo else None


def misspell(name: str, rng: random.Random, taken: set[str]) -> str | None:
    """A one-letter typo of `name` that isn't already a name. The first letter is kept."""
    options = set()
    for i in range(1, len(name)):
        options.add(name[:i] + name[i + 1 :])  # drop a letter: total -> totl
        options.add(name[:i] + name[i] + name[i:])  # double a letter: total -> tottal
        if i + 1 < len(name) and name[i] != name[i + 1]:
            options.add(name[:i] + name[i + 1] + name[i] + name[i + 2 :])  # swap: total -> toatl
    options = sorted(
        o for o in options if o.isidentifier() and not keyword.iskeyword(o) and o not in taken
    )
    return rng.choice(options) if options else None


def defined_names(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
    return names
