"""TypeError: text and a number meet at a `+`.

Three kinds of edit, picked from the AST (one kind at random, then one site):
- store a number as text:   count = 0        -> count = "0"
  (the crash then happens later, where `count` is added to a number)
- quote a number in a plain addition or `+=`:
                            total = total + 1 -> total = total + "1"
                            count += 1        -> count += "1"
  Only where the addition is the whole right-hand side (or return value) and adds a
  name and a number: quotes inside range() or a bigger formula look nothing like a
  learner's mistake.
- drop a str() next to a `+`: "n=" + str(n)  -> "n=" + n
  (the textbook case, but only a handful of MBPP programs build text like this)
"""

from __future__ import annotations

import ast
import random

from .base import MutationResult, on_one_line, pick, replace_node, source


def apply(code: str, program_lines: int, rng: random.Random) -> MutationResult | None:
    tree = ast.parse(code)
    sites: dict[str, list[tuple[ast.AST, str]]] = {"text": [], "quote": [], "unwrap": []}
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if is_number(value):
                sites["text"].append((value, quoted(code, value)))
            elif is_plain_addition(value):
                number = value.left if is_number(value.left) else value.right
                sites["quote"].append((number, quoted(code, number)))
        elif isinstance(node, ast.Return) and is_plain_addition(value):
            number = value.left if is_number(value.left) else value.right
            sites["quote"].append((number, quoted(code, number)))
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            if is_number(value):
                sites["quote"].append((value, quoted(code, value)))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            for x in (node.left, node.right):
                if is_str_call(x):
                    sites["unwrap"].append((x, source(code, x.args[0])))
    sites = {
        kind: [(n, t) for n, t in found if on_one_line(n) and n.lineno <= program_lines]
        for kind, found in sites.items()
    }
    site = pick(sites, rng)
    return replace_node(code, *site) if site else None


def quoted(code: str, node: ast.AST) -> str:
    return f'"{source(code, node)}"'


def is_plain_addition(node: ast.AST) -> bool:
    """`name + 1` or `1 + name`."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
        return False
    pair = (node.left, node.right)
    return any(isinstance(a, ast.Name) and is_number(b) for a, b in (pair, pair[::-1]))


def is_number(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int | float)
        and not isinstance(node.value, bool)
    )


def is_str_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
        and len(node.args) == 1
        and not node.keywords
    )
