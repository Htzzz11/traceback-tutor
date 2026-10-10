"""IndexError: an index or a loop bound goes one step too far.

- loop bound:  range(len(xs))     -> range(len(xs) + 1)
               range(len(xs) - 1) -> range(len(xs))
- last item:   xs[len(xs) - 1]    -> xs[len(xs)]
- next item:   xs[i]              -> xs[i + 1]
- 1-based:     pair[1]            -> pair[2]   (counting positions from 1, not 0)
"""

from __future__ import annotations

import ast
import random

from .base import MutationResult, on_one_line, replace_node, source


def apply(code: str, program_lines: int, rng: random.Random) -> MutationResult | None:
    tree = ast.parse(code)
    sites: list[tuple[ast.AST, str]] = []  # (node to replace, new text)
    for node in ast.walk(tree):
        if is_call_to(node, "range") and len(node.args) in (1, 2, 3):
            stop = node.args[0] if len(node.args) == 1 else node.args[1]
            if is_call_to(stop, "len"):
                sites.append((stop, f"{source(code, stop)} + 1"))
            elif is_len_minus_one(stop):
                sites.append((stop, source(code, stop.left)))
        elif isinstance(node, ast.Subscript):
            index = node.slice
            if is_len_minus_one(index):
                sites.append((index, source(code, index.left)))
            elif isinstance(index, ast.Name):
                sites.append((index, f"{index.id} + 1"))
            elif is_int(index) and index.value >= 0:
                sites.append((index, str(index.value + 1)))
    sites = [(n, text) for n, text in sites if on_one_line(n) and n.lineno <= program_lines]
    if not sites:
        return None
    node, new_text = rng.choice(sites)
    return replace_node(code, node, new_text)


def is_call_to(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def is_int(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    )


def is_len_minus_one(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and is_call_to(node.left, "len")
        and is_int(node.right)
        and node.right.value == 1
    )
