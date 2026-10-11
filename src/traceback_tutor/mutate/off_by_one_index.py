"""IndexError: an index or a loop bound goes one step too far.

Four kinds of edit, picked from the AST:
- loop bound:  range(len(xs))     -> range(len(xs) + 1)
               range(len(xs) - 1) -> range(len(xs))
               i < len(xs)        -> i <= len(xs)
  Always used when the program has one: it's the classic off-by-one, and the only
  kind where the crash is usually on a later line than the bug. Only about 5% of
  MBPP programs have one, so the other kinds would otherwise crowd it out.
Otherwise one of the other kinds at random, then one site of it:
- last item:   xs[len(xs) - 1]    -> xs[len(xs)]
- next item:   xs[i]              -> xs[i + 1]
- 1-based:     pair[1]            -> pair[2]   (counting positions from 1, not 0)
"""

from __future__ import annotations

import ast
import random

from .base import MutationResult, on_one_line, pick, replace_node, source


def apply(code: str, program_lines: int, rng: random.Random) -> MutationResult | None:
    tree = ast.parse(code)
    sites: dict[str, list[tuple[ast.AST, str]]] = {
        "loop bound": [],
        "last item": [],
        "next item": [],
        "1-based": [],
    }
    for node in ast.walk(tree):
        if is_call_to(node, "range") and len(node.args) in (1, 2, 3):
            stop = node.args[0] if len(node.args) == 1 else node.args[1]
            if is_call_to(stop, "len"):
                sites["loop bound"].append((stop, f"{source(code, stop)} + 1"))
            elif is_len_minus_one(stop):
                sites["loop bound"].append((stop, source(code, stop.left)))
        elif is_less_than_len(node):
            new_text = f"{source(code, node.left)} <= {source(code, node.comparators[0])}"
            sites["loop bound"].append((node, new_text))
        elif isinstance(node, ast.Subscript):
            index = node.slice
            if is_len_minus_one(index):
                sites["last item"].append((index, source(code, index.left)))
            elif isinstance(index, ast.Name):
                sites["next item"].append((index, f"{index.id} + 1"))
            elif is_int(index) and index.value >= 0:
                sites["1-based"].append((index, str(index.value + 1)))
    sites = {
        kind: [(n, t) for n, t in found if on_one_line(n) and n.lineno <= program_lines]
        for kind, found in sites.items()
    }
    site = pick({"loop bound": sites["loop bound"]}, rng) or pick(sites, rng)
    return replace_node(code, *site) if site else None


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


def is_less_than_len(node: ast.AST) -> bool:
    """`i < len(xs)`: a single `<` against len()."""
    return (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Lt)
        and is_call_to(node.comparators[0], "len")
    )
