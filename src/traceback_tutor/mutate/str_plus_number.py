"""TypeError: text and a number meet at a `+`.

Three ways in, all picked from the AST:
- drop a str() conversion next to a `+`:   "n=" + str(n)  -> "n=" + n
- put quotes round a number next to a `+`: total + 1      -> total + "1"
- store a number as text:                  count = 0      -> count = "0"
  (the crash then happens later, where `count` is added to a number)
"""

from __future__ import annotations

import ast
import random

from .base import MutationResult, on_one_line, replace_node, source


def apply(code: str, program_lines: int, rng: random.Random) -> MutationResult | None:
    tree = ast.parse(code)
    sites: list[tuple[ast.AST, str]] = []  # (node to replace, new text)
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            operands = [node.left, node.right]
        elif isinstance(node, ast.AugAssign) and isinstance(node.op, ast.Add):
            operands = [node.value]
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            operands = [node.value]  # only numbers qualify, see below
        else:
            continue
        for x in operands:
            if not on_one_line(x) or x.lineno > program_lines:
                continue
            if is_number(x):
                sites.append((x, f'"{source(code, x)}"'))
            elif is_str_call(x) and not isinstance(node, ast.Assign):
                sites.append((x, source(code, x.args[0])))
    if not sites:
        return None
    node, new_text = rng.choice(sites)
    return replace_node(code, node, new_text)


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
