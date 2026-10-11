"""TypeError: a call passes one argument too few or too many.

  area(4, 3)            -> area(4)             (leave an argument out)
  text.replace("a", "") -> text.replace("a")
  helper(xs, n)         -> helper(xs, n, n)    (one too many: own functions only)

An argument is left out only of a call that has two or more, or of a call to the
student's own function: `len()` or `range()` with nothing inside isn't a mistake
learners make. The argument left out is short (at most MAX_DROPPED_CHARS), never a
whole generator expression. "One too many" is only used on calls to the student's
own functions inside the program: `len(word, word)` isn't a learner's mistake either. A test call (`assert area(4) == 12`) is picked only
TEST_CALL_SHARE of the time, and never just because the program has no call of its
own: test calls always crash cleanly, so they would crowd out bugs in the program.
Calls that still run, or fail for another reason, are dropped later by
build_dataset's error check.
"""

from __future__ import annotations

import ast
import random

from .base import MutationResult, on_one_line, replace_node, source

# Test calls always crash cleanly, so they survive the error check far more often than
# calls in the program: 0.05 here ended up as 40% of examples, 0.02 gives about 20%.
TEST_CALL_SHARE = 0.02
MAX_DROPPED_CHARS = 20


def apply(code: str, program_lines: int, rng: random.Random) -> MutationResult | None:
    tree = ast.parse(code)
    own_functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    def is_own(call: ast.Call) -> bool:
        return isinstance(call.func, ast.Name) and call.func.id in own_functions

    def droppable(call: ast.Call) -> list[int]:
        return [i for i, a in enumerate(call.args) if len(source(code, a)) <= MAX_DROPPED_CHARS]

    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and on_one_line(node)
        and not any(isinstance(a, ast.Starred) for a in node.args)
        and all(kw.arg is not None for kw in node.keywords)  # no **kwargs
        and (is_own(node) or len(node.args) >= 2)
        and droppable(node)
    ]
    in_program = [c for c in sites if c.lineno <= program_lines]
    in_tests = [c for c in sites if c.lineno > program_lines]
    sites = in_tests if in_tests and rng.random() < TEST_CALL_SHARE else in_program
    if not sites:
        return None
    call = rng.choice(sites)
    args = [source(code, a) for a in call.args]
    if is_own(call) and call.lineno <= program_lines and rng.random() < 0.5:
        args.append(args[-1])
    else:
        del args[rng.choice(droppable(call))]
    args += [source(code, kw) for kw in call.keywords]
    return replace_node(code, call, f"{source(code, call.func)}({', '.join(args)})")
