"""TypeError: a call passes one argument too few or too many.

  area(4, 3)       -> area(4)          (drop an argument)
  xs.append(x)     -> xs.append()
  area(4, 3)       -> area(4, 3, 3)    (repeat the last argument)

Any call qualifies: the student's own functions, builtins and methods. A test call
(`assert area(4) == 12`) is picked only TEST_CALL_SHARE of the time: those always
crash cleanly, so without a cap they would crowd out bugs inside the program.
Calls that still run, or fail for another reason, are dropped later by
build_dataset's error check.
"""

from __future__ import annotations

import ast
import random

from .base import MutationResult, on_one_line, replace_node, source

TEST_CALL_SHARE = 0.1


def apply(code: str, program_lines: int, rng: random.Random) -> MutationResult | None:
    tree = ast.parse(code)
    sites = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and on_one_line(node)
        and not any(isinstance(a, ast.Starred) for a in node.args)
        and all(kw.arg is not None for kw in node.keywords)  # no **kwargs
    ]
    in_program = [c for c in sites if c.lineno <= program_lines]
    in_tests = [c for c in sites if c.lineno > program_lines]
    if not in_program or (in_tests and rng.random() < TEST_CALL_SHARE):
        sites = in_tests
    else:
        sites = in_program
    if not sites:
        return None
    call = rng.choice(sites)
    args = [source(code, a) for a in call.args]
    if rng.random() < 0.5:
        del args[rng.randrange(len(args))]
    else:
        args.append(args[-1])
    args += [source(code, kw) for kw in call.keywords]
    return replace_node(code, call, f"{source(code, call.func)}({', '.join(args)})")
