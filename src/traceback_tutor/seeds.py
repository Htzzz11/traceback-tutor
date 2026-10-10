"""Stream A: load correct seed programs and their test cases (MBPP, CC-BY-4.0)."""

from __future__ import annotations

import ast
import difflib
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .sandbox import run_sandboxed

MBPP = "google-research-datasets/mbpp"
MAX_LINES = 50  # v1 scope: programs of about 50 lines or fewer
# Seeds must pass well inside the sandbox's 5 s default, so a mutant's timeout only
# catches real infinite loops. All but one MBPP seed runs in < 0.5 s even under load;
# that one takes 2-5 s depending on load and would make builds non-reproducible.
SEED_TIMEOUT = 1.0
# Programs whose AST node sequences are at least this similar count as near-duplicates.
# Lower thresholds chain lookalike one-liners into huge groups (0.85: one group of 174
# seeds), which would wreck the split ratios; at 0.95 the largest group is 21.
SIMILARITY = 0.95


@dataclass
class Seed:
    seed_id: str  # e.g. "mbpp-11"
    code: str  # correct reference solution
    test_calls: list[str]  # statements that exercise the code, e.g. MBPP's assert lines

    @property
    def program_lines(self) -> int:
        return len(self.code.split("\n"))

    def with_tests(self) -> str:
        """The whole file that runs: program, a blank line, then the test calls."""
        return self.code + "\n\n" + "\n".join(self.test_calls) + "\n"


def load_seeds(limit: int | None = None, workers: int = 8) -> list[Seed]:
    """Load MBPP seed programs whose reference solution passes its own tests.

    Drop seeds that fail, use imports beyond the standard library, need MBPP's
    test_setup_code, are longer than MAX_LINES, take longer than SEED_TIMEOUT, or
    duplicate an earlier program up to identifier names. Returns seeds in task_id
    order; `limit` keeps the first n.
    """
    from datasets import load_dataset  # slow import; only needed here

    rows = [row for split in load_dataset(MBPP, "full").values() for row in split]
    rows.sort(key=lambda r: r["task_id"])

    candidates, fingerprints = [], set()
    for row in rows:
        if row["test_setup_code"].strip():
            continue
        code = clean_code(row["code"])
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        fp = fingerprint(tree)
        if len(code.splitlines()) > MAX_LINES or not stdlib_only(tree) or fp in fingerprints:
            continue
        fingerprints.add(fp)
        tests = [t.strip() for t in row["test_list"]]
        candidates.append(Seed(f"mbpp-{row['task_id']}", code, tests))

    with ThreadPoolExecutor(workers) as pool:
        results = list(
            pool.map(lambda s: run_sandboxed(s.with_tests(), timeout=SEED_TIMEOUT), candidates)
        )
    passed = [not (r.crashed or r.timed_out or r.out_of_memory) for r in results]
    seeds = [s for s, ok in zip(candidates, passed) if ok]
    return seeds[:limit]


def clean_code(code: str) -> str:
    """Normalise MBPP formatting: CRLF line ends, tab indents, trailing spaces.

    Safe because every seed is re-run after cleaning and dropped if it now fails.
    """
    lines = code.replace("\r\n", "\n").expandtabs(4).split("\n")
    return "\n".join(line.rstrip() for line in lines).strip("\n")


def stdlib_only(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        if any(n.split(".")[0] not in sys.stdlib_module_names for n in names):
            return False
    return True


def fingerprint(tree: ast.AST) -> str:
    """AST dump with every identifier blanked, so renamed copies of a program match."""
    tree = ast.parse(ast.unparse(tree))  # fresh copy; we edit it in place
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = "_"
        elif isinstance(node, ast.arg):
            node.arg = "_"
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            node.name = "_"
    return ast.dump(tree)


def near_duplicate_groups(seeds: list[Seed]) -> dict[str, str]:
    """Map each seed_id to a group id (the group's first seed_id) for splitting.

    MBPP has many near-copies, e.g. the same program with nsmallest/nlargest, == or !=,
    'a'/'z' or 'A'/'Z'. Two programs share a group if they have the same shape(), or if
    their AST node sequences are at least SIMILARITY alike; groups are joined
    transitively. Splitting by group keeps every near-copy on one side of the split.
    """
    trees = {s.seed_id: ast.parse(s.code) for s in seeds}
    ids = list(trees)
    position = {i: n for n, i in enumerate(ids)}
    parent = {i: i for i in ids}  # union-find; each group's root is its first seed

    def root(i: str) -> str:
        while parent[i] != i:
            i = parent[i]
        return i

    def join(a: str, b: str) -> None:
        ra, rb = sorted((root(a), root(b)), key=position.get)
        parent[rb] = ra

    first_with_shape: dict[str, str] = {}
    for i in ids:
        join(i, first_with_shape.setdefault(shape(trees[i]), i))

    types = {i: node_types(trees[i]) for i in ids}
    for n, a in enumerate(ids):
        matcher = difflib.SequenceMatcher(None, b=types[a], autojunk=False)  # caches b
        for b in ids[n + 1 :]:
            matcher.set_seq1(types[b])
            # the quick upper bounds skip most pairs before the slow exact ratio
            if (
                matcher.real_quick_ratio() >= SIMILARITY
                and matcher.quick_ratio() >= SIMILARITY
                and matcher.ratio() >= SIMILARITY
            ):
                join(a, b)
    return {i: root(i) for i in ids}


def shape(tree: ast.AST) -> str:
    """AST dump with names, attributes, constants and operators blanked and imports dropped."""
    tree = ast.parse(ast.unparse(tree))  # fresh copy; we edit it in place
    for node in ast.walk(tree):
        if isinstance(getattr(node, "body", None), list):
            imports = (ast.Import, ast.ImportFrom)
            node.body = [s for s in node.body if not isinstance(s, imports)] or [ast.Pass()]
        if isinstance(node, ast.Name):
            node.id = "_"
        elif isinstance(node, ast.arg):
            node.arg = "_"
        elif isinstance(node, ast.FunctionDef | ast.ClassDef):
            node.name = "_"
        elif isinstance(node, ast.Attribute):
            node.attr = "_"
        elif isinstance(node, ast.keyword):
            node.arg = "_"
        elif isinstance(node, ast.Constant):
            node.value = "_"
        elif isinstance(node, ast.Compare):
            node.ops = [ast.Eq() for _ in node.ops]
        elif isinstance(node, ast.BinOp | ast.AugAssign):
            node.op = ast.Add()
        elif isinstance(node, ast.BoolOp):
            node.op = ast.And()
        elif isinstance(node, ast.UnaryOp):
            node.op = ast.Not()
    return ast.dump(tree)


def node_types(tree: ast.AST) -> list[str]:
    """AST node class names in source order, without imports or Load/Store markers."""
    out = []

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Import | ast.ImportFrom | ast.expr_context):
            return
        out.append(type(node).__name__)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return out
