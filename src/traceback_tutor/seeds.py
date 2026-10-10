"""Stream A: load correct seed programs and their test cases (MBPP, CC-BY-4.0)."""

from __future__ import annotations

import ast
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
