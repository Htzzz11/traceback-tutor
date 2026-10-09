"""The data contract shared by every stage: one Example per buggy program.

Every pipeline stage reads and writes JSONL files of Examples. Change this file
only on purpose, and update data/fixtures/examples.jsonl and the tests with it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Literal

Split = Literal["train", "val", "test", "test_ood"]


@dataclass(frozen=True)
class BugType:
    name: str
    expected_error: str  # exception class name the mutation should raise
    in_v0: bool  # part of the first 5-type slice


BUG_TYPES: dict[str, BugType] = {
    b.name: b
    for b in [
        BugType("misspelt_variable", "NameError", True),
        BugType("str_plus_number", "TypeError", True),
        BugType("off_by_one_index", "IndexError", True),
        BugType("wrong_arg_count", "TypeError", True),
        BugType("missing_colon", "SyntaxError", True),
        BugType("missing_dict_key", "KeyError", False),
        BugType("method_on_wrong_type", "AttributeError", False),
        BugType("bad_conversion", "ValueError", False),
        BugType("assignment_shadows_global", "UnboundLocalError", False),
        BugType("divide_by_zero", "ZeroDivisionError", False),
        BugType("unclosed_bracket", "SyntaxError", False),
        BugType("bad_indentation", "IndentationError", False),
        BugType("call_non_function", "TypeError", False),
    ]
}


@dataclass
class Example:
    id: str  # unique, e.g. "mbpp-11-misspelt_variable-0"
    seed_id: str  # source program id (e.g. MBPP task_id); splits are made on this
    bug_type: str  # key of BUG_TYPES
    code: str  # the full file that ran: buggy program + test calls appended
    traceback: str  # real traceback, file path normalised to "main.py"
    error_type: str  # exception class actually raised, e.g. "NameError"
    error_line: int  # 1-based line in `code` where the traceback says it failed
    bug_line: int  # 1-based line in `code` the mutation changed (may differ, e.g. off-by-one)
    explanation: str | None = None  # teacher: what went wrong, in plain words
    hint: str | None = None  # teacher: one guiding question, no fix
    split: Split | None = None
    teacher_model: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> Example:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown Example fields: {sorted(unknown)}")
        return cls(**d)


def read_jsonl(path: str | Path) -> Iterator[Example]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Example.from_dict(json.loads(line))


def write_jsonl(path: str | Path, examples: Iterable[Example]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(ex.to_json() + "\n")
            n += 1
    return n
