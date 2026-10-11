"""Stream B: convert split Examples into mlx-lm chat JSONL (train.jsonl / valid.jsonl).

Each line: {"messages": [system, user, assistant]} built with prompt.build_messages
and prompt.format_response(bug_line, explanation, hint). mlx-lm (0.32) reads
{train,valid,test}.jsonl from the --data directory; train with mask_prompt so the
loss is only on the assistant turn.

Only the train and val splits are written. test and test_ood never go into the
training directory: evaluation reads those Examples directly.

Usage:
    uv run python -m traceback_tutor.to_chat --in data/v0/labelled.jsonl --out data/v0/chat
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from .prompt import build_messages, format_response
from .schema import Example, read_jsonl

SPLIT_FILES = {"train": "train.jsonl", "val": "valid.jsonl"}


def to_chat_row(example: Example) -> dict:
    if example.explanation is None or example.hint is None:
        raise ValueError(f"{example.id} has no teacher explanation/hint")
    target = format_response(example.bug_line, example.explanation, example.hint)
    messages = build_messages(example.code, example.traceback)
    return {"messages": [*messages, {"role": "assistant", "content": target}]}


def export(examples: Iterable[Example], out_dir: Path) -> dict[str, int]:
    """Write train.jsonl and valid.jsonl into out_dir. Returns rows written per file."""
    rows: dict[str, list[dict]] = {name: [] for name in SPLIT_FILES.values()}
    for ex in examples:
        name = SPLIT_FILES.get(ex.split)
        if name is not None:
            rows[name].append(to_chat_row(ex))
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in rows.items():
        with open(out_dir / name, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in lines)
    return {name: len(lines) for name, lines in rows.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export labelled Examples to mlx-lm chat JSONL.")
    parser.add_argument("--in", dest="in_paths", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True, help="directory for train/valid.jsonl")
    args = parser.parse_args(argv)
    examples = [ex for path in args.in_paths for ex in read_jsonl(path)]
    counts = export(examples, args.out)
    print(f"{len(examples)} examples -> {counts} in {args.out}")


if __name__ == "__main__":
    main()
