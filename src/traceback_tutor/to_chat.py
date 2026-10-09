"""Stream B: convert split Examples into mlx-lm chat JSONL (train.jsonl / valid.jsonl / test.jsonl).

Each line: {"messages": [system, user, assistant]} built with prompt.build_messages
and prompt.format_response(bug_line, explanation, hint). Train with mlx-lm's
--mask-prompt so loss is only on the assistant turn.
"""

from __future__ import annotations

from pathlib import Path


def export(examples_dir: Path, out_dir: Path) -> None:
    raise NotImplementedError
