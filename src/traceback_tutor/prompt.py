"""The model-facing contract: what the model sees and what it must write back.

Training data, the teacher, evaluation and the CLI all go through these
functions, so the format stays identical everywhere.

Response format (the assistant turn), exactly:

    Line <N>: <explanation in plain words>

    Hint: <one guiding question>
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SYSTEM_PROMPT = (
    "You are a patient Python tutor for complete beginners. You are given a short "
    "program and the error it raised. Explain what went wrong in plain words, say "
    "which line is responsible, and give one hint as a question. Never write the "
    "corrected code.\n\n"
    "Answer in exactly this format:\n"
    "Line <number>: <what went wrong, 1-3 sentences>\n\n"
    "Hint: <one question that nudges the student toward the fix>"
)


def number_lines(code: str) -> str:
    """Prefix each line with its 1-based number so a small model can cite lines."""
    lines = code.splitlines()
    width = len(str(len(lines)))
    return "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(lines, start=1))


def build_user_message(code: str, traceback: str) -> str:
    return f"Program:\n```\n{number_lines(code)}\n```\n\nError:\n```\n{traceback.rstrip()}\n```"


def build_messages(code: str, traceback: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(code, traceback)},
    ]


def format_response(line: int, explanation: str, hint: str) -> str:
    return f"Line {line}: {explanation.strip()}\n\nHint: {hint.strip()}"


@dataclass
class ParsedResponse:
    line: int | None  # None if the model didn't follow the format
    explanation: str
    hint: str | None


_LINE_RE = re.compile(r"^\s*Line\s+(\d+)\s*:\s*", re.IGNORECASE)
_HINT_RE = re.compile(r"^\s*Hint\s*:\s*", re.IGNORECASE | re.MULTILINE)


def parse_response(text: str) -> ParsedResponse:
    """Lenient parse of a model response; never raises."""
    text = text.strip()
    hint = None
    m = _HINT_RE.search(text)
    if m:
        hint = text[m.end() :].strip() or None
        text = text[: m.start()].strip()
    line = None
    m = _LINE_RE.match(text)
    if m:
        line = int(m.group(1))
        text = text[m.end() :].strip()
    return ParsedResponse(line=line, explanation=text, hint=hint)
