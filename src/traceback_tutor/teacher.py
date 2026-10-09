"""Stream B: a frontier model writes the target explanation and hint for each Example.

The style guide lives in docs/style-guide.md. The teacher must answer in the
prompt.py response format; parse it with prompt.parse_response.
"""

from __future__ import annotations

from .schema import Example


def explain(example: Example) -> Example:
    """Return a copy of `example` with explanation, hint and teacher_model filled in."""
    raise NotImplementedError


def passes_filters(example: Example) -> tuple[bool, str]:
    """Automatic quality filters. Returns (ok, reason).

    v0 filters: cited line is bug_line or error_line; no code block or corrected
    line of code in the response; explanation under ~80 words; hint is a question.
    """
    raise NotImplementedError
