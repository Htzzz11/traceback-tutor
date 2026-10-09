"""Stream C: automatic metrics that need no judge.

All metrics take an Example (with ground truth) and the raw model response text.
"""

from __future__ import annotations

from ..schema import Example


def score(example: Example, response: str) -> dict[str, float | int | bool]:
    """Return per-response metrics:

    - follows_format: response parses with a line number and a hint
    - line_correct: cited line == bug_line or error_line
    - error_type_named: the response names the exception or its plain-word meaning
    - gives_fix: response contains corrected code (should be False)
    - n_words: length of the response
    """
    raise NotImplementedError


def aggregate(rows: list[dict]) -> dict[str, float]:
    """Mean of each metric, plus bootstrap 95% CI for the boolean ones."""
    raise NotImplementedError
