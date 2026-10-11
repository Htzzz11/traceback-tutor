"""Stream C: automatic metrics that need no judge.

All metrics take an Example (with ground truth) and the raw model response text.
`error_type_named` and `gives_fix` are keyword/rule approximations; the LLM judge
in v1 replaces them. Their blind spots are listed in their docstrings.

None of these check the body of a response: a wrong line number, a miscount or a
wrong claim inside the explanation or hint goes unnoticed (see eval/audit_sheet.py).
"""

from __future__ import annotations

import ast
import difflib
import math
import random
import re
from collections import defaultdict

from ..prompt import parse_response
from ..schema import Example

BOOL_METRICS = (
    "follows_format",
    "line_correct",
    "line_is_bug_line",
    "error_type_named",
    "gives_fix",
)
NUMERIC_METRICS = ("n_words",)
# Reported, but too weak to compare systems on: echoing the traceback's last line scores
# True, and the keyword list favours the teacher's phrasing (see error_type_named).
WEAK_METRICS = ("error_type_named",)

# Plain-word ways of naming each exception, matched as lowercase substrings of the whole
# response. The exception's own name ("nameerror") always counts too. One TypeError list
# covers both v0 TypeError bugs (str + number, wrong number of arguments). The second
# line of each v0 list is the teacher's plain-word style, taken from train/val references
# only (never from test), so good answers in that style count.
# fmt: off
ERROR_KEYWORDS: dict[str, list[str]] = {
    "NameError": [
        "not defined", "undefined", "doesn't exist", "does not exist", "misspel", "typo",
        "spelled differently", "spelt differently", "no variable", "never created",
        "never defined", "never assigned", "unknown name",
        "doesn't know", "does not know", "different name", "spelling differs",
    ],
    "TypeError": [
        "type", "concatenat", "text and a number", "string and a number", "str and int",
        "number and text", "number to text", "number to a string", "can't add",
        "argument", "parameter", "missing", "too many", "too few", "not enough",
        "only gives", "only passes", "expects",
        "text rather than a number", "that text", "as text", "gives only", "gives it only",
        "gives no value", "needs one value", "needs two values", "needs three values",
        "needs at least", "enough values",
    ],
    "IndexError": [
        "out of range", "index", "position", "too far", "past the end", "beyond the end",
        "doesn't exist", "does not exist", "off by one", "off-by-one",
        "one more", "empty",
    ],
    "SyntaxError": [
        "syntax", "colon", "':'", "isn't finished", "not finished", "unfinished",
        "incomplete", "can't even start", "before it runs", "before running", "grammar",
        "invalid",
    ],
    "KeyError": ["key", "not in the dictionary", "not in the dict", "no entry"],
    "AttributeError": ["attribute", "no method", "doesn't have a method", "does not have"],
    "ValueError": [
        "can't be converted", "cannot be converted", "can't convert", "cannot convert",
        "not a valid number", "invalid literal", "isn't a number", "not a number",
    ],
    "UnboundLocalError": [
        "local variable", "before it", "before assign", "before a value", "before giving",
    ],
    "ZeroDivisionError": ["zero", "divide by 0", "division by 0"],
    "IndentationError": ["indent", "spaces", "line up", "lined up"],
}
# fmt: on

# Phrases that hand over the fix in words. Each is a case seen or expected in model output.
_FIX_PHRASES = [
    # "change totl to total", "replace `x` with `y`": one code-ish token on each side
    r"\b(?:change|rename|replace)\s+`?[\w.()\[\]]+`?\s+(?:to|with|by)\s+`?[\w.()\[\]'\"]",
    # "change the loop condition to `i < len(xs)`": a few words, then code in backticks
    r"\b(?:change|rename|replace)\b[^.?!\n]{0,60}?\b(?:to|with|by)\s+`",
    # "it should be `total`", "It should read: `...`", "Should it be `total`?"
    r"\bshould\s+(?:\w+\s+){0,2}(?:be|read|say)\s*:?\s*`",
    r"\bdid you mean\b",  # echoing Python's "Did you mean: 'total'?" names the fix
    # "The correct name is `count`", "The correct range should be from 0 to `len(s)`"
    r"\bcorrect\s+(?:\w+\s+){1,2}(?:is|should|would)\b[^.?!\n]{0,30}`",
    r"\bto fix (?:this|it|the error)\b|\bthe fix is\b",  # "To fix this, convert `i` ..."
]
_FIX_PHRASE_RE = re.compile("|".join(_FIX_PHRASES), re.IGNORECASE)
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)(?:```|$)", re.DOTALL)  # unclosed fence runs to end
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+\s*\|\s?")  # echoed "5 | code" from the prompt
_LABEL_RE = re.compile(r"^(?:Hint|Line\s+\d+)\s*:\s*", re.IGNORECASE)  # "Hint: <code>"
_ELLIPSIS = {"...", "\u2026"}  # "..." standing for elided code in a quoted block


def _normalise(text: str) -> str:
    return text.lower().replace("’", "'")  # curly apostrophe -> straight


def error_type_named(example: Example, response: str) -> bool:
    """Does the response name the exception, or say what it means in plain words?

    A weak sanity check (WEAK_METRICS): report it, but don't claim a difference between
    systems with it. Echoing the traceback's last line scores True, and on v0 its sign
    flipped with the keyword list (base 0.90 both times; LoRA 0.53 -> 0.98 after the
    teacher's phrasings were added).

    Approximation until the v1 LLM judge. Blind spots: a keyword can appear without
    actually explaining the error ("type" or "missing" in an unrelated sentence counts),
    and a correct paraphrase not in the list doesn't count. The list includes the
    teacher's phrasings, so it favours answers in the teacher's style. Keyed by the
    exception actually raised, so overlapping keywords between exceptions don't matter.
    """
    text = _normalise(response)
    keywords = [example.error_type.lower(), *ERROR_KEYWORDS.get(example.error_type, [])]
    return any(k in text for k in keywords)


def _code_line(line: str) -> str:
    return _LINE_NUMBER_PREFIX_RE.sub("", line).strip()


def _squash(text: str) -> str:
    return "".join(text.split())  # compare code ignoring whitespace


def _is_nontrivial_python(line: str) -> bool:
    """A statement (assignment, return, def/if/for header, ...) or an expression with a call.

    Bare names and operators are trivial, so "total" or "not defined" don't count.
    """
    source = line + "\n    pass" if line.endswith(":") else line  # let block headers parse
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.Expr):
            if any(isinstance(n, ast.Call) for n in ast.walk(node)):
                return True
        elif isinstance(node, ast.AnnAssign) and node.value is None:
            continue  # prose like "Note: x" parses as a bare annotation
        else:
            return True
    return False


def _is_edited_copy(span: str, code: str) -> bool:
    """Is `span` a program line with a small edit, i.e. corrected code in backticks?

    Spans that are copied from the program, or that only point at code (`i = 2`,
    `list(...)`, `nlargest()`), aren't close enough to any line to count.
    """
    span = _squash(span)
    if span in _squash(code):
        return False
    lines = [_squash(line) for line in code.splitlines() if line.strip()]
    return any(difflib.SequenceMatcher(None, span, line).ratio() >= 0.8 for line in lines)


def gives_fix(example: Example, response: str) -> bool:
    """Does the response hand over corrected code? (It should never.)

    True if any of these rules fires:
    1. A fenced code block contains a line that isn't copied from the program or the
       traceback, or "..." (quoting the buggy line is fine; writing new code isn't).
    2. Any response line, minus a "Hint:" / "Line N:" label, parses as non-trivial
       Python (see _is_nontrivial_python) and isn't copied from the program.
    3. Outside fences, an inline `code` span is an edited copy of a program line
       (_is_edited_copy): "write `return sm//count` instead" for `return sm//coount`.
    4. Outside fences, a fix phrase (_FIX_PHRASES): "change X to Y", "replace X with Y",
       "should be `...`", "Should it be `...`?", "did you mean", "the correct name is",
       "to fix this".

    Blind spots: a fix described in words only ("add a colon at the end of line 2",
    "use one less in the range", "convert `age` to a string"), naming the right
    identifier without a fix phrase ("`count` is defined on line 2"), new code in inline
    backticks that isn't close to a program line ("use `str(age)`"), and spelling out
    exactly which value to pass are not caught. "change it to a string" is caught, which
    is arguably right. "Missing a colon" is not a fix here: it restates Python's own
    "expected ':'" message, which the style guide allows.
    """
    program = {line.strip() for line in example.code.splitlines()}
    copied = program | {line.strip() for line in example.traceback.splitlines()} | _ELLIPSIS

    for block in _FENCE_RE.findall(response):
        for line in block.splitlines():
            line = _code_line(line)
            if line and line not in copied:
                return True
    for line in response.splitlines():
        line = _LABEL_RE.sub("", _code_line(line))
        if line and line not in program and _is_nontrivial_python(line):
            return True
    # Fences were checked above; a quoted traceback may contain Python's own "Did you mean".
    prose = _FENCE_RE.sub("", response)
    if any(_is_edited_copy(span, example.code) for span in _INLINE_CODE_RE.findall(prose)):
        return True
    return bool(_FIX_PHRASE_RE.search(prose))


def score(example: Example, response: str) -> dict[str, bool | int]:
    """Per-response metrics.

    - follows_format: parses with a "Line N:" header and a "Hint:" section
    - line_correct: the "Line N:" number is bug_line or error_line. Only the header
      counts; numbers in the body are ignored. A missing hint doesn't affect it
      (that's follows_format's job).
    - line_is_bug_line: the "Line N:" number is bug_line (stricter)
    - error_type_named: see error_type_named()
    - gives_fix: see gives_fix() (should be False)
    - n_words: whitespace-separated words in the whole response
    """
    parsed = parse_response(response)
    return {
        "follows_format": parsed.line is not None and parsed.hint is not None,
        "line_correct": parsed.line in (example.bug_line, example.error_line),
        "line_is_bug_line": parsed.line == example.bug_line,
        "error_type_named": error_type_named(example, response),
        "gives_fix": gives_fix(example, response),
        "n_words": len(response.split()),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% interval for a proportion.

    Unlike a percentile bootstrap it doesn't collapse to [1, 1] for 60/60: it gives
    [0.94, 1.00], an honest statement of what 60 examples can show.
    """
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(
    values: list[float], n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean, resampling examples with replacement.

    Used for paired differences between two systems (eval/compare.py).
    """
    rng = random.Random(seed)
    n = len(values)
    means = sorted(_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples))
    return means[round(0.025 * (n_resamples - 1))], means[round(0.975 * (n_resamples - 1))]


def aggregate(rows: list[dict]) -> dict:
    """Summarise per-example rows ({"bug_type": ..., "metrics": score(...)}).

    Returns {"n", "metrics": {name: {"mean", "count"?, "ci95"?}}, "by_bug_type": ...}.
    Boolean metrics get their count of True and a Wilson 95% CI; n_words only a mean.
    """
    if not rows:
        raise ValueError("no rows to aggregate")
    metrics: dict[str, dict] = {}
    for name in BOOL_METRICS + NUMERIC_METRICS:
        values = [float(r["metrics"][name]) for r in rows]
        metrics[name] = {"mean": _mean(values)}
        if name in BOOL_METRICS:
            count = int(sum(values))
            metrics[name]["count"] = count
            metrics[name]["ci95"] = list(wilson_ci(count, len(values)))

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r["bug_type"]].append(r)
    by_bug_type = {
        bug_type: {
            "n": len(group),
            **{
                name: _mean([float(r["metrics"][name]) for r in group])
                for name in BOOL_METRICS + NUMERIC_METRICS
            },
        }
        for bug_type, group in sorted(groups.items())
    }
    return {"n": len(rows), "metrics": metrics, "by_bug_type": by_bug_type}
