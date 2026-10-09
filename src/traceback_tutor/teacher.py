"""Stream B: a frontier model writes the target explanation and hint for each Example.

The style guide lives in docs/style-guide.md. The teacher answers in the
prompt.py response format and we parse it with prompt.parse_response.

Every request is a fixed prefix (system prompt + style guide + 3 worked examples,
identical on every call so OpenAI can cache it) followed by the per-example part.
The teacher is labelling bugs we made on purpose, so it is told the true bug line
and error type instead of having to diagnose them.

Usage:
    uv run --env-file .env python -m traceback_tutor.teacher \
        --in data/v0/examples.jsonl --out data/v0/labelled.jsonl --limit 10
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from openai import OpenAI

from .prompt import SYSTEM_PROMPT, build_user_message, format_response, parse_response
from .schema import Example, read_jsonl

MODEL = "gpt-6.1-sol"
REASONING_EFFORT = "low"  # lowest effort gpt-6.1-sol supports ("none"/"minimal" are not)
MAX_OUTPUT_TOKENS = 300  # includes reasoning tokens, which are billed as output
MAX_RETRIES = 8  # the SDK retries rate limits and server errors with exponential backoff

REPO = Path(__file__).resolve().parents[2]
STYLE_GUIDE = REPO / "docs" / "style-guide.md"
FIXTURES = REPO / "data" / "fixtures" / "examples.jsonl"
# One same-line NameError, one off-by-one where bug_line != error_line, one SyntaxError.
FEW_SHOT_IDS = (
    "fixture-1-misspelt_variable-0",
    "fixture-3-off_by_one_index-0",
    "fixture-5-missing_colon-0",
)

# US dollars per 1M tokens for gpt-6.1-sol.
PRICE_INPUT = 2.00
PRICE_CACHED_INPUT = 0.10
PRICE_CACHE_WRITE = 2.50
PRICE_OUTPUT = 10.00

MAX_WORDS = 80


# ---------------------------------------------------------------- request


def build_task(example: Example) -> str:
    """The per-example part of the request: program, traceback and the known labels."""
    if example.bug_line == example.error_line:
        where = f"The bug is on line {example.bug_line}, the same line Python reported."
    else:
        where = (
            f"The bug is on line {example.bug_line}, but Python reported the error on "
            f"line {example.error_line}. Mention both lines."
        )
    return (
        f"{build_user_message(example.code, example.traceback)}\n\n"
        f"Known facts: {where} The error type is {example.error_type}. "
        f"Start your answer with 'Line {example.bug_line}:'."
    )


def build_prefix(style_guide: str, shots: list[Example]) -> str:
    """The fixed part of every request. Must not depend on the example being labelled."""
    parts = [
        SYSTEM_PROMPT,
        (
            "You are writing the reference answers a small model will be trained to imitate. "
            "Follow this style guide exactly."
        ),
        style_guide.strip(),
        "Here are worked examples. Each shows the task you are given and the answer to write.",
    ]
    for i, shot in enumerate(shots, start=1):
        answer = format_response(shot.bug_line, shot.explanation, shot.hint)
        parts.append(f"## Worked example {i}\n\n{build_task(shot)}\n\nAnswer:\n{answer}")
    return "\n\n".join(parts)


def load_prefix() -> str:
    shots = {ex.id: ex for ex in read_jsonl(FIXTURES)}
    return build_prefix(STYLE_GUIDE.read_text(encoding="utf-8"), [shots[i] for i in FEW_SHOT_IDS])


def build_request(prefix: str, example: Example, model: str = MODEL) -> dict:
    """Keyword arguments for client.responses.create: fixed prefix first, example last."""
    return {
        "model": model,
        "input": [
            {
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": prefix,
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            },
            {"role": "user", "content": build_task(example)},
        ],
        "reasoning": {"effort": REASONING_EFFORT},
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }


# ---------------------------------------------------------------- usage and cost


@dataclass
class Usage:
    input: int = 0  # all input tokens, including cached and cache-write ones
    cached_input: int = 0
    cache_write: int = 0
    output: int = 0  # all output tokens, including reasoning ones
    reasoning: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(*(a + b for a, b in zip(asdict(self).values(), asdict(other).values())))

    def cost(self) -> float:
        uncached = self.input - self.cached_input - self.cache_write
        dollars = (
            uncached * PRICE_INPUT
            + self.cached_input * PRICE_CACHED_INPUT
            + self.cache_write * PRICE_CACHE_WRITE
            + self.output * PRICE_OUTPUT
        )
        return dollars / 1_000_000


def usage_from_response(response) -> Usage:
    u = response.usage
    if u is None:
        return Usage()
    inp = u.input_tokens_details
    out = u.output_tokens_details
    return Usage(
        input=u.input_tokens,
        cached_input=getattr(inp, "cached_tokens", 0) or 0,
        cache_write=getattr(inp, "cache_write_tokens", 0) or 0,
        output=u.output_tokens,
        reasoning=getattr(out, "reasoning_tokens", 0) or 0,
    )


# ---------------------------------------------------------------- one call


@dataclass
class TeacherResult:
    example: Example  # with explanation, hint and teacher_model filled in
    response: str  # raw text, kept so the filters and humans can check it
    status: str  # "completed", or "incomplete" if it hit max_output_tokens
    usage: Usage


def explain(client: OpenAI, prefix: str, example: Example, model: str = MODEL) -> TeacherResult:
    """Ask the teacher for one Example's explanation and hint."""
    response = client.responses.create(**build_request(prefix, example, model))
    text = response.output_text or ""
    parsed = parse_response(text)
    labelled = replace(
        example, explanation=parsed.explanation, hint=parsed.hint, teacher_model=model
    )
    return TeacherResult(labelled, text, response.status, usage_from_response(response))


# ---------------------------------------------------------------- filters

_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
_SENTENCE_RE = re.compile(r"[^.?!\n]+")
# Ways of spelling out the edit: "change `totl` to `total`", "it should be `total`",
# "`total` instead", or a sentence that starts with an edit command ("Add a colon ...").
_FIX_RES = [
    re.compile(r"\b(change|replace|rename|swap)\b.*\b(to|with|into)\b.*`", re.IGNORECASE),
    re.compile(r"\b(change|replace|rename|swap)\b.*`.*\b(to|with|into)\b", re.IGNORECASE),
    re.compile(r"\bshould\s+(be|say|read)\s+`", re.IGNORECASE),
    re.compile(r"`\s+instead\b", re.IGNORECASE),
    re.compile(r"^\s*(add|insert|remove|delete|put|use|change|replace|rename)\b", re.IGNORECASE),
]


def _looks_like_rewrite(span: str, code: str, bug_src: str) -> bool:
    """Is this code span new code (a fix) rather than a quote of the student's code?

    New code either calls a function the student never calls (`str(age)`), or is
    mostly the bug line but isn't in the student's code (`return total`).
    """
    span = span.strip()
    if span in code:
        return False
    if any(f"{name}(" not in code for name in re.findall(r"\b(\w+)\(", span)):
        return True
    if len(span) < 6:
        return False
    matcher = difflib.SequenceMatcher(None, span, bug_src, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return matched / len(span) >= 0.8


def passes_filters(example: Example, response: str) -> tuple[bool, str]:
    """Automatic quality filters on a raw teacher response. Returns (ok, reason).

    Rejects: no "Line N:" prefix; cited line is neither bug_line nor error_line;
    a code block or a corrected line of code; spelling out the edit; explanation
    over MAX_WORDS words; missing hint, more than one hint, or a hint that isn't a question.
    """
    parsed = parse_response(response)
    if parsed.line is None:
        return False, "bad_format"
    if parsed.line not in (example.bug_line, example.error_line):
        return False, "wrong_line"
    if "```" in response or re.search(r"^( {4}|\t)\S", response, re.MULTILINE):
        return False, "code_block"
    code_lines = example.code.splitlines()
    bug_src = (
        code_lines[example.bug_line - 1].strip() if example.bug_line <= len(code_lines) else ""
    )
    if any(_looks_like_rewrite(s, example.code, bug_src) for s in _CODE_SPAN_RE.findall(response)):
        return False, "corrected_code"
    sentences = _SENTENCE_RE.findall(f"{parsed.explanation}\n{parsed.hint or ''}")
    if any(r.search(s) for s in sentences for r in _FIX_RES):
        return False, "gives_fix"
    if len(parsed.explanation.split()) > MAX_WORDS:
        return False, "too_long"
    if not parsed.hint:
        return False, "no_hint"
    if re.search(r"\bhint\s*:", parsed.hint, re.IGNORECASE):
        return False, "multiple_hints"
    if not parsed.hint.endswith("?"):
        return False, "hint_not_question"
    return True, "ok"


# ---------------------------------------------------------------- CLI


def _ids_in(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--in", dest="in_path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="labelled Examples that pass")
    parser.add_argument("--limit", type=int, default=None, help="max new API calls this run")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)

    # Every call (passed or rejected) is logged next to --out, so a rerun never pays twice.
    log_path = args.out.with_name(args.out.stem + ".log.jsonl")
    done = _ids_in(args.out) | _ids_in(log_path)
    todo = [ex for ex in read_jsonl(args.in_path) if ex.id not in done][: args.limit]
    print(f"{len(done)} already done, {len(todo)} to label with {args.model}")

    client = OpenAI(max_retries=MAX_RETRIES)
    prefix = load_prefix()
    total, reasons = Usage(), Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as out, open(log_path, "a", encoding="utf-8") as log:
        for ex in todo:
            result = explain(client, prefix, ex, args.model)
            if result.status != "completed":
                ok, reason = False, "incomplete"
            else:
                ok, reason = passes_filters(ex, result.response)
            reasons[reason] += 1
            total += result.usage
            log.write(
                json.dumps(
                    {
                        "id": ex.id,
                        "ok": ok,
                        "reason": reason,
                        "status": result.status,
                        "response": result.response,
                        "usage": asdict(result.usage),
                    }
                )
                + "\n"
            )
            log.flush()
            if ok:
                out.write(result.example.to_json() + "\n")
                out.flush()
            u = result.usage
            print(
                f"{ex.id}: {reason} (in {u.input}, cached {u.cached_input}, "
                f"out {u.output}, reasoning {u.reasoning})"
            )

    print(f"\nkept {reasons['ok']} / {len(todo)}")
    for reason, n in reasons.most_common():
        if reason != "ok":
            print(f"  rejected {reason}: {n}")
    print(
        f"tokens: input {total.input} (cached {total.cached_input}, cache write "
        f"{total.cache_write}), output {total.output} (reasoning {total.reasoning})"
    )
    print(f"cost: ${total.cost():.4f}")


if __name__ == "__main__":
    main()
