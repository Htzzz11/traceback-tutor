"""Hand-audit sheet for a run: what the automatic metrics can't check.

The metrics only look at the "Line N:" header and a few fix patterns. This writes a
Markdown checklist per response (code, traceback, response, and the code of every line
the response mentions) to tick by hand, then counts the ticks.

Make:      uv run python -m traceback_tutor.eval.audit_sheet runs/<name>
Summarise: uv run python -m traceback_tutor.eval.audit_sheet --summarise runs/<name>/audit.md
"""

from __future__ import annotations

import argparse
import json
import re
from itertools import pairwise
from pathlib import Path

from ..prompt import number_lines
from ..schema import read_jsonl
from .metrics import wilson_ci

# Each check is ticked when it is TRUE for the response.
CHECKS = {
    "line_refs_ok": "every line number in the text points at what the text says is there",
    "facts_ok": "no wrong claim about the code, its values (counts, lengths) or Python",
    "no_fix": "doesn't give away the fix, in code or in words (incl. the exact value to "
    "pass); pointing at Python's own message, like expected ':', is fine",
    "hint_helps": "the hint is one question that would nudge a beginner toward the fix",
}

_LINE_MENTION_RE = re.compile(r"\blines?\s+(\d+(?:\s*(?:,|and|or|to)\s*\d+)*)", re.IGNORECASE)
_SECTION_RE = re.compile(r"^## \d+\. (\S+)", re.MULTILINE)
_BOX_RE = re.compile(r"^- \[([ xX])\] (\w+):", re.MULTILINE)


def mentioned_lines(response: str) -> list[int]:
    """Line numbers a response mentions: "Line 5:", "line 3", "lines 9 and 12"."""
    found = set()
    for m in _LINE_MENTION_RE.finditer(response):
        found.update(int(n) for n in re.findall(r"\d+", m.group(1)))
    return sorted(found)


def make_sheet(run_dir: Path) -> str:
    config = json.loads((run_dir / "metrics.json").read_text())["config"]
    examples = {ex.id: ex for ex in read_jsonl(config["data"])}
    rows = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text().splitlines()]

    out = [
        f"# Audit: {config['name']}",
        "",
        (
            f"Model `{config['model']}`, adapter `{config['adapter']}`, {len(rows)} responses "
            f"from `{config['data']}` (split {config['split']})."
        ),
        "",
        (
            "Tick `audited` once you've read a response; only audited ones are counted. "
            "Tick each other box when the statement is TRUE:"
        ),
        "",
        *[f"- **{name}**: {meaning}" for name, meaning in CHECKS.items()],
        "",
        "Automatic scores are left out on purpose, so they don't steer the reading.",
    ]
    for i, row in enumerate(rows, start=1):
        ex = examples[row["id"]]
        code_lines = ex.code.splitlines()
        mentions = [
            f"- {n} → `{code_lines[n - 1].strip()}`"
            if 1 <= n <= len(code_lines)
            else f"- {n} → (no such line)"
            for n in mentioned_lines(row["response"])
        ]
        out += [
            "",
            "---",
            "",
            f"## {i}. {ex.id}",
            "",
            (
                f"{ex.bug_type}, {ex.error_type}. Bug on line {ex.bug_line}, "
                f"Python fails on line {ex.error_line}."
            ),
            "",
            "```",
            number_lines(ex.code),
            "```",
            "",
            "```",
            ex.traceback.rstrip(),
            "```",
            "",
            "**Response**",
            "",
            *[f"> {line}" for line in row["response"].strip().splitlines()],
            "",
            "**Lines it mentions**",
            "",
            *(mentions or ["- (none)"]),
            "",
            "- [ ] audited:",
            *[f"- [ ] {name}:" for name in CHECKS],
            "",
            "Notes:",
        ]
    return "\n".join(out) + "\n"


def summarise(sheet: str) -> dict:
    """Count ticked checks over the audited sections."""
    starts = [m.start() for m in _SECTION_RE.finditer(sheet)] + [len(sheet)]
    audited = []
    for start, end in pairwise(starts):
        section = sheet[start:end]
        boxes = {name: tick != " " for tick, name in _BOX_RE.findall(section)}
        if boxes.get("audited"):
            audited.append((_SECTION_RE.match(section).group(1), boxes))
    n = len(audited)
    result = {"n_sections": len(starts) - 1, "n_audited": n, "checks": {}}
    for name in CHECKS:
        count = sum(boxes.get(name, False) for _, boxes in audited)
        failed = [id_ for id_, boxes in audited if not boxes.get(name, False)]
        result["checks"][name] = {
            "count": count,
            "ci95": list(wilson_ci(count, n)) if n else None,
            "failed": failed,
        }
    return result


def format_summary(result: dict) -> str:
    n = result["n_audited"]
    lines = [f"audited {n}/{result['n_sections']}"]
    for name, c in result["checks"].items():
        ci = f"[{c['ci95'][0]:.2f}, {c['ci95'][1]:.2f}]" if c["ci95"] else ""
        lines.append(f"{name:<12} {c['count']:>3}/{n:<3} {ci}")
        lines += [f"    failed: {id_}" for id_ in c["failed"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("path", type=Path, help="run directory, or the sheet with --summarise")
    p.add_argument("--summarise", action="store_true", help="count the ticks in a filled sheet")
    p.add_argument("--overwrite", action="store_true", help="replace an existing sheet")
    args = p.parse_args(argv)
    if args.summarise:
        print(format_summary(summarise(args.path.read_text(encoding="utf-8"))))
        return
    out = args.path / "audit.md"
    if out.exists() and not args.overwrite:
        raise SystemExit(f"{out} already exists (it may hold your ticks); pass --overwrite")
    out.write_text(make_sheet(args.path), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
