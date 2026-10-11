"""Stream C: generate responses for a split with a given model and score them.

Run: uv run python -m traceback_tutor.eval.run_eval --model <hf id or path> \
    [--adapter adapters/v0] --data <examples.jsonl> [--split test] --name <run name>
Writes runs/<name>/responses.jsonl and runs/<name>/metrics.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from ..generate import GENERATION_SETTINGS, GenerateFn, load, make_generate_fn
from ..prompt import build_messages
from ..schema import Example, read_jsonl
from .metrics import BOOL_METRICS, NUMERIC_METRICS, WEAK_METRICS, aggregate, score


def select_examples(
    data: str | Path, split: str | None = None, limit: int | None = None
) -> list[Example]:
    """Examples in file order, filtered to `split` if given, then the first `limit`."""
    examples = [ex for ex in read_jsonl(data) if split is None or ex.split == split]
    return examples[:limit] if limit is not None else examples


def run(examples: list[Example], generate_fn: GenerateFn, out_dir: Path, config: dict) -> dict:
    """Generate and score every example, write responses.jsonl and metrics.json, return the latter."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out_dir / "responses.jsonl", "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples, start=1):
            gen = generate_fn(build_messages(ex.code, ex.traceback))
            row = {
                "id": ex.id,
                "bug_type": ex.bug_type,
                "bug_line": ex.bug_line,
                "error_line": ex.error_line,
                "response": gen.text,
                "metrics": score(ex, gen.text),
                "seconds": gen.seconds,
                "n_tokens": gen.n_tokens,
                "finish_reason": gen.finish_reason,
            }
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(examples)}] {ex.id}: {gen.n_tokens} tokens in {gen.seconds:.1f}s")
    return write_summary(rows, config, out_dir)


def write_summary(rows: list[dict], config: dict, out_dir: Path) -> dict:
    """Aggregate scored rows, add speed stats and config, and write metrics.json."""
    total_seconds = sum(r["seconds"] for r in rows)
    total_tokens = sum(r["n_tokens"] for r in rows)
    result = {
        "config": {**config, "n": len(rows)},
        **aggregate(rows),
        "speed": {
            "seconds_per_response": total_seconds / len(rows),
            "tokens_per_response": total_tokens / len(rows),
            "tokens_per_second": total_tokens / total_seconds if total_seconds else None,
            # Hit max_tokens: these responses are cut off and may lose their hint.
            "truncated_rate": sum(r["finish_reason"] == "length" for r in rows) / len(rows),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def format_table(result: dict) -> str:
    n = result["n"]
    lines = [f"n = {n}", f"{'metric':<18} {'count':>7} {'mean':>6}  95% CI (Wilson)"]
    for name, m in result["metrics"].items():
        count = f"{m['count']}/{n}" if "count" in m else ""
        ci = f"[{m['ci95'][0]:.2f}, {m['ci95'][1]:.2f}]" if "ci95" in m else ""
        weak = "  weak, don't compare" if name in WEAK_METRICS else ""
        lines.append(f"{name:<18} {count:>7} {m['mean']:>6.2f}  {ci}{weak}")

    names = BOOL_METRICS + NUMERIC_METRICS
    lines += ["", "by bug type (means)", f"{'bug_type':<20} {'n':>3} " + " ".join(names)]
    for bug_type, m in result["by_bug_type"].items():
        cells = " ".join(f"{m[name]:>{len(name)}.2f}" for name in names)
        lines.append(f"{bug_type:<20} {m['n']:>3} {cells}")

    s = result["speed"]
    speed = (
        f"{s['seconds_per_response']:.2f} s/response, "
        f"{s['tokens_per_response']:.0f} tokens/response, "
        f"{s['tokens_per_second']:.1f} tokens/s, truncated {s['truncated_rate']:.0%}"
    )
    lines += ["", speed]
    return "\n".join(lines)


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_info() -> dict:
    status = _git("status", "--porcelain")
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def build_config(args: argparse.Namespace) -> dict:
    return {
        "name": args.name,
        "model": args.model,
        "adapter": args.adapter,
        "data": str(args.data),
        "data_sha256": file_sha256(args.data),
        "split": args.split,
        "limit": args.limit,
        "generation": {**GENERATION_SETTINGS, "max_tokens": args.max_tokens},
        **git_info(),
    }


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, help="hf id or local path")
    p.add_argument("--adapter", help="LoRA adapter directory")
    p.add_argument("--data", required=True, type=Path, help="Examples JSONL")
    p.add_argument("--split", help="keep only examples with this split")
    p.add_argument("--name", required=True, help="run name; output goes to <runs-dir>/<name>/")
    p.add_argument("--limit", type=int, help="only the first N examples")
    p.add_argument("--max-tokens", type=int, default=300)
    p.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p.add_argument("--overwrite", action="store_true", help="replace an existing run")
    return p.parse_args(argv)


def main(argv: list[str] | None = None, generate_fn: GenerateFn | None = None) -> dict:
    """CLI entry point; pass `generate_fn` to run without loading a real model (tests)."""
    args = parse_args(argv)
    out_dir = args.runs_dir / args.name
    if out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} already exists; pick another --name or pass --overwrite")
    examples = select_examples(args.data, args.split, args.limit)
    if not examples:
        raise SystemExit(f"no examples in {args.data} with split={args.split!r}")
    config = build_config(args)
    if generate_fn is None:
        generate_fn = make_generate_fn(load(args.model, args.adapter), args.max_tokens)
    result = run(examples, generate_fn, out_dir, config)
    print(format_table(result))
    print(f"\nwrote {out_dir}/responses.jsonl and {out_dir}/metrics.json")
    return result


if __name__ == "__main__":
    main()
