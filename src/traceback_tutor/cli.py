"""Stream C: `explain buggy.py` runs the file, catches the traceback and asks the model.

If the file doesn't crash, say so and exit. Uses sandbox.run_sandboxed and
prompt.build_messages, generates with mlx-lm, prints the parsed response.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

from .generate import GenerateFn, load, make_generate_fn
from .prompt import build_messages, parse_response
from .sandbox import run_sandboxed

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_ADAPTER = Path("adapters/v0")
TIMEOUT = 5.0


def format_explanation(code: str, response: str) -> str:
    """Pretty-print a parsed response; fall back to the raw text if it doesn't parse."""
    parsed = parse_response(response)
    if parsed.line is None or parsed.hint is None:
        return response.strip()
    code_lines = code.splitlines()
    header = f"Line {parsed.line}"
    if 1 <= parsed.line <= len(code_lines):
        header += f":  {code_lines[parsed.line - 1].strip()}"
    return "\n\n".join(
        [header, textwrap.fill(parsed.explanation, 80), textwrap.fill(f"Hint: {parsed.hint}", 80)]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="explain", description="Explain why a beginner's Python program crashed."
    )
    p.add_argument("file", type=Path, help="the Python file to run")
    p.add_argument("--model", default=DEFAULT_MODEL, help="hf id or local path")
    p.add_argument(
        "--adapter", help=f"LoRA adapter directory (default: {DEFAULT_ADAPTER} if present)"
    )
    p.add_argument("--no-adapter", action="store_true", help="use the base model")
    p.add_argument("--max-tokens", type=int, default=300)
    return p.parse_args(argv)


def main(argv: list[str] | None = None, generate_fn: GenerateFn | None = None) -> int:
    """Exit code 0 if the program ran fine, 1 if it crashed (and was explained), 2 if it was
    stopped for running too long or using too much memory."""
    args = parse_args(argv)
    code = args.file.read_text(encoding="utf-8")
    result = run_sandboxed(code, timeout=TIMEOUT)
    if result.timed_out:
        print(f"Your program didn't finish within {TIMEOUT:.0f} seconds (an endless loop?).")
        return 2
    if result.out_of_memory:
        print("Your program was stopped for using too much memory.")
        return 2
    if not result.crashed:
        print("Your program ran without errors.")
        return 0

    if generate_fn is None:
        adapter = args.adapter
        if adapter is None and not args.no_adapter:
            if DEFAULT_ADAPTER.exists():
                adapter = str(DEFAULT_ADAPTER)
            else:
                print(f"(No adapter at {DEFAULT_ADAPTER}; using the base model.)", file=sys.stderr)
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # no "Fetching 7 files" bar
        generate_fn = make_generate_fn(load(args.model, adapter), args.max_tokens)

    print(f"Your program crashed with {result.error_type}.\n")
    response = generate_fn(build_messages(code, result.traceback)).text
    print(format_explanation(code, response))
    return 1
