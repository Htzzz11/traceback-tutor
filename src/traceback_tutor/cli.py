"""Stream C: `explain buggy.py` runs the file, catches the traceback and asks the model.

If the file doesn't crash, say so and exit. Uses sandbox.run_sandboxed and
prompt.build_messages, generates with mlx-lm, prints the parsed response.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from . import sandbox
from .generate import GenerateFn, load, make_generate_fn
from .prompt import build_messages, parse_response
from .sandbox import RunResult

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEFAULT_ADAPTER = Path("adapters/v0")
TIMEOUT = 5.0


# TODO: remove once stream A's sandbox lands
def _run_fallback(code: str, timeout: float = TIMEOUT) -> RunResult:
    """Minimal stand-in for sandbox.run_sandboxed: run as main.py in a temp dir, no limits."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / sandbox.FILENAME
        path.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", sandbox.FILENAME],
                cwd=tmp,
                capture_output=True,
                text=True,
                timeout=timeout,
                stdin=subprocess.DEVNULL,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout or ""
            return RunResult(
                crashed=False,
                timed_out=True,
                error_type=None,
                error_line=None,
                traceback="",
                stdout=stdout,
            )
        if proc.returncode == 0:
            return RunResult(
                crashed=False,
                timed_out=False,
                error_type=None,
                error_line=None,
                traceback="",
                stdout=proc.stdout,
            )
        tb = proc.stderr
        # macOS: /private/var/... resolves from /var/...; replace the longer path first
        for p in sorted({str(path), str(path.resolve())}, key=len, reverse=True):
            tb = tb.replace(p, sandbox.FILENAME)
        tb = tb[tb.find("Traceback (most recent call last):") :] if "Traceback (" in tb else tb
        error_type = re.match(r"(\w+)", tb.strip().splitlines()[-1]) if tb.strip() else None
        lines = re.findall(rf'File "{sandbox.FILENAME}", line (\d+)', tb)
        return RunResult(
            crashed=True,
            timed_out=False,
            error_type=error_type.group(1) if error_type else None,
            error_line=int(lines[-1]) if lines else None,
            traceback=tb,
            stdout=proc.stdout,
        )


def run_file(code: str) -> RunResult:
    try:
        return sandbox.run_sandboxed(code, timeout=TIMEOUT)
    except NotImplementedError:
        return _run_fallback(code)  # TODO: remove once stream A's sandbox lands


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
    """Exit code 0 if the program ran fine, 1 if it crashed (and was explained), 2 on timeout."""
    args = parse_args(argv)
    code = args.file.read_text(encoding="utf-8")
    result = run_file(code)
    if result.timed_out:
        print(f"Your program didn't finish within {TIMEOUT:.0f} seconds (an endless loop?).")
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
