"""Stream A: run untrusted (mutated) programs safely and capture the traceback."""

from __future__ import annotations

from dataclasses import dataclass

FILENAME = "main.py"  # every traceback is normalised to this file name


@dataclass
class RunResult:
    crashed: bool
    timed_out: bool
    error_type: str | None  # e.g. "NameError"; None if it didn't crash
    error_line: int | None  # 1-based line in main.py of the innermost frame in main.py
    traceback: str  # normalised: temp paths replaced by "main.py"; "" if no crash
    stdout: str


def run_sandboxed(code: str, timeout: float = 5.0, memory_mb: int = 256) -> RunResult:
    """Run `code` as main.py in a fresh subprocess and report how it ended.

    Requirements:
    - Use the same interpreter as the project venv (sys.executable), so tracebacks
      have one consistent format.
    - Temp directory as cwd, isolated mode (-I), timeout, memory limit via
      resource.setrlimit in preexec_fn, no network if practical, stdin closed.
    - Replace the temp file path with "main.py" in the traceback.
    """
    raise NotImplementedError
