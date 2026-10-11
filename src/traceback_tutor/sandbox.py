"""Stream A: run untrusted (mutated) programs safely and capture the traceback.

Each program runs as main.py in a fresh temp directory, in a separate process with:
- a wall-clock timeout,
- a memory limit, enforced by polling the child's resident memory (RSS) and killing it.
  macOS rejects every resource.setrlimit memory limit (RLIMIT_AS/DATA/RSS), so a
  watchdog is the one mechanism that works on the machine this project runs on,
- on macOS, `sandbox-exec` to block the network and any file writes outside the temp dir,
- stdin closed and a minimal environment.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

FILENAME = "main.py"  # every traceback is normalised to this file name

# Stdlib frames show the interpreter's install path (which contains the user's home
# directory); replace it with a typical Linux path so no local path leaks into the data.
STDLIB_DIR = sysconfig.get_paths()["stdlib"]
STDLIB_PLACEHOLDER = "/usr/lib/python3.12"

_POLL_SECONDS = 0.1
_FRAME_RE = re.compile(rf'File "{FILENAME}", line (\d+)')
_SANDBOX_PROFILE = (
    "(version 1)(allow default)(deny network*)"
    '(deny file-write*)(allow file-write* (subpath "{dir}"))'
)


@dataclass
class RunResult:
    crashed: bool
    timed_out: bool
    error_type: str | None  # e.g. "NameError"; None if it didn't crash
    error_line: int | None  # 1-based line in main.py of the innermost frame in main.py
    traceback: str  # normalised: temp paths replaced by "main.py"; "" if no crash
    stdout: str
    out_of_memory: bool = False  # killed for going over memory_mb


def run_sandboxed(code: str, timeout: float = 5.0, memory_mb: int = 256) -> RunResult:
    """Run `code` as main.py in a fresh subprocess and report how it ended."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp).resolve()  # macOS: /var/... is a symlink to /private/var/...
        script = tmp_dir / FILENAME
        script.write_text(code, encoding="utf-8")
        proc = subprocess.Popen(
            _command(tmp_dir),
            cwd=tmp_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={"PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
        )
        stdout, stderr, killed_for = _wait(proc, timeout, memory_mb)

    if killed_for is not None or proc.returncode == 0:
        return RunResult(
            crashed=False,
            timed_out=killed_for == "timeout",
            error_type=None,
            error_line=None,
            traceback="",
            stdout=stdout,
            out_of_memory=killed_for == "memory",
        )
    tb = normalise_traceback(stderr, script)
    return RunResult(
        crashed=True,
        timed_out=False,
        error_type=parse_error_type(tb),
        error_line=parse_error_line(tb),
        traceback=tb,
        stdout=stdout,
    )


def _command(tmp_dir: Path) -> list[str]:
    # -s: ignore user site-packages. Not -I: it would also ignore PYTHONHASHSEED, and a
    # fixed hash seed keeps set/dict ordering (and so which mutants crash) reproducible.
    # The environment is passed explicitly, which covers the rest of what -I protects.
    # -W ignore::SyntaxWarning: MBPP regexes like '\d+' trigger "invalid escape
    # sequence" warnings, and 3.12 re-emits them *inside* a traceback while drawing the
    # ^^^ markers (quoting the wrong line), which garbles the traceback.
    cmd = [sys.executable, "-s", "-W", "ignore::SyntaxWarning", FILENAME]
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        cmd = ["sandbox-exec", "-p", _SANDBOX_PROFILE.format(dir=tmp_dir), *cmd]
    return cmd


def _wait(proc: subprocess.Popen, timeout: float, memory_mb: int) -> tuple[str, str, str | None]:
    """Wait for `proc`, killing it on timeout or when its memory goes over the limit.

    Returns (stdout, stderr, killed_for) with killed_for None, "timeout" or "memory".
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            stdout, stderr = proc.communicate(timeout=_POLL_SECONDS)
            return stdout, stderr, None
        except subprocess.TimeoutExpired:
            if time.monotonic() > deadline:
                killed_for = "timeout"
            elif _rss_mb(proc.pid) > memory_mb:
                killed_for = "memory"
            else:
                continue
            proc.kill()
            stdout, stderr = proc.communicate()
            return stdout, stderr, killed_for


def _rss_mb(pid: int) -> float:
    """Resident memory of `pid` in MB; 0 if it has already exited. sandbox-exec execs
    Python in place, so `pid` is the Python process either way."""
    # check=False: ps exits 1 when the process is already gone, which just means 0 MB.
    cmd = ["ps", "-o", "rss=", "-p", str(pid)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    return sum(int(x) for x in out.split()) / 1024  # ps reports KB


def normalise_traceback(stderr: str, script: Path) -> str:
    tb = stderr.replace(str(script), FILENAME)
    return tb.replace(STDLIB_DIR, STDLIB_PLACEHOLDER)


def parse_error_type(tb: str) -> str | None:
    """Exception class name from the last line, e.g. "NameError: name 'x' ..." -> "NameError"."""
    lines = [line for line in tb.splitlines() if line.strip()]
    if not lines:
        return None
    name = lines[-1].split(":", 1)[0].strip()
    name = name.rsplit(".", 1)[-1]  # "json.decoder.JSONDecodeError" -> "JSONDecodeError"
    return name if name.isidentifier() else None


def parse_error_line(tb: str) -> int | None:
    """Line of the innermost main.py frame (the last one printed)."""
    lines = _FRAME_RE.findall(tb)
    return int(lines[-1]) if lines else None
