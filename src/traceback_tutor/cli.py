"""Stream C: `explain buggy.py` runs the file, catches the traceback and asks the model.

If the file doesn't crash, say so and exit. Uses sandbox.run_sandboxed and
prompt.build_messages, generates with mlx-lm, prints the parsed response.
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError
