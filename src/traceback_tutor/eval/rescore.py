"""Re-score saved runs with the current metric code, without generating again.

Run: uv run python -m traceback_tutor.eval.rescore runs/<name> [runs/<name> ...]
Rewrites responses.jsonl and metrics.json in place. The generation config is kept;
metrics.json gets a "rescored" entry with the commit the new scores come from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..schema import read_jsonl
from .metrics import score
from .run_eval import file_sha256, format_table, git_info, write_summary


def rescore(run_dir: Path) -> dict:
    config = json.loads((run_dir / "metrics.json").read_text())["config"]
    if file_sha256(config["data"]) != config["data_sha256"]:
        raise SystemExit(f"{config['data']} changed since {run_dir} was generated; rerun instead")
    examples = {ex.id: ex for ex in read_jsonl(config["data"])}

    rows = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text().splitlines()]
    for row in rows:
        ex = examples[row["id"]]
        row["bug_line"], row["error_line"] = ex.bug_line, ex.error_line
        row["metrics"] = score(ex, row["response"])
    with open(run_dir / "responses.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return write_summary(rows, {**config, "rescored": git_info()}, run_dir)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("runs", nargs="+", type=Path, help="run directories")
    for run_dir in p.parse_args(argv).runs:
        print(f"== {run_dir}")
        print(format_table(rescore(run_dir)))


if __name__ == "__main__":
    main()
