"""Compare two runs on the same examples: paired differences with 95% CIs.

Run: uv run python -m traceback_tutor.eval.compare runs/<baseline> runs/<candidate>
Prints a table and writes runs/compare_<baseline>_vs_<candidate>.json next to the runs.

Both runs score the same examples, so each example gives a pair of results. The
difference (candidate - baseline) gets a paired bootstrap CI (resampling examples), and
an exact McNemar test on the examples where the two systems disagree.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from .metrics import BOOL_METRICS, WEAK_METRICS, bootstrap_ci


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    config = json.loads((run_dir / "metrics.json").read_text())["config"]
    rows = [json.loads(line) for line in (run_dir / "responses.jsonl").read_text().splitlines()]
    return config, rows


def mcnemar_p(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value from the two kinds of disagreement."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare_metric(a: list[bool], b: list[bool]) -> dict:
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    diffs = [float(y) - float(x) for x, y in zip(a, b)]
    return {
        "n": len(a),
        "baseline": sum(a),
        "candidate": sum(b),
        "diff": sum(diffs) / len(diffs),
        "diff_ci95": list(bootstrap_ci(diffs)),
        "only_baseline": only_a,
        "only_candidate": only_b,
        "mcnemar_p": mcnemar_p(only_a, only_b),
    }


def check_comparable(config_a: dict, config_b: dict, rows_a: list[dict], rows_b: list[dict]):
    """Refuse to compare runs on different examples or settings; warn on scoring commit."""
    if [r["id"] for r in rows_a] != [r["id"] for r in rows_b]:
        raise SystemExit("the runs don't cover the same examples in the same order")
    for key in ("data_sha256", "split", "generation"):
        if config_a.get(key) != config_b.get(key):
            raise SystemExit(
                f"the runs differ in {key}: {config_a.get(key)} vs {config_b.get(key)}"
            )
    scored_a = config_a.get("rescored", config_a)["git_commit"]
    scored_b = config_b.get("rescored", config_b)["git_commit"]
    if scored_a != scored_b:
        print(f"warning: scored with different commits ({scored_a} vs {scored_b}); rescore both")


def compare(run_a: Path, run_b: Path) -> dict:
    config_a, rows_a = load_run(run_a)
    config_b, rows_b = load_run(run_b)
    check_comparable(config_a, config_b, rows_a, rows_b)

    metrics = {
        name: compare_metric(
            [r["metrics"][name] for r in rows_a], [r["metrics"][name] for r in rows_b]
        )
        for name in BOOL_METRICS
    }
    # Where the bug and the crash are on different lines, citing the bug line means the
    # model looked past the traceback's last frame.
    split_lines = [i for i, r in enumerate(rows_a) if r["bug_line"] != r["error_line"]]
    metrics["line_is_bug_line_when_bug_differs_from_crash"] = compare_metric(
        [rows_a[i]["metrics"]["line_is_bug_line"] for i in split_lines],
        [rows_b[i]["metrics"]["line_is_bug_line"] for i in split_lines],
    )
    return {"baseline": str(run_a), "candidate": str(run_b), "metrics": metrics}


def format_comparison(result: dict) -> str:
    header = f"{'metric':<46} {'baseline':>9} {'candidate':>9}  diff [95% CI]          McNemar p"
    lines = [f"baseline: {result['baseline']}   candidate: {result['candidate']}", header]
    for name, m in result["metrics"].items():
        n = m["n"]
        label = name + (" (lower is better)" if name == "gives_fix" else "")
        label += " *" if name in WEAK_METRICS else ""
        lo, hi = m["diff_ci95"]
        lines.append(
            f"{label:<46} {m['baseline']:>5}/{n:<3} {m['candidate']:>5}/{n:<3}"
            f"  {m['diff']:+.2f} [{lo:+.2f}, {hi:+.2f}]   {m['mcnemar_p']:.2g}"
        )
    lines.append("* weak keyword check: echoing the traceback passes it; don't claim a difference")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("baseline", type=Path)
    p.add_argument("candidate", type=Path)
    args = p.parse_args(argv)
    result = compare(args.baseline, args.candidate)
    print(format_comparison(result))
    out = args.candidate.parent / f"compare_{args.baseline.name}_vs_{args.candidate.name}.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out}")
    return result


if __name__ == "__main__":
    main()
