import pytest
from conftest import FIXTURES

from traceback_tutor.eval import audit_sheet, run_eval
from traceback_tutor.generate import Generation

RESPONSE = "Line 5: `totl` is unknown; compare lines 2 and 4.\n\nHint: Same spelling?"


def test_mentioned_lines():
    assert audit_sheet.mentioned_lines(RESPONSE) == [2, 4, 5]
    assert audit_sheet.mentioned_lines("See lines 9, 11 and 12, and line 40.") == [9, 11, 12, 40]
    assert audit_sheet.mentioned_lines("No numbers here.") == []


def make_run(tmp_path):
    def generate_fn(messages):
        return Generation(RESPONSE, n_tokens=5, seconds=0.1, finish_reason="stop")

    argv = ["--model", "m", "--data", str(FIXTURES), "--name", "t", "--runs-dir", str(tmp_path)]
    run_eval.main([*argv, "--limit", "2"], generate_fn=generate_fn)
    return tmp_path / "t"


def test_sheet_shows_mentioned_code_and_counts_ticks(tmp_path):
    sheet = audit_sheet.make_sheet(make_run(tmp_path))
    assert "## 1. fixture-1-misspelt_variable-0" in sheet
    assert "- 5 → `return totl`" in sheet
    assert "- 4 → `total = total + n`" in sheet
    assert sheet.count("- [ ] audited:") == 2

    # tick the first section: audited, line refs and no_fix OK, facts wrong
    first, second = sheet.split("## 2.")
    first = (
        first.replace("- [ ] audited:", "- [x] audited:")
        .replace("- [ ] line_refs_ok:", "- [x] line_refs_ok:")
        .replace("- [ ] no_fix:", "- [x] no_fix:")
    )
    result = audit_sheet.summarise(first + "## 2." + second)
    assert result["n_sections"] == 2 and result["n_audited"] == 1
    assert result["checks"]["line_refs_ok"]["count"] == 1
    assert result["checks"]["facts_ok"]["failed"] == ["fixture-1-misspelt_variable-0"]


def test_sheet_is_not_overwritten(tmp_path):
    run_dir = make_run(tmp_path)
    audit_sheet.main([str(run_dir)])
    (run_dir / "audit.md").write_text("my ticks")
    with pytest.raises(SystemExit):
        audit_sheet.main([str(run_dir)])
    assert (run_dir / "audit.md").read_text() == "my ticks"
