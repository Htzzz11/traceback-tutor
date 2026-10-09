from pathlib import Path

import pytest

from traceback_tutor.schema import read_jsonl

FIXTURES = Path(__file__).parent.parent / "data" / "fixtures" / "examples.jsonl"


@pytest.fixture
def fixture_examples():
    return list(read_jsonl(FIXTURES))
