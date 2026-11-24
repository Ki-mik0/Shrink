"""Golden files: the output on reference inputs must not change by accident.

To accept an intended change, regenerate the files:
    shrink -l N < tests/fixtures/messy.md > tests/fixtures/messy.levelN.md
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shrink.compressors.text import compress_text

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.mark.parametrize("level", [1, 2, 3])
def test_messy_markdown(level: int) -> None:
    source = (FIXTURES / "messy.md").read_bytes().decode()
    expected = (FIXTURES / f"messy.level{level}.md").read_bytes().decode()
    assert compress_text(source, level) == expected
