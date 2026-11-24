"""Code mode on real files: this project's own source and a slice of the stdlib.

For every file and level: the output parses (validation already demands it),
running the level again changes nothing, and the output is never longer than
the input. Files the running interpreter cannot parse are skipped.
"""

from __future__ import annotations

import ast
import sysconfig
from pathlib import Path

import pytest

from shrink.compressors.python_code import CodeCompressor
from shrink.document import Document

ROOT = Path(__file__).resolve().parents[2]
STDLIB = Path(sysconfig.get_paths()["stdlib"])
STDLIB_MODULES = [
    "textwrap.py",
    "argparse.py",
    "dataclasses.py",
    "json/__init__.py",
    "json/decoder.py",
    "functools.py",
    "pathlib/__init__.py",
    "pathlib.py",
    "typing.py",
    "enum.py",
]


def corpus() -> list[Path]:
    files = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py"))
    files += [STDLIB / name for name in STDLIB_MODULES if (STDLIB / name).is_file()]
    return files


@pytest.mark.parametrize("path", corpus(), ids=lambda p: p.name)
def test_every_level_is_valid_idempotent_and_never_longer(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    try:
        ast.parse(text)
    except SyntaxError:
        pytest.skip("not parseable by this interpreter")
    compressor = CodeCompressor()
    previous = len(text)
    for level in (1, 2, 3):
        out = compressor.compress(Document(text, path=path.name), level).text
        ast.parse(out)
        assert len(out) <= previous
        again = compressor.compress(Document(out, path=path.name), level).text
        assert again == out, f"level {level} is not idempotent"
        previous = len(out)
