from __future__ import annotations

import pytest
from click.testing import CliRunner

from shrink.cli import cli
from shrink.compressors.nodocs import NoDocsCompressor
from shrink.compressors.python_code import drop_docstrings
from shrink.compressors.web import drop_js_docs
from shrink.document import Document

PYTHON = '''"""Module docs."""

import os


class Store:
    """Keeps things."""

    def get(self, key: str) -> str:
        """Return the value.

        Longer text.
        """
        return os.environ[key]


class Empty:
    """Only a docstring."""


def stub() -> None:
    """Only a docstring too."""


def skeleton(x: int) -> int:
    """Kept by level 3."""
    ...


def inline() -> None: """Same line."""
'''

EXPECTED = """import os


class Store:
    def get(self, key: str) -> str:
        return os.environ[key]


class Empty:
    ...


def stub() -> None:
    ...


def skeleton(x: int) -> int:
    ...


def inline() -> None: ...
"""


def test_drop_docstrings() -> None:
    assert drop_docstrings(PYTHON) == EXPECTED


def test_drop_docstrings_keeps_crlf_and_strings_that_are_not_docstrings() -> None:
    src = 'def f():\r\n    """Doc."""\r\n\r\n    x = """not a docstring"""\r\n    return x\r\n'
    assert drop_docstrings(src) == 'def f():\r\n    x = """not a docstring"""\r\n    return x\r\n'


def test_drop_docstrings_without_any_is_unchanged() -> None:
    assert drop_docstrings("x = 1\n") == "x = 1\n"


def test_drop_js_docs_removes_jsdoc_and_keeps_pragmas() -> None:
    src = (
        "/*! license */\n/** Load a user. */\n"
        "export function load(id) {/* ... */}\n// @ts-ignore\nconst s = '/** no */';\n"
    )
    assert drop_js_docs(src) == (
        "/*! license */\nexport function load(id) {/* ... */}\n// @ts-ignore\n"
        "const s = '/** no */';\n"
    )


@pytest.mark.parametrize(
    ("path", "text", "expected"),
    [
        ("a.py", '"""Docs."""\nx = 1\n', "x = 1\n"),
        ("a.js", "/** Docs. */\nf();\n", "f();\n"),
        ("s.css", "/* keep: not a doc */\na{}\n", "/* keep: not a doc */\na{}\n"),
        ("notes.md", '"""Not code."""\n', '"""Not code."""\n'),
        ("bad.py", 'def f(:\n    """x"""\n', 'def f(:\n    """x"""\n'),
    ],
)
def test_no_docs_compressor(path: str, text: str, expected: str) -> None:
    out = NoDocsCompressor().compress(Document(text, path=path), 3)
    assert out.text == expected


def test_cli_no_docs_flag_runs_after_the_mode(runner: CliRunner) -> None:
    packed = 'a.py\n---\ndef f(x):\n    """Docs."""\n    return x\n---\n'
    result = runner.invoke(cli, ["-l", "3", "--no-docs", "--stats"], input=packed)
    assert result.stdout == "a.py\n---\ndef f(x):\n    ...\n---\n"
    assert "no-docs" in result.stderr
