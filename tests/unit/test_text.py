from __future__ import annotations

import pytest

from shrink.compressors.text import TextCompressor, compress_text
from shrink.document import Document


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("", ""),
        ("a\n", "a\n"),
        ("a", "a"),
        ("a   \t\nb  \n", "a\nb\n"),
        ("a\n\n\n\nb\n", "a\n\nb\n"),
        ("\n\n  \na\n\n\n", "a\n"),
        ("   \n\t\n", ""),
        ("a  b\tc\n", "a  b\tc\n"),  # inner whitespace untouched at level 1
        ("one\r\ntwo   \r\n\r\n\r\nthree\r\n", "one\r\ntwo\r\n\r\nthree\r\n"),
    ],
)
def test_level_1(source: str, expected: str) -> None:
    assert compress_text(source, 1) == expected


def test_mixed_line_endings_become_lf() -> None:
    assert compress_text("a\r\nb\nc\r\n", 1) == "a\nb\nc\n"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("a  b\t\tc\n", "a b c\n"),
        ("    indented   code\n", "    indented code\n"),
        ("- item  one\n  - nested   item\n", "- item one\n  - nested item\n"),
    ],
)
def test_level_2_inner_whitespace(source: str, expected: str) -> None:
    assert compress_text(source, 2) == expected


def test_level_2_drops_repeated_paragraphs() -> None:
    para = "This paragraph is long enough to count."
    source = f"{para}\n\nother\n\n{para}\n\nend\n"
    assert compress_text(source, 2) == f"{para}\n\nother\n\nend\n"


def test_duplicates_compare_normalised_whitespace() -> None:
    source = "a long enough paragraph here\nwrapped\n\na  long enough paragraph here wrapped\n"
    assert compress_text(source, 2) == "a long enough paragraph here\nwrapped\n"


def test_short_paragraphs_and_headings_are_kept() -> None:
    source = "Yes.\n\n## Usage of the tool in CI\n\nYes.\n\n## Usage of the tool in CI\n"
    assert compress_text(source, 2) == source


def test_level_1_keeps_duplicates() -> None:
    para = "This paragraph is long enough to count."
    source = f"{para}\n\n{para}\n"
    assert compress_text(source, 1) == source


def test_fenced_code_is_untouched_at_level_2_and_3() -> None:
    source = (
        "Some   text\n"
        "\n"
        "```python\n"
        "x  =  1   # <b>keep</b> [a](b)\n"
        "```\n"
        "\n"
        "~~~~\n"
        "y  =  2\n"
        "```\n"
        "still  code\n"
        "~~~~\n"
    )
    expected = source.replace("Some   text", "Some text")
    assert compress_text(source, 2) == expected
    assert compress_text(source, 3) == expected


def test_unclosed_fence_runs_to_the_end() -> None:
    source = "```\na   b\n\nc   d\n"
    assert compress_text(source, 3) == source


def test_duplicate_code_blocks_are_kept() -> None:
    block = "```\nprint('the same code block twice')\n```\n"
    source = f"{block}\n{block}"
    assert compress_text(source, 3) == source


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("a <!-- hidden --> b\n", "a b\n"),
        ("a\n<!--\nmulti\nline\n-->\nb\n", "a\n\nb\n"),
        ("<div class='x'>Hello</div> <b>world</b>\n", "Hello world\n"),
        ("one<br>two<br/>three\n", "one two three\n"),
        ("See [the docs](https://example.com) now\n", "See the docs now\n"),
        ("![logo](img/logo.png) Title\n", "logo Title\n"),
        ("[[x](y)](z)\n", "x\n"),
        ("<<b>b>text\n", "text\n"),
        ("a\n\n---\n\nb\n", "a\n\nb\n"),
        ("a\n\n* * *\n\nb\n", "a\n\nb\n"),
        ("Title\n---\n", "Title\n---\n"),  # setext heading underline, not a rule
        ("use `<div>` and `[a](b)` here\n", "use `<div>` and `[a](b)` here\n"),
        ("2 < 3 and 5 > 4\n", "2 < 3 and 5 > 4\n"),
        ("<div>   <h1>Title</h1></div>\n", "Title\n"),  # no fake indent left behind
        ("  - <b>item</b>\n", "  - item\n"),  # real indentation kept
    ],
)
def test_level_3_noise(source: str, expected: str) -> None:
    assert compress_text(source, 3) == expected


def test_level_3_keeps_level_2_rules() -> None:
    para = "This paragraph is long enough to count."
    source = f"<p>{para}</p>\n\n{para}\n"
    assert compress_text(source, 3) == f"{para}\n"


def test_invalid_level() -> None:
    with pytest.raises(ValueError, match="level"):
        compress_text("x", 7)


def test_compressor_keeps_path() -> None:
    out = TextCompressor().compress(Document("a  \n", path="notes.md"), 1)
    assert out == Document("a\n", path="notes.md")


def test_help_mentions_every_level() -> None:
    for level in ("level 1", "level 2", "level 3"):
        assert level in TextCompressor.help


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/a.py", True),
        (r"C:\proj\main.TS", True),
        ("Makefile", True),
        ("docs/readme.md", False),
        ("notes.txt", False),
        (None, False),
    ],
)
def test_is_code_path(path: str | None, expected: bool) -> None:
    from shrink.compressors.text import is_code_path

    assert is_code_path(path) is expected
