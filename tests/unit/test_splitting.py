from __future__ import annotations

import pytest

from shrink.document import Document
from shrink.splitting.delimiter import DelimiterSplitter
from shrink.splitting.factory import AutoSplitter, make_splitter
from shrink.splitting.formats import FORMATS, detect, looks_like_path
from shrink.splitting.framed import FramedSplitter

F2P = "src/a.py\n---\nprint(1)\n\n---\nREADME.md\n---\n# Hi\n\n---\n"
F2P_NO_BLANK = "src/a.py\n---\nprint(1)\n---\nREADME.md\n---\n# Hi\n---\n"
CXML = (
    "<documents>\n"
    '<document index="1">\n<source>src/a.py</source>\n<document_content>\n'
    "print(1)\n</document_content>\n</document>\n"
    '<document index="2">\n<source>README.md</source>\n<document_content>\n'
    "# Hi\n</document_content>\n</document>\n</documents>\n"
)
REPOMIX_XML = (
    "This file is a merged representation of the codebase.\n\n"
    "<file_summary>\nsummary\n</file_summary>\n\n"
    "<directory_structure>\nsrc/\n</directory_structure>\n\n"
    "<files>\nThis section contains the contents.\n\n"
    '<file path="src/a.py">\nprint(1)\n</file>\n\n'
    '<file path="README.md">\n# Hi\n</file>\n\n</files>\n'
)
REPOMIX_MD = (
    "# File Summary\n\nsummary\n\n# Directory Structure\n\n```\nsrc/\n```\n\n# Files\n\n"
    "## File: src/a.py\n```python\nprint(1)\n```\n\n"
    "## File: README.md\n```markdown\n# Hi\n```\n"
)
REPOMIX_PLAIN = (
    "================\nFile Summary\n================\n\nsummary\n\n"
    "================\nDirectory Structure\n================\nsrc/\n\n"
    "================\nFiles\n================\n\n"
    "================\nFile: src/a.py\n================\nprint(1)\n\n"
    "================\nFile: README.md\n================\n# Hi\n\n"
)

CASES = {
    "files-to-prompt": F2P,
    "cxml": CXML,
    "repomix-xml": REPOMIX_XML,
    "repomix-markdown": REPOMIX_MD,
    "repomix-plain": REPOMIX_PLAIN,
}


def split(text: str, fmt: str) -> list[Document]:
    return list(FramedSplitter(FORMATS[fmt]).split(text.splitlines(keepends=True)))


def joined(docs: list[Document]) -> str:
    return "".join(doc.full_text for doc in docs)


@pytest.mark.parametrize("fmt", CASES)
def test_split_and_join_reproduces_the_input(fmt: str) -> None:
    assert joined(split(CASES[fmt], fmt)) == CASES[fmt]


@pytest.mark.parametrize("fmt", CASES)
def test_each_format_yields_two_files_with_paths_and_content(fmt: str) -> None:
    docs = split(CASES[fmt], fmt)
    assert [d.path for d in docs] == ["src/a.py", "README.md"]
    assert docs[0].text == "print(1)\n"
    assert docs[1].text == "# Hi\n"


@pytest.mark.parametrize("fmt", CASES)
def test_auto_detects_each_format(fmt: str) -> None:
    found = detect(CASES[fmt].splitlines(keepends=True))
    assert found is not None
    assert found.name == fmt


def test_files_to_prompt_without_the_blank_line_before_the_rule() -> None:
    docs = split(F2P_NO_BLANK, "files-to-prompt")
    assert [d.path for d in docs] == ["src/a.py", "README.md"]
    assert joined(docs) == F2P_NO_BLANK


def test_preamble_and_epilogue_stay_in_the_framing() -> None:
    docs = split(REPOMIX_XML, "repomix-xml")
    assert docs[0].head.startswith("This file is a merged representation")
    assert docs[0].head.endswith('<file path="src/a.py">\n')
    assert docs[-1].tail.endswith("</files>\n")


def test_markdown_fence_longer_than_three_backticks_is_matched() -> None:
    text = "## File: a.md\n````markdown\n```\ninner\n```\n````\n"
    docs = split(text, "repomix-markdown")
    assert docs[0].text == "```\ninner\n```\n"
    assert joined(docs) == text


def test_plain_style_other_sections_end_the_last_file() -> None:
    text = REPOMIX_PLAIN + "================\nGit Logs\n================\nabc\n"
    docs = split(text, "repomix-plain")
    assert docs[-1].text == "# Hi\n"
    assert docs[-1].tail.endswith("Git Logs\n================\nabc\n")
    assert joined(docs) == text


def test_crlf_framing_is_preserved() -> None:
    text = CXML.replace("\n", "\r\n")
    docs = split(text, "cxml")
    assert docs[0].text == "print(1)\r\n"
    assert joined(docs) == text


def test_unterminated_last_file_is_still_returned() -> None:
    text = "a.py\n---\nprint(1)\n"
    docs = split(text, "files-to-prompt")
    assert [d.text for d in docs] == ["print(1)\n"]
    assert joined(docs) == text


@pytest.mark.parametrize("fmt", CASES)
def test_text_without_files_comes_back_as_one_document(fmt: str) -> None:
    text = "just some\nprose\n"
    docs = split(text, fmt)
    assert docs == [Document(text)]


def test_empty_input_yields_nothing() -> None:
    assert split("", "cxml") == []


@pytest.mark.parametrize("text", ["Intro\n---\nbody\n", "title: x\n---\nbody\n", "---\nbody\n"])
def test_markdown_headings_and_front_matter_are_not_file_headers(text: str) -> None:
    assert detect(text.splitlines(keepends=True)) is None


def test_front_matter_inside_a_file_does_not_split_it() -> None:
    text = "docs/a.md\n---\n---\ntitle: x\n---\nbody\n\n---\n"
    docs = split(text, "files-to-prompt")
    assert [d.path for d in docs] == ["docs/a.md"]
    assert joined(docs) == text


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("src/a.py", True),
        ("a.py", True),
        ("C:\\x\\y", True),
        ("Intro", False),
        ("v1.2", False),
        ("title: x", False),
        ("# a.md", False),
        (" a.py", False),
        ("", False),
    ],
)
def test_looks_like_path(name: str, expected: bool) -> None:
    assert looks_like_path(name) is expected


def test_auto_splitter_streams_a_recognised_format() -> None:
    docs = list(AutoSplitter().split(F2P.splitlines(keepends=True)))
    assert [d.path for d in docs] == ["src/a.py", "README.md"]


def test_auto_splitter_falls_back_to_the_whole_stream() -> None:
    docs = list(AutoSplitter().split(["a\n", "b\n"]))
    assert docs == [Document("a\nb\n")]


def test_make_splitter_none_ignores_packed_formats() -> None:
    docs = list(make_splitter("none").split(F2P.splitlines(keepends=True)))
    assert len(docs) == 1
    assert docs[0].text == F2P


def test_delimiter_splitter_keeps_the_delimiter_line_as_head() -> None:
    text = "intro\n=== a.py ===\nx\n=== b.md ===\ny\nz\n"
    docs = list(DelimiterSplitter(r"^=== (?P<path>\S+) ===$").split(text.splitlines(keepends=True)))
    assert [(d.head, d.path, d.text) for d in docs] == [
        ("", None, "intro\n"),
        ("=== a.py ===\n", "a.py", "x\n"),
        ("=== b.md ===\n", "b.md", "y\nz\n"),
    ]
    assert joined(docs) == text


def test_delimiter_without_path_group_leaves_the_path_empty() -> None:
    docs = list(DelimiterSplitter(r"^----$").split(["----\n", "a\n"]))
    assert docs == [Document("a\n", path=None, head="----\n")]


def test_delimiter_wins_over_format() -> None:
    splitter = make_splitter("cxml", r"^X$")
    assert [d.text for d in splitter.split(["X\n", "a\n"])] == ["a\n"]
