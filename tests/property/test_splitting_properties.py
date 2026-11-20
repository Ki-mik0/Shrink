from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from shrink.splitting.delimiter import DelimiterSplitter
from shrink.splitting.factory import AutoSplitter
from shrink.splitting.formats import FORMATS
from shrink.splitting.framed import FramedSplitter

# Lines drawn from the fragments the formats look for, so that headers, closing
# markers and near-misses actually occur in the generated streams.
FRAGMENTS = [
    "---",
    "a.py",
    "src/b.md",
    "title: x",
    "================",
    "File: a.py",
    "File Summary",
    "Files",
    "<documents>",
    '<document index="1">',
    "<source>a.py</source>",
    "<document_content>",
    "</document_content>",
    "</document>",
    '<file path="a.py">',
    "</file>",
    "## File: a.py",
    "```python",
    "```",
    "````",
    "text",
    "",
    "  ",
]
LINES = st.lists(
    st.sampled_from(FRAGMENTS) | st.text(max_size=12),
    max_size=40,
).map(lambda parts: [part + "\n" for part in parts])


@given(LINES)
@settings(max_examples=200, deadline=None)
def test_every_format_returns_exactly_the_bytes_it_was_given(lines: list[str]) -> None:
    text = "".join(lines)
    for fmt in FORMATS.values():
        docs = list(FramedSplitter(fmt).split(lines))
        assert "".join(d.full_text for d in docs) == text


@given(LINES)
@settings(max_examples=200, deadline=None)
def test_auto_splitter_returns_exactly_the_bytes_it_was_given(lines: list[str]) -> None:
    docs = list(AutoSplitter().split(lines))
    assert "".join(d.full_text for d in docs) == "".join(lines)


@given(LINES)
@settings(max_examples=100, deadline=None)
def test_delimiter_splitter_returns_exactly_the_bytes_it_was_given(lines: list[str]) -> None:
    docs = list(DelimiterSplitter(r"^---$").split(lines))
    assert "".join(d.full_text for d in docs) == "".join(lines)
