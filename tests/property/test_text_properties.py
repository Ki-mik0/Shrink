"""Properties the text mode must hold for any input."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from shrink.compressors.text import compress_text

# Markdown/HTML-heavy alphabet finds structural edge cases much faster than
# arbitrary unicode, so mix both.
MARKUP = st.text(alphabet=st.sampled_from(list("ab #-*_`~<>/![]()\n\r\t =")), max_size=200)
ANY = st.text(max_size=200)
TEXT = st.one_of(MARKUP, ANY)
LEVEL = st.integers(min_value=1, max_value=3)


@settings(max_examples=400)
@given(TEXT, LEVEL)
def test_idempotent(text: str, level: int) -> None:
    once = compress_text(text, level)
    assert compress_text(once, level) == once


@settings(max_examples=400)
@given(TEXT, LEVEL)
def test_never_longer(text: str, level: int) -> None:
    assert len(compress_text(text, level)) <= len(text)


@given(TEXT, LEVEL)
def test_deterministic(text: str, level: int) -> None:
    assert compress_text(text, level) == compress_text(text, level)


@given(TEXT)
def test_higher_level_never_longer(text: str) -> None:
    sizes = [len(compress_text(text, level)) for level in (1, 2, 3)]
    assert sizes[0] >= sizes[1] >= sizes[2]


@given(st.lists(st.text(alphabet="abc ", min_size=1, max_size=20), max_size=10))
def test_crlf_input_stays_crlf(lines: list[str]) -> None:
    text = "\r\n".join(lines) + "\r\n"
    out = compress_text(text, 2)
    assert "\n" not in out.replace("\r\n", "")
