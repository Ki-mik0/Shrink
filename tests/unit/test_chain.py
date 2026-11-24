from __future__ import annotations

import pytest

from shrink.compressors.base import Step, Traceable
from shrink.compressors.chain import ChainCompressor
from shrink.compressors.registry import ModeSpec, Registry, default_registry
from shrink.compressors.text import STEP_NAMES, TextCompressor
from shrink.document import Document


class Suffix:
    summary = ""
    help = "adds a suffix"

    def __init__(self, name: str, suffix: str) -> None:
        self.name = name
        self._suffix = suffix

    def compress(self, doc: Document, level: int) -> Document:
        return doc.with_text(doc.text + self._suffix)


class TwoStep(Suffix):
    def trace(self, doc: Document, level: int) -> list[Step]:
        first = doc.with_text(doc.text + "1")
        return [Step("one", first), Step("two", first.with_text(first.text + "2"))]

    def compress(self, doc: Document, level: int) -> Document:
        return self.trace(doc, level)[-1].after


def test_chain_runs_members_left_to_right() -> None:
    chain = ChainCompressor([Suffix("a", "A"), Suffix("b", "B")])
    assert chain.compress(Document("x"), 2) == Document("xAB")
    assert chain.name == "a+b"
    assert "[a]" in chain.help
    assert "[b]" in chain.help


def test_chain_needs_a_member() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ChainCompressor([])


def test_chain_trace_has_a_step_per_plain_member_and_per_traced_step() -> None:
    chain = ChainCompressor([Suffix("a", "A"), TwoStep("t", ""), Suffix("b", "B")])
    steps = chain.trace(Document("x"), 2)
    assert [s.name for s in steps] == ["a", "one", "two", "b"]
    assert [s.after.text for s in steps] == ["xA", "xA1", "xA12", "xA12B"]
    assert steps[-1].after == chain.compress(Document("x"), 2)


def test_chain_is_traceable_and_a_plain_compressor_is_not() -> None:
    assert isinstance(ChainCompressor([Suffix("a", "A")]), Traceable)
    assert not isinstance(Suffix("a", "A"), Traceable)


def test_registry_create_chain() -> None:
    registry = Registry([ModeSpec("t", "shrink.compressors.text:TextCompressor", "t")])
    assert registry.create_chain(["t"]).name == "text"
    assert registry.create_chain(["t", "t"]).name == "text+text"


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize(
    "text",
    ["", "a   b\n\n\n\n<b>[x](y)</b>  z\n", "para one is long enough\n\npara one is long enough\n"],
)
def test_text_trace_ends_with_the_compress_result(text: str, level: int) -> None:
    compressor = TextCompressor()
    doc = Document(text)
    steps = compressor.trace(doc, level)
    assert [s.name for s in steps] == [STEP_NAMES[k] for k in range(1, level + 1)]
    assert steps[-1].after == compressor.compress(doc, level)


def test_text_trace_of_a_code_file_stops_at_level_1() -> None:
    steps = TextCompressor().trace(Document("x  \n", path="a.py"), 3)
    assert [s.name for s in steps] == ["whitespace"]


def test_text_trace_keeps_the_framing() -> None:
    doc = Document("x  \n", path="a.md", head="a.md\n---\n", tail="---\n")
    assert all(s.after.head == "a.md\n---\n" for s in TextCompressor().trace(doc, 2))


def test_default_registry_chain_of_text() -> None:
    chain = default_registry.create_chain(["text", "text"])
    assert chain.compress(Document("a   b\n"), 2).text == "a b\n"
