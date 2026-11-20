from __future__ import annotations

import io

from shrink.document import Document
from shrink.io.reader import StdinReader
from shrink.io.writer import StdoutWriter
from shrink.splitting.documents import WholeStreamSplitter


def test_reader_keeps_line_endings() -> None:
    reader = StdinReader(io.BytesIO(b"a\r\nb\nc"))
    assert list(reader.lines()) == ["a\r\n", "b\n", "c"]


def test_invalid_utf8_round_trips_byte_for_byte() -> None:
    raw = bytes(range(256)) + "привіт\n".encode()
    text = "".join(StdinReader(io.BytesIO(raw)).lines())
    out = io.BytesIO()
    StdoutWriter(out).write(Document(text))
    assert out.getvalue() == raw


def test_writer_skips_empty_documents() -> None:
    out = io.BytesIO()
    StdoutWriter(out).write(Document(""))
    assert out.getvalue() == b""


def test_whole_stream_splitter() -> None:
    splitter = WholeStreamSplitter()
    assert list(splitter.split(["a\n", "b"])) == [Document("a\nb")]
    assert list(splitter.split([])) == []


def test_document_label_and_with_text() -> None:
    assert Document("x").label == "<stdin>"
    doc = Document("x", path="a.py")
    assert doc.label == "a.py"
    assert doc.with_text("y") == Document("y", path="a.py")


def test_writer_writes_framing_around_the_text() -> None:
    out = io.BytesIO()
    StdoutWriter(out).write(Document("body\n", head="a.py\n---\n", tail="---\n"))
    assert out.getvalue() == b"a.py\n---\nbody\n---\n"


def test_writer_writes_framing_even_when_the_text_is_empty() -> None:
    out = io.BytesIO()
    StdoutWriter(out).write(Document("", head="a.py\n---\n", tail="---\n"))
    assert out.getvalue() == b"a.py\n---\n---\n"


def test_with_text_keeps_the_framing() -> None:
    doc = Document("x", path="a.py", head="h", tail="t").with_text("y")
    assert (doc.head, doc.tail, doc.full_text) == ("h", "t", "hyt")
