"""Split a packed stream (files-to-prompt, Repomix, ...) into documents.

A packed stream is a sequence of files, each wrapped in framing: a header that
names the file and, for most formats, a closing marker. The framing is kept in
``Document.head`` / ``Document.tail`` and written back byte for byte; only the
file content between them is handed to a compressor. So splitting and joining
without compression reproduces the input exactly, whatever the format.

Only one document (plus a three-line lookahead) is held in memory. If a stream
turns out to contain no file at all, it is returned as one document, so a wrong
format guess costs nothing.
"""

from __future__ import annotations

import itertools
from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Protocol

from shrink.document import Document

LOOKAHEAD = 3  # the longest header (or header + closing probe) is three lines


@dataclass(frozen=True)
class Header:
    """A recognised header.

    ``size`` lines belong to the header. ``is_file`` is False for section
    headings that end a file without starting another one (Repomix ``plain``).
    ``state`` carries what the closing probe needs (the Markdown fence).
    """

    size: int
    path: str | None = None
    is_file: bool = True
    state: str = ""


class Format(Protocol):
    name: str
    summary: str
    #: Content ends where the next header begins (no closing marker).
    implicit_end: bool
    #: Blank lines just before the end belong to the framing, not the content.
    blank_to_tail: bool

    def header(self, ahead: list[str]) -> Header | None:
        """Does a header start at ``ahead[0]``? ``ahead`` holds up to three lines."""

    def content_end(self, ahead: list[str], header: Header) -> int | None:
        """Number of closing lines if the content ends at ``ahead[0]``, else None."""


class Lookahead:
    """An iterator over lines that can peek a few lines ahead."""

    def __init__(self, lines: Iterable[str]) -> None:
        self._it = iter(lines)
        self._buf: deque[str] = deque()

    def peek(self, n: int) -> list[str]:
        while len(self._buf) < n:
            try:
                self._buf.append(next(self._it))
            except StopIteration:
                break
        return list(itertools.islice(self._buf, n))

    def take(self, n: int = 1) -> list[str]:
        got = self.peek(n)
        for _ in got:
            self._buf.popleft()
        return got


def line_text(line: str) -> str:
    return line.rstrip("\r\n")


class FramedSplitter:
    def __init__(self, fmt: Format) -> None:
        self._fmt = fmt

    def split(self, lines: Iterable[str]) -> Iterator[Document]:
        src = Lookahead(lines)
        between: list[str] = []  # framing outside any file: preamble, or the tail of `prev`
        prev: Document | None = None
        while src.peek(1):
            header = self._fmt.header(src.peek(LOOKAHEAD))
            if header is None:
                between.extend(src.take(1))
            elif not header.is_file:
                between.extend(src.take(header.size))
            else:
                head_lines = src.take(header.size)
                if prev is None:
                    head = "".join(between + head_lines)
                else:
                    yield replace(prev, tail=prev.tail + "".join(between))
                    head = "".join(head_lines)
                between = []
                prev = self._read_file(src, header, head)
        if prev is not None:
            yield replace(prev, tail=prev.tail + "".join(between))
        elif between:
            yield Document("".join(between))

    def _read_file(self, src: Lookahead, header: Header, head: str) -> Document:
        fmt = self._fmt
        body: list[str] = []
        closing: list[str] = []
        while True:
            ahead = src.peek(LOOKAHEAD)
            if not ahead:
                break
            size = fmt.content_end(ahead, header)
            if size is not None:
                closing = src.take(size)
                break
            if fmt.implicit_end:
                nxt = fmt.header(ahead)
                if nxt is not None:
                    break
            body.extend(src.take(1))
        moved: list[str] = []
        if fmt.blank_to_tail:
            while body and not body[-1].strip():
                moved.insert(0, body.pop())
        return Document("".join(body), path=header.path, head=head, tail="".join(moved + closing))
