"""Read the input stream line by line, without newline translation."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import BinaryIO

from shrink.io import ENCODING, ERRORS


class StdinReader:
    """Yields decoded lines (with their line endings) from a binary stream.

    Decoding per line is safe: in UTF-8 the byte ``0x0A`` never occurs inside a
    multi-byte sequence, so a line boundary never splits a character.
    """

    def __init__(self, stream: BinaryIO | None = None) -> None:
        self._stream = stream

    def _source(self) -> BinaryIO:
        if self._stream is not None:
            return self._stream
        return sys.stdin.buffer

    def lines(self) -> Iterator[str]:
        for raw in self._source():
            yield raw.decode(ENCODING, ERRORS)
