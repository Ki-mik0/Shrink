"""Write documents to the output stream as soon as they are ready."""

from __future__ import annotations

import sys
from typing import BinaryIO

from shrink.document import Document
from shrink.io import ENCODING, ERRORS


class StdoutWriter:
    """Encodes each document and flushes it, so the next program sees it at once."""

    def __init__(self, stream: BinaryIO | None = None) -> None:
        self._stream = stream

    def _target(self) -> BinaryIO:
        if self._stream is not None:
            return self._stream
        return sys.stdout.buffer

    def write(self, doc: Document) -> None:
        text = doc.full_text
        if not text:
            return
        target = self._target()
        target.write(text.encode(ENCODING, ERRORS))
        target.flush()
