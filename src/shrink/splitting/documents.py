"""Stream → documents.

Stage 1 ships only :class:`WholeStreamSplitter`. Delimiter-based splitters
(Repomix, files-to-prompt, ``--delimiter``) plug into the same protocol later.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol

from shrink.document import Document


class Splitter(Protocol):
    def split(self, lines: Iterable[str]) -> Iterator[Document]: ...


class WholeStreamSplitter:
    """Treats the entire input as a single document. Empty input yields nothing."""

    def split(self, lines: Iterable[str]) -> Iterator[Document]:
        text = "".join(lines)
        if text:
            yield Document(text)
