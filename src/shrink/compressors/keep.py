"""``--keep``: functions that stay whole at the skeleton level.

A model asked to change a function it only sees as ``...`` has to guess its
code (see :mod:`shrink.restore`). Naming the functions the task changes keeps
exactly those bodies in the context and lets the rest stay skeletons.
"""

from __future__ import annotations

from dataclasses import replace

from shrink.compressors.base import Compressor, Step, Traceable
from shrink.document import Document


class KeepCompressor:
    """Another compressor, with ``names`` added to every document's ``keep``."""

    def __init__(self, inner: Compressor, names: frozenset[str]) -> None:
        self.inner = inner
        self.names = names
        self.name = inner.name
        self.summary = inner.summary
        self.help = inner.help

    def _marked(self, doc: Document) -> Document:
        return replace(doc, keep=doc.keep | self.names)

    def compress(self, doc: Document, level: int) -> Document:
        return self.inner.compress(self._marked(doc), level)

    def trace(self, doc: Document, level: int) -> list[Step]:
        if isinstance(self.inner, Traceable):
            return self.inner.trace(self._marked(doc), level)
        return [Step(self.name, self.compress(doc, level))]
