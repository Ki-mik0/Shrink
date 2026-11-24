"""Run several compressors one after another, as one compressor."""

from __future__ import annotations

from collections.abc import Sequence

from shrink.compressors.base import Compressor, Step, Traceable
from shrink.document import Document


class ChainCompressor:
    """``a`` then ``b`` then ``c``: each gets the previous one's output.

    The chain is a compressor itself, so the pipeline does not know the
    difference (Liskov). It is also traceable: the per-step report shows what
    every member saved.
    """

    def __init__(self, members: Sequence[Compressor]) -> None:
        if not members:
            raise ValueError("a chain needs at least one compressor")
        self._members = tuple(members)
        self.name = "+".join(member.name for member in self._members)
        self.summary = " then ".join(member.name for member in self._members)
        self.help = "\n\n".join(f"[{member.name}]\n{member.help}" for member in self._members)

    def compress(self, doc: Document, level: int) -> Document:
        for member in self._members:
            doc = member.compress(doc, level)
        return doc

    def trace(self, doc: Document, level: int) -> list[Step]:
        steps: list[Step] = []
        for member in self._members:
            if isinstance(member, Traceable):
                part = member.trace(doc, level)
            else:
                part = [Step(member.name, member.compress(doc, level))]
            steps.extend(part)
            if part:
                doc = part[-1].after
        return steps
