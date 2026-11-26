"""The pipeline: read → split → compress → write, with optional stats.

Every collaborator is passed in (Dependency Inversion), so tests can swap any
step for a fake. Only one document is held in memory at a time, except with a
planner (``--query`` / ``--budget``), which has to see every document first.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Protocol

from shrink import messages
from shrink.compressors.base import Compressor, Step, Traceable
from shrink.document import Document
from shrink.splitting.documents import Splitter

if TYPE_CHECKING:
    from shrink.retrieval.planner import Plan


class Reader(Protocol):
    def lines(self) -> Iterator[str]: ...


class Writer(Protocol):
    def write(self, doc: Document) -> None: ...


class Reporter(Protocol):
    def record(self, before: Document, after: Document, steps: Sequence[Step] = ()) -> None: ...

    def finish(self) -> None: ...


class Planner(Protocol):
    def plan(
        self, docs: list[Document], level: int, compress: Callable[[Document, int], Document]
    ) -> list[Plan]: ...


class Pipeline:
    def __init__(
        self,
        reader: Reader,
        splitter: Splitter,
        compressor: Compressor,
        writer: Writer,
        reporter: Reporter | None = None,
        *,
        planner: Planner | None = None,
        trailer: Document | None = None,
        warn: Callable[[str], None] = messages.warn,
    ) -> None:
        self._reader = reader
        self._splitter = splitter
        self._compressor = compressor
        self._writer = writer
        self._reporter = reporter
        self._planner = planner
        self._trailer = trailer  # written after the last document (--brief)
        self._warn = warn
        self._warned: set[str] = set()

    def run(self, level: int) -> int:
        """Process the whole stream; return the number of documents written."""
        want_steps = self._reporter is not None and getattr(self._reporter, "wants_steps", False)
        count = 0
        for doc, doc_level, ready in self._jobs(level):
            steps: list[Step] = []
            if ready is not None and not want_steps:
                out = ready
            else:
                out, steps = self._compress(doc, doc_level, want_steps)
            self._writer.write(out)
            if self._reporter is not None:
                self._reporter.record(doc, out, steps)
            count += 1
        if self._trailer is not None and count:
            self._writer.write(self._trailer)
            if self._reporter is not None:
                empty = Document("", path=self._trailer.path)
                self._reporter.record(empty, self._trailer, [Step("brief note", self._trailer)])
        if self._reporter is not None:
            self._reporter.finish()
        return count

    def _jobs(self, level: int) -> Iterator[tuple[Document, int, Document | None]]:
        """(document, level, output if already computed), in input order."""
        docs = self._splitter.split(self._reader.lines())
        if self._planner is None:
            for doc in docs:
                yield doc, level, None
            return

        def compress(doc: Document, doc_level: int) -> Document:
            return self._compress(doc, doc_level, False)[0]

        for plan in self._planner.plan(list(docs), level, compress):
            yield plan.doc, plan.level, plan.output

    def _compress(self, doc: Document, level: int, want_steps: bool) -> tuple[Document, list[Step]]:
        try:
            if want_steps and isinstance(self._compressor, Traceable):
                steps = self._compressor.trace(doc, level)
                return (steps[-1].after if steps else doc), steps
            return self._compressor.compress(doc, level), []
        except Exception as exc:  # fail-open: a bad file must not break the pipe
            message = (
                f"{doc.label}: {self._compressor.name} mode failed "
                f"({type(exc).__name__}: {exc}); passed through unchanged"
            )
            if message not in self._warned:  # a planner may try several levels
                self._warned.add(message)
                self._warn(message)
            return doc, []
