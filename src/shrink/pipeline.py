"""The pipeline: read → split → compress → write, with optional stats.

Every collaborator is passed in (Dependency Inversion), so tests can swap any
step for a fake. Only one document is held in memory at a time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Protocol

from shrink import messages
from shrink.compressors.base import Compressor, Step, Traceable
from shrink.document import Document
from shrink.splitting.documents import Splitter


class Reader(Protocol):
    def lines(self) -> Iterator[str]: ...


class Writer(Protocol):
    def write(self, doc: Document) -> None: ...


class Reporter(Protocol):
    def record(self, before: Document, after: Document, steps: Sequence[Step] = ()) -> None: ...

    def finish(self) -> None: ...


class Pipeline:
    def __init__(
        self,
        reader: Reader,
        splitter: Splitter,
        compressor: Compressor,
        writer: Writer,
        reporter: Reporter | None = None,
        *,
        warn: Callable[[str], None] = messages.warn,
    ) -> None:
        self._reader = reader
        self._splitter = splitter
        self._compressor = compressor
        self._writer = writer
        self._reporter = reporter
        self._warn = warn

    def run(self, level: int) -> int:
        """Process the whole stream; return the number of documents written."""
        want_steps = self._reporter is not None and getattr(self._reporter, "wants_steps", False)
        count = 0
        for doc in self._splitter.split(self._reader.lines()):
            out, steps = self._compress(doc, level, want_steps)
            self._writer.write(out)
            if self._reporter is not None:
                self._reporter.record(doc, out, steps)
            count += 1
        if self._reporter is not None:
            self._reporter.finish()
        return count

    def _compress(self, doc: Document, level: int, want_steps: bool) -> tuple[Document, list[Step]]:
        try:
            if want_steps and isinstance(self._compressor, Traceable):
                steps = self._compressor.trace(doc, level)
                return (steps[-1].after if steps else doc), steps
            return self._compressor.compress(doc, level), []
        except Exception as exc:  # fail-open: a bad file must not break the pipe
            self._warn(
                f"{doc.label}: {self._compressor.name} mode failed "
                f"({type(exc).__name__}: {exc}); passed through unchanged"
            )
            return doc, []
