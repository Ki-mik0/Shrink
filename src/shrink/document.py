"""The unit of work that flows through the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Document:
    """One file (or the whole stream) of text.

    ``text`` is the part a compressor may change. ``head`` and ``tail`` are the
    framing the stream put around it (file headers, closing tags, separators);
    they are written back byte for byte and never compressed.

    ``path`` is the file name taken from the stream framing, or ``None`` when the
    input was not split into files.
    """

    text: str
    path: str | None = None
    head: str = ""
    tail: str = ""
    #: Functions (``name`` or ``Class.method``) to keep whole at the skeleton level;
    #: set by ``--query`` for the parts of a file that match the question.
    keep: frozenset[str] = frozenset()

    @property
    def label(self) -> str:
        """Human-readable name for messages on stderr."""
        return self.path if self.path is not None else "<stdin>"

    @property
    def full_text(self) -> str:
        """Everything this document contributes to the output stream."""
        return self.head + self.text + self.tail

    def with_text(self, text: str) -> Document:
        return replace(self, text=text)
