"""The Strategy interface every compression mode implements."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from shrink.document import Document

MIN_LEVEL = 1
MAX_LEVEL = 3
DEFAULT_LEVEL = 2

LEVEL_NAMES = {1: "light", 2: "medium", 3: "skeleton"}


class Compressor(Protocol):
    """A compression mode.

    Contract (Liskov): takes a :class:`Document`, returns a :class:`Document`.
    It may raise on input it cannot handle; the pipeline then passes the
    document through unchanged (fail-open).
    """

    name: str
    summary: str
    help: str

    def compress(self, doc: Document, level: int) -> Document: ...


@dataclass(frozen=True)
class Step:
    """The document after one named step of a compressor (for the per-step report)."""

    name: str
    after: Document


@runtime_checkable
class Traceable(Protocol):
    """A compressor that can show its intermediate results.

    ``trace(doc, level)[-1].after`` must equal ``compress(doc, level)``; the steps
    only explain where the savings came from.
    """

    def trace(self, doc: Document, level: int) -> list[Step]: ...


def check_level(level: int) -> None:
    if not MIN_LEVEL <= level <= MAX_LEVEL:
        raise ValueError(f"level must be between {MIN_LEVEL} and {MAX_LEVEL}, got {level}")
