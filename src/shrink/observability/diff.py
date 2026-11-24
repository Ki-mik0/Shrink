"""``--diff``: show on stderr what was removed, then the usual savings report.

The diff is a plain unified diff of each document's text (the framing around a
file is never changed, so it is left out), written raw with no ``shrink:``
prefix, so that ``2> changes.diff`` gives a file other tools can read.
"""

from __future__ import annotations

import difflib
import sys
from collections.abc import Sequence
from typing import TextIO

from shrink.compressors.base import Step
from shrink.document import Document
from shrink.observability.stats import StatsReporter

MAX_DIFF_CHARS = 200_000  # difflib is quadratic in the worst case


class DiffReporter:
    """A :class:`StatsReporter` that also prints each document's diff."""

    wants_steps = True

    def __init__(self, stats: StatsReporter, stream: TextIO | None = None) -> None:
        self._stats = stats
        self._stream = stream

    def _out(self) -> TextIO:
        return self._stream if self._stream is not None else sys.stderr

    def record(self, before: Document, after: Document, steps: Sequence[Step] = ()) -> None:
        self._write_diff(before, after)
        self._stats.record(before, after, steps)

    def finish(self) -> None:
        self._stats.finish()

    def _write_diff(self, before: Document, after: Document) -> None:
        out = self._out()
        label = before.label
        if before.text == after.text:
            out.write(f"shrink: {label}: no changes\n")
        elif len(before.text) + len(after.text) > MAX_DIFF_CHARS:
            out.write(f"shrink: {label}: diff skipped (larger than {MAX_DIFF_CHARS} characters)\n")
        else:
            lines = difflib.unified_diff(
                before.text.splitlines(),
                after.text.splitlines(),
                fromfile=f"{label} (before)",
                tofile=f"{label} (after)",
                n=1,
                lineterm="",
            )
            out.write("".join(line + "\n" for line in lines))
        out.flush()
