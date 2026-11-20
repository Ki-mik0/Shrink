"""Choose the splitter for the ``--format`` and ``--delimiter`` options."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator

from shrink.document import Document
from shrink.splitting.documents import Splitter, WholeStreamSplitter
from shrink.splitting.formats import DETECT_LINES, FORMATS, detect
from shrink.splitting.framed import FramedSplitter

AUTO = "auto"
NONE = "none"
FORMAT_NAMES = [AUTO, NONE, *FORMATS]


class AutoSplitter:
    """Looks at the first lines, then streams with the format it recognised."""

    def split(self, lines: Iterable[str]) -> Iterator[Document]:
        source = iter(lines)
        head = list(itertools.islice(source, DETECT_LINES))
        fmt = detect(head)
        stream = itertools.chain(head, source)
        splitter: Splitter = FramedSplitter(fmt) if fmt is not None else WholeStreamSplitter()
        yield from splitter.split(stream)


def make_splitter(fmt: str = AUTO, delimiter: str | None = None) -> Splitter:
    """``delimiter`` wins over ``fmt``. Raises ``re.error`` for a bad pattern."""
    if delimiter is not None:
        from shrink.splitting.delimiter import DelimiterSplitter

        return DelimiterSplitter(delimiter)
    if fmt == NONE:
        return WholeStreamSplitter()
    if fmt == AUTO:
        return AutoSplitter()
    return FramedSplitter(FORMATS[fmt])
