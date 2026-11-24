"""Split on a user-supplied delimiter line (``--delimiter REGEX``)."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator

from shrink.document import Document


class DelimiterSplitter:
    """Every line that matches ``pattern`` starts a new document.

    The matching line is the document's ``head`` and is written back unchanged.
    A named group ``path`` in the pattern gives the document its file name.
    Lines before the first delimiter form a document of their own.
    """

    def __init__(self, pattern: str) -> None:
        self._pattern = re.compile(pattern)

    def split(self, lines: Iterable[str]) -> Iterator[Document]:
        head = ""
        path: str | None = None
        body: list[str] = []
        for line in lines:
            match = self._pattern.search(line.rstrip("\r\n"))
            if match is None:
                body.append(line)
                continue
            if head or body:
                yield Document("".join(body), path=path, head=head)
            head = line
            groups = match.groupdict()
            path = groups.get("path")
            body = []
        if head or body:
            yield Document("".join(body), path=path, head=head)
