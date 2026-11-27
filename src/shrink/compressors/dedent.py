"""``--dedent``: the context in a layout that costs fewer tokens.

After the mode, Python is re-indented with one space per level and JavaScript,
HTML and CSS lose their indentation (see :mod:`shrink.indent` for why that
saves tokens and how it is checked). A model shown code this way tends to answer
in the same layout, and is also asked to (``--dedent`` adds a note);
``shrink apply`` gives the answer its usual indentation back.

Not part of the levels: the context stays exact, but people read it less easily.
A file that fails the check keeps what the mode made of it.
"""

from __future__ import annotations

from shrink import indent
from shrink.compressors.auto import choose_mode
from shrink.document import Document


def _kind(doc: Document) -> str | None:
    kind = indent.kind_of(doc.path)
    if kind is None and doc.path is None and choose_mode(doc) == "code":
        return "python"
    return kind


class DedentCompressor:
    name = "dedent"
    summary = "Python with one space per indentation level, JS/HTML/CSS without indentation"
    help = __doc__ or ""

    def compress(self, doc: Document, level: int) -> Document:
        kind = _kind(doc)
        if kind is None:
            return doc
        try:
            return doc.with_text(indent.compact(doc.text, kind))
        except (SyntaxError, ValueError):  # parse or check failed: keep the mode's output
            return doc
