"""``--no-docs``: drop docstrings (Python) and JSDoc (JavaScript) after the mode.

At level 3 a docstring's first sentence is often the largest part of what is
left of a function, and a model that sees docstrings everywhere writes them in
its answer too. Names, parameters and types usually say enough. This step is
not part of the levels because it throws away prose that can matter; it runs
only when asked for.

Each file is checked like the levels are (Python: same syntax tree apart from
docstrings; JavaScript: same code tokens). A file that fails the check keeps
what the mode made of it.
"""

from __future__ import annotations

from collections.abc import Callable

from shrink.compressors.auto import choose_mode
from shrink.document import Document


def _drop_python(text: str) -> str:
    from shrink.compressors.python_code import drop_docstrings

    return drop_docstrings(text)


def _drop_js(text: str) -> str:
    from shrink.compressors.web import drop_js_docs

    return drop_js_docs(text)


def _dropper(doc: Document) -> Callable[[str], str] | None:
    mode = choose_mode(doc)
    if mode == "code":
        return _drop_python
    if mode == "web":
        from shrink.compressors.web import kind_of

        return _drop_js if kind_of(doc.path) == "js" else None
    return None


class NoDocsCompressor:
    name = "no-docs"
    summary = "drop Python docstrings and JSDoc comments"
    help = __doc__ or ""

    def compress(self, doc: Document, level: int) -> Document:
        drop = _dropper(doc)
        if drop is None:
            return doc
        try:
            return doc.with_text(drop(doc.text))
        except (SyntaxError, ValueError):  # parse or check failed: keep the mode's output
            return doc
