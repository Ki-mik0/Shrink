"""``--brief``: a short note after the context asking the model for a terse answer.

Compression only shrinks what the model reads. In tasks where the model writes a
lot (code generation), most of the tokens are in the answer, and a large part of
them is prose around the code, comments and docstrings, files rewritten after
they were already given, and existing files repeated in full for a few changed
lines. The note asks the model to leave those out and to give changes to
existing files as SEARCH/REPLACE blocks (:mod:`shrink.edits` applies them); it
costs about 140 tokens of input once.
"""

from __future__ import annotations

from shrink.document import Document

BRIEF_NOTE = (
    "\n[Answer tersely: no preamble, no restating the task, no closing summary or "
    "explanations unless asked. In code you write: no comments or docstrings. Give "
    "each new file once, complete and final; never rewrite a file you already gave. "
    "Do not repeat a file that already exists, even if asked for whole files: give "
    "only its changes as\n## Edit: <path>\n```\n<<<<<<< SEARCH\nexact lines shown "
    "above\n=======\nnew lines\n>>>>>>> REPLACE\n```\n(several blocks per file are "
    "fine). If the lines to change are not shown (a body given as `...`), give that "
    "file in full instead. Keep every required feature.]\n"
)

BRIEF_PATH = "(brief note)"


def brief_document() -> Document:
    """The note as a document of its own, so --stats and --diff show what it costs."""
    return Document(BRIEF_NOTE, path=BRIEF_PATH)
