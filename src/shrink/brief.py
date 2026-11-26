"""``--brief``: a short note after the context asking the model for a terse answer.

Compression only shrinks what the model reads. In tasks where the model writes a
lot (code generation), most of the tokens are in the answer, and a large part of
them is prose around the code, comments and docstrings, and files repeated
without changes, or rewritten after it was already given. The note asks the
model to leave those out; it costs about 60 tokens of input once.

:data:`EDITS_NOTE` is an experiment for the end of a prompt (where an agent
proxy could add it): give existing files as SEARCH/REPLACE edits instead of
whole copies; :mod:`shrink.edits` applies them. Placed before the task, the
model ignored it in favour of the task's own "return every file in full".
"""

from __future__ import annotations

from shrink.document import Document

BRIEF_NOTE = (
    "\n[Answer tersely: no preamble, no restating the task, no closing summary or "
    "explanations unless asked. In code you write: no comments or docstrings, and "
    "no files you do not change. Give each file once, complete and final; never "
    "rewrite a file you already gave. Keep every required feature.]\n"
)

EDITS_NOTE = (
    "\n\n[Exception to the answer format above, to save tokens: do not return a file "
    "that already exists in full. Return only its changes, as\n## Edit: <path>\n```\n"
    "<<<<<<< SEARCH\nexact lines you were shown\n=======\nnew lines\n>>>>>>> REPLACE\n"
    "```\nSeveral blocks per file are fine. If the lines to change were not shown "
    "(a body given as `...`), return that file in full instead.]\n"
)

BRIEF_PATH = "(brief note)"


def brief_document() -> Document:
    """The note as a document of its own, so --stats and --diff show what it costs."""
    return Document(BRIEF_NOTE, path=BRIEF_PATH)
