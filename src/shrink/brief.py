"""Notes after the context that make the model's answer shorter.

Compression only shrinks what the model reads. In tasks where the model writes a
lot (code generation), most of the tokens are in the answer:

* ``--brief`` (:data:`BRIEF_NOTE`): no prose around the code, no comments or
  docstrings, no files repeated without changes. About 60 tokens of input.
* ``--lazy`` (:data:`LAZY_NOTE`): an existing file comes back the way the model
  saw it, with stub bodies and ``...`` gaps for what did not change;
  ``shrink apply`` fills them in from the repository (:mod:`shrink.restore`).
* ``--patch`` (:data:`PATCH_NOTE`): for an existing file only the definitions
  added or changed, as ``## Patch:``; ``shrink apply`` merges them by name.
* ``--dedent`` (:data:`DEDENT_NOTE`): code in the compact layout of the context
  (Python with one space per level, JavaScript/HTML/CSS without indentation);
  ``shrink apply`` re-indents it (:mod:`shrink.indent`).

The notes come after the context; a format the task itself demands ("return
every file in full") wins over a note placed before it, so the lazy and dedent
notes say that they are an exception to such a rule.

:data:`EDITS_NOTE` is an earlier experiment kept for the eval: SEARCH/REPLACE
edits. At level 3 the model copies the shortened view into SEARCH, so the edits
do not apply; :data:`LAZY_NOTE` matches by name instead.
"""

from __future__ import annotations

from shrink.document import Document

BRIEF_NOTE = (
    "\n[Answer tersely: no preamble, no restating the task, no closing summary or "
    "explanations unless asked. In code you write: no comments or docstrings, and "
    "no files you do not change. Give each file once, complete and final; never "
    "rewrite a file you already gave. Keep every required feature.]\n"
)

LAZY_NOTE = (
    "\n[To save tokens, even where complete files are asked for: return a file that "
    "already exists the way you were shown it, with your changes. Keep a body you do "
    "not change as `...` (Python) or `{/* ... */}` (JavaScript), and write a line "
    "`...` (Python), `/* ... */` (JavaScript, CSS) or `<!-- ... -->` (HTML) in place "
    "of unchanged code you leave out. Left-out parts are filled in from the "
    "repository by name (HTML elements by id). Write new code in full.]\n"
)

PATCH_NOTE = (
    "\n[Exception to the answer format above, to save tokens: for a file that already "
    "exists, write `## Patch: <path>` instead of `## File:` and give only the top-level "
    "definitions you add or change (imports, variables, functions, classes; in HTML the "
    "changed elements), each one whole. Everything else stays as it is. Never write a "
    "body shown to you as `...`. New files stay `## File:`, and each file holds its own "
    "code: never inline it in another file.]\n"
)

DEDENT_NOTE = (
    "\n[Write code in the layout of the files above, to save tokens: Python indented "
    "with one space per level; JavaScript, HTML and CSS without indentation. It is "
    "re-indented afterwards.]\n"
)

EDITS_NOTE = (
    "\n\n[Exception to the answer format above, to save tokens: do not return a file "
    "that already exists in full. Return only its changes, as\n## Edit: <path>\n```\n"
    "<<<<<<< SEARCH\nexact lines you were shown\n=======\nnew lines\n>>>>>>> REPLACE\n"
    "```\nSeveral blocks per file are fine. If the lines to change were not shown "
    "(a body given as `...`), return that file in full instead.]\n"
)

BRIEF_PATH = "(brief note)"
NOTE_PATH = "(answer notes)"


def brief_document() -> Document:
    """The note as a document of its own, so --stats and --diff show what it costs."""
    return Document(BRIEF_NOTE, path=BRIEF_PATH)


def answer_notes(
    *, brief: bool = False, lazy: bool = False, dedent: bool = False, patch: bool = False
) -> str:
    """The notes asked for, in a fixed order."""
    chosen = [
        (brief, BRIEF_NOTE),
        (dedent, DEDENT_NOTE),
        (lazy, LAZY_NOTE),
        (patch, PATCH_NOTE),
    ]
    return "".join(note for wanted, note in chosen if wanted)


def notes_document(
    *, brief: bool = False, lazy: bool = False, dedent: bool = False, patch: bool = False
) -> Document | None:
    text = answer_notes(brief=brief, lazy=lazy, dedent=dedent, patch=patch)
    if not text:
        return None
    return Document(text, path=BRIEF_PATH if text == BRIEF_NOTE else NOTE_PATH)
