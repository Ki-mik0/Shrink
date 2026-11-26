"""SEARCH/REPLACE edits in a model's answer, and applying them to the real files.

``--brief`` asks the model not to repeat a file that already exists but to give
only its changes, in the block format agent tools such as aider use::

    ## Edit: backend/app/main.py
    ```
    <<<<<<< SEARCH
    from app.routers import health
    =======
    from app.routers import auth, health, todos
    >>>>>>> REPLACE
    ```

The model quotes what it saw, and with shrink in between it saw a compressed
copy: comments and blank lines may be missing and indentation may differ. So a
SEARCH text is matched exactly first, then line by line ignoring indentation,
trailing spaces, blank lines and comment-only lines of the real file. An edit
that matches nowhere, or in more than one place, fails; it is never guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_EDIT_HEADER = re.compile(r"^## Edit: *(?P<path>.+?) *$", re.M)
_ANY_HEADER = re.compile(r"^## (?:File|Edit): ", re.M)
_BLOCK = re.compile(
    r"^<{7} SEARCH[ \t]*\r?\n(?P<search>.*?)^={7}[ \t]*\r?\n(?P<replace>.*?)^>{7} REPLACE[ \t]*$",
    re.M | re.S,
)
_COMMENT_ONLY = re.compile(r"^\s*(#|//|/\*|\*|<!--)")


@dataclass(frozen=True)
class Edit:
    search: str
    replace: str


@dataclass(frozen=True)
class FileEdits:
    path: str
    edits: list[Edit]


class EditError(ValueError):
    pass


def split_edits(text: str) -> tuple[str, list[FileEdits]]:
    """``(answer without the edit sections, the edits)``."""
    found: list[FileEdits] = []
    kept: list[str] = []
    pos = 0
    for header in _EDIT_HEADER.finditer(text):
        if header.start() < pos:
            continue
        nxt = _ANY_HEADER.search(text, header.end())
        end = nxt.start() if nxt else len(text)
        kept.append(text[pos : header.start()])
        body = text[header.end() : end]
        edits = [Edit(m["search"], m["replace"]) for m in _BLOCK.finditer(body)]
        found.append(FileEdits(header["path"].strip().strip("`"), edits))
        pos = end
    kept.append(text[pos:])
    return "".join(kept), found


def apply_edits(original: str, edits: list[Edit]) -> str:
    """Apply the edits in order; raise :class:`EditError` if one does not match once."""
    text = original
    for number, edit in enumerate(edits, start=1):
        text = _apply_one(text, edit, number)
    return text


def _apply_one(text: str, edit: Edit, number: int) -> str:
    search, replace = edit.search, edit.replace
    if not search.strip():  # nothing to find: append
        sep = "" if not text or text.endswith("\n") else "\n"
        return text + sep + replace
    # exact match, starting at a line start (a match inside an indented line would
    # leave the replacement without the line's indentation)
    starts = [
        m.start()
        for m in re.finditer(re.escape(search), text)
        if m.start() == 0 or text[m.start() - 1] == "\n"
    ]
    if len(starts) > 1:
        raise EditError(f"edit {number}: SEARCH text occurs {len(starts)} times")
    if starts:
        return text[: starts[0]] + replace + text[starts[0] + len(search) :]
    start, end = _loose_span(text, search, number)
    replace = _reindent(replace, _indent(search), _indent(text[start:end]))
    if replace and not replace.endswith("\n") and text[start:end].endswith("\n"):
        replace += "\n"
    return text[:start] + replace + text[end:]


def _indent(block: str) -> str:
    """Indentation of the first significant line."""
    for line in block.splitlines():
        if line.strip() and not _COMMENT_ONLY.match(line):
            return line[: len(line) - len(line.lstrip())]
    return ""


def _reindent(block: str, seen: str, real: str) -> str:
    """Shift ``block`` from the indentation the model saw to the file's real one."""
    if seen == real:
        return block
    out = []
    for line in block.splitlines(keepends=True):
        shifted = line.strip() and line.startswith(seen)
        out.append(real + line.removeprefix(seen) if shifted else line)
    return "".join(out)


def _significant(lines: list[str]) -> list[tuple[int, str]]:
    return [
        (i, line.strip())
        for i, line in enumerate(lines)
        if line.strip() and not _COMMENT_ONLY.match(line)
    ]


def _loose_span(text: str, search: str, number: int) -> tuple[int, int]:
    """Character span of the lines matching ``search`` loosely (see the module docs)."""
    lines = text.splitlines(keepends=True)
    wanted = [line for _, line in _significant(search.splitlines())]
    if not wanted:
        raise EditError(f"edit {number}: SEARCH has only blank or comment lines")
    have = _significant(lines)
    hits = [
        k
        for k in range(len(have) - len(wanted) + 1)
        if all(have[k + j][1] == wanted[j] for j in range(len(wanted)))
    ]
    if len(hits) != 1:
        problem = "not found" if not hits else f"found {len(hits)} times"
        raise EditError(f"edit {number}: SEARCH text {problem}")
    first = have[hits[0]][0]
    last = have[hits[0] + len(wanted) - 1][0]
    start = sum(len(line) for line in lines[:first])
    end = sum(len(line) for line in lines[: last + 1])
    return start, end
