"""Code mode for Python: comments, docstrings and function bodies.

Every change is an edit on the original text (start, end, replacement), never a
regeneration from the tree, so formatting and everything not targeted stays as
written. ``ast`` reports columns as UTF-8 byte offsets; they are converted to
character offsets per line, which keeps non-ASCII source (Cyrillic, emoji)
intact.

Each level is checked before its result is used (see :mod:`shrink.validation`):
the output must parse and must keep every signature. A file that cannot be
parsed, or whose output fails a check, raises; the pipeline then passes it
through unchanged with a warning (fail-open).
"""

from __future__ import annotations

import ast
import functools
import io
import re
import tokenize
from collections.abc import Callable

from shrink import validation
from shrink.compressors.base import Step, check_level
from shrink.document import Document

HELP = """\
Python source. Every level keeps the code valid: the result is parsed again and
must have the same signatures (and, at levels 1-2, the same code) as the input.
Files that do not parse are passed through unchanged with a warning.

  level 1  light     remove comments (shebang, encoding line and pragmas such as
                     `# type: ignore`, `# noqa` are kept), trailing whitespace and
                     extra blank lines; strings are never touched
  level 2  medium    + shorten docstrings to their first line
  level 3  skeleton  + replace function bodies with `...`, keeping signatures,
                     type hints, decorators and the one-line docstring; classes,
                     imports and module-level code stay

Generators become plain functions at level 3: the skeleton describes an API, it
is not meant to run."""

STEP_NAMES = {1: "comments", 2: "docstrings", 3: "bodies"}

_PRAGMA = re.compile(
    r"#\s*(type:|noqa|pragma|fmt:|pylint:|mypy:|isort:|ruff:|nosec|pyright:|pyre-|flake8)",
    re.IGNORECASE,
)
_CODING = re.compile(r"^[ \t\f]*#.*?coding[:=][ \t]*[-\w.]+")
_LONE_CR = re.compile(r"\r(?!\n)")
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
MAX_SUMMARY_CHARS = 200
_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_STRING_START = {
    getattr(tokenize, name)
    for name in ("FSTRING_START", "TSTRING_START")
    if hasattr(tokenize, name)
}
_STRING_END = {
    getattr(tokenize, name) for name in ("FSTRING_END", "TSTRING_END") if hasattr(tokenize, name)
}


class CodeCompressor:
    name = "code"
    summary = "Python: drop comments, shorten docstrings, reduce functions to signatures"
    help = HELP

    def compress(self, doc: Document, level: int) -> Document:
        return self.trace(doc, level)[-1].after

    def trace(self, doc: Document, level: int) -> list[Step]:
        check_level(level)
        steps: list[Step] = []
        text = doc.text
        for k in range(1, level + 1):
            new = skeletonize(text, doc.keep) if k == 3 else _LEVELS[k](text)
            if len(new) <= len(text):  # guard: a step never makes the file longer
                text = new
            steps.append(Step(STEP_NAMES[k], doc.with_text(text)))
        return steps


# ------------------------------------------------------------------ source helpers


class _Source:
    """Text plus the conversions between ``ast`` positions and string offsets."""

    def __init__(self, text: str) -> None:
        if _LONE_CR.search(text):
            raise ValueError("lone CR line endings are not supported")
        self.text = text
        self.starts = [0] + [m.end() for m in re.finditer("\n", text)]

    def line(self, lineno: int) -> str:
        start = self.starts[lineno - 1]
        end = self.starts[lineno] if lineno < len(self.starts) else len(self.text)
        return self.text[start:end]

    def offset(self, lineno: int, byte_col: int) -> int:
        prefix = self.line(lineno).encode("utf-8")[:byte_col]
        return self.starts[lineno - 1] + len(prefix.decode("utf-8"))

    def start_of(self, node: ast.stmt | ast.expr) -> int:
        return self.offset(node.lineno, node.col_offset)

    def end_of(self, node: ast.stmt | ast.expr) -> int:
        assert node.end_lineno is not None
        assert node.end_col_offset is not None
        return self.offset(node.end_lineno, node.end_col_offset)


def _apply(text: str, edits: list[tuple[int, int, str]]) -> str:
    out: list[str] = []
    pos = 0
    for start, end, replacement in sorted(edits):
        if start < pos:
            raise ValueError("overlapping edits")
        out.append(text[pos:start])
        out.append(replacement)
        pos = end
    out.append(text[pos:])
    return "".join(out)


_BOM = "\ufeff"


def _with_bom(transform: Callable[[str], str]) -> Callable[[str], str]:
    """Keep a leading BOM out of the parser's way and put it back afterwards."""

    @functools.wraps(transform)
    def wrapper(text: str) -> str:
        bom = _BOM if text.startswith(_BOM) else ""
        body = text[len(bom) :]
        if not body.strip():
            return text
        return bom + transform(body)

    return wrapper


def _docstring(body: list[ast.stmt]) -> ast.Expr | None:
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return body[0]
    return None


def _scopes(
    tree: ast.AST,
) -> list[ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node for node in ast.walk(tree) if isinstance(node, (ast.Module, ast.ClassDef, *_FUNCTIONS))
    ]


# ------------------------------------------------------------------ level 1


@_with_bom
def strip_comments(text: str) -> str:
    """Comments, trailing whitespace and repeated blank lines, outside strings."""
    before = ast.parse(text)
    _Source(text)  # rejects line endings the row arithmetic below cannot handle
    lines = text.split("\n")
    comments, keep_ws, keep_blank = _scan_tokens(text, lines)
    out: list[str] = []
    for row, raw in enumerate(lines, start=1):
        cr = raw.endswith("\r")
        body = raw[:-1] if cr else raw
        if row in comments:
            code = body[: comments[row]]
            if not code.strip():
                continue  # a comment-only line disappears entirely
            body = code.rstrip()
        elif row not in keep_ws:
            body = body.rstrip()
        if not body and row not in keep_blank and (not out or out[-1] in ("", "\r")):
            continue
        out.append(body + ("\r" if cr else ""))
    while out and out[-1] in ("", "\r"):
        out.pop()
    result = "\n".join(out)
    if text.endswith("\n") and result:
        result += "\n"  # a CR, if the file uses CRLF, is still on the last line
    validation.check_same_code(before, result)
    return result


def _scan_tokens(text: str, lines: list[str]) -> tuple[dict[int, int], set[int], set[int]]:
    """Removable comments (row -> column) and the rows that lie inside strings."""
    comments: dict[int, int] = {}
    keep_ws: set[int] = set()  # rows whose trailing whitespace belongs to a string
    keep_blank: set[int] = set()  # blank rows that belong to a string
    open_strings: list[int] = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            if not _keep_comment(tok.string, row, lines):
                comments[row] = col
        elif tok.type == tokenize.STRING and tok.start[0] != tok.end[0]:
            _protect(tok.start[0], tok.end[0], keep_ws, keep_blank)
        elif tok.type in _STRING_START:
            open_strings.append(tok.start[0])
        elif tok.type in _STRING_END and open_strings:
            start = open_strings.pop()
            if start != tok.end[0]:
                _protect(start, tok.end[0], keep_ws, keep_blank)
    return comments, keep_ws, keep_blank


def _keep_comment(comment: str, row: int, lines: list[str]) -> bool:
    if row == 1 and comment.startswith("#!"):
        return True
    if row <= 2 and _CODING.match(lines[row - 1]):
        return True
    return _PRAGMA.match(comment) is not None


def _protect(first: int, last: int, keep_ws: set[int], keep_blank: set[int]) -> None:
    keep_ws.update(range(first, last))
    keep_blank.update(range(first + 1, last))


# ------------------------------------------------------------------ level 2


@_with_bom
def shorten_docstrings(text: str) -> str:
    """Module, class and function docstrings keep only their first line."""
    tree = ast.parse(text)
    src = _Source(text)
    edits: list[tuple[int, int, str]] = []
    for scope in _scopes(tree):
        doc = _docstring(scope.body)
        if doc is None:
            continue
        assert isinstance(doc.value, ast.Constant)
        content = str(doc.value.value).strip()
        if "\n" not in content:
            continue  # already one line: leave its quoting alone
        start, end = src.start_of(doc), src.end_of(doc)
        literal = _one_line_docstring(docstring_summary(content))
        if len(literal) < end - start:
            edits.append((start, end, literal))
    result = _apply(text, edits)
    validation.check_same_code(tree, result, ignore_docstrings=True)
    return result


def docstring_summary(content: str) -> str:
    """The first sentence of the first paragraph, or the first line if it runs on.

    PEP 257 asks for a one-line summary, but many docstrings wrap their first
    sentence over two lines; cutting at the line would end mid-sentence.
    """
    first_line = content.split("\n", 1)[0].strip()
    paragraph = " ".join(line.strip() for line in content.split("\n\n", 1)[0].split("\n"))
    match = _SENTENCE_END.search(paragraph)
    if match is not None and match.end() <= MAX_SUMMARY_CHARS:
        return paragraph[: match.end()]
    return first_line


def _one_line_docstring(summary: str) -> str:
    """A triple-quoted one-liner when that is safe to write, otherwise ``repr``."""
    quote = '"' * 3
    if quote in summary or summary.endswith('"') or chr(92) in summary:
        return repr(summary)
    return quote + summary + quote


# ------------------------------------------------------------------ level 3


def skeletonize(text: str, keep: frozenset[str] = frozenset()) -> str:
    """Function bodies become ``...``; the docstring (if any) stays in front of it.

    Functions named in ``keep`` (``name`` or ``Class.method``) stay whole.
    """
    return _with_bom(functools.partial(_skeleton, keep=keep))(text)


def _skeleton(text: str, keep: frozenset[str]) -> str:
    tree = ast.parse(text)
    src = _Source(text)
    edits: list[tuple[int, int, str]] = []
    _collect_bodies(tree, src, edits, keep, "")
    result = _apply(text, edits)
    validation.check_same_signatures(tree, result)
    return result


def _collect_bodies(
    node: ast.AST,
    src: _Source,
    edits: list[tuple[int, int, str]],
    keep: frozenset[str],
    prefix: str,
) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _FUNCTIONS):
            if prefix + child.name in keep:
                continue
            edit = _body_edit(child, src)
            if edit is not None:
                edits.append(edit)
        elif isinstance(child, ast.ClassDef):
            _collect_bodies(child, src, edits, keep, f"{prefix}{child.name}.")
        else:
            _collect_bodies(child, src, edits, keep, prefix)


def functions(text: str) -> list[tuple[str, str]]:
    """``(qualified name, source)`` of every outermost function and method."""
    tree = ast.parse(text)
    found: list[tuple[str, str]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FUNCTIONS):
                found.append((prefix + child.name, ast.get_source_segment(text, child) or ""))
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)

    visit(tree, "")
    return found


def _body_edit(
    func: ast.FunctionDef | ast.AsyncFunctionDef, src: _Source
) -> tuple[int, int, str] | None:
    body = func.body[1:] if _docstring(func.body) is not None else func.body
    if not body:
        return None
    if len(body) == 1 and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and value.value is Ellipsis:
            return None  # already a stub
    return _statement_start(body[0], src), src.end_of(body[-1]), "..."


def _statement_start(stmt: ast.stmt, src: _Source) -> int:
    decorators = getattr(stmt, "decorator_list", None)
    if not decorators:
        return src.start_of(stmt)
    first: int = min(d.lineno for d in decorators)
    line = src.line(first)
    indent = len(line) - len(line.lstrip(" \t\f"))
    return src.starts[first - 1] + indent


_LEVELS = {1: strip_comments, 2: shorten_docstrings, 3: skeletonize}
