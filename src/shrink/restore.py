"""``shrink apply``: put back what a model left out of its answer.

At level 3 the model sees function bodies as ``...`` (Python) and
``{/* ... */}`` (JavaScript). Asked to change such a file it either rewrites it
in full, inventing the bodies it never saw, or copies the shortened view. With
``--lazy`` the second is what it is asked to do: return an existing file the way
it was shown, keep a body it does not change as a stub, and leave unchanged
parts out with a ``...`` line (``/* ... */`` in JavaScript and CSS,
``<!-- ... -->`` in HTML). This module merges such an answer with the real file.

Parts are matched by name, never by their text, so a shortened docstring or a
stub body cannot make a match fail (that is what broke SEARCH/REPLACE edits at
level 3):

* Python and JavaScript: top-level definitions by name (functions, classes,
  variables), imports by module, class members by name; a stub body is
  replaced by the body of the function of the same (qualified) name.
* HTML: elements by ``id``, otherwise by a tag that occurs once (``<title>``,
  ``<nav>``, ``<main>`` ...); new scripts go before ``</body>``.
* CSS: rules by selector; an empty rule body is the stub.

Without a gap marker the answer is the whole file (stubs are still filled in);
with one, everything the answer does not mention stays as it was, so a partial
answer cannot delete code. Every result is parsed again (Python) before it is
used. This is the structure-aware edit format idea of Cheng et al. (2026, "To
Diff or Not to Diff?"), with symbols as anchors instead of copied lines.
"""

from __future__ import annotations

import ast
import itertools
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field

from shrink import indent
from shrink.compressors import web

_GAP_JS = re.compile(r"^\s*(/\*\s*\.\.\.[^*]*\*/|//\s*\.\.\..*)\s*$")
_GAP_HTML = re.compile(r"^<!--\s*\.\.\.[^>]*-->$")
_STUB_JS = "/* ... */"
_UNIQUE_TAGS = ("title", "head", "body", "header", "nav", "main", "footer", "h1", "form")
_HEAD_TAGS = ("link", "meta", "style", "title", "base")


class RestoreError(ValueError):
    """The answer cannot be merged with the file (it does not parse, for example)."""


@dataclass
class Report:
    """What was put back into one file."""

    path: str
    status: str = "new"  # new | changed | unchanged
    replaced: int = 0  # parts of the file the answer rewrote
    inserted: int = 0  # parts the answer added
    kept: int = 0  # parts left out of the answer and taken from the file
    bodies: int = 0  # stub bodies filled in from the file
    unresolved: list[str] = field(default_factory=list)  # stubs with nothing to fill in
    #: functions the model only saw as stubs but answered with another body; the
    #: file's own code was kept (see ``shown`` in :func:`restore`)
    reverted: list[str] = field(default_factory=list)

    @property
    def restored(self) -> bool:
        return bool(self.kept or self.bodies or self.reverted)

    def describe(self) -> str:
        parts = []
        if self.replaced:
            parts.append(f"{self.replaced} replaced")
        if self.inserted:
            parts.append(f"{self.inserted} added")
        if self.kept:
            parts.append(f"{self.kept} kept")
        if self.bodies:
            parts.append(f"{self.bodies} bodies restored")
        if self.reverted:
            parts.append(
                f"{len(self.reverted)} invented bodies rejected: {', '.join(self.reverted)}"
            )
        if self.unresolved:
            parts.append(f"{len(self.unresolved)} left as stubs: {', '.join(self.unresolved)}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Restored:
    text: str
    report: Report


# ------------------------------------------------------------------ generic merge


@dataclass
class Item:
    """A top-level part of a file: its text includes the blank lines and comments
    in front of it."""

    key: str | None
    text: str
    gap: bool = False
    header: bool = False  # import / module docstring: new ones go after the last import
    node: object = None
    row: int = 1  # first row of ``text`` in its file


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _match(orig: list[Item], answer: list[Item]) -> list[int | None]:
    """For every answer item, the index of the original item it stands for."""
    taken: set[int] = set()
    found: list[int | None] = []
    for item in answer:
        hit = None
        if not item.gap:
            for i, old in enumerate(orig):
                if i in taken or old.gap:
                    continue
                if item.key is not None and old.key == item.key:
                    hit = i
                    break
                if item.key is None and old.key is None and _norm(old.text) == _norm(item.text):
                    hit = i
                    break
        if hit is not None:
            taken.add(hit)
        found.append(hit)
    return found


def merge_items(
    orig: list[Item],
    answer: list[Item],
    report: Report,
    combine: Callable[[Item, Item], str] | None = None,
    *,
    partial: bool = False,
) -> str | None:
    """The merged text, or ``None`` when the answer has no gap (it is the whole file).
    With ``partial`` the answer only ever adds to and replaces parts of the file.

    Answer items replace the original items they match; new ones go next to their
    neighbours in the answer (after the previous matched item, or before the next
    one, or at the end of the gap they follow); everything else is kept.
    """
    if not partial and not any(item.gap for item in answer):
        return None
    matches = _match(orig, answer)
    replace: dict[int, str] = {}
    # New items by where they go: ("before"/"after", original index) or ("end", -1).
    slots: dict[tuple[str, int], list[str]] = defaultdict(list)
    header_end = max((i for i, item in enumerate(orig) if item.header), default=-1)
    for j, item in enumerate(answer):
        hit = matches[j]
        if item.gap:
            continue
        if hit is not None:
            new = combine(orig[hit], item) if combine else item.text
            replace[hit] = _keep_lead(orig[hit].text, new)
            continue
        report.inserted += 1
        text = _spaced(item.text, _usual_gap(orig))
        if partial and not item.header:
            slot = _patch_slot(orig, matches, j)
        else:
            slot = _slot(answer, matches, j, header_end, bool(orig))
        slots[slot].append(text)
    pieces: list[str] = []
    for i, old in enumerate(orig):
        pieces.extend(slots.get(("before", i), []))
        text = replace.get(i, old.text)
        _count(report, old.text, text)
        pieces.append(text)
        pieces.extend(slots.get(("after", i), []))
    pieces.extend(slots.get(("end", -1), []))
    return _join(pieces)


def _patch_slot(orig: list[Item], matches: list[int | None], j: int) -> tuple[str, int]:
    """A new definition in a patch goes after the definition it follows, otherwise
    to the end: placed after an import it could use names defined further down."""
    for k in range(j - 1, -1, -1):
        hit = matches[k]
        if hit is not None:
            return ("end", -1) if orig[hit].header else ("after", hit)
    return "end", -1


def _slot(
    answer: list[Item], matches: list[int | None], j: int, header_end: int, has_orig: bool
) -> tuple[str, int]:
    """Where the new answer item ``j`` goes in the original's order."""
    if answer[j].header and header_end >= 0:
        return "after", header_end
    prev, gap_before = _neighbour(answer, matches, range(j - 1, -1, -1))
    next_, gap_after = _neighbour(answer, matches, range(j + 1, len(answer)))
    if prev is not None and not gap_before:
        return "after", prev
    if next_ is not None and (gap_before or not gap_after):
        return "before", next_
    if gap_before or not has_orig:
        return "end", -1
    if header_end >= 0:
        return "after", header_end
    return "before", 0


def _neighbour(
    answer: list[Item], matches: list[int | None], order: range
) -> tuple[int | None, bool]:
    """The nearest matched item in ``order`` and whether a gap lies before it."""
    gap = False
    for k in order:
        if answer[k].gap:
            gap = True
        elif matches[k] is not None:
            return matches[k], gap
    return None, gap


def whole_items(
    orig: list[Item],
    answer: list[Item],
    report: Report,
    combine: Callable[[Item, Item], str] | None = None,
) -> str:
    """The answer is the whole file: matched items still go through ``combine``
    (which may take the original back), the rest is the answer's."""
    matches = _match(orig, answer)
    pieces: list[str] = []
    for item, hit in zip(answer, matches, strict=True):
        if hit is None:
            report.inserted += not item.gap
            pieces.append(item.text)
            continue
        text = _keep_lead(orig[hit].text, combine(orig[hit], item) if combine else item.text)
        _count(report, orig[hit].text, text)
        pieces.append(text)
    return _join(pieces)


_COMMENT_START = ("#", "//", "/*", "<!--")


def _lead(text: str) -> tuple[str, str]:
    """(blank and comment lines in front of an item, the rest)."""
    lines = text.splitlines(keepends=True)
    k = 0
    in_block = False
    while k < len(lines):
        stripped = lines[k].strip()
        if in_block or stripped.startswith("/*"):
            in_block = "*/" not in stripped
        elif stripped and not stripped.startswith(_COMMENT_START):
            break
        k += 1
    return "".join(lines[:k]), "".join(lines[k:])


def _keep_lead(old: str, new: str) -> str:
    """A rewritten item keeps the comments that stood in front of the original
    when the answer gives none of its own (the model never saw them)."""
    if new == old:
        return new
    old_lead, _ = _lead(old)
    new_lead, body = _lead(new)
    return new if new_lead.strip() else old_lead + body


def _usual_gap(orig: list[Item]) -> int:
    """How many blank lines usually separate the original's items."""
    gaps = [len(re.match(r"\n*", item.text).group(0)) for item in orig[1:]]  # type: ignore[union-attr]
    return max(set(gaps), key=gaps.count) if gaps else 0


def _spaced(text: str, blank: int) -> str:
    """A new item with at least the usual blank lines in front of it."""
    have = len(text) - len(text.lstrip("\n"))
    return "\n" * max(blank - have, 0) + text


def _count(report: Report, old: str, new: str) -> None:
    if _norm(new) == _norm(old):
        report.kept += 1
    else:
        report.replaced += 1


Views = dict[tuple[str, ...], set[str]]


def views_of(original: str, path: str) -> list[str]:
    """What shrink may have shown the model of this file: every level, with and
    without docstrings. An answer part equal to one of them was not changed."""
    from shrink.compressors.auto import AutoCompressor
    from shrink.compressors.nodocs import NoDocsCompressor
    from shrink.document import Document

    doc = Document(original, path=path)
    found: list[str] = [original]
    for level in (1, 2, 3):
        try:
            view = AutoCompressor().compress(doc, level)
        except (SyntaxError, ValueError):  # a file the mode cannot read has no views
            continue
        # NoDocsCompressor keeps its input when it cannot drop the docs.
        found += [view.text, NoDocsCompressor().compress(view, level).text]
    return list(dict.fromkeys(found))


def _index(views: list[str], items_of: Callable[[str], list[Item]]) -> Views:
    index: Views = defaultdict(set)
    for view in views:
        try:
            items = items_of(view)
        except (SyntaxError, ValueError):
            continue
        for item in items:
            if item.key is not None:
                index[(item.key,)].add(_norm(item.text))
    return index


def _seen(views: Views, path: tuple[str, ...], item: Item) -> bool:
    if item.key is None:
        return False
    where = (*path, item.key)
    return _norm(item.text) in views.get(where, set())


def _join(pieces: list[str]) -> str:
    out: list[str] = []
    for piece in pieces:
        if out and not out[-1].endswith("\n"):
            out[-1] += "\n"
        out.append(piece)
    return "".join(out)


# ------------------------------------------------------------------ Python


class _PySource:
    def __init__(self, text: str) -> None:
        self.text = text
        self.starts = [0] + [m.end() for m in re.finditer("\n", text)]
        self.tree = ast.parse(text)
        self.strings, _ = indent._python_rows(text)

    def line_start(self, row: int) -> int:
        return self.starts[row - 1] if row - 1 < len(self.starts) else len(self.text)

    def line_end(self, row: int) -> int:
        """Offset just after the line break ending ``row``."""
        return self.starts[row] if row < len(self.starts) else len(self.text)

    def indent_of(self, row: int) -> int:
        start = self.line_start(row)
        line = self.text[start : self.line_end(row)]
        return len(line) - len(line.lstrip(" \t"))


def _first_row(stmt: ast.stmt) -> int:
    decorators: list[ast.expr] = getattr(stmt, "decorator_list", None) or []
    return min([stmt.lineno, *(d.lineno for d in decorators)])


def _is_ellipsis(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    )


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _py_key(stmt: ast.stmt, index: int) -> tuple[str | None, bool]:
    """(key, is a header item): definitions and variables by name, imports by module."""
    key: str | None = None
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        key = f"def:{stmt.name}"
    elif isinstance(stmt, ast.Assign) and all(isinstance(t, ast.Name) for t in stmt.targets):
        key = "var:" + ",".join(t.id for t in stmt.targets if isinstance(t, ast.Name))
    elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        key = f"var:{stmt.target.id}"
    elif isinstance(stmt, ast.If) and "__main__" in ast.unparse(stmt.test):
        key = "main"
    elif isinstance(stmt, ast.Import):
        return "import:" + ",".join(sorted(a.name for a in stmt.names)), True
    elif isinstance(stmt, ast.ImportFrom):
        return f"from:{'.' * stmt.level}{stmt.module or ''}", True
    elif index == 0 and _is_docstring(stmt):
        return "doc", True
    return key, False


def _py_items(src: _PySource, body: list[ast.stmt], start: int, end: int) -> list[Item]:
    """``body`` cut into items covering ``src.text[start:end]`` without a gap."""
    items: list[Item] = []
    pos = start
    for index, stmt in enumerate(body):
        assert stmt.end_lineno is not None
        stop = src.line_end(stmt.end_lineno)
        if index == len(body) - 1:
            stop = max(stop, end)
        key, header = _py_key(stmt, index)
        row = src.text.count("\n", 0, pos) + 1
        items.append(Item(key, src.text[pos:stop], _is_ellipsis(stmt), header, stmt, row))
        pos = stop
    return items


def _shift(src: _PySource, text: str, first_row: int, delta: int) -> str:
    """Move every line of ``text`` (which starts at ``first_row`` of ``src``) by
    ``delta`` columns, except lines inside multi-line strings."""
    if delta == 0:
        return text
    out: list[str] = []
    for offset, line in enumerate(text.splitlines(keepends=True)):
        if first_row + offset in src.strings or not line.strip():
            out.append(line)
        elif delta > 0:
            out.append(" " * delta + line)
        else:
            lead = len(line) - len(line.lstrip(" "))
            out.append(line[min(lead, -delta) :])
    return "".join(out)


def _class_parts(src: _PySource, node: ast.ClassDef, item: Item) -> tuple[str, int, int] | None:
    """(header text, offset where the members start, member indentation) of a
    class item, or ``None`` for a one-line class."""
    first = _first_row(node.body[0])
    if first <= node.lineno:
        return None
    header_end = first - 1
    while header_end > node.lineno:
        code = src.text[src.line_start(header_end) : src.line_end(header_end)].split("#")[0]
        if code.rstrip().endswith(":"):
            break
        header_end -= 1
    member_start = src.line_end(header_end)
    item_start = src.line_start(item.row)
    return src.text[item_start:member_start], member_start, src.indent_of(node.body[0].lineno)


def _py_index(views: list[str]) -> Views:
    """Normalised text of every top-level item and class member (and class header)
    in every view, by key path."""
    index: Views = defaultdict(set)
    for view in views:
        try:
            src = _PySource(view)
        except SyntaxError:
            continue
        _index_body(src, src.tree.body, (0, len(src.text)), (), index)
    return index


def _index_body(
    src: _PySource,
    body: list[ast.stmt],
    span: tuple[int, int],
    prefix: tuple[str, ...],
    index: Views,
) -> None:
    for item in _py_items(src, body, *span):
        if item.key is None:
            continue
        path = (*prefix, item.key)
        index[path].add(_norm(item.text))
        if isinstance(item.node, ast.ClassDef):
            parts = _class_parts(src, item.node, item)
            if parts is None:
                continue
            header, member_start, _ = parts
            header_path = (*path, "<header>")
            index[header_path].add(_norm(header))
            end = src.line_start(item.row) + len(item.text)
            _index_body(src, item.node.body, (member_start, end), path, index)


@dataclass
class _PyMerge:
    """Merges a Python answer with the original, body by body."""

    orig: _PySource
    answer: _PySource
    views: Views
    report: Report
    partial: bool = False

    def module(self) -> str:
        orig_items = _py_items(self.orig, self.orig.tree.body, 0, len(self.orig.text))
        ans_items = _py_items(self.answer, self.answer.tree.body, 0, len(self.answer.text))
        return self._body(orig_items, ans_items, ())

    def _body(self, orig_items: list[Item], ans_items: list[Item], prefix: tuple[str, ...]) -> str:
        def combine(old: Item, new: Item) -> str:
            if _seen(self.views, prefix, new):
                return old.text  # unchanged from what the model was shown
            if isinstance(old.node, ast.ClassDef) and isinstance(new.node, ast.ClassDef):
                return self._class(old, new, prefix)
            if self.partial and isinstance(old.node, ast.ImportFrom):
                return _join_imports(old, new)
            return new.text

        merged = merge_items(orig_items, ans_items, self.report, combine, partial=self.partial)
        if merged is None:
            merged = whole_items(orig_items, ans_items, self.report, combine)
        return merged

    def _class(self, old: Item, new: Item, prefix: tuple[str, ...]) -> str:
        assert isinstance(old.node, ast.ClassDef) and isinstance(new.node, ast.ClassDef)
        old_parts = _class_parts(self.orig, old.node, old)
        new_parts = _class_parts(self.answer, new.node, new)
        if old_parts is None or new_parts is None:
            return new.text
        old_header, old_start, old_indent = old_parts
        header, new_start, new_indent = new_parts
        path = (*prefix, str(new.key))
        header_path = (*path, "<header>")
        if _norm(header) in self.views.get(header_path, set()):
            header = old_header
        old_end = self.orig.line_start(old.row) + len(old.text)
        new_end = self.answer.line_start(new.row) + len(new.text)
        # Original members move to the answer's indentation if it differs.
        shifted = [
            Item(
                item.key,
                _shift(self.orig, item.text, item.row, new_indent - old_indent),
                item.gap,
                False,
                item.node,
                item.row,
            )
            for item in _py_items(self.orig, old.node.body, old_start, old_end)
        ]
        ans_items = _py_items(self.answer, new.node.body, new_start, new_end)
        for item in ans_items:  # a class docstring is a member like any other
            item.header = False
        return header + self._body(shifted, ans_items, path)


def _join_imports(old: Item, new: Item) -> str:
    """``from m import a`` patched with ``from m import b`` imports both."""
    assert isinstance(old.node, ast.ImportFrom) and isinstance(new.node, ast.ImportFrom)
    names = {(a.name, a.asname) for a in [*old.node.names, *new.node.names]}
    if names == {(a.name, a.asname) for a in old.node.names}:
        return old.text
    listed = ", ".join(f"{n} as {a}" if a else n for n, a in sorted(names, key=lambda x: x[0]))
    lead, _ = _lead(old.text)
    return f"{lead}from {'.' * old.node.level}{old.node.module or ''} import {listed}\n"


def _code_of(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """The function's syntax tree without its docstring: what it does."""
    body = func.body[1:] if _is_docstring(func.body[0]) else func.body
    parts: list[ast.AST] = [func.args, *func.decorator_list, *body]
    return "|".join(ast.dump(part) for part in parts)


def _reject_python(original: str, text: str, shown: str, report: Report) -> str:
    """Put the file's own code back wherever ``text`` rewrites a function that
    ``shown`` (the model's view) had as a stub: the model never saw that code."""
    try:
        orig, merged = _PySource(original), _PySource(text)
    except SyntaxError:
        return text  # not Python before the change: nothing was shown as a skeleton
    try:
        view = _PySource(shown)
    except SyntaxError as exc:
        raise RestoreError(
            f"the file's text in the context is not valid Python (line {exc.lineno}), "
            "so the answer cannot be checked against what the model saw"
        ) from None
    hidden = {name for name, func in _functions(view).items() if _stub_body(func)}
    originals, results = _functions(orig), _functions(merged)
    edits: list[tuple[int, int, str]] = []
    for name in sorted(hidden & originals.keys() & results.keys()):
        source, func = originals[name], results[name]
        if _code_of(source) == _code_of(func):
            continue
        assert source.end_lineno is not None and func.end_lineno is not None
        first = _first_row(source)
        block = orig.text[orig.line_start(first) : orig.line_end(source.end_lineno)]
        if not block.endswith("\n"):
            block += "\n"
        row = _first_row(func)
        block = _shift(orig, block, first, merged.indent_of(row) - orig.indent_of(first))
        edits.append((merged.line_start(row), merged.line_end(func.end_lineno), block))
        report.reverted.append(name)
    return _splice(text, edits)


def _splice(text: str, edits: list[tuple[int, int, str]]) -> str:
    out: list[str] = []
    pos = 0
    for start, end, replacement in sorted(edits):
        if start < pos:
            continue  # inside a part already replaced
        out += [text[pos:start], replacement]
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _functions(src: _PySource) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.setdefault(prefix + stmt.name, stmt)
            elif isinstance(stmt, ast.ClassDef):
                visit(stmt.body, f"{prefix}{stmt.name}.")

    visit(src.tree.body, "")
    return found


def _stub_body(func: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    body = func.body[1:] if _is_docstring(func.body[0]) and len(func.body) > 1 else func.body
    return len(body) == 1 and _is_ellipsis(body[0])


def _fill_python_stubs(orig: _PySource | None, text: str, report: Report) -> str:
    answer = _PySource(text)
    originals = _functions(orig) if orig is not None else {}
    edits: list[tuple[int, int, str]] = []
    for name, func in _functions(answer).items():
        if not _stub_body(func):
            continue
        source = originals.get(name)
        if source is None or orig is None or _stub_body(source):
            if source is None:
                report.unresolved.append(name)
            continue
        assert source.end_lineno is not None and func.end_lineno is not None
        body_row = source.body[0].lineno
        block = orig.text[orig.line_start(body_row) : orig.line_end(source.end_lineno)]
        if not block.endswith("\n"):
            block += "\n"
        first = func.body[0]
        if first.lineno == func.lineno:  # def f(): ...  -> a block of its own
            target = (
                answer.indent_of(func.lineno)
                + orig.indent_of(body_row)
                - orig.indent_of(source.lineno)
            )
            block = _shift(orig, block, body_row, target - orig.indent_of(body_row))
            start = answer.line_start(first.lineno) + len(
                answer.text[answer.line_start(first.lineno) :]
                .encode("utf-8")[: first.col_offset]
                .decode("utf-8")
                .rstrip()
            )
            edits.append((start, answer.line_end(func.end_lineno), "\n" + block))
        else:
            delta = answer.indent_of(first.lineno) - orig.indent_of(body_row)
            block = _shift(orig, block, body_row, delta)
            edits.append((answer.line_start(first.lineno), answer.line_end(func.end_lineno), block))
        report.bodies += 1
    out: list[str] = []
    pos = 0
    for start, end, replacement in sorted(edits):
        out.append(text[pos:start])
        out.append(replacement)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def restore_python(
    original: str | None,
    answer: str,
    report: Report,
    views: list[str] | None = None,
    partial: bool = False,
) -> str:
    try:
        ans = _PySource(answer)
    except SyntaxError as exc:
        raise RestoreError(
            f"the answer is not valid Python: {exc.msg} (line {exc.lineno})"
        ) from None
    orig = None
    if original is not None:
        try:
            orig = _PySource(original)
        except SyntaxError:
            orig = None
    text = answer
    if orig is not None:
        text = _PyMerge(orig, ans, _py_index(views or [orig.text]), report, partial).module()
    text = _fill_python_stubs(orig, text, report)
    try:
        ast.parse(text)
    except SyntaxError as exc:
        raise RestoreError(
            f"the merged file does not parse: {exc.msg} (line {exc.lineno})"
        ) from None
    return text


# ------------------------------------------------------------------ JavaScript


def _offsets(toks: list[web.Tok]) -> list[int]:
    out = []
    pos = 0
    for tok in toks:
        out.append(pos)
        pos += len(tok.text)
    return out


def _prev(toks: list[web.Tok], i: int) -> int:
    i -= 1
    while i >= 0 and toks[i].kind in ("ws", "nl", "line_comment", "block_comment"):
        i -= 1
    return i


def _next_sig(toks: list[web.Tok], i: int) -> int:
    i += 1
    while i < len(toks) and toks[i].kind in ("ws", "nl", "line_comment", "block_comment"):
        i += 1
    return i


def _open_paren(toks: list[web.Tok], close: int) -> int:
    depth = 0
    for k in range(close, -1, -1):
        if toks[k].kind == "punct" and toks[k].text == ")":
            depth += 1
        elif toks[k].kind == "punct" and toks[k].text == "(":
            depth -= 1
            if depth == 0:
                return k
    return -1


def _assigned_name(toks: list[web.Tok], i: int) -> str | None:
    """The name in ``name = <i>``, ``name: <i>`` or ``const name = <i>``."""
    k = _prev(toks, i)
    if k >= 0 and toks[k].kind == "word" and toks[k].text == "async":
        k = _prev(toks, k)
    if k >= 0 and toks[k].text in ("=", ":"):
        n = _prev(toks, k)
        if n >= 0 and toks[n].kind in ("word", "string"):
            return toks[n].text.strip("'\"`")
    return None


def _line_prefix(toks: list[web.Tok], i: int) -> str:
    k = i
    while k > 0 and toks[k - 1].kind != "nl":
        k -= 1
    return _norm("".join(t.text for t in toks[k:i]))


def _js_bodies(src: str) -> list[tuple[str, int, int]]:
    """(key, start, end) of every ``function`` and block-arrow body (the ``{...}``)."""
    toks = web.lex_js(src)
    offs = _offsets(toks)
    found: list[tuple[str, int, int]] = []
    for i, tok in enumerate(toks):
        key: str | None = None
        body = None
        if tok.kind == "word" and tok.text == "function":
            close = web._matching_after(toks, i, "(", ")")
            if close is None:
                continue
            body = web._block_after(toks, close)
            n = _next_sig(toks, i)
            if n < len(toks) and toks[n].text == "*":
                n = _next_sig(toks, n)
            if n < len(toks) and toks[n].kind == "word":
                key = toks[n].text
            else:
                key = _assigned_name(toks, i)
        elif tok.text == "=" and i + 1 < len(toks) and toks[i + 1].text == ">":
            body = web._block_after(toks, i + 1)
            p = _prev(toks, i)
            start = _open_paren(toks, p) if p >= 0 and toks[p].text == ")" else p
            key = _assigned_name(toks, start) if start >= 0 else None
        if body is None:
            continue
        open_, close_ = body
        found.append((key or _line_prefix(toks, open_), offs[open_], offs[close_] + 1))
    return found


def _fill_js_stubs(
    originals: dict[str, str], text: str, report: Report, orig_text: str | None
) -> str:
    edits: list[tuple[int, int, str]] = []
    seen: dict[str, int] = defaultdict(int)
    for key, start, end in _js_bodies(text):
        if _norm(text[start + 1 : end - 1]) != _STUB_JS:
            continue
        seen[key] += 1
        body = originals.get(key)
        if body is None or _norm(body[1:-1]) == _STUB_JS:
            report.unresolved.append(key)
            continue
        line_start = text.rfind("\n", 0, start) + 1
        here = len(text[line_start:start]) - len(text[line_start:start].lstrip(" \t"))
        edits.append((start, end, _reindent_block(body, here, orig_text)))
        report.bodies += 1
    out: list[str] = []
    pos = 0
    for start, end, replacement in sorted(edits):
        out.append(text[pos:start])
        out.append(replacement)
        pos = end
    out.append(text[pos:])
    return "".join(out)


def _js_code(body: str) -> list[tuple[str, str]]:
    return [(kind, token) for kind, token, _ in web._signature(web.lex_js(body))]


def _reject_js(original: str, text: str, shown: str, report: Report) -> str:
    """:func:`_reject_python` for the ``{/* ... */}`` bodies of JavaScript."""
    try:
        hidden = {key for key, s, e in _js_bodies(shown) if _norm(shown[s + 1 : e - 1]) == _STUB_JS}
        originals = {key: original[s:e] for key, s, e in _js_bodies(original)}
        results = _js_bodies(text)
    except web.LexError:
        return text
    edits: list[tuple[int, int, str]] = []
    done = 0
    for key, start, end in results:
        if start < done or key not in hidden or key not in originals:
            continue
        if _js_code(text[start:end]) == _js_code(originals[key]):
            continue
        line_start = text.rfind("\n", 0, start) + 1
        here = len(text[line_start:start]) - len(text[line_start:start].lstrip(" \t"))
        edits.append((start, end, _reindent_block(originals[key], here, original)))
        report.reverted.append(key)
        done = end
    return _splice(text, edits)


def _reindent_block(body: str, target: int, orig_text: str | None) -> str:
    """Shift the lines after the first of a ``{...}`` body so its closing brace
    lines up with the line it now starts on."""
    lines = body.split("\n")
    if len(lines) == 1:
        return body
    last = lines[-1]
    base = len(last) - len(last.lstrip(" \t"))
    delta = target - base
    if delta == 0:
        return body
    shifted = [lines[0]]
    for line in lines[1:]:
        if not line.strip():
            shifted.append(line)
        elif delta > 0:
            shifted.append(" " * delta + line)
        else:
            lead = len(line) - len(line.lstrip(" "))
            shifted.append(line[min(lead, -delta) :])
    return "\n".join(shifted)


_JS_DECL = ("import", "export", "function", "async", "class", "const", "let", "var")


def _js_items(src: str) -> list[Item]:
    """Top-level statements, each with the comments and blank lines in front of it."""
    lines: list[list[web.Tok]] = [[]]
    for tok in web.lex_js(src):
        lines[-1].append(tok)
        if tok.kind == "nl":
            lines.append([])
    items: list[Item] = []
    for a, b in itertools.pairwise([*_js_starts(lines), len(lines)]):
        text = "".join(t.text for line in lines[a:b] for t in line)
        if not text:
            continue
        rows = [row for row in text.splitlines() if row.strip()]
        gap = any(_GAP_JS.match(row) for row in rows) and all(
            _GAP_JS.match(row) or row.strip().startswith(("//", "/*", "*")) for row in rows
        )
        key, header = (None, False) if gap else _js_key(text)
        items.append(Item(key, text, gap, header, None, a + 1))
    return items


def _js_starts(lines: list[list[web.Tok]]) -> list[int]:
    """Line numbers where a top-level item starts: a declaration, a gap marker or a
    statement after a finished one; comments and blank lines join the next item."""
    starts: list[int] = [0]
    depth = 0
    last_code = ""  # last significant token of the previous code line
    pending: int | None = None  # first comment/blank line in front of the next item
    for number, line in enumerate(lines):
        text = "".join(t.text for t in line)
        code = [t for t in line if t.kind not in ("ws", "nl", "line_comment", "block_comment")]
        if depth == 0:
            gap = bool(text.strip()) and _GAP_JS.match(text.rstrip("\r\n")) is not None
            if gap or (code and (code[0].text in _JS_DECL or last_code in (";", "}", ""))):
                starts.append(pending if pending is not None else number)
                pending = None
                last_code = ";" if gap else last_code
            elif not code:
                pending = number if pending is None else pending
            else:
                pending = None
        for tok in code:
            if tok.kind == "punct" and tok.text in "([{":
                depth += 1
            elif tok.kind == "punct" and tok.text in ")]}":
                depth = max(depth - 1, 0)
        if code:
            last_code = code[-1].text
    return sorted(set(starts))


def _js_key(text: str) -> tuple[str | None, bool]:
    code = re.sub(r"/\*.*?\*/|//[^\n]*", " ", text, flags=re.S).strip()
    imported = re.match(r"import\b[^;]*?['\"]([^'\"]+)['\"]", code, re.S)
    if imported:
        return f"import:{imported.group(1)}", True
    words = re.findall(r"[\w$*]+|\S", code[:200])
    while words and words[0] in ("export", "async", "default"):
        if words[0] == "default" and (len(words) == 1 or words[1] not in ("function", "class")):
            return "default", False
        words.pop(0)
    if len(words) >= 2 and words[0] in ("function", "function*", "class"):
        name = words[2] if words[1] == "*" and len(words) > 2 else words[1]
        return f"def:{name}", False
    if len(words) >= 2 and words[0] in ("const", "let", "var") and re.match(r"[\w$]+$", words[1]):
        return f"def:{words[1]}", False
    return None, False


def restore_js(
    original: str | None,
    answer: str,
    report: Report,
    views: list[str] | None = None,
    partial: bool = False,
) -> str:
    try:
        ans_items = _js_items(answer)
    except web.LexError as exc:
        raise RestoreError(f"the answer cannot be read as JavaScript: {exc}") from None
    text = answer
    originals: dict[str, str] = {}
    if original is not None:
        try:
            orig_items = _js_items(original)
            originals = {key: original[s:e] for key, s, e in _js_bodies(original)}
        except web.LexError:
            orig_items = []
        index = _index(views or [original], _js_items)

        def combine(old: Item, new: Item) -> str:
            return old.text if _seen(index, (), new) else new.text

        if orig_items:
            merged = merge_items(orig_items, ans_items, report, combine, partial=partial)
            text = (
                merged
                if merged is not None
                else whole_items(orig_items, ans_items, report, combine)
            )
    return _fill_js_stubs(originals, text, report, original)


# ------------------------------------------------------------------ HTML


@dataclass(frozen=True)
class _Element:
    name: str
    attrs: str
    start: int
    end: int  # after the closing tag (or the start tag of a void element)
    depth: int


def _html_elements(src: str) -> list[_Element]:
    masked = list(src)
    for match in re.finditer(r"(<(script|style)\b[^>]*>)(.*?)(</\2\s*>)", src, re.I | re.S):
        for k in range(match.start(3), match.end(3)):
            masked[k] = " "
    for match in re.finditer(r"<!--.*?-->", src, re.S):
        for k in range(match.start(), match.end()):
            masked[k] = " "
    text = "".join(masked)
    elements: list[_Element] = []
    stack: list[tuple[str, str, int]] = []
    for match in re.finditer(r"<(/?)([a-zA-Z][\w:-]*)\b([^<>]*?)(/?)>", text):
        slash, name, attrs, self_closing = match.groups()
        name = name.lower()
        if slash:
            for k in range(len(stack) - 1, -1, -1):
                if stack[k][0] == name:
                    open_name, open_attrs, open_start = stack[k]
                    del stack[k:]
                    elements.append(_Element(open_name, open_attrs, open_start, match.end(), k))
                    break
        elif self_closing or name in indent._VOID:
            elements.append(_Element(name, attrs, match.start(), match.end(), len(stack)))
        else:
            stack.append((name, attrs, match.start()))
    return sorted(elements, key=lambda e: e.start)


def _attr(attrs: str, name: str) -> str | None:
    found = re.search(rf"""\b{name}\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))""", attrs, re.I)
    if not found:
        return None
    return next(group for group in found.groups() if group is not None)


def _html_top(src: str) -> list[tuple[str, _Element | None, str]]:
    """Top-level nodes of an answer fragment: (text, element or None, kind)."""
    elements = [e for e in _html_elements(src) if e.depth == 0]
    nodes: list[tuple[str, _Element | None, str]] = []
    pos = 0
    for element in elements:
        between = src[pos : element.start]
        for comment in re.finditer(r"<!--.*?-->", between, re.S):
            kind = "gap" if _GAP_HTML.match(comment.group(0)) else "comment"
            nodes.append((comment.group(0), None, kind))
        nodes.append((src[element.start : element.end], element, "element"))
        pos = element.end
    for comment in re.finditer(r"<!--.*?-->", src[pos:], re.S):
        kind = "gap" if _GAP_HTML.match(comment.group(0)) else "comment"
        nodes.append((comment.group(0), None, kind))
    return nodes


def _line_indent(src: str, pos: int) -> str:
    start = src.rfind("\n", 0, pos) + 1
    line = src[start:pos]
    return line[: len(line) - len(line.lstrip(" \t"))]


def _place(fragment: str, pad: str) -> str:
    lines = fragment.strip("\n").split("\n")
    base = min((len(line) - len(line.lstrip(" ")) for line in lines[1:] if line.strip()), default=0)
    first = lines[0].lstrip(" ")
    rest = [
        pad + line[min(base, len(line) - len(line.lstrip(" "))) :] if line.strip() else line
        for line in lines[1:]
    ]
    return "\n".join([first, *rest])


def restore_html(
    original: str | None,
    answer: str,
    report: Report,
    views: list[str] | None = None,
    partial: bool = False,  # a fragment is merged by element either way
) -> str:
    text = answer
    if original is not None:
        whole = re.search(r"<html\b|<!doctype", answer, re.I) is not None
        whole_orig = re.search(r"<html\b|<!doctype", original, re.I) is not None
        nodes = _html_top(answer)
        if any(kind == "gap" for _, _, kind in nodes) or (whole_orig and not whole):
            text = _merge_html(original, nodes, report)
    originals: dict[str, str] = {}
    if original is not None:
        for code in _scripts(original):
            originals.update({key: code[s:e] for key, s, e in _js_bodies(code)})
    pieces: list[str] = []
    pos = 0
    for match in re.finditer(
        r"(<script\b(?![^>]*\bsrc=)[^>]*>)(.*?)(</script\s*>)", text, re.I | re.S
    ):
        pieces.append(text[pos : match.start(2)])
        try:
            pieces.append(_fill_js_stubs(originals, match.group(2), report, None))
        except web.LexError:
            pieces.append(match.group(2))
        pos = match.end(2)
    pieces.append(text[pos:])
    return "".join(pieces)


def _scripts(src: str) -> list[str]:
    return [
        m.group(1)
        for m in re.finditer(r"<script\b(?![^>]*\bsrc=)[^>]*>(.*?)</script\s*>", src, re.I | re.S)
    ]


def _html_target(
    original: str, elements: list[_Element], text: str, element: _Element
) -> _Element | None:
    """The element of the original an answer element stands for, if any."""
    ident = _attr(element.attrs, "id")
    if ident is not None:
        same_id = [e for e in elements if _attr(e.attrs, "id") == ident]
        if same_id:
            return same_id[0]
    same_tag = [e for e in elements if e.name == element.name]
    if element.name in _UNIQUE_TAGS and len(same_tag) == 1:
        return same_tag[0]
    if element.name != "script":
        return None
    src_attr = _attr(element.attrs, "src")
    for candidate in same_tag:
        same_src = src_attr is not None and _attr(candidate.attrs, "src") == src_attr
        if same_src or _norm(original[candidate.start : candidate.end]) == _norm(text):
            return candidate
    return None


def _html_insert(original: str, element: _Element, last_end: int | None) -> tuple[int, str]:
    """(position, indentation) for a new element: scripts before ``</body>``, head
    tags before ``</head>``, anything else after the last element it followed."""
    if element.name != "script" and element.name not in _HEAD_TAGS and last_end is not None:
        return last_end, ""
    tag = "head" if element.name in _HEAD_TAGS else "body"
    found = re.search(rf"</{tag}\s*>", original, re.I)
    where = found.start() if found else len(original)
    return original.rfind("\n", 0, where) + 1, _line_indent(original, where) + "  "


def _merge_html(
    original: str, nodes: list[tuple[str, _Element | None, str]], report: Report
) -> str:
    elements = _html_elements(original)
    changes: list[tuple[int, int, str]] = []
    last_end: int | None = None
    replaced: list[tuple[int, int]] = []
    for text, element, kind in nodes:
        if kind != "element" or element is None:
            continue
        target = _html_target(original, elements, text, element)
        if target is not None and any(a <= target.start < b for a, b in replaced):
            continue  # inside an element the answer already gave in full
        if target is not None:
            replaced.append((target.start, target.end))
            new = _place(text, _line_indent(original, target.start))
            report.replaced += _norm(new) != _norm(original[target.start : target.end])
            changes.append((target.start, target.end, new))
            last_end = target.end
            continue
        report.inserted += 1
        position, pad = _html_insert(original, element, last_end)
        if pad:
            changes.append((position, position, pad + _place(text, pad) + "\n"))
        else:
            pad = _line_indent(original, position)
            changes.append((position, position, "\n" + pad + _place(text, pad)))
    report.kept += 1  # everything the fragment does not mention
    # Replacements first by position, the outermost of overlapping ones wins.
    out: list[str] = []
    pos = 0
    for start, end, replacement in sorted(changes, key=lambda c: (c[0], -c[1])):
        if start < pos:
            continue
        out += [original[pos:start], replacement]
        pos = end
    out.append(original[pos:])
    return "".join(out)


# ------------------------------------------------------------------ CSS


def _css_items(src: str) -> list[Item]:
    items: list[Item] = []
    depth = 0
    pos = 0
    current = 0
    for kind, text in web._lex_css(src):
        end = pos + len(text)
        if kind == "punct" and text == "{":
            depth += 1
        elif kind == "punct" and text == "}":
            depth -= 1
            if depth == 0:
                items.append(_css_item(src[current:end], current, src))
                current = end
        elif kind == "punct" and text == ";" and depth == 0:
            items.append(_css_item(src[current:end], current, src))
            current = end
        elif kind == "comment" and depth == 0 and _norm(text).startswith("/* ..."):
            gap_end = end
            items.append(Item(None, src[current:gap_end], True))
            current = gap_end
        pos = end
    if current < len(src):
        if items:
            items[-1].text += src[current:]
        elif src[current:].strip():
            items.append(Item(None, src[current:]))
    return items


def _css_item(text: str, _start: int, _src: str) -> Item:
    code = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    prelude = code.split("{", 1)[0] if "{" in code else code
    return Item("rule:" + _norm(prelude), text, False, prelude.strip().startswith("@import"))


def restore_css(
    original: str | None,
    answer: str,
    report: Report,
    views: list[str] | None = None,
    partial: bool = False,
) -> str:
    if original is None:
        return answer
    index = _index(views or [original], _css_items)
    try:
        orig_items = _css_items(original)
        ans_items = _css_items(answer)
    except web.LexError as exc:
        raise RestoreError(f"the answer cannot be read as CSS: {exc}") from None

    def body(item: Item) -> str:
        code = re.sub(r"/\*.*?\*/", "", item.text, flags=re.S)
        return _norm(code.split("{", 1)[1].rsplit("}", 1)[0]) if "{" in code else "x"

    def combine(old: Item, new: Item) -> str:
        if _seen(index, (), new):
            return old.text
        if body(new) == "" and body(old) != "":
            report.bodies += 1
            return old.text
        return new.text

    merged = merge_items(orig_items, ans_items, report, combine, partial=partial)
    return merged if merged is not None else whole_items(orig_items, ans_items, report, combine)


# ------------------------------------------------------------------ fragments


def is_fragment(block: str, path: str) -> bool:
    """A patch block of lines from inside a function: Python or JavaScript whose
    first line is indented (top-level definitions start at the left edge)."""
    if indent.kind_of(path) not in ("python", "js"):
        return False
    first = next((line for line in block.splitlines() if line.strip()), "")
    return first[:1] in (" ", "\t")


def _width(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def insert_fragment(original: str, block: str) -> str:
    """Put the new lines of ``block`` into ``original`` next to the lines it
    shares with it (anchors), as models write "add these lines here":

        app.include_router(health.router)     <- already in the file: anchor
        app.include_router(auth.router)       <- new, goes after the anchor

    An anchor must occur exactly once in the file. New lines keep their depth
    relative to the anchor, in the file's own indentation step.
    """
    lines = original.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    where: dict[str, list[int]] = defaultdict(list)
    for number, line in enumerate(lines):
        where[line.strip()].append(number)
    rows = [line for line in block.splitlines() if line.strip()]
    anchors = {
        k: where[row.strip()][0] for k, row in enumerate(rows) if len(where[row.strip()]) == 1
    }
    if not anchors:
        raise RestoreError("no line of the patch block matches the file, so it has no place")
    widths = sorted({_width(row) for row in rows} | {_width(lines[i]) for i in anchors.values()})
    step = min((b - a for a, b in itertools.pairwise(widths) if b > a), default=1)
    unit = indent.detect_unit(original, "python") or step
    inserts: dict[int, list[str]] = defaultdict(list)  # after original line number
    for k, row in enumerate(rows):
        if k in anchors:
            continue
        before = [j for j in anchors if j < k]
        after = [j for j in anchors if j > k]
        anchor = max(before) if before else min(after)
        line_no = anchors[anchor] if before else anchors[anchor] - 1
        depth = round((_width(row) - _width(rows[anchor])) / step)
        pad = max(_width(lines[anchors[anchor]]) + depth * unit, 0)
        inserts[line_no].append(" " * pad + row.strip() + "\n")
    out = list(inserts.get(-1, []))
    for number, line in enumerate(lines):
        out.append(line)
        out += inserts.get(number, [])
    return "".join(out)


# ------------------------------------------------------------------ entry point

_RESTORERS = {"python": restore_python, "js": restore_js, "html": restore_html, "css": restore_css}
_REJECTORS = {"python": _reject_python, "js": _reject_js}


def restore(
    original: str | None,
    answer: str,
    path: str,
    *,
    partial: bool = False,
    shown: str | None = None,
) -> Restored:
    """Merge a (possibly shortened) answer for ``path`` with the file's current text.

    The answer's indentation is first given back its usual width (see
    :mod:`shrink.indent`), matching the original file when there is one.

    ``partial``: the answer is a patch; it replaces and adds definitions and never
    removes one. ``shown``: the text of this file the model was given. Then nothing
    is removed either, and a function it saw only as a stub keeps the file's own
    code even if the answer has a body for it, because that body can only be a
    guess. To let the model change a function, show it (``shrink --keep``).
    """
    report = Report(path)
    kind = indent.kind_of(path)
    text = answer
    if kind is not None:
        unit = indent.detect_unit(original, kind) if original else None
        try:
            text = indent.expand(answer, kind, unit)
        except (SyntaxError, ValueError):
            text = answer
    views = views_of(original, path) if original is not None else []
    if original is not None and _norm(text) in {_norm(view) for view in views}:
        text = original  # the file as it was shown: nothing changed
    else:
        restorer = _RESTORERS.get(kind or "")
        if restorer is not None:
            text = restorer(original, text, report, views, partial or shown is not None)
        rejector = _REJECTORS.get(kind or "")
        if rejector is not None and shown is not None and original is not None:
            text = rejector(original, text, shown, report)
    if original is None:
        report.status = "new"
    else:
        report.status = "unchanged" if text == original else "changed"
    return Restored(text, report)
