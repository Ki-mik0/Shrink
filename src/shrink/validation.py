"""Checks that a Python transformation kept what it promised to keep.

Each check parses the output again (so it is valid Python for this interpreter)
and compares syntax trees, which ignore formatting and comments:

* :func:`check_same_code`: the program is unchanged (optionally ignoring
  docstrings);
* :func:`check_same_signatures`: everything outside function bodies is
  unchanged, so names, parameters, defaults, annotations, decorators, classes,
  imports and module-level code all survived.

A failed check raises :class:`ValidationError`; the pipeline then passes the
file through unchanged (fail-open).
"""

from __future__ import annotations

import ast
import copy


class ValidationError(ValueError):
    pass


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPES = (ast.Module, ast.ClassDef, *_FUNCTIONS)


def check_same_code(
    before: ast.Module,
    after_text: str,
    *,
    ignore_docstrings: bool = False,
    ignore_stubs: bool = False,
) -> None:
    """``ignore_stubs``: a body that is only ``...`` counts as empty (where a removed
    docstring had to leave something behind)."""
    after = _parse(after_text)
    if _dump(before, ignore_docstrings, ignore_stubs) != _dump(
        after, ignore_docstrings, ignore_stubs
    ):
        what = "code other than docstrings" if ignore_docstrings else "the code"
        raise ValidationError(f"output changed {what}")


def check_same_signatures(before: ast.Module, after_text: str) -> None:
    after = _parse(after_text)
    if _dump(_hollow(before), True) != _dump(_hollow(after), True):
        raise ValidationError("output changed something outside function bodies")


def _parse(text: str) -> ast.Module:
    try:
        return ast.parse(text)
    except SyntaxError as exc:
        raise ValidationError(f"output is not valid Python: {exc.msg}") from exc


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _dump(tree: ast.Module, ignore_docstrings: bool, ignore_stubs: bool = False) -> str:
    if ignore_docstrings or ignore_stubs:
        tree = copy.deepcopy(tree)
        for node in ast.walk(tree):
            if not isinstance(node, _SCOPES):
                continue
            if ignore_docstrings and node.body and _is_docstring(node.body[0]):
                node.body = node.body[1:]
            if ignore_stubs and len(node.body) == 1 and _is_ellipsis(node.body[0]):
                node.body = []
    return ast.dump(tree)


def _is_ellipsis(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and stmt.value.value is Ellipsis
    )


def _hollow(tree: ast.Module) -> ast.Module:
    """A copy where every outermost function keeps only its docstring."""
    tree = copy.deepcopy(tree)
    _hollow_children(tree)
    return tree


def _hollow_children(node: ast.AST) -> None:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, _FUNCTIONS):
            child.body = child.body[:1] if child.body and _is_docstring(child.body[0]) else []
        else:
            _hollow_children(child)
