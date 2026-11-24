"""Auto mode: pick a mode per document, from its file name or its content."""

from __future__ import annotations

import ast
import importlib

from shrink.compressors.base import Compressor, Step, Traceable
from shrink.document import Document

PYTHON_EXTENSIONS = (".py", ".pyi", ".pyw")
JSON_EXTENSIONS = (".json", ".jsonl", ".ndjson", ".geojson")
LOG_EXTENSIONS = (".log",)
_DEFINITIONS = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)

_TARGETS = {
    "code": "shrink.compressors.python_code:CodeCompressor",
    "json": "shrink.compressors.json_mode:JsonCompressor",
    "log": "shrink.compressors.log:LogCompressor",
    "text": "shrink.compressors.text:TextCompressor",
}

HELP = """\
Chooses a mode for every document separately:

  code   files ending in .py, .pyi or .pyw; unnamed input (plain stdin) that
         parses as Python and has an import, def or class at the top level
  json   files ending in .json, .jsonl, .ndjson or .geojson; unnamed input that
         starts with { or [ and parses as JSON or JSON Lines
  log    files ending in .log; unnamed input where most lines start with a
         timestamp
  text   everything else (source files in other languages only get text level 1)

The level is passed on unchanged; see `--mode-help code` and the others."""


def is_python(doc: Document) -> bool:
    if doc.path is not None:
        return doc.path.lower().endswith(PYTHON_EXTENSIONS)
    try:
        tree = ast.parse(doc.text)
    except (SyntaxError, ValueError):
        return False
    return any(isinstance(node, _DEFINITIONS) for node in tree.body)


def choose_mode(doc: Document) -> str:
    """The mode name auto mode uses for this document."""
    if doc.path is not None:
        path = doc.path.lower()
        by_extension = [
            (PYTHON_EXTENSIONS, "code"),
            (JSON_EXTENSIONS, "json"),
            (LOG_EXTENSIONS, "log"),
        ]
        return next((mode for exts, mode in by_extension if path.endswith(exts)), "text")
    from shrink.compressors.json_mode import looks_like_json

    if looks_like_json(doc.text):
        return "json"
    if is_python(doc):
        return "code"
    from shrink.compressors.log import looks_like_log

    return "log" if looks_like_log(doc.text) else "text"


class AutoCompressor:
    name = "auto"
    summary = "choose code, json, log or text for each file"
    help = HELP

    def __init__(self, **overrides: Compressor) -> None:
        # Compressors are created on first use, so prose never imports the others.
        self._made: dict[str, Compressor] = dict(overrides)

    def choose(self, doc: Document) -> Compressor:
        mode = choose_mode(doc)
        if mode not in self._made:
            module_name, _, class_name = _TARGETS[mode].partition(":")
            factory = getattr(importlib.import_module(module_name), class_name)
            self._made[mode] = factory()
        return self._made[mode]

    def compress(self, doc: Document, level: int) -> Document:
        return self.choose(doc).compress(doc, level)

    def trace(self, doc: Document, level: int) -> list[Step]:
        chosen = self.choose(doc)
        if isinstance(chosen, Traceable):
            return chosen.trace(doc, level)
        return [Step(chosen.name, chosen.compress(doc, level))]
