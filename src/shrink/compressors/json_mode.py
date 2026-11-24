"""JSON mode: minify, then shorten long arrays and strings.

The document is parsed with hooks that keep numbers as their original text and
objects as lists of pairs, so minifying is exact: no float rounding, duplicate
keys and key order survive. Strings are written with ``ensure_ascii=False``:
``"\\u0456"`` costs several tokens, the letter itself one or two. JSON Lines
(one value per line) is handled line by line.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from shrink.compressors.base import Step, check_level
from shrink.document import Document

HELP = """\
JSON and JSON Lines (one value per line). The output is always valid JSON.

  level 1  light     minify: whitespace outside strings goes, \\uXXXX escapes become
                     the characters; numbers, key order and duplicate keys are
                     kept exactly
  level 2  medium    + arrays longer than 8 items keep the first 5 and a string
                     "... +N more items"; strings longer than 300 characters keep
                     200 and "... (+N chars)"
  level 3  skeleton  + arrays keep their first item only, strings longer than 60
                     characters keep 40; the shape of the data stays visible

Input that is not JSON is passed through unchanged with a warning."""

STEP_NAMES = {1: "minify", 2: "long values", 3: "shape only"}
_BOM = chr(0xFEFF)
_LINES = re.compile(r"[^\n]*\n|[^\n]+$")
_MORE = re.compile(r"^\.\.\. \+(\d+) more items$")

# level -> (array length that triggers cutting, items kept, string length that triggers, chars kept)
_LIMITS: dict[int, tuple[int, int, int, int] | None] = {
    1: None,
    2: (8, 5, 300, 200),
    3: (1, 1, 60, 40),
}


class _Number(str):
    """A number exactly as it was written."""


class _Object(list[tuple[str, object]]):
    """An object as its (key, value) pairs, duplicates and order included."""


def _parse(text: str) -> object:
    return json.loads(
        text,
        object_pairs_hook=_Object,
        parse_float=_Number,
        parse_int=_Number,
        parse_constant=_Number,
    )


class JsonCompressor:
    name = "json"
    summary = "JSON / JSON Lines: minify, shorten long arrays and strings"
    help = HELP

    def compress(self, doc: Document, level: int) -> Document:
        return doc.with_text(compress_json(doc.text, level))

    def trace(self, doc: Document, level: int) -> list[Step]:
        check_level(level)
        return [
            Step(STEP_NAMES[k], doc.with_text(compress_json(doc.text, k)))
            for k in range(1, level + 1)
        ]


def compress_json(text: str, level: int) -> str:
    """Raises ``ValueError`` when the text is neither JSON nor JSON Lines."""
    check_level(level)
    bom = _BOM if text.startswith(_BOM) else ""
    body = text[len(bom) :]
    if not body.strip():
        return text
    dump = _dumper(level)
    try:
        value = _parse(body)
    except ValueError:
        out = _json_lines(body, dump)
    else:
        out = dump(value) + ("\n" if body.endswith("\n") else "")
    result = bom + out
    return result if len(result) <= len(text) else text


def _json_lines(body: str, dump: Callable[[object], str]) -> str:
    out: list[str] = []
    for raw in _LINES.findall(body):  # only LF ends a line: U+2028 may sit inside a string
        line = raw.rstrip("\r\n")
        if not line.strip():
            continue
        out.append(dump(_parse(line)) + raw[len(line) :])  # raises ValueError if not JSON
    return "".join(out)


def _dumper(level: int) -> Callable[[object], str]:
    limits = _LIMITS[level]

    def dump(value: object) -> str:
        if isinstance(value, _Number):
            return str(value)
        if isinstance(value, str):
            if limits is not None and len(value) > limits[2]:
                value = f"{value[: limits[3]]}... (+{len(value) - limits[3]} chars)"
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, _Object):
            return "{" + ",".join(f"{dump(k)}:{dump(v)}" for k, v in value) + "}"
        if isinstance(value, list):
            return "[" + ",".join(_array(value, limits, dump)) + "]"
        return json.dumps(value)  # True, False, None

    return dump


def _array(
    items: list[object],
    limits: tuple[int, int, int, int] | None,
    dump: Callable[[object], str],
) -> list[str]:
    """The array's items as JSON, cut to the limit only when that is shorter."""
    hidden = 0
    last = items[-1] if items else None
    if isinstance(last, str) and not isinstance(last, _Number):
        mark = _MORE.match(last)
        if mark is not None:  # cut before: keep counting from there (idempotence)
            hidden = int(mark.group(1))
            items = items[:-1]
    parts = [dump(item) for item in items]  # each item once: nested arrays stay linear
    full = parts + ([_more(hidden)] if hidden else [])
    if limits is None or len(items) + (1 if hidden else 0) <= limits[0]:
        return full
    kept = limits[1]
    cut = [*parts[:kept], _more(hidden + max(len(items) - kept, 0))]
    return cut if len(",".join(cut)) < len(",".join(full)) else full


def _more(hidden: int) -> str:
    return json.dumps(f"... +{hidden} more items")


def looks_like_json(text: str) -> bool:
    body = text.lstrip(_BOM).lstrip()
    if not body.startswith(("{", "[")):
        return False
    try:
        compress_json(text, 1)
    except ValueError:
        return False
    return True
