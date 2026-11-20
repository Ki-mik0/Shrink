"""Text mode: whitespace, duplicates and markup noise.

The transform is deterministic and idempotent (running it on its own output
changes nothing). Each rule only ever removes characters or swaps a tab for a
space, so the output is never longer than the input; the rules are applied
until a fixed point is reached, which makes idempotence hold even when one rule
uncovers work for another (e.g. ``<b>[x](y)</b>``).
"""

from __future__ import annotations

import re
from collections.abc import Callable

from shrink.compressors.base import Step, check_level
from shrink.document import Document

_MAX_PASSES = 16
MIN_DUPLICATE_CHARS = 20  # shorter paragraphs ("Yes.", "---") may repeat on purpose

_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INNER_WS = re.compile(r"[ \t]+")
_INLINE_CODE = re.compile(r"(`[^`\n]*`)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG = re.compile(r"</?[A-Za-z][A-Za-z0-9-]*(?:\s[^<>]*)?/?>")
_MD_IMAGE = re.compile(r"!\[([^\]\n]*)\]\([^)\n]*\)")
_MD_LINK = re.compile(r"\[([^\]\n]*)\]\([^)\n]*\)")
_HR = re.compile(r"^ {0,3}([-*_])(?:[ \t]*\1){2,}[ \t]*$")

HELP = f"""\
Cleans prose, Markdown and other plain text. Line endings are kept: pure CRLF
input stays CRLF, anything else becomes LF.

  level 1  light     strip trailing whitespace, squeeze runs of blank lines
                     into one, drop leading/trailing blank lines
  level 2  medium    + squeeze spaces and tabs inside lines (indentation is
                     kept), drop repeated paragraphs of {MIN_DUPLICATE_CHARS}+ characters
                     (headings are never dropped)
  level 3  skeleton  + remove HTML comments and tags, turn Markdown links and
                     images into their text, drop horizontal rules

Fenced code blocks (``` or ~~~) are left alone at levels 2 and 3. Inline code
spans are left alone at level 3.

In a packed stream, files with a source-code extension (.py, .js, .json, ...) only
get level 1, because the higher levels would change what the code means."""


# Source files found in a packed stream. Dropping "duplicate paragraphs" or stripping
# tags would change what the code means, so for these only level 1 (whitespace) runs.
CODE_EXTENSIONS = frozenset(
    {
        ".py", ".pyi", ".js", ".jsx", ".mjs", ".cjs", ".ts", ".tsx", ".java", ".kt", ".go",
        ".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".scala",
        ".sh", ".bash", ".ps1", ".sql", ".lua", ".r", ".pl", ".vue", ".svelte", ".html",
        ".htm", ".xml", ".css", ".scss", ".json", ".yaml", ".yml", ".toml", ".ini", ".mk",
    }
)  # fmt: skip


def is_code_path(path: str | None) -> bool:
    if path is None:
        return False
    name = path.replace("\\", "/").rsplit("/", 1)[-1].lower()
    dot = name.rfind(".")
    return (dot >= 0 and name[dot:] in CODE_EXTENSIONS) or name in {"makefile", "dockerfile"}


STEP_NAMES = {1: "whitespace", 2: "spaces+duplicates", 3: "markup"}


class TextCompressor:
    name = "text"
    summary = "normalise whitespace, drop duplicate paragraphs and markup noise"
    help = HELP

    def compress(self, doc: Document, level: int) -> Document:
        return doc.with_text(compress_text(doc.text, self._effective_level(doc, level)))

    def trace(self, doc: Document, level: int) -> list[Step]:
        """One step per level up to the one in use: the levels are cumulative, so
        step ``k`` is what level ``k`` alone gives, and the last step is the result."""
        effective = self._effective_level(doc, level)
        return [
            Step(STEP_NAMES[k], doc.with_text(compress_text(doc.text, k)))
            for k in range(1, effective + 1)
        ]

    @staticmethod
    def _effective_level(doc: Document, level: int) -> int:
        check_level(level)
        return min(level, 1) if is_code_path(doc.path) else level


def compress_text(text: str, level: int) -> str:
    check_level(level)
    if not text:
        return text
    pure_crlf = "\r\n" in text and "\n" not in text.replace("\r\n", "")
    body = text.replace("\r\n", "\n")
    for _ in range(_MAX_PASSES):
        new = _one_pass(body, level)
        if new == body:
            break
        body = new
    return body.replace("\n", "\r\n") if pure_crlf else body


def _one_pass(body: str, level: int) -> str:
    lines = [line.rstrip() for line in body.split("\n")]
    if level >= 2:
        lines = _rewrite_prose(lines, level)
    lines = _squeeze_blank_lines(lines)
    result = "\n".join(lines)
    if result and body.endswith("\n"):
        result += "\n"
    return result


# ------------------------------------------------------------------ structure


def _segments(lines: list[str]) -> list[tuple[bool, list[str]]]:
    """Split lines into (is_code, lines) runs; fences belong to the code run."""
    segments: list[tuple[bool, list[str]]] = []
    prose: list[str] = []
    i = 0
    while i < len(lines):
        match = _FENCE_OPEN.match(lines[i])
        if match is None:
            prose.append(lines[i])
            i += 1
            continue
        if prose:
            segments.append((False, prose))
            prose = []
        fence = match.group(1)
        closing = re.compile(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
        code = [lines[i]]
        i += 1
        while i < len(lines):
            code.append(lines[i])
            i += 1
            if closing.match(code[-1]):
                break
        segments.append((True, code))
    if prose:
        segments.append((False, prose))
    return segments


def _rewrite_prose(lines: list[str], level: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for is_code, segment in _segments(lines):
        if is_code:
            out.extend(segment)
            continue
        prose = _strip_noise(segment) if level >= 3 else segment
        prose = [_squeeze_inner_whitespace(line) for line in prose]
        out.extend(_drop_duplicate_paragraphs(prose, seen))
    return out


# ------------------------------------------------------------------ level 1


def _squeeze_blank_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    while out and not out[-1]:
        out.pop()
    return out


# ------------------------------------------------------------------ level 2


def _squeeze_inner_whitespace(line: str) -> str:
    body = line.lstrip(" \t")
    indent = line[: len(line) - len(body)]
    return indent + _INNER_WS.sub(" ", body)


def _paragraph_key(paragraph: list[str]) -> str:
    return " ".join(" ".join(paragraph).split())


def _drop_duplicate_paragraphs(lines: list[str], seen: set[str]) -> list[str]:
    out: list[str] = []
    paragraph: list[str] = []

    def flush() -> None:
        key = _paragraph_key(paragraph)
        if len(key) >= MIN_DUPLICATE_CHARS and not key.startswith("#"):
            if key in seen:
                paragraph.clear()
                return
            seen.add(key)
        out.extend(paragraph)
        paragraph.clear()

    for line in lines:
        if line:
            paragraph.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return out


# ------------------------------------------------------------------ level 3


def _outside_code(text: str, transform: Callable[[str], str]) -> str:
    """Apply ``transform`` to everything except inline code spans."""
    parts = _INLINE_CODE.split(text)
    # Odd indices are the code spans captured by the split.
    return "".join(part if i % 2 else transform(part) for i, part in enumerate(parts))


def _strip_markup(text: str) -> str:
    text = _HTML_BREAK.sub(" ", text)
    text = _HTML_TAG.sub("", text)
    text = _MD_IMAGE.sub(r"\1", text)
    return _MD_LINK.sub(r"\1", text)


def _strip_line(line: str) -> str:
    content = line.lstrip(" \t")
    indent = line[: len(line) - len(content)]
    # Whitespace exposed at the start by removing a tag is not indentation.
    return (indent + _outside_code(content, _strip_markup).lstrip(" \t")).rstrip()


def _strip_noise(lines: list[str]) -> list[str]:
    # Comments may span lines, so they go first, on the whole segment.
    text = _outside_code("\n".join(lines), lambda part: _HTML_COMMENT.sub("", part))
    out: list[str] = []
    for raw in text.split("\n"):
        line = _strip_line(raw)
        if _HR.match(line) and (not out or not out[-1]):
            continue
        out.append(line)
    return out
