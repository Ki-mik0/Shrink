"""``--dedent``: indentation that costs fewer tokens, and the way back.

BPE tokenizers such as o200k_base give a run of spaces a token of its own,
unless it is a single space, which merges with the word after it (``" return"``
is one token). So Python indented with one space per level costs about 5% fewer
tokens than with four, and JavaScript, HTML and CSS without any indentation
8-12% fewer (measured on the answers of the generation eval). Blank lines cost
nothing extra (``"\\n\\n"`` is one token), so they stay.

Indentation is part of Python's syntax, so Python keeps one space per level;
in JavaScript, HTML and CSS it is only for people. The ``compact_*`` functions
prepare the context, the ``expand_*`` functions give an answer written that way
its usual indentation back (``shrink apply``). Both keep the code: Python is
checked by comparing syntax trees, JavaScript by comparing tokens, HTML and CSS
by comparing everything but whitespace. Lines that start inside a string, a
template literal or a comment, and ``<pre>``/``<textarea>`` blocks, are never
touched.

Pan et al. (2025), "The Hidden Cost of Readability", measured the same effect
across ten models: formatting is 24.5% of code tokens, removing it from the
input does not hurt, and models asked to write unformatted code can.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import re
import tokenize
from collections import Counter

from shrink.compressors import web

PYTHON_UNIT = 4  # PEP 8
WEB_UNIT = 2  # what most JavaScript, HTML and CSS style guides use
_COMPACT_SHARE = 0.2  # an answer counts as compact if at most this share of lines is indented
_VOID = frozenset(
    [
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    ]
)
_FLAT = frozenset({"html"})  # <head> and <body> usually sit at the left edge
_RAW = re.compile(r"<(pre|textarea)\b.*?</\1\s*>", re.I | re.S)
# The code of an inline block; the indentation in front of its closing tag is not part of it.
_CODE_BLOCK = re.compile(r"(<(script|style)\b[^>]*>)(.*?\n?)([ \t]*)(?=</\2\s*>)", re.I | re.S)
_TAG = re.compile(r"<!--.*?-->|<!.*?>|<(/?)([a-zA-Z][\w:-]*)\b([^<>]*?)(/?)>", re.S)


class IndentError(ValueError):
    """The text could not be re-indented without changing the code."""


# ------------------------------------------------------------------ Python


def _python_rows(text: str) -> tuple[set[int], set[int]]:
    """Rows (1-based) that start inside a multi-line string, and rows that start a
    logical line (whose indentation is syntax)."""
    inside: set[int] = set()
    logical: set[int] = set()
    open_strings: list[int] = []
    starts = {
        getattr(tokenize, n) for n in ("FSTRING_START", "TSTRING_START") if hasattr(tokenize, n)
    }
    ends = {getattr(tokenize, n) for n in ("FSTRING_END", "TSTRING_END") if hasattr(tokenize, n)}
    new_line = True
    skip = {tokenize.NL, tokenize.COMMENT, tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.NEWLINE:
            new_line = True
            continue
        if new_line and tok.type not in skip and tok.type != tokenize.ENDMARKER:
            logical.add(tok.start[0])
            new_line = False
        if tok.type == tokenize.STRING and tok.start[0] != tok.end[0]:
            inside.update(range(tok.start[0] + 1, tok.end[0] + 1))
        elif tok.type in starts:
            open_strings.append(tok.start[0])
        elif tok.type in ends and open_strings:
            first = open_strings.pop()
            inside.update(range(first + 1, tok.end[0] + 1))
    return inside, logical


def python_unit(text: str) -> int | None:
    """The indentation step of a Python file (greatest common divisor of the
    indentation of its logical lines), ``None`` without indentation or with tabs."""
    inside, logical = _python_rows(text)
    widths: list[int] = []
    for row, line in enumerate(text.splitlines(), start=1):
        if row in inside or row not in logical:
            continue
        lead = line[: len(line) - len(line.lstrip(" \t"))]
        if "\t" in lead:
            return None
        if lead:
            widths.append(len(lead))
    return math.gcd(*widths) if widths else None


def _rescale_python(text: str, have: int, want: int) -> str:
    inside, _ = _python_rows(text)
    out: list[str] = []
    for row, line in enumerate(text.splitlines(keepends=True), start=1):
        body = line.lstrip(" ")
        if row in inside or not body.strip():
            out.append(line if row in inside else body if body.strip() else line.lstrip(" \t"))
            continue
        width = len(line) - len(body)
        levels, rest = divmod(width, have)
        out.append(" " * (levels * want + rest) + body)
    return "".join(out)


def _same_python(before: ast.AST, after: str) -> None:
    if ast.dump(ast.parse(after)) != ast.dump(before):
        raise IndentError("re-indenting changed the Python code")


def compact_python(text: str) -> str:
    """One space per indentation level."""
    if not text.strip():
        return text
    tree = ast.parse(text)
    have = python_unit(text)
    if have is None or have == 1:
        return text
    result = _rescale_python(text, have, 1)
    _same_python(tree, result)
    return result


def expand_python(text: str, unit: int = PYTHON_UNIT) -> str:
    """``unit`` spaces per level for an answer indented with one space (or tabs
    used as levels); anything else is left as it is."""
    if not text.strip():
        return text
    tree = ast.parse(text)
    have = python_unit(text)
    if have is None and "\n\t" in text:
        text = re.sub(r"(?m)^\t+", lambda m: " " * len(m.group(0)), text)
        have = 1
    if have != 1 or unit == 1:
        return text
    result = _rescale_python(text, 1, unit)
    _same_python(tree, result)
    return result


# ------------------------------------------------------------------ JavaScript


def compact_js(src: str) -> str:
    """No indentation: whitespace at the start of a line goes, unless the line
    starts inside a template literal or comment (those are single tokens)."""
    if not src.strip():
        return src
    toks = web.lex_js(src)
    out: list[web.Tok] = []
    line_start = True
    for tok in toks:
        if line_start and tok.kind == "ws":
            continue
        out.append(tok)
        line_start = tok.kind == "nl"
    result = "".join(tok.text for tok in out)
    if web._signature(web.lex_js(result)) != web._signature(toks):
        raise IndentError("removing indentation changed the JavaScript")
    return result


def _js_line_indents(toks: list[web.Tok]) -> list[tuple[int, int]]:
    """For every line: (index of its first token, indentation level). The level is
    the number of earlier lines that opened a bracket still open, so ``f(g({``
    on one line indents the next line once, not three times."""
    result: list[tuple[int, int]] = []
    stack: list[int] = []  # line number of every open bracket
    line = 0
    first = 0
    while first < len(toks):
        j = first
        while j < len(toks) and toks[j].kind == "ws":
            j += 1
        closers = 0
        k = j
        while k < len(toks) and toks[k].kind == "punct" and toks[k].text in ")]}":
            closers += 1
            k += 1
            while k < len(toks) and toks[k].kind == "ws":
                k += 1
        remaining = stack[: max(len(stack) - closers, 0)]
        result.append((first, len(set(remaining))))
        end = first
        while end < len(toks) and toks[end].kind != "nl":
            tok = toks[end]
            if tok.kind == "punct" and tok.text in "([{":
                stack.append(line)
            elif tok.kind == "punct" and tok.text in ")]}" and stack:
                stack.pop()
            end += 1
        line += 1
        first = end + 1
    return result


def _mostly_flush(lines: list[tuple[bool, int]]) -> bool:
    """``(starts with whitespace, level)`` per non-blank line: compact if at most a
    small share of the lines that would be indented already are."""
    deep = [indented for indented, level in lines if level > 0]
    return not deep or sum(deep) <= _COMPACT_SHARE * len(deep)


def expand_js(src: str, unit: int = WEB_UNIT) -> str:
    """Indent an answer written without indentation by bracket depth."""
    if not src.strip():
        return src
    toks = web.lex_js(src)
    indents = _js_line_indents(toks)
    lines = []
    for first, level in indents:
        if first < len(toks) and toks[first].kind != "nl":
            lines.append((toks[first].kind == "ws", level))
    if not _mostly_flush(lines):
        return src
    out: list[str] = []
    starts = dict(indents)
    i = 0
    while i < len(toks):
        if i in starts:
            j = i
            while j < len(toks) and toks[j].kind == "ws":
                j += 1
            if j < len(toks) and toks[j].kind != "nl":
                out.append(" " * (unit * starts[i]))
            i = j
            if i >= len(toks):
                break
        out.append(toks[i].text)
        i += 1
    result = "".join(out)
    if web._signature(web.lex_js(result)) != web._signature(toks):
        raise IndentError("re-indenting changed the JavaScript")
    return result


# ------------------------------------------------------------------ CSS


def _no_space(text: str) -> str:
    return re.sub(r"\s+", "", text)


def compact_css(src: str) -> str:
    if not src.strip():
        return src
    out: list[str] = []
    for kind, text in web._lex_css(src):
        out.append(text[: text.rfind("\n") + 1] if kind == "ws" and "\n" in text else text)
    result = "".join(out)
    if _no_space(result) != _no_space(src):
        raise IndentError("removing indentation changed the CSS")
    return result


def expand_css(src: str, unit: int = WEB_UNIT) -> str:
    if not src.strip():
        return src
    tokens = web._lex_css(src)
    lines: list[list[tuple[str, str]]] = [[]]
    for kind, text in tokens:
        if kind == "ws" and "\n" in text:
            for _ in range(text.count("\n")):
                lines[-1].append(("nl", "\n"))
                lines.append([])
            continue
        lines[-1].append((kind, text))
    depth = 0
    plan: list[tuple[bool, int]] = []
    levels: list[int] = []
    for line in lines:
        body = [t for t in line if t[0] not in ("ws", "nl")]
        level = depth - (1 if body and body[0][1] == "}" else 0)
        levels.append(max(level, 0))
        if body:
            plan.append((bool(line) and line[0][0] == "ws", max(level, 0)))
        for _, text in body:
            depth += {"{": 1, "}": -1}.get(text, 0)
    if not _mostly_flush(plan):
        return src
    out: list[str] = []
    for line, level in zip(lines, levels, strict=True):
        first = next((k for k, tok in enumerate(line) if tok[0] != "ws"), len(line))
        if first < len(line) and line[first][0] != "nl":
            out.append(" " * (unit * level))
        out.extend(text for _, text in line[first:])
    result = "".join(out)
    if _no_space(result) != _no_space(src):
        raise IndentError("re-indenting changed the CSS")
    return result


# ------------------------------------------------------------------ HTML


def _raw_spans(src: str) -> list[tuple[int, int]]:
    """``<pre>``, ``<textarea>`` and comments: their lines keep their whitespace."""
    return [m.span() for m in _RAW.finditer(src)] + [
        m.span() for m in re.finditer(r"<!--.*?-->", src, re.S)
    ]


def _in_spans(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < pos < end for start, end in spans)


def _code_blocks(src: str, raw: list[tuple[int, int]]) -> list[re.Match[str]]:
    """Inline ``<script>``/``<style>`` blocks that hold code (not inside ``<pre>``)."""
    return [
        m
        for m in _CODE_BLOCK.finditer(src)
        if m.end(3) > m.start(3) and not _in_spans(m.start(), raw)
    ]


def compact_html(src: str) -> str:
    """No indentation in markup; inline scripts and styles as in JavaScript and CSS."""
    if not src.strip():
        return src
    raw = _raw_spans(src)
    blocks = {m.start(3): m for m in _code_blocks(src, raw)}
    out: list[str] = []
    pos = 0
    line_start = True
    while pos < len(src):
        if pos in blocks:
            match = blocks[pos]
            code = match.group(3)
            try:
                is_js = match.group(2).lower() == "script"
                code = compact_js(code) if is_js else compact_css(code)
            except (web.LexError, IndentError):
                pass
            out.append(code)
            pos = match.end(3)
            line_start = src[pos - 1] == "\n"
            continue
        if line_start and not _in_spans(pos, raw):
            while pos < len(src) and src[pos] in " \t":
                pos += 1
            line_start = False
            continue
        out.append(src[pos])
        line_start = src[pos] == "\n"
        pos += 1
    result = "".join(out)
    if _no_space(result) != _no_space(src):
        raise IndentError("removing indentation changed the HTML")
    return result


def _tag_events(markup: str) -> list[tuple[int, int, bool]]:
    """(position, depth change, is a closing tag) for every tag in ``markup``."""
    events: list[tuple[int, int, bool]] = []
    for match in _TAG.finditer(markup):
        slash, name, _attrs, self_closing = match.groups()
        if name is None:
            continue
        name = name.lower()
        if slash:
            if name not in _FLAT:
                events.append((match.start(), -1, True))
        elif not self_closing and name not in _VOID and name not in _FLAT:
            events.append((match.start(), 1, False))
    return events


class _HtmlLayout:
    """Element depth at every position of an HTML text (inline code masked out)."""

    def __init__(self, src: str) -> None:
        self.src = src
        self.raw = _raw_spans(src)
        self.blocks = _code_blocks(src, self.raw)
        masked = list(src)
        for match in self.blocks:  # so a "<" in code never reads as a tag
            for k in range(match.start(3), match.end(3)):
                if masked[k] != "\n":
                    masked[k] = " "
        self.events = _tag_events("".join(masked))
        self.in_block = [(m.start(3), m.end(3)) for m in self.blocks]

    def depth_at(self, pos: int) -> int:
        return sum(delta for at, delta, _ in self.events if at < pos)

    def line_levels(self) -> tuple[dict[int, int], list[tuple[bool, int]]]:
        """{line start: level} for markup lines, and (was indented, level) for each."""
        src = self.src
        levels: dict[int, int] = {}
        flags: list[tuple[bool, int]] = []
        for start in [0] + [m.end() for m in re.finditer("\n", src)]:
            if start >= len(src) or _in_spans(start, self.raw) or _in_spans(start, self.in_block):
                continue
            k = start
            while k < len(src) and src[k] in " \t":
                k += 1
            if k >= len(src) or src[k] == "\n":
                continue
            level = max(self.depth_at(start) - _leading_closers(src[k:]), 0)
            levels[start] = level
            flags.append((k > start, level))
        return levels, flags


def _leading_closers(rest: str) -> int:
    count = 0
    while True:
        closing = re.match(r"</([a-zA-Z][\w:-]*)\s*>[ \t]*", rest)
        if not closing:
            return count
        count += closing.group(1).lower() not in _FLAT
        rest = rest[closing.end() :]


def _expand_block(match: re.Match[str], base: int, unit: int) -> str:
    """The code of an inline ``<script>``/``<style>`` re-indented below its tag."""
    code = match.group(3)
    with contextlib.suppress(web.LexError, IndentError):  # unreadable code stays as it is
        code = expand_js(code) if match.group(2).lower() == "script" else expand_css(code)
    pad = " " * (unit * base)
    code = re.sub(r"(?m)^(?=[^\n])", pad, code) if pad else code
    if not match.group(3).startswith("\n") and code.startswith(pad):
        code = code[len(pad) :]
    return code


def expand_html(src: str, unit: int = WEB_UNIT) -> str:
    """Indent markup by element depth; inline scripts and styles by their own rules."""
    if not src.strip():
        return src
    layout = _HtmlLayout(src)
    levels, flags = layout.line_levels()
    if not _mostly_flush(flags):
        return src
    block_at = {m.start(3): m for m in layout.blocks}
    out: list[str] = []
    pos = 0
    while pos < len(src):
        if pos in block_at:
            match = block_at[pos]
            out.append(_expand_block(match, layout.depth_at(match.start(3)), unit))
            pos = match.end(3)
        elif pos in levels:
            out.append(" " * (unit * levels.pop(pos)))
            while pos < len(src) and src[pos] in " \t":
                pos += 1
        else:
            out.append(src[pos])
            pos += 1
    result = "".join(out)
    if _no_space(result) != _no_space(src):
        raise IndentError("re-indenting changed the HTML")
    return result


# ------------------------------------------------------------------ by file kind


def kind_of(path: str | None, text: str = "") -> str | None:
    """``python``, ``js``, ``css``, ``html`` or ``None`` for other files."""
    name = (path or "").lower()
    if name.endswith((".py", ".pyi")):
        return "python"
    if name.endswith(web.WEB_EXTENSIONS):
        return {"js": "js", "css": "css", "html": "html"}[web.kind_of(name)]
    return None


def compact(text: str, kind: str | None) -> str:
    functions = {
        "python": compact_python,
        "js": compact_js,
        "css": compact_css,
        "html": compact_html,
    }
    return functions[kind](text) if kind in functions else text


def expand(text: str, kind: str | None, unit: int | None = None) -> str:
    if kind == "python":
        return expand_python(text, unit or PYTHON_UNIT)
    functions = {"js": expand_js, "css": expand_css, "html": expand_html}
    return functions[kind](text, unit or WEB_UNIT) if kind in functions else text


def detect_unit(text: str, kind: str | None) -> int | None:
    """The indentation step an existing file uses, so an answer can match it."""
    if kind == "python":
        try:
            return python_unit(text)
        except (SyntaxError, tokenize.TokenError):
            return None
    widths = [
        len(line) - len(line.lstrip(" "))
        for line in text.splitlines()
        if line.startswith(" ") and line.strip() and not line.lstrip().startswith("*")
    ]
    if not widths:
        return None
    counts = Counter(widths)
    frequent = [w for w, n in counts.items() if n >= max(2, len(widths) // 10)]
    return min(frequent or widths)
