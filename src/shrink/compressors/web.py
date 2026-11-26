"""Web mode: JavaScript, CSS and HTML.

JavaScript is split into tokens by a small lexer that knows strings, template
literals (with nested ``${...}``), regular expressions and comments, so a ``//``
inside a string or a URL is never taken for a comment. Every level is checked:
the output is lexed again and must contain the same code tokens, with a line
break wherever the input had one (JavaScript inserts semicolons at some line
breaks, so they are part of the meaning). Level 3 replaces function bodies and
is checked for balanced brackets instead. Anything that fails a check raises,
and the pipeline passes the file through unchanged (fail-open).

CSS is lexed for comments, strings and ``url(...)``; HTML keeps ``<pre>`` and
``<textarea>`` as they are and compresses inline ``<script>`` and ``<style>``
blocks with the JavaScript and CSS rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from shrink.compressors.base import Step, check_level
from shrink.document import Document

HELP = """\
JavaScript (.js .mjs .cjs .jsx), CSS and HTML. The result is lexed again and must
keep the same code; a file that fails the check is passed through unchanged.

  level 1  light     remove comments (license `/*! */`, `// @ts-...`, eslint and
                     bundler pragmas stay), trailing whitespace, extra blank lines;
                     HTML comments go, <pre> and <textarea> stay as they are
  level 2  medium    + JSDoc `/** ... */` shortened to its first sentence;
                     CSS minified (whitespace, last `;` of a block)
  level 3  skeleton  + bodies of `function` declarations and block arrow functions
                     become `{/* ... */}`, keeping names, parameters and defaults;
                     CSS keeps selectors and custom properties (`--name: value`)

Inline <script> and <style> blocks in HTML get the same treatment."""

STEP_NAMES = {1: "comments", 2: "doc comments+minify", 3: "bodies"}
JS_EXTENSIONS = (".js", ".mjs", ".cjs", ".jsx")
CSS_EXTENSIONS = (".css",)
HTML_EXTENSIONS = (".html", ".htm")
WEB_EXTENSIONS = JS_EXTENSIONS + CSS_EXTENSIONS + HTML_EXTENSIONS

_REGEX_AFTER_WORDS = frozenset({
    "return", "typeof", "instanceof", "in", "of", "new", "delete", "void", "throw",
    "case", "do", "else", "yield", "await",
})  # fmt: skip
_JS_PRAGMA = re.compile(
    r"^(/\*!|//[#@]|///\s*<|/[/*]\s*(@ts-|eslint|prettier-ignore|istanbul|c8 |jshint|global |"
    r"@vite-ignore|webpack|@license|@preserve))"
)
_SPACE = " \t\f\v" + chr(0xA0) + chr(0xFEFF)  # no-break space, BOM
_HTML_RAW = re.compile(r"(<(pre|textarea)\b.*?</\2\s*>)", re.I | re.S)
_HTML_SCRIPT = re.compile(r"(<script\b([^>]*)>)(.*?)(</script\s*>)", re.I | re.S)
_HTML_STYLE = re.compile(r"(<style\b[^>]*>)(.*?)(</style\s*>)", re.I | re.S)
_HTML_COMMENT = re.compile(r"<!--(?!\[if).*?-->", re.S)
_SCRIPT_TYPES = ("", "module", "text/javascript", "application/javascript")


class LexError(ValueError):
    pass


@dataclass(frozen=True)
class Tok:
    kind: str  # word punct string template regex line_comment block_comment ws nl shebang
    text: str


_REMOVED = Tok("removed", "")
BODY_OMITTED = Tok("block_comment", "/* ... */")


# ------------------------------------------------------------------ JavaScript lexer


def lex_js(src: str) -> list[Tok]:
    toks: list[Tok] = []
    i = 0
    if src.startswith("#!"):
        i = _line_end(src, 0)
        toks.append(Tok("shebang", src[:i]))
    while i < len(src):
        tok, i = _next_token(src, i, toks)
        toks.append(tok)
    return toks


def _line_end(src: str, i: int) -> int:
    """Index of the line break after ``i`` (before a CR of CRLF), or the end."""
    end = src.find("\n", i)
    end = len(src) if end < 0 else end
    return end - 1 if end > i and src[end - 1] == "\r" else end


def _next_token(src: str, i: int, toks: list[Tok]) -> tuple[Tok, int]:
    c = src[i]
    if c in "\r\n":
        size = 2 if src.startswith("\r\n", i) else 1
        return Tok("nl", src[i : i + size]), i + size
    if c in _SPACE:
        j = i
        while j < len(src) and src[j] in _SPACE:
            j += 1
        return Tok("ws", src[i:j]), j
    found = _comment_or_literal(src, i, toks)
    if found is not None:
        return found
    if c.isalnum() or c in "_$" or ord(c) > 127:
        j = i
        while j < len(src) and _word_char(src[j]):
            j += 1
        return Tok("word", src[i:j]), j
    return Tok("punct", c), i + 1


def _word_char(ch: str) -> bool:
    return ch.isalnum() or ch in "_$" or (ord(ch) > 127 and ch not in _SPACE)


def _comment_or_literal(src: str, i: int, toks: list[Tok]) -> tuple[Tok, int] | None:
    c = src[i]
    if src.startswith("//", i):
        j = _line_end(src, i)
        return Tok("line_comment", src[i:j]), j
    if src.startswith("/*", i):
        j = src.find("*/", i + 2)
        if j < 0:
            raise LexError("unterminated comment")
        return Tok("block_comment", src[i : j + 2]), j + 2
    if c in "'\"":
        j = _scan_string(src, i, c)
        return Tok("string", src[i:j]), j
    if c == "`":
        j = _scan_template(src, i)
        return Tok("template", src[i:j]), j
    if c == "/" and _regex_allowed(toks):
        j = _scan_regex(src, i)
        return Tok("regex", src[i:j]), j
    return None


def _scan_string(src: str, i: int, quote: str) -> int:
    j = i + 1
    while j < len(src):
        ch = src[j]
        if ch == "\\":
            j += 2
            continue
        if ch == quote:
            return j + 1
        if ch in "\r\n":
            break
        j += 1
    raise LexError("unterminated string")


def _scan_template(src: str, i: int) -> int:
    j = i + 1
    while j < len(src):
        ch = src[j]
        if ch == "\\":
            j += 2
        elif ch == "`":
            return j + 1
        elif src.startswith("${", j):
            j = _scan_template_expression(src, j + 2)
        else:
            j += 1
    raise LexError("unterminated template literal")


def _scan_template_expression(src: str, j: int) -> int:
    depth = 1
    while j < len(src):
        ch = src[j]
        if ch in "'\"":
            j = _scan_string(src, j, ch)
        elif ch == "`":
            j = _scan_template(src, j)
        elif src.startswith("/*", j):
            end = src.find("*/", j + 2)
            if end < 0:
                raise LexError("unterminated comment")
            j = end + 2
        elif ch == "{":
            depth += 1
            j += 1
        elif ch == "}":
            depth -= 1
            j += 1
            if depth == 0:
                return j
        else:
            j += 1
    raise LexError("unterminated template expression")


def _scan_regex(src: str, i: int) -> int:
    j, in_class = i + 1, False
    while j < len(src):
        ch = src[j]
        if ch == "\\":
            j += 2
            continue
        if ch in "\r\n":
            break
        if in_class:
            in_class = ch != "]"
        elif ch == "[":
            in_class = True
        elif ch == "/":
            j += 1
            while j < len(src) and src[j].isalnum():
                j += 1
            return j
        j += 1
    raise LexError("unterminated regular expression")


def _significant(toks: list[Tok]) -> Tok | None:
    for tok in reversed(toks):
        if tok.kind not in ("ws", "nl", "line_comment", "block_comment", "removed"):
            return tok
    return None


def _regex_allowed(toks: list[Tok]) -> bool:
    """Is a '/' here the start of a regular expression rather than a division?"""
    last = _significant(toks)
    if last is None:
        return True
    if last.kind == "word":
        return last.text in _REGEX_AFTER_WORDS
    if last.kind in ("string", "template", "regex"):
        return False
    return last.text not in ")]}"


# ------------------------------------------------------------------ JavaScript transforms


def _is_doc(tok: Tok) -> bool:
    return tok.kind == "block_comment" and tok.text.startswith("/**") and tok.text != "/**/"


def _keep_comment(tok: Tok) -> bool:
    # the level 3 body marker stays, so compressing a skeleton again changes nothing
    return tok.text == BODY_OMITTED.text or bool(_JS_PRAGMA.match(tok.text))


def _doc_summary(comment: str) -> str:
    """``/** First sentence. */`` from a JSDoc block, or "" if it has no prose."""
    from shrink.compressors.python_code import docstring_summary

    lines = [line.strip().lstrip("*").strip() for line in comment[3:-2].splitlines()]
    prose: list[str] = []
    for line in lines:
        if line.startswith("@"):
            break
        prose.append(line)
    text = "\n".join(prose).strip()
    if not text:
        return ""
    summary = docstring_summary(text).replace("*/", "* /")
    return f"/** {summary} */"


def _wordish(tok: Tok | None) -> bool:
    return tok is not None and tok.kind in ("word", "string", "template", "regex")


def strip_js_comments(toks: list[Tok], level: int, *, keep_docs: bool = True) -> list[Tok]:
    """Drop comments (level 1) and shorten JSDoc (level 2), keeping line breaks;
    ``keep_docs=False`` drops JSDoc as well."""
    out: list[Tok] = []
    for index, tok in enumerate(toks):
        if tok.kind == "line_comment" and not _keep_comment(tok):
            out.append(_REMOVED)
            continue
        if tok.kind == "block_comment" and not _keep_comment(tok):
            if _is_doc(tok) and level < 2 and keep_docs:
                out.append(tok)
                continue
            replacement = _doc_summary(tok.text) if _is_doc(tok) and keep_docs else ""
            if replacement:
                out.append(Tok("block_comment", replacement))
                continue
            out.append(_REMOVED)
            if "\n" in tok.text and not _line_ends_after(toks, index):
                # a line break inside a comment still ends a statement: keep one
                out.append(Tok("nl", "\n"))
            elif (
                _wordish(_significant(out))
                and index + 1 < len(toks)
                and _wordish(_next(toks, index))
            ):
                out.append(Tok("ws", " "))  # keep `a/**/b` from turning into `ab`
            continue
        out.append(tok)
    return out


def _line_ends_after(toks: list[Tok], index: int) -> bool:
    """True if only spaces separate token ``index`` from the next line break (or the end)."""
    for tok in toks[index + 1 :]:
        if tok.kind == "nl":
            return True
        if tok.kind != "ws":
            return False
    return True


def _next(toks: list[Tok], index: int) -> Tok | None:
    for tok in toks[index + 1 :]:
        if tok.kind not in ("ws", "nl", "line_comment", "block_comment"):
            return tok
    return None


def skeleton_js(toks: list[Tok]) -> list[Tok]:
    """Bodies of ``function`` declarations/expressions and ``=> {...}`` become
    ``{/* ... */}`` (an empty body stays ``{}``)."""
    out: list[Tok] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        body = None
        if tok.kind == "word" and tok.text == "function":
            close = _matching_after(toks, i, "(", ")")
            if close is not None:
                body = _block_after(toks, close)
        elif tok.text == "=" and i + 1 < len(toks) and toks[i + 1].text == ">":
            body = _block_after(toks, i + 1)
            if body is not None:
                out.extend(toks[i : i + 2])
                i += 2
                tok = toks[i]
        if body is None:
            out.append(tok)
            i += 1
            continue
        start, end = body
        out.extend(toks[i:start])
        out.append(Tok("punct", "{"))
        if any(t.kind not in ("ws", "nl") for t in toks[start + 1 : end]):
            # a bare `{}` reads as "empty, implement me"; mark the body as left out
            out.append(BODY_OMITTED)
        out.append(Tok("punct", "}"))
        i = end + 1
    return out


def _matching_after(toks: list[Tok], i: int, opener: str, closer: str) -> int | None:
    """Index of the ``closer`` matching the first ``opener`` after position ``i``."""
    depth = 0
    for j in range(i + 1, len(toks)):
        if toks[j].kind != "punct":
            continue
        if toks[j].text == opener:
            depth += 1
        elif toks[j].text == closer:
            depth -= 1
            if depth == 0:
                return j
        if depth == 0 and toks[j].text in ";{}" and opener != "{":
            return None
    return None


def _block_after(toks: list[Tok], i: int) -> tuple[int, int] | None:
    """(index of '{', index of matching '}') if the next significant token opens a block."""
    j = i + 1
    while j < len(toks) and toks[j].kind in ("ws", "nl", "line_comment", "block_comment"):
        j += 1
    if j >= len(toks) or toks[j].text != "{":
        return None
    depth = 0
    for k in range(j, len(toks)):
        if toks[k].kind != "punct":
            continue
        if toks[k].text == "{":
            depth += 1
        elif toks[k].text == "}":
            depth -= 1
            if depth == 0:
                return j, k
    return None


def tidy_lines(toks: list[Tok]) -> str:
    """Join tokens; strip trailing spaces, drop lines that held only removed comments,
    squeeze blank lines. Multi-line strings and templates are single tokens, so their
    insides are never touched."""
    lines: list[list[Tok]] = [[]]
    endings: list[str] = []
    for tok in toks:
        if tok.kind == "nl":
            endings.append(tok.text)
            lines.append([])
        else:
            lines[-1].append(tok)
    endings.append("")
    out: list[str] = []
    for line, ending in zip(lines, endings, strict=True):
        removed = any(tok.kind == "removed" for tok in line)
        while line and line[-1].kind in ("ws", "removed"):
            line.pop()
        text = "".join(tok.text for tok in line)
        if not text.strip() and (removed or (out and out[-1].strip() == "")):
            continue
        out.append(text + ending)
    while out and not out[-1].strip():
        out.pop()
    while out and not out[0].strip():
        out.pop(0)
    return "".join(out)


def _signature(toks: list[Tok]) -> list[tuple[str, str, bool]]:
    """Code tokens with 'a line break came before me' flags, comments and spaces left out."""
    out: list[tuple[str, str, bool]] = []
    newline = False
    for tok in toks:
        if tok.kind in ("nl",):
            newline = True
        elif tok.kind in ("ws", "line_comment", "block_comment", "removed"):
            if tok.kind == "block_comment" and "\n" in tok.text:
                newline = True
        else:
            out.append((tok.kind, tok.text, newline and bool(out)))
            newline = False
    return out


def _balanced(toks: list[Tok]) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for tok in toks:
        if tok.kind != "punct":
            continue
        if tok.text in "([{":
            stack.append(tok.text)
        elif tok.text in pairs and (not stack or stack.pop() != pairs[tok.text]):
            return False
    return not stack


def compress_js(src: str, level: int) -> str:
    check_level(level)
    if not src.strip():
        return src
    result = _without_comments(src, level, keep_docs=True)
    if level >= 3:
        skeleton = skeleton_js(lex_js(result))
        if not _balanced(skeleton):
            raise LexError("skeleton has unbalanced brackets")
        result = tidy_lines(skeleton)
        if src.endswith("\n") and result and not result.endswith("\n"):
            result += "\n"
        lex_js(result)  # must still lex
    return result


def drop_js_docs(src: str) -> str:
    """Every comment goes, JSDoc included (pragmas and the level 3 body marker stay)."""
    if not src.strip():
        return src
    return _without_comments(src, 2, keep_docs=False)


def _without_comments(src: str, level: int, *, keep_docs: bool) -> str:
    """Comments removed, checked: the code tokens and line breaks must be the same."""
    toks = lex_js(src)
    if not _balanced(toks):
        raise LexError("unbalanced brackets in the input")
    result = tidy_lines(strip_js_comments(toks, level, keep_docs=keep_docs))
    if src.endswith(("\n", "\r\n")) and result and not result.endswith("\n"):
        result += "\r\n" if src.endswith("\r\n") else "\n"
    if _signature(lex_js(result)) != _signature(toks):
        raise LexError("comment removal changed the code")
    return result


# ------------------------------------------------------------------ CSS


_CSS_TOKEN = re.compile(
    r"""(?P<comment>/\*.*?\*/)
      |(?P<string>"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*')
      |(?P<url>url\(\s*[^'")\s][^)]*\))
      |(?P<ws>\s+)
      |(?P<punct>[{};:,])
      |(?P<other>[^\s/"'u{};:,]+|.)""",
    re.S | re.X | re.I,
)


def _lex_css(src: str) -> list[tuple[str, str]]:
    tokens = []
    for match in _CSS_TOKEN.finditer(src):
        kind = match.lastgroup or "other"
        tokens.append((kind, match.group(0)))
    if src.count("/*") > src.count("*/"):
        raise LexError("unterminated CSS comment")
    return tokens


def _css_code(tokens: list[tuple[str, str]]) -> str:
    """The CSS with comments and whitespace removed, last ``;`` of blocks dropped: for checks."""
    text = "".join(t for kind, t in tokens if kind not in ("comment", "ws"))
    return text.replace(";}", "}")


def compress_css(src: str, level: int) -> str:
    check_level(level)
    if not src.strip():
        return src
    tokens = _lex_css(src)
    kept = [(k, t) for k, t in tokens if k != "comment" or t.startswith("/*!")]
    if level == 1:
        text = "".join(t for _, t in kept)
        lines = [line.rstrip() for line in text.splitlines()]
        out: list[str] = []
        for line in lines:
            if line or (out and out[-1]):
                out.append(line)
        result = "\n".join(out).strip("\n") + ("\n" if src.endswith("\n") else "")
    else:
        result = _css_minify(kept)
    if _css_code(_lex_css(result)) != _css_code(kept):
        raise LexError("CSS minification changed the rules")
    if level >= 3:
        result = css_skeleton(result)
    return result


def _css_minify(tokens: list[tuple[str, str]]) -> str:
    """No spaces around ``{ } ; , :`` and no last ``;`` in a block; one rule per line."""
    pieces: list[str] = []
    for kind, text in tokens:
        if kind == "ws":
            if pieces and pieces[-1] not in ("{", "}", ";", ",", " ", ":"):
                pieces.append(" ")
            continue
        if text in ("{", "}", ";", ",") and pieces and pieces[-1] == " ":
            pieces.pop()
        if text == "}" and pieces and pieces[-1] == ";":
            pieces.pop()
        pieces.append(text)
        if text == "}":
            pieces.append("\n")
    return "".join(pieces).replace("\n ", "\n").strip() + "\n"


def css_skeleton(src: str) -> str:
    """Selectors and custom properties (``--name: value``) stay; other declarations go.

    The level 3 idea for code (names and signatures, no bodies) applied to CSS:
    the class names and the theme variables are what new markup and styles use.
    """
    tokens = _lex_css(src)
    out, index = _css_block(tokens, 0, top=True)
    if index != len(tokens):
        raise LexError("unbalanced braces in CSS")
    result = "".join(out).strip() + "\n"
    if result.count("{") != result.count("}"):
        raise LexError("CSS skeleton has unbalanced braces")
    return result


def _css_block(tokens: list[tuple[str, str]], i: int, *, top: bool) -> tuple[list[str], int]:
    """Skeleton of the items from ``i`` to the closing ``}`` (or the end at the top)."""
    out: list[str] = []
    item: list[str] = []

    def flush_declaration() -> None:
        text = "".join(item).strip()
        if text and top:  # @import, @charset
            out.append(text + ";\n")
        elif text.startswith("--"):
            out.append(text + ";")
        item.clear()

    while i < len(tokens):
        kind, text = tokens[i]
        if kind == "comment":
            out.append(text + "\n")
        elif kind == "ws":
            item.extend([" "] if item else [])
        elif text == ";":
            flush_declaration()
        elif text == "{":
            prelude = "".join(item).strip()
            item.clear()
            inner, i = _css_block(tokens, i + 1, top=False)
            body = "".join(inner)
            if body.endswith(";"):
                body = body[:-1]
            out.append(f"{prelude}{{{body}}}" + ("\n" if top else ""))
        elif text == "}" and not top:
            flush_declaration()
            return out, i
        else:  # a stray "}" at the top level ends up here and fails the brace check
            item.append(text)
        i += 1
    if not top:
        raise LexError("unbalanced braces in CSS")
    flush_declaration()
    return out, i


# ------------------------------------------------------------------ HTML


def compress_html(src: str, level: int) -> str:
    check_level(level)
    if not src.strip():
        return src
    pieces = _HTML_RAW.split(src)
    out: list[str] = []
    # split() with two groups yields: text, whole raw block, tag name, text, ...
    for index in range(0, len(pieces), 3):
        out.append(_html_text(pieces[index], level))
        if index + 1 < len(pieces):
            out.append(pieces[index + 1])
    return "".join(out)


def _html_text(text: str, level: int) -> str:
    def script(match: re.Match[str]) -> str:
        attrs = match.group(2)
        kind = re.search(r"""type\s*=\s*["']?([^"'\s>]*)""", attrs, re.I)
        if "src=" in attrs.lower() or (
            kind is not None and kind.group(1).lower() not in _SCRIPT_TYPES
        ):
            return match.group(0)
        try:
            body = compress_js(match.group(3), level)
        except LexError:
            return match.group(0)
        return match.group(1) + ("\n" + body if body.strip() else "") + match.group(4)

    def style(match: re.Match[str]) -> str:
        try:
            body = compress_css(match.group(2), level)
        except LexError:
            return match.group(0)
        return match.group(1) + ("\n" + body if body.strip() else "") + match.group(3)

    # Comments outside script/style blocks; the blocks are handled separately.
    parts = re.split(r"(<script\b.*?</script\s*>|<style\b.*?</style\s*>)", text, flags=re.I | re.S)
    for i in range(0, len(parts), 2):
        parts[i] = _HTML_COMMENT.sub("", parts[i])
    text = "".join(parts)
    text = _HTML_SCRIPT.sub(script, text)
    text = _HTML_STYLE.sub(style, text)
    lines = [line.rstrip() for line in text.split("\n")]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out)


# ------------------------------------------------------------------ the mode


def kind_of(path: str | None) -> str:
    name = (path or "").lower()
    if name.endswith(CSS_EXTENSIONS):
        return "css"
    if name.endswith(HTML_EXTENSIONS):
        return "html"
    return "js"


_COMPRESSORS = {"js": compress_js, "css": compress_css, "html": compress_html}


class WebCompressor:
    name = "web"
    summary = "JavaScript, CSS, HTML: comments, JSDoc, CSS minification, function bodies"
    help = HELP

    def compress(self, doc: Document, level: int) -> Document:
        return self.trace(doc, level)[-1].after

    def trace(self, doc: Document, level: int) -> list[Step]:
        check_level(level)
        compress = _COMPRESSORS[kind_of(doc.path)]
        steps: list[Step] = []
        text = doc.text
        for k in range(1, level + 1):
            new = compress(doc.text, k)
            if len(new) <= len(text):  # guard: a step never makes the file longer
                text = new
            steps.append(Step(STEP_NAMES[k], doc.with_text(text)))
        return steps
