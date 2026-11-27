"""--dedent layout: compact indentation for the context, and the way back."""

from __future__ import annotations

import ast

import pytest

from shrink import indent
from shrink.compressors import web
from shrink.compressors.dedent import DedentCompressor
from shrink.document import Document

PYTHON = '''\
"""Module."""

SQL = """
    SELECT 1
        FROM t
"""


class A:
    def f(self, x):
        if x:
            return [
                1,
                2,
            ]
        return None
'''

JS = """\
export function f(a) {
  if (a) {
    return g({
      b: 1,
    });
  }
  const t = `line one
      kept as is`;
  return t;
}
"""

HTML = """\
<!doctype html>
<html lang="en">
<head>
  <title>T</title>
</head>
<body>
  <nav id="nav">
    <a href="/">Home</a>
  </nav>
  <pre>
    kept   as is
  </pre>
  <script type="module">
    import { f } from "./f.js";
    if (f(1)) {
      console.log("<b>not a tag</b>");
    }
  </script>
</body>
</html>
"""

CSS = """\
:root {
  --gap: 12px;
}
.a {
  color: red;
}
"""


def test_python_one_space_per_level_keeps_the_code() -> None:
    compact = indent.compact_python(PYTHON)
    assert ast.dump(ast.parse(compact)) == ast.dump(ast.parse(PYTHON))
    assert "\n def f(self, x):\n  if x:\n   return [" in compact
    assert '\n    SELECT 1\n        FROM t\n"""' in compact  # string lines untouched
    assert indent.expand_python(compact) == PYTHON


def test_python_unit_and_tabs() -> None:
    assert indent.python_unit(PYTHON) == 4
    assert indent.python_unit("x = 1\n") is None
    assert indent.python_unit("if x:\n\ty = 1\n") is None
    # tabs used as levels expand to spaces
    assert indent.expand_python("if x:\n\ty = 1\n") == "if x:\n    y = 1\n"


def test_python_already_compact_or_normal_is_left_alone() -> None:
    assert indent.compact_python("x = 1\n") == "x = 1\n"
    assert indent.compact_python("") == ""
    normal = "if x:\n    y = 1\n"
    assert indent.expand_python(normal) == normal  # four spaces: not an answer to expand
    assert indent.expand_python("if x:\n y = 1\n", unit=2) == "if x:\n  y = 1\n"


def test_python_that_does_not_parse_raises() -> None:
    with pytest.raises(SyntaxError):
        indent.compact_python("def (:\n")


def test_js_without_indentation_round_trips() -> None:
    compact = indent.compact_js(JS)
    assert "\nif (a) {\nreturn g({\nb: 1,\n});\n}" in compact
    assert "      kept as is`" in compact  # template literal untouched
    assert indent.expand_js(compact) == JS


def test_js_brackets_on_one_line_indent_once() -> None:
    assert indent.expand_js("f(g({\na: 1,\n}));\n") == "f(g({\n  a: 1,\n}));\n"


def test_js_that_is_already_indented_is_left_alone() -> None:
    assert indent.expand_js(JS) == JS
    assert indent.expand_js("") == ""
    assert indent.compact_js("  ") == "  "


def test_css_round_trips() -> None:
    compact = indent.compact_css(CSS)
    assert "\n--gap: 12px;\n" in compact
    assert indent.expand_css(compact) == CSS
    assert indent.expand_css(CSS) == CSS
    assert indent.compact_css("") == ""


def test_html_round_trips_and_keeps_pre_and_code() -> None:
    compact = indent.compact_html(HTML)
    assert '\n<nav id="nav">\n<a href' in compact
    assert "    kept   as is\n" in compact
    assert '\nconsole.log("<b>not a tag</b>");' in compact
    assert indent.expand_html(compact) == HTML
    assert indent.expand_html(HTML) == HTML  # already indented
    assert indent.compact_html("") == indent.expand_html("") == ""


def test_html_script_on_one_line_and_empty_script() -> None:
    html = '<body>\n<script src="a.js"></script>\n<script>f()</script>\n</body>\n'
    assert indent.compact_html(html) == html
    assert indent.expand_html(html) == (
        '<body>\n  <script src="a.js"></script>\n  <script>f()</script>\n</body>\n'
    )


def test_kind_detect_and_dispatch() -> None:
    assert indent.kind_of("a/b.py") == "python"
    assert indent.kind_of("x.mjs") == "js"
    assert indent.kind_of("s.css") == "css"
    assert indent.kind_of("p.htm") == "html"
    assert indent.kind_of("README.md") is None
    assert indent.compact("text", None) == "text"
    assert indent.expand("text", None) == "text"
    assert indent.detect_unit(PYTHON, "python") == 4
    assert indent.detect_unit(JS, "js") == 2
    assert indent.detect_unit("x", "js") is None
    assert indent.detect_unit("def (:", "python") is None
    assert indent.expand(indent.compact(JS, "js"), "js", 4).startswith(
        "export function f(a) {\n    if (a) {"
    )


def test_checks_catch_a_broken_rewrite(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(indent, "_rescale_python", lambda text, have, want: "x = 2\n")
    with pytest.raises(indent.IndentError):
        indent.compact_python("if x:\n    y = 1\n")
    real_lex = web.lex_js
    calls = {"n": 0}

    def lex(src: str) -> list[web.Tok]:
        calls["n"] += 1
        return real_lex(src if calls["n"] == 1 else src + " extra")

    monkeypatch.setattr(web, "lex_js", lex)
    with pytest.raises(indent.IndentError):
        indent.compact_js("f();\n")


def test_dedent_compressor() -> None:
    step = DedentCompressor()
    py = step.compress(Document(PYTHON, path="a.py"), 3)
    assert "\n def f(self, x):" in py.text
    assert step.compress(Document("# notes\n", path="notes.md"), 3).text == "# notes\n"
    sniffed = step.compress(Document("def f():\n    return 1\n"), 3)  # stdin, no path
    assert sniffed.text == "def f():\n return 1\n"
    broken = Document("def (:\n    x\n", path="b.py")
    assert step.compress(broken, 3) == broken  # fail-open
