from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from shrink.compressors.auto import choose_mode
from shrink.compressors.web import (
    STEP_NAMES,
    LexError,
    WebCompressor,
    compress_css,
    compress_html,
    compress_js,
    lex_js,
)
from shrink.document import Document

STARTER = Path(__file__).resolve().parents[2] / "eval" / "generation" / "starter" / "frontend"
NODE = shutil.which("node")


def kinds(src: str) -> list[tuple[str, str]]:
    return [(t.kind, t.text) for t in lex_js(src) if t.kind not in ("ws", "nl")]


# ------------------------------------------------------------------ lexer


def test_slashes_inside_strings_are_not_comments() -> None:
    assert kinds('const u = "http://x"; // c') == [
        ("word", "const"),
        ("word", "u"),
        ("punct", "="),
        ("string", '"http://x"'),
        ("punct", ";"),
        ("line_comment", "// c"),
    ]


@pytest.mark.parametrize(
    ("src", "expected"),
    [
        ("x = a / b / c", "punct"),
        ("x = /ab+c/gi.test(s)", "regex"),
        ("return /x/", "regex"),
        ("f(/[/]/)", "regex"),
        ("a[0] / 2", "punct"),
    ],
)
def test_regex_or_division(src: str, expected: str) -> None:
    slashes = [kind for kind, text in kinds(src) if text.startswith("/")]
    assert slashes[0] == expected


def test_template_literals_with_nested_expressions() -> None:
    src = "const s = `a ${b ? `c ${d}` : '}'} // not a comment`;"
    assert ("template", src[10:-1]) in kinds(src)


def test_unterminated_input_raises() -> None:
    for broken in ('"abc', "`abc", "/* abc", "x = /abc"):
        with pytest.raises(LexError):
            lex_js(broken)


# ------------------------------------------------------------------ JavaScript levels


def test_level_1_removes_comments_but_keeps_jsdoc_and_pragmas() -> None:
    src = (
        "#!/usr/bin/env node\n"
        "/*! license */\n"
        "// plain comment\n"
        "/** Docs. */\n"
        "const a = 1; // trailing\n"
        "// @ts-ignore\n"
        "const b = 2;\n"
    )
    assert compress_js(src, 1) == (
        "#!/usr/bin/env node\n/*! license */\n/** Docs. */\n"
        "const a = 1;\n// @ts-ignore\nconst b = 2;\n"
    )


def test_level_2_shortens_jsdoc_to_its_first_sentence() -> None:
    src = (
        "/**\n * Load a user. Uses the cache.\n *\n * @param {number} id\n */\n"
        "function load(id) {}\n"
    )
    assert compress_js(src, 2) == "/** Load a user. */\nfunction load(id) {}\n"


def test_jsdoc_with_only_tags_disappears() -> None:
    assert compress_js("/** @param {x} y */\nf();\n", 2) == "f();\n"


def test_line_break_inside_a_removed_comment_is_kept() -> None:
    # `return` followed by a line break returns undefined: the break must survive
    assert compress_js("return /*\n*/ x\n", 1) == "return\n x\n"


def test_comment_between_words_keeps_them_apart() -> None:
    assert compress_js("let/**/a = 1;\n", 1) == "let a = 1;\n"


def test_level_3_replaces_function_bodies() -> None:
    src = (
        "export async function load(id, { cache = true } = {}) {\n"
        "  const r = await fetch(`/u/${id}`);\n"
        "  return r.json();\n"
        "}\n"
        "const f = (x) => {\n  return x + 1;\n};\n"
        "const g = (x) => ({ x });\n"
        "const h = () => {\n};\n"
    )
    # a left-out body is marked, so the model does not take it for an empty stub
    assert compress_js(src, 3) == (
        "export async function load(id, { cache = true } = {}) {/* ... */}\n"
        "const f = (x) => {/* ... */};\nconst g = (x) => ({ x });\nconst h = () => {};\n"
    )


def test_strings_and_templates_are_never_touched() -> None:
    src = "const s = `line one   \n\n\n// not a comment\n`;\nconst t = '/* no */';\n"
    assert compress_js(src, 2) == src


def test_crlf_and_blank_lines() -> None:
    assert compress_js("a();\r\n\r\n\r\n// c\r\nb();\r\n", 1) == "a();\r\n\r\nb();\r\n"


def test_unbalanced_input_raises() -> None:
    with pytest.raises(LexError):
        compress_js("function f() {\n", 1)


@pytest.mark.parametrize("level", [1, 2, 3])
@pytest.mark.parametrize("name", ["api.js", "ui.js"])
def test_starter_scripts_are_idempotent_and_shorter(name: str, level: int) -> None:
    src = (STARTER / "js" / name).read_text(encoding="utf-8")
    once = compress_js(src, level)
    assert len(once) < len(src)
    assert compress_js(once, level) == once


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
@pytest.mark.parametrize("level", [1, 2, 3])
def test_output_is_valid_javascript_for_node(tmp_path: Path, level: int) -> None:
    for script in (STARTER / "js").glob("*.js"):
        out = tmp_path / f"{script.stem}-{level}.mjs"
        out.write_text(compress_js(script.read_text(encoding="utf-8"), level), encoding="utf-8")
        result = subprocess.run(
            [str(NODE), "--check", str(out)], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


# ------------------------------------------------------------------ CSS and HTML


def test_css_levels() -> None:
    src = "/* theme */\n.a  >  .b ,\n.c {\n  color: red;\n  margin: 0 auto;\n}\n\n\n/*! keep */\n"
    assert (
        compress_css(src, 1)
        == ".a  >  .b ,\n.c {\n  color: red;\n  margin: 0 auto;\n}\n\n/*! keep */\n"
    )
    assert compress_css(src, 2) == ".a > .b,.c{color:red;margin:0 auto}\n/*! keep */\n"


def test_css_strings_and_urls_are_kept() -> None:
    src = 'a { content: "x  /* y */  z"; background: url(a//b.png); }\n'
    out = compress_css(src, 2)
    assert '"x  /* y */  z"' in out
    assert "url(a//b.png)" in out


def test_html_comments_scripts_styles_and_pre() -> None:
    src = (
        "<!-- remove me -->\n<!--[if IE]>keep<![endif]-->\n"
        "<pre>  keep   <!-- this -->  </pre>\n"
        '<script type="module">\n// js comment\nrun();\n</script>\n'
        '<script src="x.js"></script>\n'
        "<style>\n/* css */\nb { color: red; }\n</style>\n"
    )
    out = compress_html(src, 2)
    assert "remove me" not in out
    assert "<!--[if IE]>" in out
    assert "<pre>  keep   <!-- this -->  </pre>" in out
    assert "js comment" not in out
    assert "run();" in out
    assert '<script src="x.js"></script>' in out
    assert "css" not in out.replace("</style>", "")
    assert "b{color:red}" in out


def test_broken_inline_script_is_left_alone() -> None:
    src = '<script>\nconst s = "unterminated\n</script>\n'
    assert "unterminated" in compress_html(src, 1)


# ------------------------------------------------------------------ the mode


@pytest.mark.parametrize(
    ("path", "mode"),
    [("a.js", "web"), ("b.MJS", "web"), ("s.css", "web"), ("i.html", "web"), ("x.ts", "text")],
)
def test_auto_routes_web_files(path: str, mode: str) -> None:
    assert choose_mode(Document("x", path=path)) == mode


@pytest.mark.parametrize("level", [1, 2, 3])
def test_trace_ends_with_compress(level: int) -> None:
    doc = Document((STARTER / "js" / "ui.js").read_text(encoding="utf-8"), path="ui.js")
    steps = WebCompressor().trace(doc, level)
    assert [s.name for s in steps] == [STEP_NAMES[k] for k in range(1, level + 1)]
    assert steps[-1].after == WebCompressor().compress(doc, level)


def test_css_and_html_documents_use_their_rules() -> None:
    css = WebCompressor().compress(Document("a {\n  b: c;\n}\n", path="s.css"), 2)
    assert css.text == "a{b:c}\n"
    html = WebCompressor().compress(Document("<!-- x -->\n<p>hi</p>\n", path="i.html"), 1)
    assert html.text == "<p>hi</p>\n"


def test_css_skeleton_keeps_selectors_custom_properties_and_nesting() -> None:
    assert compress_css(".a  >  .b ,\n.c {\n  color: red;\n}\n/*! keep */\n", 3) == (
        ".a > .b,.c{}\n/*! keep */\n"
    )
    src = (
        "@import url(a.css);\n:root { --gap: 12px; color: red }\n"
        "@media (max-width: 600px) {\n  .a { color: red; --x: 1px; }\n}\n"
        "@keyframes k { from { opacity: 0 } to { opacity: 1 } }\n"
    )
    assert compress_css(src, 3) == (
        "@import url(a.css);\n:root{--gap:12px}\n@media (max-width:600px){.a{--x:1px}}\n"
        "@keyframes k{from{}to{}}\n"
    )
