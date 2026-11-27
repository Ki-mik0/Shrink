"""shrink apply's merge: shortened answers are filled in from the original file."""

from __future__ import annotations

import ast

import pytest

from shrink import indent
from shrink.compressors.auto import AutoCompressor
from shrink.document import Document
from shrink.restore import Report, RestoreError, restore

DB = '''\
"""Database helpers.

Longer module documentation that level 2 drops.
"""

from __future__ import annotations

import sqlite3

# Every schema change is appended here.
MIGRATIONS: list[str] = [
    "CREATE TABLE users (id INTEGER)",
]


def connect(path: str) -> sqlite3.Connection:
    """Open a connection.

    With foreign keys on.
    """
    conn = sqlite3.connect(path)  # one per request
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


class Store:
    """Owns one database."""

    def __init__(self, path: str) -> None:
        self.path = path

    def open(self) -> sqlite3.Connection:
        """A new connection."""
        return connect(self.path)


def audit(conn: sqlite3.Connection, event: str) -> None:
    conn.execute("INSERT INTO audit VALUES (?)", (event,))
'''


def view(text: str, path: str, level: int = 3) -> str:
    return AutoCompressor().compress(Document(text, path=path), level).text


def test_every_level_of_the_view_restores_exactly() -> None:
    for level in (1, 2, 3):
        result = restore(DB, view(DB, "db.py", level), "db.py")
        assert result.text == DB
        assert result.report.status == "unchanged"


def test_partial_answer_replaces_one_item_and_keeps_the_rest() -> None:
    answer = (
        "...\n\nMIGRATIONS: list[str] = [\n"
        '    "CREATE TABLE users (id INTEGER)",\n'
        '    "CREATE TABLE todos (id INTEGER)",\n]\n\n...\n'
    )
    result = restore(DB, answer, "db.py")
    assert result.text == DB.replace(
        '"CREATE TABLE users (id INTEGER)",\n',
        '"CREATE TABLE users (id INTEGER)",\n    "CREATE TABLE todos (id INTEGER)",\n',
    )
    assert result.report.replaced == 1
    assert result.report.kept > 5
    assert result.report.status == "changed"


def test_new_items_go_next_to_their_neighbours() -> None:
    answer = (
        "import json\n\n...\n\n"
        "def helper() -> int:\n    return 1\n\n"  # after a gap, before the next match
        "def audit(conn: sqlite3.Connection, event: str) -> None:\n    ...\n\n"
        "def last() -> None:\n    pass\n"  # right after a matched item
    )
    result = restore(DB, answer, "db.py")
    text = result.text
    assert text.index("import sqlite3") < text.index("import json") < text.index("# Every")
    assert text.index("class Store") < text.index("def helper") < text.index("def audit")
    assert text.index('VALUES (?)", (event,))') < text.index("def last")
    assert result.report.inserted == 3
    ast.parse(text)


def test_new_item_after_a_leading_gap_goes_to_the_end() -> None:
    result = restore(DB, "...\n\ndef extra() -> None:\n    pass\n", "db.py")
    assert result.text.endswith("def extra() -> None:\n    pass\n")
    assert result.text.startswith(DB.rstrip("\n"))


def test_class_with_a_gap_gets_a_new_method() -> None:
    answer = (
        "...\n\nclass Store:\n    ...\n\n"
        "    def close(self) -> None:\n        self.path = ''\n\n...\n"
    )
    text = restore(DB, answer, "db.py").text
    assert '    """Owns one database."""' in text
    assert text.index("def open") < text.index("def close") < text.index("def audit")
    ast.parse(text)


def test_class_members_move_to_the_answers_indentation() -> None:
    answer = "...\n\nclass Store:\n  ...\n\n  def close(self) -> None:\n    pass\n\n...\n"
    text = restore(DB, answer, "db.py").text
    assert "\n  def open(self) -> sqlite3.Connection:\n" in text
    assert "\n  def close(self) -> None:\n" in text
    ast.parse(text)


def test_whole_answer_with_a_changed_function_and_stubs() -> None:
    answer = view(DB, "db.py").replace(
        "def audit(conn: sqlite3.Connection, event: str) -> None:\n    ...",
        "def audit(conn: sqlite3.Connection, event: str) -> None:\n    print(event)",
    )
    result = restore(DB, answer, "db.py")
    assert "    print(event)\n" in result.text
    assert "Longer module documentation" in result.text  # docstring taken back
    assert "# one per request" in result.text  # comment inside an unchanged function
    assert result.report.replaced == 1


def test_whole_answer_drops_what_it_leaves_out() -> None:
    answer = view(DB, "db.py").split("\n\nclass Store")[0] + "\n"
    text = restore(DB, answer, "db.py").text
    assert "class Store" not in text and "def audit" not in text


def test_stub_with_a_new_signature_gets_the_old_body() -> None:
    answer = view(DB, "db.py").replace(
        "def connect(path: str) -> sqlite3.Connection:",
        "def connect(path: str, timeout: float = 5.0) -> sqlite3.Connection:",
    )
    result = restore(DB, answer, "db.py")
    assert "timeout: float = 5.0" in result.text
    assert 'conn.execute("PRAGMA foreign_keys = ON")' in result.text
    assert result.report.bodies >= 1


def test_one_line_stub_becomes_a_block() -> None:
    answer = "...\n\ndef audit(conn: sqlite3.Connection, event: str, extra: int = 0) -> None: ...\n"
    text = restore(DB, answer, "db.py").text
    assert (
        "def audit(conn: sqlite3.Connection, event: str, extra: int = 0) -> None:\n"
        '    conn.execute("INSERT INTO audit VALUES (?)", (event,))\n'
    ) in text


def test_stub_without_an_original_is_reported() -> None:
    result = restore(None, "def todo() -> None:\n    ...\n", "new.py")
    assert result.report.status == "new"
    assert result.report.unresolved == ["todo"]
    assert "left as stubs: todo" in result.report.describe()


def test_compact_answer_is_reindented_like_the_file() -> None:
    compact = indent.compact_python(view(DB, "db.py"))
    assert restore(DB, compact, "db.py").text == DB
    two = "if x:\n  y = 1\n"
    result = restore(two, "if x:\n y = 2\n", "two.py")
    assert result.text == "if x:\n  y = 2\n"


def test_answers_that_do_not_parse() -> None:
    with pytest.raises(RestoreError, match="not valid Python"):
        restore(DB, "def broken(:\n", "db.py")
    # a broken original is not used, the answer still is
    assert restore("def (:\n", "x = 1\n", "a.py").text == "x = 1\n"


JS = """\
// Talking to the API.
import { el } from "./ui.js";

/** Key of the token. */
export const TOKEN_KEY = "t";

/** Store the token. */
export function saveToken(token) {
  localStorage.setItem(TOKEN_KEY, token);
}

export const load = () => {
  return localStorage.getItem(TOKEN_KEY);
};

const handlers = {
  click: function () {
    el("b");
  },
};
"""


def test_js_view_restores_exactly() -> None:
    for level in (1, 2, 3):
        assert restore(JS, view(JS, "api.js", level), "api.js").text == JS


def test_js_stubs_are_filled_by_name() -> None:
    answer = view(JS, "api.js").replace(
        "export function saveToken(token)", "export function saveToken(token, x)"
    )
    result = restore(JS, answer, "api.js")
    assert "export function saveToken(token, x) {\n  localStorage.setItem" in result.text
    assert result.report.bodies >= 1


def test_js_partial_answer_adds_an_export() -> None:
    answer = (
        'import { el, clear } from "./ui.js";\n/* ... */\n'
        "export function clearToken() {\n  localStorage.removeItem(TOKEN_KEY);\n}\n"
    )
    text = restore(JS, answer, "api.js").text
    assert 'import { el, clear } from "./ui.js";' in text
    assert "// Talking to the API." in text
    assert text.rstrip().endswith("localStorage.removeItem(TOKEN_KEY);\n}")
    assert "export const load" in text


def test_js_unknown_stub_and_unreadable_answer() -> None:
    result = restore(JS, "function nothing() {/* ... */}\n", "api.js")
    assert result.report.unresolved == ["nothing"]
    with pytest.raises(RestoreError):
        restore(JS, "const s = 'unterminated\n", "api.js")


HTML = """\
<!doctype html>
<html lang="en">
<head>
  <title>Board</title>
</head>
<body>
  <!-- the shell -->
  <header class="topbar">
    <nav id="nav" class="nav"></nav>
  </header>
  <main id="app"><p>Welcome</p></main>
  <script type="module">
    function show() {
      document.title = "x";
    }
    show();
  </script>
</body>
</html>
"""


def test_html_fragment_replaces_by_id_and_adds_a_script() -> None:
    fragment = (
        '<nav id="nav" class="nav">\n<a href="login.html">Sign in</a>\n</nav>\n'
        "<!-- ... -->\n"
        '<script type="module" src="js/nav.js"></script>\n'
        '<link rel="icon" href="i.png">\n'
    )
    result = restore(HTML, fragment, "index.html")
    text = result.text
    assert (
        '    <nav id="nav" class="nav">\n      <a href="login.html">Sign in</a>\n    </nav>' in text
    )
    assert '  <script type="module" src="js/nav.js"></script>\n</body>' in text
    assert '  <link rel="icon" href="i.png">\n</head>' in text
    assert "<!-- the shell -->" in text and "<p>Welcome</p>" in text
    assert result.report.replaced == 1 and result.report.inserted == 2


def test_html_unique_tags_and_outer_replacement_wins() -> None:
    fragment = (
        "<title>Todos</title>\n"
        '<header class="topbar"><nav id="nav">A</nav></header>\n'
        '<nav id="nav">B</nav>\n'
        "<section>new</section>\n"
    )
    text = restore(HTML, fragment, "index.html").text
    assert "<title>Todos</title>" in text
    assert '<header class="topbar"><nav id="nav">A</nav></header>' in text
    assert ">B<" not in text
    assert "</header>\n  <section>new</section>" in text


def test_html_whole_file_fills_inline_script_stubs() -> None:
    whole = view(HTML, "index.html").replace("<title>Board</title>", "<title>New</title>")
    assert "{/* ... */}" in whole
    text = restore(HTML, whole, "index.html").text
    assert "<title>New</title>" in text
    assert 'document.title = "x";' in text


def test_html_view_restores_exactly() -> None:
    assert restore(HTML, view(HTML, "index.html"), "index.html").text == HTML


CSS = """\
/* Styles. */
:root {
  --gap: 12px;
  font-family: sans-serif;
}

.card {
  padding: var(--gap);
}
"""


def test_css_view_restores_and_empty_rules_are_stubs() -> None:
    assert restore(CSS, view(CSS, "s.css"), "s.css").text == CSS
    result = restore(CSS, ":root{--gap:12px}\n.card{}\n.new{color:red}\n", "s.css")
    assert "padding: var(--gap);" in result.text
    assert ".new{color:red}" in result.text


def test_css_partial_answer() -> None:
    text = restore(CSS, "/* ... */\n.card {\n  padding: 0;\n}\n", "s.css").text
    assert "padding: 0;" in text and "--gap: 12px;" in text and "/* Styles. */" in text


def test_other_files_are_taken_as_they_are() -> None:
    result = restore("old\n", "new\n", "notes.txt")
    assert result.text == "new\n"
    assert result.report.status == "changed"
    assert Report("x").describe() == ""
