from __future__ import annotations

import pytest
from click.testing import CliRunner

from shrink.cli import cli
from shrink.compressors.auto import AutoCompressor
from shrink.compressors.python_code import functions, skeletonize
from shrink.document import Document
from shrink.pipeline import Pipeline
from shrink.retrieval.bm25 import BM25, tokenize
from shrink.retrieval.planner import Planner

AUTH = '''\
def login(user: str, password: str) -> bool:
    """Check the password and open a session."""
    session = open_session(user)
    return verify_password(user, password) and session.ok


def render_page(name: str) -> str:
    template = load_template(name)
    return template.render()
'''

NOTES = "Deployment notes.\n\nUse docker compose to start the stack.\n"


def compress_with_auto(doc: Document, level: int) -> Document:
    return AutoCompressor().compress(doc, level)


# ------------------------------------------------------------------ bm25


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("getUserName", ["get", "user", "name"]),
        ("user_id = 1", ["user", "id", "1"]),
        ("HTTPServer started", ["http", "server", "started"]),
        ("Як працює підрахунок?", ["як", "працює", "підрахунок"]),
        ("the def of a self", []),
    ],
)
def test_tokenize(text: str, tokens: list[str]) -> None:
    assert tokenize(text) == tokens


def test_bm25_ranks_the_matching_chunk_first() -> None:
    chunks = ["password check and session", "template rendering", "docker deployment"]
    scores = BM25(chunks).scores("check the password")
    assert scores[0] > 0
    assert scores[1] == scores[2] == 0


def test_bm25_rare_terms_weigh_more() -> None:
    chunks = ["user login", "user profile", "user settings"]
    scores = BM25(chunks).scores("user login")
    assert scores[0] > scores[1] == scores[2] > 0


def test_bm25_of_nothing() -> None:
    assert BM25([]).scores("x") == []


# ------------------------------------------------------------------ skeleton with keep


def test_functions_lists_qualified_names() -> None:
    code = (
        "def a():\n    pass\n\nclass K:\n    def m(self):\n        def inner():\n            pass\n"
    )
    assert [name for name, _ in functions(code)] == ["a", "K.m"]


def test_skeleton_keeps_named_functions_whole() -> None:
    out = skeletonize(AUTH, frozenset({"login"}))
    assert "verify_password" in out
    assert "load_template" not in out
    assert "def render_page(name: str) -> str:\n    ..." in out


def test_skeleton_keeps_methods_by_class_name() -> None:
    code = (
        "class A:\n    def keep(self):\n        return 1\n\n    def drop(self):\n        return 2\n"
    )
    out = skeletonize(code, frozenset({"A.keep"}))
    assert "return 1" in out
    assert "return 2" not in out


# ------------------------------------------------------------------ planner: query


def test_query_keeps_relevant_functions_and_lightens_relevant_text() -> None:
    docs = [
        Document(AUTH, path="auth.py"),
        Document(NOTES, path="notes.md"),
        Document("Unrelated   text here.\n", path="other.md"),
    ]
    said: list[str] = []
    plans = Planner(query="password session", report=said.append).plan(docs, 3, compress_with_auto)
    assert plans[0].doc.keep == frozenset({"login"})
    assert plans[0].level == 3
    assert plans[0].output is not None
    assert "verify_password" in plans[0].output.text
    assert "load_template" not in plans[0].output.text
    assert plans[1].level == 3  # notes do not mention it
    assert said[0].startswith("query: 1 of 3 file(s) relevant; kept: auth.py:login")


def test_query_lightens_a_relevant_text_document() -> None:
    docs = [Document(AUTH, path="auth.py"), Document(NOTES, path="notes.md")]
    plans = Planner(query="docker compose").plan(docs, 3, compress_with_auto)
    assert plans[1].level == 1
    assert plans[1].relevance == 1.0
    assert plans[0].doc.keep == frozenset()


def test_query_without_matches_changes_nothing() -> None:
    said: list[str] = []
    plans = Planner(query="kubernetes", report=said.append).plan(
        [Document(NOTES, path="notes.md")], 2, compress_with_auto
    )
    assert plans[0].level == 2
    assert said == ["query: nothing in 1 file(s) matches; compressing as usual"]


def test_query_on_empty_input() -> None:
    assert Planner(query="x").plan([], 2, compress_with_auto) == []


def test_unparseable_python_is_ranked_by_paragraphs() -> None:
    docs = [Document("def broken(:\n\npassword here\n", path="bad.py")]
    plans = Planner(query="password").plan(docs, 2, lambda d, _: d)
    assert plans[0].relevance == 1.0


# ------------------------------------------------------------------ planner: budget


def fake_compress(doc: Document, level: int) -> Document:
    """Each level halves the text; kept functions add 100 characters at level 3."""
    size = len(doc.text) // (2 ** (level - 1))
    if level == 3 and doc.keep:
        size += 100
    return doc.with_text("x" * size)


def test_budget_tightens_the_least_relevant_document_first() -> None:
    docs = [Document("a" * 400, path="a.md"), Document("b" * 400, path="b.md")]
    planner = Planner(budget=500, count=len)
    plans = planner.plan(docs, 1, fake_compress)
    assert sum(p.tokens for p in plans) <= 500
    assert [p.level for p in plans] == [2, 2]


def test_budget_uses_relevance_order() -> None:
    docs = [Document("relevant words " * 20, path="a.md"), Document("other " * 60, path="b.md")]
    plans = Planner(query="relevant words", budget=500, count=len).plan(docs, 1, fake_compress)
    assert plans[0].level == 1
    assert plans[1].level > 1


def test_budget_drops_kept_functions_only_at_the_last_step() -> None:
    doc = Document("x" * 400, path="a.py", keep=frozenset({"f"}))
    plans = Planner(budget=150, count=len).plan([doc], 1, fake_compress)
    assert plans[0].level == 3
    assert plans[0].doc.keep == frozenset()
    assert plans[0].tokens == 100


def test_unreachable_budget_warns_and_cuts_nothing() -> None:
    warnings: list[str] = []
    planner = Planner(budget=10, count=len, warn=warnings.append)
    plans = planner.plan([Document("y" * 400, path="a.md")], 1, fake_compress)
    assert plans[0].tokens == 100
    assert warnings == [
        "budget of 10 tokens not reached: 100 tokens at the strongest compression; "
        "nothing was cut off"
    ]


def test_budget_already_met_takes_no_step() -> None:
    said: list[str] = []
    plans = Planner(budget=1000, count=len, report=said.append).plan(
        [Document("z" * 100)], 2, fake_compress
    )
    assert plans[0].level == 2
    assert said == ["budget: 50 of 1000 tokens after 0 step(s)"]


def test_budget_needs_a_counter() -> None:
    with pytest.raises(ValueError, match="counter"):
        Planner(budget=10)


# ------------------------------------------------------------------ pipeline and CLI


class ListReader:
    def __init__(self, text: str) -> None:
        self._text = text

    def lines(self):  # type: ignore[no-untyped-def]
        yield from self._text.splitlines(keepends=True)


class ListWriter:
    def __init__(self) -> None:
        self.docs: list[Document] = []

    def write(self, doc: Document) -> None:
        self.docs.append(doc)


def test_pipeline_with_a_planner_writes_in_input_order() -> None:
    from shrink.splitting.factory import make_splitter

    packed = f"auth.py\n---\n{AUTH}\n---\nnotes.md\n---\n{NOTES}\n---\n"
    writer = ListWriter()
    Pipeline(
        ListReader(packed),
        make_splitter(),
        AutoCompressor(),
        writer,
        planner=Planner(query="password"),
    ).run(3)
    assert [d.path for d in writer.docs] == ["auth.py", "notes.md"]
    assert "verify_password" in writer.docs[0].text


def test_failed_document_warns_once_although_tried_at_several_levels() -> None:
    class Broken:
        name = "broken"
        summary = ""
        help = ""

        def compress(self, doc: Document, level: int) -> Document:
            raise ValueError("nope")

    warnings: list[str] = []
    Pipeline(
        ListReader("text\n"),
        make_whole(),
        Broken(),
        ListWriter(),
        planner=Planner(budget=1, count=len, warn=lambda _: None),
        warn=warnings.append,
    ).run(1)
    assert len(warnings) == 1


def make_whole():  # type: ignore[no-untyped-def]
    from shrink.splitting.documents import WholeStreamSplitter

    return WholeStreamSplitter()


def test_cli_query_keeps_the_matching_function(runner: CliRunner) -> None:
    packed = f"auth.py\n---\n{AUTH}\n---\n"
    result = runner.invoke(cli, ["-l", "3", "--query", "password", "--stats"], input=packed)
    assert result.exit_code == 0
    assert "verify_password" in result.stdout
    assert "load_template" not in result.stdout
    assert "shrink: query: 1 of 1 file(s) relevant" in result.stderr


def test_cli_budget_reports_and_warns(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["-l", "1", "--budget", "1"], input=AUTH)
    assert result.exit_code == 0
    assert "budget of 1 tokens not reached" in result.stderr
    assert "def login" in result.stdout


def test_cli_budget_must_be_positive(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--budget", "0"], input="x")
    assert result.exit_code == 2
