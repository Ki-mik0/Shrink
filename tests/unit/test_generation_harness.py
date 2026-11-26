"""The generation eval script (no network: the model is replaced by a fake)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

import run_generation as gen

FENCE = "`" * 3

COMPLETE_ANSWER = "\n".join(
    [
        "## File: backend/app/routers/auth.py",
        f"{FENCE}python",
        "from app.security import hash_password, verify_password, get_current_user",
        "@router.post('/register')",
        "def register(): ...",
        "@router.post('/login')",
        "def login(): ...",
        "@router.get('/me')",
        "def me(): ...",
        FENCE,
        "",
        "## File: backend/app/routers/todos.py",
        f"{FENCE}python",
        "from app.pagination import paginate",
        "from app.security import get_current_user",
        "@router.patch('/{todo_id}')",
        "def update(): ...",
        "@router.delete('/{todo_id}')",
        "def delete(): ...",
        FENCE,
        "",
        "## File: backend/app/db.py",
        f"{FENCE}python",
        "MIGRATIONS = ['CREATE TABLE todos (id INTEGER)']",
        FENCE,
        "",
        "## File: backend/app/main.py",
        f"{FENCE}python",
        "app.include_router(auth.router)",
        "app.include_router(todos.router)",
        FENCE,
        "",
        "## File: frontend/login.html",
        f"{FENCE}html",
        "<form></form>",
        FENCE,
        "## File: frontend/js/login.js",
        f"{FENCE}javascript",
        "import { apiFetch } from './api.js';",
        FENCE,
        "## File: frontend/todos.html",
        f"{FENCE}html",
        "<ul></ul>",
        FENCE,
        "## File: frontend/js/todos.js",
        f"{FENCE}javascript",
        "import { apiFetch } from './api.js';",
        FENCE,
        "",
    ]
)


def test_starter_project_is_valid_python() -> None:
    import ast

    files = dict(gen.starter_files())
    assert "backend/app/security.py" in files
    for path, text in files.items():
        if path.endswith(".py"):
            ast.parse(text)


def test_contexts_shrink_with_the_level() -> None:
    sizes = {v: len(gen.build_context(v)) for v in ("original", "L1", "L2", "L3")}
    assert sizes["original"] >= sizes["L1"] > sizes["L2"] > sizes["L3"]
    assert "Task:" in gen.build_prompt("x")


def test_parse_answer_reads_files_and_rejects_unsafe_paths() -> None:
    text = COMPLETE_ANSWER + "\n".join(
        [
            "## File: ../outside.py",
            f"{FENCE}python",
            "x = 1",
            FENCE,
            "## File: C:/Windows/evil.py",
            f"{FENCE}python",
            "x = 1",
            FENCE,
            "## File: README.md",
            f"{FENCE}md",
            "notes",
            FENCE,
            "",
        ]
    )
    answer = gen.parse_answer(text)
    assert "backend/app/routers/auth.py" in answer.files
    assert answer.files["frontend/todos.html"] == "<ul></ul>\n"
    assert sorted(answer.rejected) == ["../outside.py", "C:/Windows/evil.py", "README.md"]


@pytest.mark.parametrize(
    ("path", "ok"),
    [
        ("backend/app/x.py", True),
        ("frontend/js/a.js", True),
        ("/etc/passwd", False),
        ("backend/../../x", False),
        ("D:\\x.py", False),
        ("docs/x.md", False),
        ("", False),
    ],
)
def test_safe_path(path: str, ok: bool) -> None:
    assert gen.safe_path(path) is ok


def test_completeness_of_a_full_and_an_empty_answer() -> None:
    full = gen.completeness(gen.parse_answer(COMPLETE_ANSWER))
    assert len(full["checks_passed"]) == full["checks_total"]
    assert full["python_valid"] == full["python_files"] == 4
    empty = gen.completeness(gen.parse_answer("no files here"))
    assert empty["checks_passed"] == []
    assert empty["files"] == 0


def test_run_saves_answers_and_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen, "RESULTS", tmp_path)
    monkeypatch.setattr(gen.rq, "PAUSE", 0)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    calls: list[dict[str, Any]] = []

    def fake_ask(model: str, prompt: str, key: str, **kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        reason = "length" if len(calls) == 7 else "stop"
        return {
            "choices": [{"message": {"content": COMPLETE_ANSWER}, "finish_reason": reason}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 900},
        }

    monkeypatch.setattr(gen.rq, "ask", fake_ask)
    gen.run("fake/model", 1, dry_run=False)
    assert len(calls) == 7
    assert calls[0]["max_tokens"] == gen.MAX_TOKENS
    assert calls[0]["system"] == gen.SYSTEM
    gen.run("fake/model", 1, dry_run=False)
    assert len(calls) == 7  # saved answers are not asked again
    table = (tmp_path / "generation.md").read_text(encoding="utf-8")
    assert "| original |" in table
    assert "| L3+query |" in table
    assert "| L3+brief |" in table
    assert "stale" not in table
    assert "1/1" in table  # the last variant was cut off
    record = json.loads(next((tmp_path / "raw-generation").rglob("L3--1.json")).read_text("utf-8"))
    assert record["usage"]["completion_tokens"] == 900


def test_redo_moves_answers_aside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen, "RESULTS", tmp_path)
    path = gen.raw_path("m/x", "L3", 1)
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    gen.archive("m/x", "L3")
    assert not path.exists()
    assert (path.parent / "previous" / "L3" / "L3--1.json").is_file()


def test_dry_run_needs_no_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    gen.run("fake/model", 1, dry_run=True)
    out = capsys.readouterr().out
    assert "original" in out
    assert "7 request(s)" in out


def test_progress_is_reported_while_the_answer_streams(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    lines = [
        b'data: {"choices": [{"delta": {"content": "abc"}}]}\n',
        b'data: {"choices": [{"delta": {"content": "de"}, "finish_reason": "stop"}]}\n',
    ]
    seen: list[int] = []
    reply = gen.rq.read_stream(lines, seen.append)
    assert seen == [3, 5]
    assert reply["choices"][0]["finish_reason"] == "stop"

    clock = iter([10.0, 10.2, 11.5])
    monkeypatch.setattr(gen.time, "perf_counter", lambda: next(clock))
    progress = gen.Progress(started=9.0)
    progress(100)  # shown
    progress(200)  # less than a second later: skipped
    progress(300)  # shown
    out = capsys.readouterr().out
    assert "100 characters" in out
    assert "200 characters" not in out
    assert "300 characters" in out


def test_brief_variant_adds_the_note_to_the_level_3_context() -> None:
    from shrink.brief import BRIEF_NOTE

    assert gen.build_context("L3+brief") == gen.build_context("L3") + BRIEF_NOTE


def test_answer_for_an_older_prompt_is_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gen, "RESULTS", tmp_path)
    monkeypatch.setattr(gen.rq, "PAUSE", 0)
    monkeypatch.setattr(gen, "VARIANTS", ["original", "L3"])
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    answer = {
        "choices": [{"message": {"content": COMPLETE_ANSWER}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 900},
    }
    asked: list[str] = []

    def fake_ask(model: str, prompt: str, key: str, **kwargs: Any) -> dict[str, Any]:
        asked.append(prompt)
        return answer

    monkeypatch.setattr(gen.rq, "ask", fake_ask)
    gen.run("m/x", 1, dry_run=False)
    old = gen.raw_path("m/x", "L3", 1)
    record = json.loads(old.read_text(encoding="utf-8"))
    record["prompt_sha256"] = "an older prompt"
    old.write_text(json.dumps(record), encoding="utf-8")
    gen.report("m/x")
    assert "| L3 (stale) |" in (tmp_path / "generation.md").read_text(encoding="utf-8")
    gen.run("m/x", 1, dry_run=False)
    assert len(asked) == 3  # only L3 again
    assert (old.parent / "previous" / "L3" / "L3--1.json").is_file()
    assert "stale" not in (tmp_path / "generation.md").read_text(encoding="utf-8")


def test_variants_limits_the_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    gen.run("fake/model", 3, dry_run=True, only=["original", "L3+brief"])
    out = capsys.readouterr().out
    assert "L3+brief" in out
    assert "L2 " not in out
    assert "6 request(s) for 3 run(s)" in out


def test_no_docs_variant_has_no_docstrings_and_the_note() -> None:
    from shrink.brief import BRIEF_NOTE

    context = gen.build_context("L3+no-docs+brief")
    assert context.endswith(BRIEF_NOTE)
    assert "Open a connection configured" not in context
    assert "/**" not in context
    assert len(context) < len(gen.build_context("L3+brief"))


def test_edits_are_applied_to_the_real_starter_files() -> None:
    answer = (
        "## File: backend/app/routers/auth.py\n```python\nrouter = 1\n```\n\n"
        "## Edit: backend/app/main.py\n```\n<<<<<<< SEARCH\nfrom app.routers import health\n"
        "=======\nfrom app.routers import auth, health\n>>>>>>> REPLACE\n"
        "<<<<<<< SEARCH\napp.include_router(health.router)\n=======\n"
        "app.include_router(health.router)\napp.include_router(auth.router)\n"
        ">>>>>>> REPLACE\n```\n\n"
        "## Edit: backend/app/db.py\n```\n<<<<<<< SEARCH\nno such line\n=======\nx\n"
        ">>>>>>> REPLACE\n```\n"
    )
    parsed = gen.parse_answer(answer)
    main = parsed.files["backend/app/main.py"]
    assert "from app.routers import auth, health" in main
    assert "    app.include_router(auth.router)" in main  # indented like the real line
    assert "def create_app(" in main  # the rest of the real file is there
    assert "backend/app/db.py" not in parsed.files
    assert (parsed.edits, parsed.edits_failed) == (3, 1)
    assert gen.completeness(parsed)["python_valid"] == 2
