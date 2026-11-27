"""shrink apply, and the --lazy / --dedent notes of the filter."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from shrink import cli as cli_module
from shrink.apply import answer_files, apply_answer, command, safe_target
from shrink.brief import BRIEF_NOTE, DEDENT_NOTE, LAZY_NOTE, answer_notes, notes_document
from shrink.cli import cli

FENCE = "`" * 3
MAIN = '''\
"""The application."""

from app.routers import health


def create_app():
    """Build the app."""
    app = make()
    app.include_router(health.router)
    return app


def build():
    return create_app()
'''

ANSWER = f"""\
Here you go.

## File: app/main.py
{FENCE}python
from app.routers import auth, health

...

def create_app():
 app = make()
 app.include_router(health.router)
 app.include_router(auth.router)
 return app

...
{FENCE}

## File: app/routers/auth.py
{FENCE}python
def login():
 return 1
{FENCE}
"""


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text(MAIN, encoding="utf-8")
    return tmp_path


def test_answer_files_and_safe_targets(tmp_path: Path) -> None:
    files = answer_files(ANSWER)
    assert [path for path, _ in files] == ["app/main.py", "app/routers/auth.py"]
    assert safe_target(tmp_path, "a/b.py") == (tmp_path / "a" / "b.py").resolve()
    for bad in ("", "/etc/passwd", "C:/x.py", "../x.py", "a/../../x.py"):
        assert safe_target(tmp_path, bad) is None


def test_apply_writes_merged_and_new_files(repo: Path) -> None:
    reports, errors = apply_answer(ANSWER, repo)
    assert errors == []
    assert [r.status for r in reports] == ["changed", "new"]
    main = (repo / "app" / "main.py").read_text(encoding="utf-8")
    assert '"""The application."""' in main
    assert "from app.routers import auth, health" in main
    assert "    app.include_router(auth.router)\n" in main  # re-indented to four spaces
    assert "def build():\n    return create_app()" in main
    assert (repo / "app" / "routers" / "auth.py").read_text(encoding="utf-8") == (
        "def login():\n    return 1\n"
    )


def test_dry_run_writes_nothing_and_diff_shows_changes(
    repo: Path, runner: CliRunner, tmp_path: Path
) -> None:
    answer = tmp_path / "answer.md"
    answer.write_text(ANSWER, encoding="utf-8")
    result = runner.invoke(command, [str(answer), "--root", str(repo), "--dry-run", "--diff"])
    assert result.exit_code == 0
    assert "+    app.include_router(auth.router)" in result.stdout
    assert "M app/main.py" in result.stderr
    assert "A app/routers/auth.py" in result.stderr
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == MAIN
    assert not (repo / "app" / "routers").exists()


def test_errors_skip_the_file_and_fail(repo: Path, runner: CliRunner) -> None:
    bad = f"## File: app/main.py\n{FENCE}python\ndef broken(:\n{FENCE}\n" + (
        f"## File: ../evil.py\n{FENCE}python\nx = 1\n{FENCE}\n"
    )
    result = runner.invoke(command, ["--root", str(repo)], input=bad)
    assert result.exit_code == 1
    assert "not valid Python" in result.stderr
    assert "outside the repository" in result.stderr
    assert (repo / "app" / "main.py").read_text(encoding="utf-8") == MAIN


def test_no_sections_and_stubs_left_warn(repo: Path, runner: CliRunner) -> None:
    result = runner.invoke(command, ["--root", str(repo)], input="just prose\n")
    assert result.exit_code == 0
    assert "no '## File: <path>' sections" in result.stderr
    stub = f"## File: app/new.py\n{FENCE}python\ndef todo():\n    ...\n{FENCE}\n"
    result = runner.invoke(command, ["--root", str(repo)], input=stub)
    assert "left as stubs" in result.stderr


def test_crlf_files_stay_crlf(repo: Path) -> None:
    target = repo / "app" / "main.py"
    target.write_bytes(MAIN.replace("\n", "\r\n").encode("utf-8"))
    apply_answer(ANSWER, repo)
    data = target.read_bytes()
    assert b"\r\n" in data and b"\n" not in data.replace(b"\r\n", b"")


def test_unchanged_file_is_not_rewritten(repo: Path) -> None:
    answer = f"## File: app/main.py\n{FENCE}python\n{MAIN}{FENCE}\n"
    reports, _ = apply_answer(answer, repo)
    assert reports[0].status == "unchanged"


def test_main_dispatches_apply(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    answer = repo / "a.md"
    answer.write_text(ANSWER, encoding="utf-8")
    assert cli_module.main(["apply", str(answer), "--root", str(repo), "--dry-run"]) == 0
    assert cli_module.main(["apply", "--no-such-option"]) == 2


def test_notes_order_and_documents() -> None:
    assert answer_notes() == ""
    assert notes_document() is None
    assert answer_notes(brief=True, lazy=True, dedent=True) == BRIEF_NOTE + DEDENT_NOTE + LAZY_NOTE
    brief = notes_document(brief=True)
    assert brief is not None and brief.path == "(brief note)"
    both = notes_document(lazy=True)
    assert both is not None and both.path == "(answer notes)"


PACKED = "app.py\n---\ndef f(x):\n    if x:\n        return 1\n    return 2\n---\n"


def test_filter_lazy_and_dedent(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--level", "1", "--dedent", "--lazy", "--stats"], input=PACKED)
    assert result.exit_code == 0
    assert "def f(x):\n if x:\n  return 1\n return 2\n" in result.stdout
    assert result.stdout.endswith(DEDENT_NOTE + LAZY_NOTE)
    assert "dedent" in result.stderr
    assert "answer notes" in result.stderr
