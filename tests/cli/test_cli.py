from __future__ import annotations

import errno
import sys

import pytest
from click.testing import CliRunner

from shrink import __version__
from shrink import cli as cli_module
from shrink.cli import cli, is_broken_pipe, main
from shrink.document import Document


def test_version(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"shrink {__version__}"


def test_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "--mode" in result.stdout
    assert "--level" in result.stdout


def test_list_modes(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--list-modes"])
    assert result.exit_code == 0
    assert [line.split()[0] for line in result.stdout.splitlines()] == [
        "auto",
        "code",
        "json",
        "log",
        "text",
        "web",
    ]


def test_mode_help(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--mode-help", "text"])
    assert result.exit_code == 0
    assert result.stdout.startswith("text - ")


@pytest.mark.parametrize(
    "args",
    [["--level", "0"], ["--level", "4"], ["--mode", "nope"], ["--no-such-flag"]],
)
def test_usage_errors_exit_2(runner: CliRunner, args: list[str]) -> None:
    result = runner.invoke(cli, args, input="x")
    assert result.exit_code == 2
    assert result.stdout == ""


def test_stdin_to_stdout(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--level", "1"], input="hello\n")
    assert result.exit_code == 0
    assert result.stdout == "hello\n"
    assert result.stderr == ""


def test_empty_input(runner: CliRunner) -> None:
    result = runner.invoke(cli, [], input="")
    assert result.exit_code == 0
    assert result.stdout == ""


def _patch_run(monkeypatch: pytest.MonkeyPatch, exc: BaseException) -> None:
    def boom(self: object, level: int) -> int:
        raise exc

    monkeypatch.setattr(cli_module.Pipeline, "run", boom)


def test_internal_error_is_one_line_without_traceback(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_run(monkeypatch, RuntimeError("kaput"))
    result = runner.invoke(cli, [], input="x")
    assert result.exit_code == 1
    assert result.stderr == "shrink: error: RuntimeError: kaput\n"


def test_debug_shows_traceback(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch, RuntimeError("kaput"))
    result = runner.invoke(cli, ["--debug"], input="x")
    assert result.exit_code == 1
    assert "Traceback" in result.stderr
    assert "shrink: error: RuntimeError: kaput" in result.stderr


def test_keyboard_interrupt_exits_130(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch, KeyboardInterrupt())
    result = runner.invoke(cli, [], input="x")
    assert result.exit_code == 130
    assert "Traceback" not in result.stderr


def test_broken_pipe_exits_quietly(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_run(monkeypatch, BrokenPipeError(errno.EPIPE, "Broken pipe"))
    result = runner.invoke(cli, [], input="x")
    assert result.exit_code == 0
    assert result.stderr == ""


def test_compressor_failure_passes_input_through(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken:
        name = "text"
        summary = ""
        help = ""

        def compress(self, doc: Document, level: int) -> Document:
            raise ValueError("cannot parse")

    monkeypatch.setattr(cli_module.default_registry, "create", lambda name: Broken())
    result = runner.invoke(cli, [], input="keep  me\n")
    assert result.exit_code == 0
    assert result.stdout == "keep  me\n"
    assert "shrink: warning: <stdin>: text mode failed" in result.stderr


def test_is_broken_pipe() -> None:
    assert is_broken_pipe(BrokenPipeError())
    assert not is_broken_pipe(ValueError())
    assert is_broken_pipe(OSError(errno.EINVAL, "x")) == (sys.platform == "win32")


class TestMain:
    def test_returns_zero_on_version(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--version"]) == 0
        assert capsys.readouterr().out.strip() == f"shrink {__version__}"

    def test_usage_error_returns_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--level", "9"]) == 2
        assert "Invalid value" in capsys.readouterr().err

    def test_interrupt_during_startup_returns_130(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def interrupted(*args: object, **kwargs: object) -> int:
            raise KeyboardInterrupt

        monkeypatch.setattr(cli, "main", interrupted)
        assert main([]) == 130

    def test_exit_code_from_command_is_returned(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "main", lambda *a, **k: 1)
        assert main([]) == 1


PACKED = "src/a.py\n---\nx = 1   \n\n\n\ny = 2\n\n---\nREADME.md\n---\nhello    world\n\n---\n"


def test_packed_input_is_split_and_the_framing_is_kept(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--level", "2"], input=PACKED)
    assert result.exit_code == 0
    assert (
        result.stdout
        == "src/a.py\n---\nx = 1\n\ny = 2\n\n---\nREADME.md\n---\nhello world\n\n---\n"
    )


def test_format_none_treats_packed_input_as_one_text(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--level", "1", "--format", "none"], input=PACKED)
    assert result.stdout.startswith("src/a.py\n---\nx = 1\n\ny = 2\n")
    assert "hello    world" in result.stdout


def test_source_files_in_a_packed_stream_only_get_level_1(runner: CliRunner) -> None:
    paragraph = "same paragraph here, long enough"
    packed = (
        "a.py\n---\ndef f():\n    pass\n\n\ndef g():\n    pass\n\n---\n"
        f"b.md\n---\n{paragraph}\n\n{paragraph}\n\n---\n"
    )
    result = runner.invoke(cli, ["--mode", "text", "--level", "3"], input=packed)
    assert "def f():\n    pass\n\ndef g():\n    pass\n" in result.stdout
    assert result.stdout.count(paragraph) == 1


def test_delimiter_option(runner: CliRunner) -> None:
    result = runner.invoke(
        cli, ["--level", "1", "--delimiter", r"^== (?P<path>\S+) ==$"], input="== a.txt ==\nx   \n"
    )
    assert result.stdout == "== a.txt ==\nx\n"


def test_invalid_delimiter_is_a_usage_error(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--delimiter", "("], input="x")
    assert result.exit_code == 2
    assert "regular expression" in result.stderr


def test_stats_include_the_framing(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--level", "1", "--stats"], input=PACKED)
    assert "2 documents" in result.stderr


def test_mode_chain_runs_each_mode(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--mode", "text,text", "--level", "2"], input="a   b\n")
    assert result.stdout == "a b\n"


def test_mode_chain_rejects_an_unknown_member(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--mode", "text,nope"], input="x")
    assert result.exit_code == 2
    assert "'nope' is not one of" in result.stderr


def test_brief_adds_the_note_after_the_output(runner: CliRunner) -> None:
    from shrink.brief import BRIEF_NOTE

    result = runner.invoke(cli, ["--level", "1", "--brief"], input=PACKED)
    assert result.stdout.endswith("---\n" + BRIEF_NOTE)
    assert result.stdout.count(BRIEF_NOTE) == 1


def test_brief_cost_is_in_the_stats(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--level", "1", "--brief", "--stats"], input=PACKED)
    assert "2 documents" in result.stderr
    assert "brief note" in result.stderr
    assert "(new," in result.stderr


def test_brief_without_input_writes_nothing(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--brief"], input="")
    assert result.stdout == ""
