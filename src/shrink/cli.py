"""Command-line interface.

Exit codes: 0 ok (also when the reader closed the pipe), 1 internal error,
2 usage error, 130 interrupted (Ctrl+C).
"""

from __future__ import annotations

import contextlib
import errno
import os
import re
import sys

import click

from shrink import __version__, messages
from shrink.compressors.base import DEFAULT_LEVEL, LEVEL_NAMES, MAX_LEVEL, MIN_LEVEL
from shrink.compressors.registry import default_registry
from shrink.io.reader import StdinReader
from shrink.io.writer import StdoutWriter
from shrink.pipeline import Pipeline, Reporter
from shrink.splitting.factory import AUTO, FORMAT_NAMES, make_splitter

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

DEFAULT_MODE = "text"

_LEVEL_HELP = ", ".join(f"{n} {name}" for n, name in LEVEL_NAMES.items())


def _parse_modes(_ctx: click.Context, _param: click.Parameter, value: str) -> list[str]:
    """``text`` or ``a,b``: one or more known modes, in the order they run."""
    names = [part.strip() for part in value.split(",")]
    known = default_registry.names()
    for name in names:
        if name not in known:
            raise click.BadParameter(f"{name!r} is not one of {', '.join(known)}.")
    return names


def _list_modes(ctx: click.Context, _param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    for spec in default_registry.specs():
        click.echo(f"{spec.name:<8} {spec.summary}")
    ctx.exit(EXIT_OK)


def _mode_help(ctx: click.Context, _param: click.Parameter, value: str | None) -> None:
    if value is None or ctx.resilient_parsing:
        return
    compressor = default_registry.create(value)
    click.echo(f"{compressor.name} - {compressor.summary}\n\n{compressor.help.strip()}")
    ctx.exit(EXIT_OK)


def is_broken_pipe(exc: BaseException) -> bool:
    """The reader went away (``| head``). Windows reports it as EINVAL."""
    if isinstance(exc, BrokenPipeError):
        return True
    return sys.platform == "win32" and isinstance(exc, OSError) and exc.errno == errno.EINVAL


def silence_stdout() -> None:
    """Point fd 1 at devnull so the interpreter's final flush cannot fail again."""
    with contextlib.suppress(OSError, ValueError, AttributeError):
        devnull = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull, sys.stdout.fileno())
        finally:
            os.close(devnull)


def _check_delimiter(_ctx: click.Context, _param: click.Parameter, value: str | None) -> str | None:
    if value is not None:
        try:
            re.compile(value)
        except re.error as exc:
            raise click.BadParameter(f"not a valid regular expression: {exc}") from exc
    return value


def _build_reporter(stats: bool, diff: bool) -> Reporter | None:
    if not (stats or diff):
        return None
    from shrink.observability.stats import StatsReporter, make_counter

    reporter = StatsReporter(make_counter())
    if not diff:
        return reporter
    from shrink.observability.diff import DiffReporter

    return DiffReporter(reporter)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "-m",
    "--mode",
    metavar="MODE[,MODE...]",
    callback=_parse_modes,
    default=DEFAULT_MODE,
    show_default=True,
    help="Compression mode; several modes separated by commas run one after another.",
)
@click.option(
    "-l",
    "--level",
    type=click.IntRange(MIN_LEVEL, MAX_LEVEL),
    default=DEFAULT_LEVEL,
    show_default=True,
    help=f"Compression level: {_LEVEL_HELP}.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(FORMAT_NAMES),
    default=AUTO,
    show_default=True,
    help="Packed-input format to split into files: auto-detect, none, files-to-prompt, "
    "cxml, repomix-xml, repomix-markdown or repomix-plain.",
)
@click.option(
    "--delimiter",
    metavar="REGEX",
    callback=_check_delimiter,
    help="Start a new file at every line matching REGEX (named group 'path' = file name). "
    "Overrides --format.",
)
@click.option(
    "--stats",
    is_flag=True,
    help="Print token counts before/after, and what each step saved, to stderr.",
)
@click.option(
    "--diff",
    is_flag=True,
    help="Print a unified diff of every file to stderr, then the --stats report.",
)
@click.option("--debug", is_flag=True, help="Show a traceback on internal errors.")
@click.option(
    "--list-modes",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=_list_modes,
    help="List available modes and exit.",
)
@click.option(
    "--mode-help",
    type=click.Choice(default_registry.names()),
    metavar="MODE",
    expose_value=False,
    is_eager=True,
    callback=_mode_help,
    help="Describe what a mode does at each level and exit.",
)
@click.version_option(
    __version__, "-V", "--version", prog_name="shrink", message="%(prog)s %(version)s"
)
def cli(  # noqa: PLR0917 - one parameter per Click option
    mode: list[str],
    level: int,
    fmt: str,
    delimiter: str | None,
    stats: bool,
    diff: bool,
    debug: bool,
) -> None:
    """Compress stdin for LLM context and write the result to stdout.

    \b
    Examples:
      cat notes.md | shrink --level 3 --stats
      files-to-prompt src | shrink --mode text | llm "review this"
      repomix --stdout | shrink --level 2 --stats > packed.txt
      shrink --diff < notes.md > /dev/null 2> changes.diff
    """
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            messages.info(
                "reading from the terminal; finish with Ctrl+D (Ctrl+Z, Enter on Windows)"
            )
        pipeline = Pipeline(
            reader=StdinReader(),
            splitter=make_splitter(fmt, delimiter),
            compressor=default_registry.create_chain(mode),
            writer=StdoutWriter(),
            reporter=_build_reporter(stats, diff),
        )
        pipeline.run(level)
    except KeyboardInterrupt:
        raise click.exceptions.Exit(EXIT_INTERRUPTED) from None
    except Exception as exc:
        if is_broken_pipe(exc):
            silence_stdout()
            raise click.exceptions.Exit(EXIT_OK) from None
        if debug:
            import traceback

            traceback.print_exc()
        messages.error(f"{type(exc).__name__}: {exc}")
        raise click.exceptions.Exit(EXIT_ERROR) from None


def _configure_stderr() -> None:
    reconfigure = getattr(sys.stderr, "reconfigure", None)
    if reconfigure is not None:
        with contextlib.suppress(OSError, ValueError):
            reconfigure(encoding="utf-8", errors="backslashreplace")


def main(argv: list[str] | None = None) -> int:
    """Console entry point; returns the process exit code."""
    _configure_stderr()
    try:
        result = cli.main(args=argv, prog_name="shrink", standalone_mode=False)
    except (KeyboardInterrupt, click.exceptions.Abort):
        return EXIT_INTERRUPTED
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    return result if isinstance(result, int) else EXIT_OK
