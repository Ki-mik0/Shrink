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
from typing import TYPE_CHECKING

import click

from shrink import __version__, messages
from shrink.compressors.base import DEFAULT_LEVEL, LEVEL_NAMES, MAX_LEVEL, MIN_LEVEL
from shrink.compressors.registry import default_registry
from shrink.io.reader import StdinReader
from shrink.io.writer import StdoutWriter
from shrink.pipeline import Pipeline, Reporter

if TYPE_CHECKING:
    from shrink.compressors.base import Compressor
    from shrink.observability.stats import Price
    from shrink.pipeline import Planner
from shrink.observability import DEFAULT_ENCODING, ENCODINGS, STATS_FORMATS
from shrink.splitting.factory import AUTO, FORMAT_NAMES, make_splitter

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_INTERRUPTED = 130

DEFAULT_MODE = "auto"
SUBCOMMANDS = ("wrap", "gain")

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


def _build_reporter(
    stats: str | None, diff: bool, encoding: str, price: Price | None
) -> Reporter | None:
    if not (stats or diff or price):
        return None
    from shrink.observability.stats import StatsReporter, make_counter

    reporter = StatsReporter(make_counter(encoding), fmt=stats or "text", price=price)
    if not diff:
        return reporter
    from shrink.observability.diff import DiffReporter

    return DiffReporter(reporter)


def _parse_price(spec: str | None, hit: float) -> Price | None:
    if spec is None:
        if hit:
            raise click.UsageError("--cache-hit needs --price INPUT/CACHED")
        return None
    from shrink.observability.stats import Price

    try:
        return Price.parse(spec, hit)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="'--price'") from exc


def _download(what: str, encoding: str) -> None:
    try:
        if what == "translate":
            from shrink.models import download_translate

            messages.info("downloading the translation model (about 630 MB, once) ...")
            folder = download_translate()
            done = "translation is now available offline (--translate)"
        else:
            from shrink.observability.stats import download_tokenizer

            messages.info(f"downloading the {encoding} tokenizer vocabulary ...")
            folder = download_tokenizer(encoding)
            done = "token counts are now exact and offline"
    except Exception as exc:
        messages.error(f"download failed: {type(exc).__name__}: {exc}")
        raise click.exceptions.Exit(EXIT_ERROR) from None
    messages.info(f"{what} ready in {folder}; {done}")


def _build_planner(
    query: str | None, budget: int | None, encoding: str, verbose: bool
) -> Planner | None:
    if not query and budget is None:
        return None
    from shrink.retrieval.planner import Planner

    count = None
    if budget is not None:
        from shrink.observability.stats import make_counter

        count = make_counter(encoding).count
    return Planner(
        query=query,
        budget=budget,
        count=count,
        report=messages.info if verbose else None,
        warn=messages.warn,
    )


def _build_compressor(mode: list[str], translate: bool, languages: str | None) -> Compressor:
    compressor = default_registry.create_chain(mode)
    if not translate:
        return compressor
    from shrink.compressors.chain import ChainCompressor
    from shrink.compressors.translate import TranslateCompressor

    only = frozenset(code.strip() for code in languages.split(",")) if languages else None
    return ChainCompressor([compressor, TranslateCompressor(only=only)])


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
    type=click.Choice(STATS_FORMATS),
    is_flag=False,
    flag_value="text",
    default=None,
    help="Print token counts before/after, and what each step saved, to stderr; "
    "--stats=json prints one JSON object instead.",
)
@click.option(
    "--encoding",
    type=click.Choice(ENCODINGS),
    default=DEFAULT_ENCODING,
    show_default=True,
    help="Tokenizer used for the counts (o200k_base: GPT-4o and newer; for Claude "
    "and other models it is an approximation).",
)
@click.option(
    "--price",
    metavar="USD[/CACHED]",
    help="Input price in USD per million tokens (optionally /price of cached input); "
    "adds the cost per call to the report.",
)
@click.option(
    "--cache-hit",
    type=click.FloatRange(0.0, 1.0),
    default=0.0,
    metavar="RATE",
    help="Share of input tokens read from the prompt cache (0-1), used with --price.",
)
@click.option(
    "--download",
    type=click.Choice(["tokenizer", "translate"]),
    help="Fetch data that needs the network once (tokenizer vocabulary or translation "
    "model) and exit.",
)
@click.option(
    "--translate",
    is_flag=True,
    help="After the mode, translate prose in any language to English with a local "
    "model, where that saves tokens (needs --download translate once).",
)
@click.option(
    "--translate-from",
    metavar="LANGS",
    help="Only translate these languages, e.g. uk,ru (fastText codes).",
)
@click.option(
    "-q",
    "--query",
    metavar="TEXT",
    help="Keep what matches TEXT (BM25 over functions and paragraphs) whole and compress "
    "the rest at --level. Reads the whole input before writing.",
)
@click.option(
    "--budget",
    type=click.IntRange(min=1),
    metavar="TOKENS",
    help="Compress the least relevant files harder until the output fits in TOKENS; "
    "never cuts text off. Reads the whole input before writing.",
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
    stats: str | None,
    encoding: str,
    price: str | None,
    cache_hit: float,
    download: str | None,
    translate: bool,
    translate_from: str | None,
    query: str | None,
    budget: int | None,
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
      shrink --download tokenizer      # once, for exact offline token counts
      cat notes_uk.md | shrink --translate --stats
      files-to-prompt src | shrink -l 3 --query "token counting" --budget 4000

    \b
    Other commands:
      shrink wrap claude    run an agent through the compressing proxy
      shrink gain           what shrink wrap has saved so far
    """
    if download is not None:
        _download(download, encoding)
        return
    parsed_price = _parse_price(price, cache_hit)
    try:
        if sys.stdin is not None and sys.stdin.isatty():
            messages.info(
                "reading from the terminal; finish with Ctrl+D (Ctrl+Z, Enter on Windows)"
            )
        pipeline = Pipeline(
            reader=StdinReader(),
            splitter=make_splitter(fmt, delimiter),
            compressor=_build_compressor(mode, translate, translate_from),
            writer=StdoutWriter(),
            reporter=_build_reporter(stats, diff, encoding, parsed_price),
            planner=_build_planner(query, budget, encoding, bool(stats or diff)),
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
    """Console entry point; returns the process exit code.

    ``shrink wrap ...`` and ``shrink gain`` are separate commands; everything else
    is the filter (it takes no positional arguments, so the two never clash).
    """
    _configure_stderr()
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] in SUBCOMMANDS:
        from shrink.integrations.commands import run

        return run(args[0], args[1:])
    try:
        result = cli.main(args=args, prog_name="shrink", standalone_mode=False)
    except (KeyboardInterrupt, click.exceptions.Abort):
        return EXIT_INTERRUPTED
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    return result if isinstance(result, int) else EXIT_OK
