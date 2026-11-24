"""``shrink wrap`` and ``shrink gain``.

``shrink wrap claude`` starts the local proxy, runs Claude Code with
``ANTHROPIC_BASE_URL`` pointing at it, and prints what was saved when the agent
exits. ``shrink wrap`` alone only runs the proxy, for any other tool.
``shrink gain`` sums up the log of all sessions.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import click

from shrink import messages


@dataclass(frozen=True)
class Api:
    variable: str  # environment variable the agent reads its base URL from
    suffix: str  # path the agent expects at the end of that URL
    default_upstream: str


APIS = {
    "anthropic": Api("ANTHROPIC_BASE_URL", "", "https://api.anthropic.com"),
    "openai": Api("OPENAI_BASE_URL", "/v1", "https://api.openai.com"),
}
AGENTS = {"claude": "anthropic", "codex": "openai"}


def _agent_api(command: tuple[str, ...], api: str | None) -> str:
    if api is not None:
        return api
    if command:
        name = Path(command[0]).name.lower()
        for ext in (".exe", ".cmd", ".bat", ".ps1"):
            name = name.removesuffix(ext)
        if name in AGENTS:
            return AGENTS[name]
        raise click.UsageError(
            f"which API does {command[0]!r} use? add --api anthropic or --api openai"
        )
    return "anthropic"


def _upstream(api: Api, explicit: str | None) -> str:
    if explicit:
        return explicit.rstrip("/")
    current = os.environ.get(api.variable, "").rstrip("/")
    if current and not current.startswith("http://127.0.0.1:"):
        return current.removesuffix(api.suffix) if api.suffix else current
    return api.default_upstream


@click.command(
    "wrap",
    context_settings={
        "help_option_names": ["-h", "--help"],
        "ignore_unknown_options": True,
        "allow_interspersed_args": False,
    },
)
@click.option(
    "-l",
    "--level",
    type=click.IntRange(1, 3),
    default=1,
    show_default=True,
    help="Compression level for tool output (1 keeps it closest to the original).",
)
@click.option("--translate", is_flag=True, help="Translate what you type to English.")
@click.option("--translate-from", metavar="LANGS", help="Only these languages, e.g. uk,ru.")
@click.option(
    "--api",
    type=click.Choice(sorted(APIS)),
    help="API the agent speaks (known for claude and codex).",
)
@click.option(
    "--upstream", metavar="URL", help="Where to forward requests (default: the real API)."
)
@click.option("--port", type=click.IntRange(0, 65535), default=0, help="Port to listen on.")
@click.option("--no-log", is_flag=True, help="Do not add this session to `shrink gain`.")
@click.argument("command", nargs=-1, type=click.UNPROCESSED)
def wrap(  # noqa: PLR0917 - one parameter per Click option
    level: int,
    translate: bool,
    translate_from: str | None,
    api: str | None,
    upstream: str | None,
    port: int,
    no_log: bool,
    command: tuple[str, ...],
) -> None:
    """Run an agent (or just the proxy) with shrink between it and the model API.

    \b
    Examples:
      shrink wrap claude                 # Claude Code, tool output compressed
      shrink wrap --translate claude     # and your prompts translated to English
      shrink wrap codex                  # OpenAI Codex CLI
      shrink wrap --port 8787            # proxy only; point any tool at it

    Only tool output (and, with --translate, your own text) is changed; results of
    tools that read or edit files, system prompts and earlier assistant turns are
    forwarded as they are.
    """
    from shrink.integrations.gain import GainLog
    from shrink.integrations.proxy import ShrinkProxy
    from shrink.integrations.shrinker import RequestShrinker
    from shrink.observability.stats import make_counter

    api_name = _agent_api(command, api)
    spec = APIS[api_name]
    translator = None
    if translate:
        from shrink.compressors.translate import TranslateCompressor

        only = frozenset(c.strip() for c in translate_from.split(",")) if translate_from else None
        translator = TranslateCompressor(only=only)
    shrinker = RequestShrinker(level=level, translate=translator, count=make_counter().count)
    log = GainLog(agent=Path(command[0]).stem if command else api_name)
    target = _upstream(spec, upstream)
    try:
        proxy = ShrinkProxy(target, shrinker, on_saving=None if no_log else log.record, port=port)
    except (OSError, ValueError) as exc:
        raise click.UsageError(f"cannot start the proxy: {exc}") from exc
    base = proxy.url + spec.suffix
    messages.info(f"proxy {base} -> {target} (level {level}{', translate' if translate else ''})")
    if not command:
        messages.info(f"point your tool at it: {spec.variable}={base}  (Ctrl+C to stop)")
        try:
            proxy.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            _finish(proxy, log)
        return
    executable = shutil.which(command[0])
    if executable is None:
        raise click.UsageError(f"command not found: {command[0]}")
    proxy.start()
    env = {**os.environ, spec.variable: base}
    try:
        child = subprocess.Popen([executable, *command[1:]], env=env)
        code = _wait(child)
    finally:
        _finish(proxy, log)
    raise click.exceptions.Exit(code)


def _wait(child: subprocess.Popen[bytes]) -> int:
    """Ctrl+C reaches the agent too; let it decide, and wait until it is gone."""
    while True:
        try:
            return child.wait()
        except KeyboardInterrupt:
            continue


def _finish(proxy: object, log: object) -> None:
    from shrink.integrations.gain import GainLog, Totals
    from shrink.integrations.proxy import ShrinkProxy

    assert isinstance(proxy, ShrinkProxy)
    assert isinstance(log, GainLog)
    proxy.stop()
    totals = Totals(log.requests, log.tokens_before, log.tokens_after)
    messages.info(
        f"session: {totals.requests} request(s) shortened, saved {totals.saved} tokens "
        f"({totals.percent()} on the changed text); `shrink gain` for all sessions"
    )


@click.command("gain", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--json", "as_json", is_flag=True, help="Print one JSON object.")
@click.option("--days", type=click.IntRange(1), default=7, show_default=True)
@click.option("--reset", is_flag=True, help="Delete the log and start counting again.")
def gain(as_json: bool, days: int, reset: bool) -> None:
    """Show what `shrink wrap` saved across all sessions."""
    import json

    from shrink.integrations.gain import gain_path, load, report, summary_json

    if reset:
        gain_path().unlink(missing_ok=True)
        messages.info("gain log cleared")
        return
    records = load()
    if as_json:
        click.echo(json.dumps(summary_json(records)))
    else:
        click.echo("\n".join(report(records, days)))


COMMANDS = {"wrap": wrap, "gain": gain}


def run(name: str, args: list[str]) -> int:
    """Entry point used by ``shrink.cli.main`` for ``shrink <name> ...``."""
    try:
        result = COMMANDS[name].main(args=args, prog_name=f"shrink {name}", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except (KeyboardInterrupt, click.exceptions.Abort):
        return 130
    return result if isinstance(result, int) else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run(sys.argv[1], sys.argv[2:]))
