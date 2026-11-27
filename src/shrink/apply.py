"""``shrink apply``: write a model's answer into the repository.

The answer gives files as ``## File: <path>`` followed by a fenced code block
(the format the generation eval and most coding prompts ask for). For every
file, what the model left out is put back from the file on disk
(:mod:`shrink.restore`: stub bodies, ``...`` gaps, compact indentation), the
result is checked, and the file is written under ``--root``. A file that cannot
be restored is not written; the exit code is then 1.

    llm "add accounts" < context.txt > answer.md
    shrink apply answer.md --dry-run     # what would change
    shrink apply answer.md               # write the files
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path
from typing import TextIO

import click

from shrink import messages
from shrink.restore import Report, RestoreError, restore
from shrink.splitting.formats import FORMATS
from shrink.splitting.framed import FramedSplitter

EXIT_OK = 0
EXIT_FAILED = 1


def answer_files(text: str) -> list[tuple[str, str]]:
    """``(path, content)`` of every ``## File:`` block, in order; a later block for
    the same path replaces an earlier one."""
    splitter = FramedSplitter(FORMATS["repomix-markdown"])
    found: dict[str, str] = {}
    for doc in splitter.split(text.splitlines(keepends=True)):
        if doc.path is None:
            continue
        found[doc.path.strip().strip("`").replace("\\", "/")] = doc.text
    return list(found.items())


def safe_target(root: Path, path: str) -> Path | None:
    """The file ``path`` names under ``root``, or ``None`` if it would leave it."""
    if not path or path.startswith("/") or ":" in path or ".." in path.split("/"):
        return None
    target = (root / path).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        return None
    return target


def _read(path: Path) -> tuple[str | None, str]:
    """The file's text with LF line breaks, and the line break it used."""
    if not path.is_file():
        return None, "\n"
    raw = path.read_bytes().decode("utf-8")
    newline = "\r\n" if "\r\n" in raw else "\n"
    return raw.replace("\r\n", "\n"), newline


def _line(report: Report, path: str) -> str:
    mark = {"new": "A", "changed": "M", "unchanged": "="}[report.status]
    details = report.describe()
    return f"{mark} {path}" + (f"  ({details})" if details else "")


def apply_answer(
    text: str, root: Path, *, dry_run: bool = False, diff: TextIO | None = None
) -> tuple[list[Report], list[str]]:
    """Restore and write every file of an answer; returns the reports and the errors."""
    reports: list[Report] = []
    errors: list[str] = []
    for path, content in answer_files(text):
        target = safe_target(root, path)
        if target is None:
            errors.append(f"{path}: outside the repository, skipped")
            continue
        try:
            original, newline = _read(target)
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"{path}: cannot read the current file: {exc}")
            continue
        try:
            result = restore(original, content, path)
        except RestoreError as exc:
            errors.append(f"{path}: {exc}")
            continue
        reports.append(result.report)
        messages.info(_line(result.report, path))
        if diff is not None and result.text != (original or ""):
            diff.writelines(
                difflib.unified_diff(
                    (original or "").splitlines(keepends=True),
                    result.text.splitlines(keepends=True),
                    f"a/{path}",
                    f"b/{path}",
                )
            )
        if dry_run or result.report.status == "unchanged":
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.text.replace("\n", newline).encode("utf-8"))
    return reports, errors


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("answer", type=click.File("r", encoding="utf-8"), default="-")
@click.option(
    "--root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    show_default=True,
    help="Repository the paths in the answer are relative to.",
)
@click.option("--dry-run", is_flag=True, help="Show what would change; write nothing.")
@click.option("--diff", is_flag=True, help="Print a unified diff of every change to stdout.")
def command(answer: TextIO, root: Path, dry_run: bool, diff: bool) -> int:
    """Write the files of a model's answer, filling in what it left out.

    ANSWER is a file with `## File: <path>` sections (default: stdin). Stub
    bodies (`...`, `{/* ... */}`), `...` gaps and compact indentation from
    `shrink --lazy --dedent` are restored from the files under --root.
    """
    text = answer.read()
    reports, errors = apply_answer(text, root, dry_run=dry_run, diff=sys.stdout if diff else None)
    for error in errors:
        messages.error(error)
    if not reports and not errors:
        messages.warn("no '## File: <path>' sections found in the answer")
    stubs = sum(len(r.unresolved) for r in reports)
    if stubs:
        messages.warn(f"{stubs} function(s) left as stubs: nothing to fill them in from")
    if errors:
        raise click.exceptions.Exit(EXIT_FAILED)
    return EXIT_OK


def run(args: list[str]) -> int:
    """Entry point used by ``shrink.cli.main`` for ``shrink apply ...``."""
    try:
        result = command.main(args=args, prog_name="shrink apply", standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except (KeyboardInterrupt, click.exceptions.Abort):
        return 130
    return result if isinstance(result, int) else EXIT_OK
