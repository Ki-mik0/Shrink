"""``shrink apply``: write a model's answer into the repository.

The answer gives files as ``## File: <path>`` followed by a fenced code block
(the format the generation eval and most coding prompts ask for), or as
``## Patch: <path>`` with only the definitions to add or change (``--patch``).
For every file, what the model left out is put back from the file on disk
(:mod:`shrink.restore`: patches, stub bodies, ``...`` gaps, compact
indentation), the result is checked, and the file is written under ``--root``.
A file that cannot be restored is not written; the exit code is then 1.

``--context`` takes the very text the model was given. A function it only saw
as a stub then keeps the file's own code, even if the answer has a body for it:
the model never saw that code, so its version is a guess.

    shrink -l 3 --keep create_app --patch < packed.txt > context.txt
    llm "add accounts" < context.txt > answer.md
    shrink apply answer.md --context context.txt --dry-run
    shrink apply answer.md --context context.txt
"""

from __future__ import annotations

import difflib
import re
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import click

from shrink import messages
from shrink.restore import (
    Report,
    Restored,
    RestoreError,
    insert_fragment,
    is_fragment,
    restore,
)
from shrink.splitting.factory import AUTO, make_splitter

EXIT_OK = 0
EXIT_FAILED = 1


_HEADING = re.compile(r"(?m)^## (File|Patch): (\S[^\n]*?)[ \t]*$")
_BLOCK = re.compile(r"(?ms)^(`{3,})[^\n`]*\n(.*?)^\1[ \t]*$")
# "```html<!doctype html>": models sometimes glue the first line to the fence.
_GLUED = re.compile(r"(?m)^(`{3,})(html|javascript|js|jsx|python|py|css)(?=[^\s`\w+-])")


@dataclass(frozen=True)
class Section:
    path: str
    content: str
    partial: bool = False  # "## Patch:"


def answer_sections(text: str) -> list[Section]:
    """Every ``## File:`` and ``## Patch:`` block, in order. A patch may hold
    several code blocks (an import, then lines for inside a function); each
    becomes a section of its own, applied in turn."""
    text = _GLUED.sub(r"\1\2\n", text.replace("\r\n", "\n"))
    headings = list(_HEADING.finditer(text))
    found: list[Section] = []
    ends = [heading.start() for heading in headings[1:]] + [len(text)] * bool(headings)
    for heading, end in zip(headings, ends, strict=True):
        body = text[heading.end() : end]
        blocks = [match.group(2) for match in _BLOCK.finditer(body)]
        path = heading.group(2).strip().strip("`").replace("\\", "/")
        if heading.group(1) == "File":
            found += [Section(path, blocks[0])] if blocks else []
        else:
            found += [Section(path, block, partial=True) for block in blocks]
    return found


def answer_files(text: str) -> list[tuple[str, str]]:
    """``(path, content)`` per file of an answer; a later block wins."""
    return list({s.path: s.content for s in answer_sections(text)}.items())


def context_files(text: str) -> dict[str, str]:
    """``{path: text}`` of the packed context shrink wrote for the model."""
    lines = _without_trailer(text.replace("\r\n", "\n").splitlines(keepends=True))
    docs = make_splitter(AUTO, None).split(lines)
    return {doc.path.replace("\\", "/"): doc.text for doc in docs if doc.path is not None}


def _without_trailer(lines: list[str]) -> list[str]:
    """Drop what follows the closing ``---`` of the last files-to-prompt file (the
    notes of ``--brief`` or ``--patch``, a task pasted after the context)."""
    if len(lines) < 2 or lines[1].rstrip("\n") != "---":
        return lines
    for index in range(len(lines) - 1, 1, -1):
        if lines[index].rstrip("\n") == "---":
            return lines[: index + 1]
    return lines


def merge_sections(
    sections: list[Section],
    read: Callable[[str], str | None],
    shown: Mapping[str, str] | None = None,
) -> tuple[dict[str, Restored], dict[str, str]]:
    """The new text of every file in an answer, and why some could not be merged.

    ``read`` gives a file's current text (``None``: it does not exist). A later
    ``## File:`` replaces an earlier block for the same path; a later
    ``## Patch:`` applies on top of it.
    """
    done: dict[str, Restored] = {}
    errors: dict[str, str] = {}
    for section in sections:
        path = section.path
        earlier = done.get(path) if section.partial else None
        try:
            base = earlier.text if earlier is not None else read(path)
            content = section.content
            if section.partial and base is not None and is_fragment(content, path):
                content = insert_fragment(base, content)
            result = restore(
                base,
                content,
                path,
                partial=section.partial,
                shown=None if shown is None else shown.get(path),
            )
        except (OSError, UnicodeDecodeError) as exc:
            errors[path] = f"cannot read the current file: {exc}"
            continue
        except RestoreError as exc:
            errors[path] = str(exc)
            continue
        if earlier is not None:
            result.report.reverted[:0] = earlier.report.reverted
        done[path] = result
        errors.pop(path, None)
    return done, errors


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
    text: str,
    root: Path,
    *,
    dry_run: bool = False,
    diff: TextIO | None = None,
    context: str | None = None,
) -> tuple[list[Report], list[str]]:
    """Restore and write every file of an answer; returns the reports and the errors."""
    errors: list[str] = []
    sections = []
    for section in answer_sections(text):
        if safe_target(root, section.path) is None:
            errors.append(f"{section.path}: outside the repository, skipped")
        else:
            sections.append(section)
    originals: dict[str, tuple[str | None, str]] = {}

    def read(path: str) -> str | None:
        if path not in originals:
            originals[path] = _read(root / path)
        return originals[path][0]

    shown = None if context is None else context_files(context)
    merged, failed = merge_sections(sections, read, shown)
    errors += [f"{path}: {why}" for path, why in failed.items()]
    reports: list[Report] = []
    for path, result in merged.items():
        if path in failed:
            continue
        original, newline = originals[path]
        report = result.report
        if original is not None:
            report.status = "unchanged" if result.text == original else "changed"
        else:
            report.status = "new"
        reports.append(report)
        messages.info(_line(report, path))
        if diff is not None and result.text != (original or ""):
            diff.writelines(
                difflib.unified_diff(
                    (original or "").splitlines(keepends=True),
                    result.text.splitlines(keepends=True),
                    f"a/{path}",
                    f"b/{path}",
                )
            )
        if dry_run or report.status == "unchanged":
            continue
        target = root / path
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
@click.option(
    "--context",
    type=click.File("r", encoding="utf-8"),
    help="The text the model was given (shrink's output). Functions it only saw as "
    "stubs keep the file's own code, and nothing is removed from existing files.",
)
@click.option("--dry-run", is_flag=True, help="Show what would change; write nothing.")
@click.option("--diff", is_flag=True, help="Print a unified diff of every change to stdout.")
def command(answer: TextIO, root: Path, context: TextIO | None, dry_run: bool, diff: bool) -> int:
    """Write the files of a model's answer, filling in what it left out.

    ANSWER is a file with `## File: <path>` or `## Patch: <path>` sections
    (default: stdin). Patches, stub bodies (`...`, `{/* ... */}`), `...` gaps and
    compact indentation (`shrink --patch --dedent`) are merged with the files
    under --root.
    """
    text = answer.read()
    reports, errors = apply_answer(
        text,
        root,
        dry_run=dry_run,
        diff=sys.stdout if diff else None,
        context=None if context is None else context.read(),
    )
    reverted = [f"{r.path}:{name}" for r in reports for name in r.reverted]
    if reverted:
        messages.warn(
            f"the model never saw the code of {', '.join(reverted)}, so its version was "
            "not used; if the task needs them changed, compress with --keep and ask again"
        )
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
