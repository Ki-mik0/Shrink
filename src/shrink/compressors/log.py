"""Log mode: terminal noise, timestamps, repeats and low-level lines.

Logs and test or build output are where agents burn most tokens: colour codes,
progress bars redrawn with carriage returns, the same line a thousand times. The
rules here only drop or merge lines; whatever is kept is the original text.
Each level runs to a fixed point, so running it twice changes nothing.
"""

from __future__ import annotations

import re

from shrink.compressors.base import Step, check_level
from shrink.document import Document

HELP = """\
Logs, test runs and build output.

  level 1  light     remove ANSI colour/cursor codes, keep only the last state of
                     carriage-return progress lines, strip trailing whitespace,
                     squeeze blank lines, merge identical neighbouring lines into
                     one line with `[xN]`
  level 2  medium    + drop leading timestamps (ISO 8601, hh:mm:ss, syslog
                     dates), then merge repeats again; a block of 2-8 lines that
                     repeats back to back is kept once, followed by
                     `[previous N lines xM]`
  level 3  skeleton  + drop DEBUG, TRACE, VERBOSE and INFO lines; each dropped run
                     leaves `[N lines dropped]` (a run shorter than that
                     marker is kept)

Line endings are kept: pure CRLF input stays CRLF."""

STEP_NAMES = {1: "ansi+repeats", 2: "timestamps+blocks", 3: "low-level lines"}
MAX_BLOCK = 8
_MAX_PASSES = 8

_ANSI = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"  # CSI: colours, cursor movement
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC: titles, hyperlinks
    r"|\x1b[@-Z\\-_]"  # other two-byte escapes
)
_TIME = r"\d{1,2}:\d{2}:\d{2}(?:[.,]\d+)?"
_ZONE = r"(?:Z|[+-]\d{2}:?\d{2}|\s?UTC)?"
_DATE = r"\d{4}[-/]\d{2}[-/]\d{2}"
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_STAMP = rf"(?:{_DATE}[T ]{_TIME}{_ZONE}|{_DATE}|{_MONTH} [ \d]\d {_TIME}|{_TIME}{_ZONE})"
_TIMESTAMP = re.compile(rf"^(?:\[{_STAMP}\]|\({_STAMP}\)|{_STAMP})(?:\s*[-|:]\s*|\s+|$)")
_LOW_LEVEL = re.compile(
    r"^\W{0,3}(?:DEBUG|TRACE|VERBOSE|INFO)\b|\blevel=(?:debug|trace|info)\b",
)
_REPEAT_MARK = re.compile(r" \[x\d+\]$")


class LogCompressor:
    name = "log"
    summary = "logs and build output: colour codes, timestamps, repeats, debug lines"
    help = HELP

    def compress(self, doc: Document, level: int) -> Document:
        return doc.with_text(compress_log(doc.text, level))

    def trace(self, doc: Document, level: int) -> list[Step]:
        check_level(level)
        return [
            Step(STEP_NAMES[k], doc.with_text(compress_log(doc.text, k)))
            for k in range(1, level + 1)
        ]


def compress_log(text: str, level: int) -> str:
    check_level(level)
    if not text:
        return text
    pure_crlf = "\r\n" in text and "\n" not in text.replace("\r\n", "")
    body = text.replace("\r\n", "\n")
    for _ in range(_MAX_PASSES):
        new = _one_pass(body, level)
        if new == body:
            break
        body = new
    if len(body) > len(text):  # guard: never longer than the input
        return text
    return body.replace("\n", "\r\n") if pure_crlf else body


def _one_pass(body: str, level: int) -> str:
    lines = [_clean(line) for line in body.split("\n")]
    if level >= 2:
        lines = [_TIMESTAMP.sub("", line, count=1) for line in lines]
    lines = _merge_repeats(lines)
    if level >= 2:
        lines = _merge_blocks(lines)
    if level >= 3:
        lines = _drop_low_level(lines)
    while lines and not lines[-1]:
        lines.pop()
    while lines and not lines[0]:
        lines.pop(0)
    result = "\n".join(lines)
    if result and body.endswith("\n"):
        result += "\n"
    return result


def _clean(line: str) -> str:
    line = _ANSI.sub("", line)
    if "\r" in line:  # a progress bar redrawn in place: only the last state is visible
        line = line.rsplit("\r", 1)[-1] or line.rstrip("\r").rsplit("\r", 1)[-1]
    return line.rstrip()


def _merge_repeats(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        j = i + 1
        while j < len(lines) and lines[j] == line:
            j += 1
        count = j - i
        if not line:
            if not out or out[-1]:
                out.append("")
        elif count == 1:
            out.append(line)
        else:
            merged = _add_count(line, count)
            # Merge only when it saves characters: "a [x2]" is longer than "a\na".
            out.extend([merged] if len(merged) < count * (len(line) + 1) - 1 else [line] * count)
        i = j
    return out


def _add_count(line: str, count: int) -> str:
    """``line [x3]``; a line that already carries a count gets the product."""
    mark = _REPEAT_MARK.search(line)
    if mark is not None:
        count *= int(mark.group(0)[3:-1])
        line = line[: mark.start()]
    return f"{line} [x{count}]"


def _merge_blocks(lines: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(lines):
        best_size, best_times = 0, 1
        for size in range(2, MAX_BLOCK + 1):
            block = lines[i : i + size]
            if len(block) < size or not any(block):
                break
            times = 1
            while lines[i + times * size : i + (times + 1) * size] == block:
                times += 1
            if times > 1 and times * size > best_times * best_size:
                best_size, best_times = size, times
        if best_times > 1:
            out.extend(lines[i : i + best_size])
            out.append(f"[previous {best_size} lines x{best_times}]")
            i += best_size * best_times
        else:
            out.append(lines[i])
            i += 1
    return out


def _drop_low_level(lines: list[str]) -> list[str]:
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if not run:
            return
        marker = f"[{len(run)} line{'s' if len(run) > 1 else ''} dropped]"
        # A run shorter than its marker is kept: dropping it would cost tokens.
        out.extend([marker] if sum(len(x) + 1 for x in run) > len(marker) + 1 else run)
        run.clear()

    for line in lines:
        if line and _LOW_LEVEL.search(line):
            run.append(line)
            continue
        flush()
        out.append(line)
    flush()
    return out


def looks_like_log(text: str) -> bool:
    """Most of the first non-empty lines start with a timestamp."""
    sample = [line for line in text.split("\n", 200)[:200] if line.strip()][:20]
    if len(sample) < 3:
        return False
    stamped = sum(1 for line in sample if _TIMESTAMP.match(_ANSI.sub("", line)))
    return stamped >= 3 and stamped * 3 >= len(sample)
