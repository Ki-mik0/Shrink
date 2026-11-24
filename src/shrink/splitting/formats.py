"""The packed-stream formats shrink understands.

Each format only answers two questions for :class:`~shrink.splitting.framed.FramedSplitter`:
does a file header start here, and does the file content end here. Layouts
checked against the tools' documentation: files-to-prompt (default and
``--cxml``) and Repomix (``xml``, ``markdown`` and ``plain`` styles).
"""

from __future__ import annotations

import re

from shrink.splitting.framed import Format, Header, line_text

_RULE = "---"
_PATH_EXT = re.compile(r"\.[A-Za-z][A-Za-z0-9]{0,9}$")
_CXML_DOCUMENT = re.compile(r'^<document index="\d+">$')
_CXML_SOURCE = re.compile(r"^<source>(.*)</source>$")
_XML_FILE = re.compile(r'^<file path="(.*)">$')
_MD_FILE = re.compile(r"^## File: (.+)$")
_MD_FENCE = re.compile(r"^(`{3,})[^`]*$")
_MD_CLOSE = re.compile(r"^ {0,3}(`{3,})[ \t]*$")
_PLAIN_RULE = re.compile(r"^={16}$")
_PLAIN_SECTIONS = frozenset(
    {"File Summary", "Directory Structure", "Files", "Git Logs", "Instruction"}
)


def looks_like_path(text: str) -> bool:
    """Conservative: a name with a slash or an extension, not a sentence or a title.

    Keeps ``Title`` / ``---`` (a Markdown heading) and ``title: x`` / ``---`` (front
    matter) from being taken for file headers.
    """
    if not text or text != text.strip() or len(text) > 1024 or "\0" in text:
        return False
    if ": " in text or text.startswith(("-", "#", "<")):
        return False
    return "/" in text or "\\" in text or _PATH_EXT.search(text) is not None


class FilesToPrompt:
    """``path`` / ``---`` / content / ``---``, as printed by ``files-to-prompt``."""

    name = "files-to-prompt"
    summary = "files-to-prompt default: path, ---, content, ---"
    implicit_end = False
    blank_to_tail = True

    def header(self, ahead: list[str]) -> Header | None:
        if (
            len(ahead) >= 2
            and line_text(ahead[1]) == _RULE
            and looks_like_path(line_text(ahead[0]))
        ):
            return Header(2, line_text(ahead[0]))
        return None

    def content_end(self, ahead: list[str], header: Header) -> int | None:
        if line_text(ahead[0]) != _RULE:
            return None
        rest = ahead[1:]
        if len(ahead) < 3 and all(not line.strip() for line in rest):
            return 1  # the stream ends after this rule (maybe with a blank line)
        if len(ahead) >= 3 and self.header(rest) is not None:
            return 1
        return None


class FilesToPromptXml:
    """``files-to-prompt --cxml``: ``<documents>`` with ``<document index="n">`` items."""

    name = "cxml"
    summary = "files-to-prompt --cxml: <documents><document>..."
    implicit_end = False
    blank_to_tail = False

    def header(self, ahead: list[str]) -> Header | None:
        if len(ahead) < 3 or not _CXML_DOCUMENT.match(line_text(ahead[0])):
            return None
        source = _CXML_SOURCE.match(line_text(ahead[1]))
        if source is None or line_text(ahead[2]) != "<document_content>":
            return None
        return Header(3, source.group(1))

    def content_end(self, ahead: list[str], header: Header) -> int | None:
        return 1 if line_text(ahead[0]) == "</document_content>" else None


class RepomixXml:
    """Repomix ``--style xml``: ``<file path="...">`` ... ``</file>``."""

    name = "repomix-xml"
    summary = "Repomix xml style: <file path=...>"
    implicit_end = False
    blank_to_tail = False

    def header(self, ahead: list[str]) -> Header | None:
        match = _XML_FILE.match(line_text(ahead[0]))
        return Header(1, match.group(1)) if match else None

    def content_end(self, ahead: list[str], header: Header) -> int | None:
        return 1 if line_text(ahead[0]) == "</file>" else None


class RepomixMarkdown:
    """Repomix ``--style markdown``: ``## File: path`` and a fenced block."""

    name = "repomix-markdown"
    summary = "Repomix markdown style: ## File: path + fenced block"
    implicit_end = False
    blank_to_tail = False

    def header(self, ahead: list[str]) -> Header | None:
        if len(ahead) < 2:
            return None
        title = _MD_FILE.match(line_text(ahead[0]))
        fence = _MD_FENCE.match(line_text(ahead[1]))
        if title is None or fence is None:
            return None
        return Header(2, title.group(1), state=fence.group(1))

    def content_end(self, ahead: list[str], header: Header) -> int | None:
        match = _MD_CLOSE.match(line_text(ahead[0]))
        if match is not None and len(match.group(1)) >= len(header.state):
            return 1
        return None


class RepomixPlain:
    """Repomix ``--style plain``: a ``=`` rule, ``File: path``, a ``=`` rule."""

    name = "repomix-plain"
    summary = "Repomix plain style: ==== / File: path / ===="
    implicit_end = True
    blank_to_tail = True

    def header(self, ahead: list[str]) -> Header | None:
        if len(ahead) < 3:
            return None
        top, title, bottom = (line_text(line) for line in ahead[:3])
        if not (_PLAIN_RULE.match(top) and _PLAIN_RULE.match(bottom)):
            return None
        if title.startswith("File: ") and len(title) > len("File: "):
            return Header(3, title[len("File: ") :])
        if title in _PLAIN_SECTIONS:
            return Header(3, None, is_file=False)
        return None

    def content_end(self, ahead: list[str], header: Header) -> int | None:
        return None


FORMATS: dict[str, Format] = {
    fmt.name: fmt
    for fmt in (
        FilesToPrompt(),
        FilesToPromptXml(),
        RepomixXml(),
        RepomixMarkdown(),
        RepomixPlain(),
    )
}

_XML_MARKERS = frozenset({"<file_summary>", "<directory_structure>", "<files>"})
_MD_MARKERS = frozenset({"# File Summary", "# Directory Structure", "# Files"})
DETECT_LINES = 60


def detect(head: list[str]) -> Format | None:
    """Guess the format from the first lines of a stream; None means plain text."""
    texts = [line_text(line) for line in head]
    if texts[:1] == ["<documents>"]:
        return FORMATS["cxml"]
    if any(text in _XML_MARKERS for text in texts):
        return FORMATS["repomix-xml"]
    if any(text in _MD_MARKERS for text in texts):
        return FORMATS["repomix-markdown"]
    plain = FORMATS["repomix-plain"]
    for i in range(len(head)):
        found = plain.header(head[i : i + 3])
        if found is not None:
            return plain
    if FORMATS["files-to-prompt"].header(head[:2]) is not None:
        return FORMATS["files-to-prompt"]
    return None
