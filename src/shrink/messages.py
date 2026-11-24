"""Diagnostics on stderr. stdout is reserved for data."""

from __future__ import annotations

import sys

PROG = "shrink"


def _emit(kind: str, message: str) -> None:
    stream = sys.stderr
    if stream is None:  # pragma: no cover - pythonw / detached process
        return
    stream.write(f"{PROG}: {kind}{message}\n")
    stream.flush()


def warn(message: str) -> None:
    _emit("warning: ", message)


def error(message: str) -> None:
    _emit("error: ", message)


def info(message: str) -> None:
    _emit("", message)
