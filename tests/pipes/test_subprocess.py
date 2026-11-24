"""End-to-end tests with a real process and real pipes."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

SHRINK = [sys.executable, "-m", "shrink"]


def run(args: list[str], data: bytes) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [*SHRINK, *args], input=data, capture_output=True, timeout=60, check=False
    )


def test_pipe_round_trip() -> None:
    proc = run(["--level", "1"], b"hello\nworld\n")
    assert proc.returncode == 0
    assert proc.stdout == b"hello\nworld\n"
    assert proc.stderr == b""


def test_empty_input() -> None:
    proc = run([], b"")
    assert proc.returncode == 0
    assert proc.stdout == b""


def test_binary_input_survives() -> None:
    raw = bytes(range(256)) * 4
    proc = run(["--level", "1"], raw)
    assert proc.returncode == 0
    assert b"Traceback" not in proc.stderr
    assert bytes(range(0x80, 0x100)) in proc.stdout  # invalid UTF-8 kept as-is


def test_output_is_deterministic() -> None:
    data = b"# Title\n\n  some   text \n\n\n\nmore\n" * 50
    first = run(["--level", "3"], data)
    second = run(["--level", "3"], data)
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout


def test_usage_error_exit_code() -> None:
    proc = run(["--level", "7"], b"")
    assert proc.returncode == 2


def test_closed_pipe_is_silent() -> None:
    data = b"x" * 79 + b"\n"
    data *= 200_000  # ~16 MB, far more than any pipe buffer
    proc = subprocess.Popen(
        [*SHRINK, "--level", "1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None and proc.stdout is not None and proc.stderr is not None
    proc.stdin.write(data)
    proc.stdin.close()
    proc.stdout.read(100)  # like `| head -c 100`
    proc.stdout.close()
    err = proc.stderr.read()
    assert proc.wait(timeout=60) == 0
    assert b"Traceback" not in err
    assert b"Exception ignored" not in err


@pytest.mark.skipif(sys.platform == "win32", reason="SIGINT cannot be sent to a child on Windows")
def test_ctrl_c_exits_130_without_traceback() -> None:
    proc = subprocess.Popen(
        SHRINK, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    time.sleep(1.5)  # let it start and block on stdin
    proc.send_signal(signal.SIGINT)
    _, err = proc.communicate(timeout=30)
    assert proc.returncode == 130
    assert b"Traceback" not in err


def test_core_import_does_not_load_optional_modules() -> None:
    code = (
        "import sys, shrink.cli; "
        "print([m for m in ('shrink.compressors.text', 'shrink.observability.stats', 'tiktoken') "
        "if m in sys.modules])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert proc.stdout.strip() == "[]"
