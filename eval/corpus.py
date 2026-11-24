"""The fixed corpus used by the benchmarks.

Python: a slice of the standard library of the interpreter that runs the
benchmark (its version is written into every result) plus shrink's own source.
Text, logs and JSON: files kept in ``eval/corpus`` and the project's own docs.
"""

from __future__ import annotations

import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = Path(__file__).resolve().parent / "corpus"
STDLIB = Path(sysconfig.get_paths()["stdlib"])

STDLIB_MODULES = [
    "argparse.py",
    "dataclasses.py",
    "textwrap.py",
    "shlex.py",
    "fnmatch.py",
    "json/decoder.py",
    "json/encoder.py",
    "json/__init__.py",
    "functools.py",
    "enum.py",
]


@dataclass(frozen=True)
class Sample:
    name: str  # unique, used in tables and as a file name for other tools
    path: Path
    kind: str  # code | text | log | json

    def read(self) -> str:
        return self.path.read_text(encoding="utf-8")


def python_samples() -> list[Sample]:
    samples = [
        Sample("stdlib/" + name, STDLIB / name, "code")
        for name in STDLIB_MODULES
        if (STDLIB / name).is_file()
    ]
    samples += [
        Sample(
            "shrink/" + str(p.relative_to(ROOT / "src" / "shrink")).replace("\\", "/"), p, "code"
        )
        for p in sorted((ROOT / "src" / "shrink").rglob("*.py"))
        if p.stat().st_size > 1500  # skip tiny __init__ files
    ]
    return samples


def other_samples() -> list[Sample]:
    return [
        Sample("README.md (uk)", ROOT / "README.md", "text"),
        Sample("docs/PLAN.md (uk)", ROOT / "docs" / "PLAN.md", "text"),
        Sample("messy.md", ROOT / "tests" / "fixtures" / "messy.md", "text"),
        Sample("pytest.log", CORPUS / "pytest.log", "log"),
        Sample("packages.json", CORPUS / "packages.json", "json"),
    ]


def python_version() -> str:
    return f"Python {sys.version.split()[0]}"
