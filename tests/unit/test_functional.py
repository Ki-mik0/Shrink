"""The eval's behaviour checks, proved on a reference answer (needs FastAPI)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# one insert per folder: the eval scripts are not a package
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval" / "generation"))

import functional
import run_generation as gen

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REFERENCE = (FIXTURES / "generation_reference.md").read_text(encoding="utf-8")
VARIANT = "L3+keep+dedent+brief+patch"


def failures(files: dict[str, str]) -> dict[str, str]:
    result = functional.run_checks(files)
    assert result is not None
    return {name: why for name, why in result.items() if why}


def test_reference_patch_answer_passes_every_check() -> None:
    answer = gen.answer_of(VARIANT, REFERENCE)
    assert answer.restore_failed == 0
    assert answer.reverted == 0  # create_app was shown, so its change is accepted
    assert answer.stubs_left == 0
    assert "def get_current_user" not in answer.files["backend/app/db.py"]
    assert "def init_db" in answer.files["backend/app/db.py"]  # kept by the merge
    assert answer.files["frontend/login.html"].startswith("<!doctype html>")  # glued fence
    assert failures(answer.files) == {}
    checks = gen.completeness(answer)
    assert len(checks["checks_passed"]) == checks["checks_total"]


def test_the_starter_alone_fails() -> None:
    failed = failures({})
    assert "app starts, health works" not in failed
    assert len(failed) == len(functional.CHECKS) - 1


def test_invented_code_is_caught_without_the_guard() -> None:
    # A model that never saw init_db() "rewrites" it with its own version table:
    # a fresh database works, a deployed one breaks.
    invented = """
## Patch: backend/app/db.py
```python
def schema_version(conn: sqlite3.Connection) -> int:
    return 0
```
"""
    files = gen.parse_answer(REFERENCE + invented).files
    from shrink.apply import answer_sections, merge_sections

    starter = dict(gen.starter_files())
    sections = answer_sections(REFERENCE + invented)
    unguarded, _ = merge_sections(sections, starter.get)
    broken = {**files, **{path: result.text for path, result in unguarded.items()}}
    assert "a database of the old version upgrades" in failures(broken)

    guarded = gen.answer_of(VARIANT, REFERENCE + invented)
    assert guarded.reverted == 1
    assert failures(guarded.files) == {}


def test_the_guard_refuses_network_and_subprocesses(tmp_path: Path) -> None:
    hostile = """
import socket, subprocess
for attempt in (
    lambda: subprocess.run(["cmd", "/c", "echo"]),
    lambda: socket.create_connection(("93.184.216.34", 80), timeout=1),
    lambda: open("../../outside.txt", "w"),
):
    try:
        attempt()
    except PermissionError:
        continue
    raise RuntimeError("not blocked")
raise RuntimeError("all blocked")
"""
    result = functional.run_checks({"backend/app/main.py": hostile})
    assert result is not None
    assert "all blocked" in str(result["app starts, health works"])
