from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from shrink.paths import cache_dir

# The user's real cache, remembered before tests redirect it (for opt-in tests).
REAL_CACHE_DIR = cache_dir()


@pytest.fixture
def real_cache_dir() -> Path:
    return REAL_CACHE_DIR


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Tests never read or write the user's downloads: counting is the estimate."""
    folder = tmp_path / "shrink-cache"
    monkeypatch.setenv("SHRINK_CACHE_DIR", str(folder))
    monkeypatch.setenv("SHRINK_STATE_DIR", str(tmp_path / "shrink-state"))
    return folder


@pytest.fixture
def runner() -> CliRunner:
    """CliRunner with stderr kept apart from stdout (Click 8.1 and 8.2+)."""
    try:
        return CliRunner(mix_stderr=False)  # type: ignore[call-arg]
    except TypeError:  # Click >= 8.2 always separates the streams
        return CliRunner()
