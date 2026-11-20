from __future__ import annotations

import pytest
from click.testing import CliRunner


@pytest.fixture
def runner() -> CliRunner:
    """CliRunner with stderr kept apart from stdout (Click 8.1 and 8.2+)."""
    try:
        return CliRunner(mix_stderr=False)  # type: ignore[call-arg]
    except TypeError:  # Click >= 8.2 always separates the streams
        return CliRunner()
