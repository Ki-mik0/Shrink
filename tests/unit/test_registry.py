from __future__ import annotations

import pytest

from shrink.compressors.base import check_level
from shrink.compressors.registry import ModeSpec, Registry, UnknownModeError, default_registry


def test_default_registry_has_text_mode() -> None:
    assert "text" in default_registry.names()
    compressor = default_registry.create("text")
    assert compressor.name == "text"
    assert compressor.summary
    assert compressor.help


def test_unknown_mode() -> None:
    with pytest.raises(UnknownModeError, match="unknown mode 'nope'"):
        default_registry.create("nope")


def test_duplicate_registration_rejected() -> None:
    spec = ModeSpec("x", "shrink.compressors.text:TextCompressor", "x")
    registry = Registry([spec])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec)


def test_names_and_specs_are_sorted() -> None:
    registry = Registry(
        [
            ModeSpec("b", "m:B", "b"),
            ModeSpec("a", "m:A", "a"),
        ]
    )
    assert registry.names() == ["a", "b"]
    assert [s.name for s in registry.specs()] == ["a", "b"]


def test_empty_registry_error_message() -> None:
    with pytest.raises(UnknownModeError, match="known: none"):
        Registry().spec("x")


@pytest.mark.parametrize("level", [0, 4, -1])
def test_check_level_rejects_out_of_range(level: int) -> None:
    with pytest.raises(ValueError, match="level must be between"):
        check_level(level)
