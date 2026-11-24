from __future__ import annotations

import io
import json
import sys
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from shrink import paths
from shrink.cli import cli
from shrink.document import Document
from shrink.observability import stats
from shrink.observability.stats import (
    ApproxCounter,
    Price,
    StatsReporter,
    TiktokenCounter,
    TokenizerMissingError,
    download_tokenizer,
    make_counter,
)


class FakeEncoding:
    def encode_ordinary(self, text: str) -> list[int]:
        return list(range(len(text.split())))


def install_fake_tiktoken(monkeypatch: pytest.MonkeyPatch, seen: list[str]) -> None:
    module = types.ModuleType("tiktoken")

    def get_encoding(name: str) -> FakeEncoding:
        import os

        seen.append(os.environ["TIKTOKEN_CACHE_DIR"])
        return FakeEncoding()

    module.get_encoding = get_encoding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tiktoken", module)


# ------------------------------------------------------------------ cache location


def test_cache_dir_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SHRINK_CACHE_DIR", str(tmp_path / "x"))
    assert paths.cache_dir() == tmp_path / "x"


@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        ("win32", {"LOCALAPPDATA": "C:/L"}, Path("C:/L/shrink/cache")),
        ("linux", {"XDG_CACHE_HOME": "/xdg"}, Path("/xdg/shrink")),
    ],
)
def test_cache_dir_defaults(
    monkeypatch: pytest.MonkeyPatch, platform: str, env: dict[str, str], expected: Path
) -> None:
    monkeypatch.delenv("SHRINK_CACHE_DIR")
    monkeypatch.setattr(paths.sys, "platform", platform)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert paths.cache_dir() == expected


def test_cache_dir_on_macos(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHRINK_CACHE_DIR")
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    assert paths.cache_dir() == Path.home() / "Library" / "Caches" / "shrink"


# ------------------------------------------------------------------ counters


def test_counter_without_downloaded_vocabulary_is_the_estimate_with_a_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_tiktoken(monkeypatch, [])
    with pytest.raises(TokenizerMissingError):
        TiktokenCounter()
    counter = make_counter()
    assert isinstance(counter, ApproxCounter)
    assert counter.hint == "for exact counts, once: shrink --download tokenizer"
    other = make_counter("cl100k_base")
    assert other.hint is not None
    assert other.hint.endswith("--encoding cl100k_base")


def test_counter_without_tiktoken_suggests_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    counter = make_counter()
    assert isinstance(counter, ApproxCounter)
    assert counter.hint is not None
    assert "shrink[tokens]" in counter.hint


def test_download_then_count_offline_from_the_shrink_cache(
    monkeypatch: pytest.MonkeyPatch, isolated_cache: Path
) -> None:
    seen: list[str] = []
    install_fake_tiktoken(monkeypatch, seen)
    monkeypatch.delenv("TIKTOKEN_CACHE_DIR", raising=False)

    folder = download_tokenizer()
    assert folder == isolated_cache / "tiktoken"
    assert (folder / "o200k_base.ready").is_file()

    counter = make_counter()
    assert isinstance(counter, TiktokenCounter)
    assert counter.count("one two three") == 3
    assert seen == [str(folder), str(folder)]
    assert "TIKTOKEN_CACHE_DIR" not in __import__("os").environ  # restored afterwards


def test_download_without_tiktoken_explains(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(RuntimeError, match="pip install"):
        download_tokenizer()


def test_damaged_cache_falls_back_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("tiktoken")

    def broken(name: str) -> None:
        raise ValueError("hash mismatch")

    module.get_encoding = broken  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tiktoken", module)
    marker = stats.tokenizer_dir() / "o200k_base.ready"
    marker.parent.mkdir(parents=True)
    marker.write_text("ok")
    counter = make_counter()
    assert isinstance(counter, ApproxCounter)
    assert counter.hint is None


def test_real_tiktoken_from_the_users_cache(
    monkeypatch: pytest.MonkeyPatch, real_cache_dir: Path
) -> None:
    pytest.importorskip("tiktoken")
    if not (real_cache_dir / "tiktoken" / "o200k_base.ready").is_file():
        pytest.skip("o200k_base not downloaded (shrink --download tokenizer)")
    monkeypatch.setenv("SHRINK_CACHE_DIR", str(real_cache_dir))
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")  # any network use would fail
    counter = TiktokenCounter()
    assert counter.count("hello world") == 2
    assert counter.count("<|endoftext|>") > 1  # plain text, not a special token
    assert counter.count("bad \udcff byte") > 0


# ------------------------------------------------------------------ price and cost


@pytest.mark.parametrize(
    ("spec", "hit", "tokens", "expected"),
    [
        ("3", 0.0, 1_000_000, 3.0),
        ("3/0.3", 0.5, 1_000_000, 1.65),
        ("3/0.3", 1.0, 2_000_000, 0.6),
    ],
)
def test_price_cost(spec: str, hit: float, tokens: int, expected: float) -> None:
    assert Price.parse(spec, hit).cost(tokens) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("spec", "hit", "message"),
    [
        ("abc", 0.0, "could not convert"),
        ("-1", 0.0, "negative"),
        ("3", 0.5, "needs a cached price"),
        ("3/0.3", 1.5, "between 0 and 1"),
    ],
)
def test_price_errors(spec: str, hit: float, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Price.parse(spec, hit)


def make_report(price: Price | None = None, fmt: str = "text") -> str:
    out = io.StringIO()
    reporter = StatsReporter(ApproxCounter("hint here"), out, fmt=fmt, price=price)
    reporter.record(Document("x" * 4_000_000), Document("x" * 1_000_000))
    reporter.finish()
    return out.getvalue()


def test_text_report_has_cost_and_hint() -> None:
    lines = make_report(Price.parse("3")).splitlines()
    assert lines[0].startswith("shrink: stats: 1 document, tokens ~1000000 -> ~250000")
    assert lines[1] == (
        "shrink: cost: $3.00 -> $0.7500 per call, saves $2.25 ($2250.00 per 1000 calls) at $3/Mtok"
    )
    assert lines[2] == "shrink: hint here"


def test_json_report() -> None:
    data = json.loads(make_report(Price.parse("3/0.3", 0.5), fmt="json"))
    assert data["tokens_before"] == 1_000_000
    assert data["tokens_saved"] == 750_000
    assert data["saved_percent"] == 75.0
    assert data["approximate"] is True
    assert data["cost_usd"]["saved"] == pytest.approx(1.2375)
    assert data["cost_usd"]["cache_hit_rate"] == 0.5


def test_json_report_of_empty_input() -> None:
    out = io.StringIO()
    StatsReporter(ApproxCounter(), out, fmt="json").finish()
    data = json.loads(out.getvalue())
    assert data["saved_percent"] == 0.0
    assert "cost_usd" not in data


# ------------------------------------------------------------------ CLI


def test_cli_stats_json(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--stats=json", "-l", "2"], input="a   b\n\n\n\nc\n")
    assert result.exit_code == 0
    data = json.loads(result.stderr)
    assert data["documents"] == 1
    assert [s["name"] for s in data["steps"]] == ["whitespace", "spaces+duplicates"]


def test_cli_price_alone_prints_the_report(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--price", "3"], input="a   b\n")
    assert result.exit_code == 0
    assert "shrink: cost:" in result.stderr


def test_cli_cache_hit_without_price_is_a_usage_error(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--cache-hit", "0.5"], input="x")
    assert result.exit_code == 2
    assert "--cache-hit needs --price" in result.stderr


def test_cli_bad_price_is_a_usage_error(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--price", "3", "--cache-hit", "0.5"], input="x")
    assert result.exit_code == 2
    assert "needs a cached price" in result.stderr


def test_cli_download(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_tiktoken(monkeypatch, [])
    result = runner.invoke(cli, ["--download", "tokenizer"])
    assert result.exit_code == 0
    assert "tokenizer ready in" in result.stderr
    assert result.stdout == ""


def test_cli_download_failure_exits_1(runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    result = runner.invoke(cli, ["--download", "tokenizer"])
    assert result.exit_code == 1
    assert "download failed" in result.stderr
