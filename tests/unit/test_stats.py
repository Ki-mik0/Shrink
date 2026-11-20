from __future__ import annotations

import io
import sys

import pytest
from click.testing import CliRunner

from shrink.cli import cli
from shrink.compressors.base import Step
from shrink.document import Document
from shrink.observability.stats import (
    ApproxCounter,
    StatsReporter,
    TiktokenCounter,
    format_change,
    make_counter,
)


@pytest.mark.parametrize(("text", "tokens"), [("", 0), ("a", 1), ("abcd", 1), ("abcde", 2)])
def test_approx_counter(text: str, tokens: int) -> None:
    assert ApproxCounter().count(text) == tokens


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [(0, 0, "0.0%"), (100, 65, "-35.0%"), (100, 100, "+0.0%"), (3, 4, "+33.3%")],
)
def test_format_change(before: int, after: int, expected: str) -> None:
    assert format_change(before, after) == expected


def test_reporter_summary_marks_estimates() -> None:
    out = io.StringIO()
    reporter = StatsReporter(ApproxCounter(), stream=out)
    reporter.record(Document("a" * 400), Document("a" * 100))
    reporter.finish()
    assert out.getvalue() == ("shrink: stats: 1 document, tokens ~100 -> ~25 (-75.0%) [approx]\n")


def test_reporter_plural_and_exact_counter() -> None:
    class Exact:
        name = "fake"
        approximate = False

        def count(self, text: str) -> int:
            return len(text)

    reporter = StatsReporter(Exact())
    reporter.record(Document("ab"), Document("a"))
    reporter.record(Document("cd"), Document("c"))
    assert reporter.summary() == "shrink: stats: 2 documents, tokens 4 -> 2 (-50.0%) [fake]"


def test_make_counter_falls_back_without_tiktoken(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)  # makes `import tiktoken` fail
    assert isinstance(make_counter(), ApproxCounter)


def test_tiktoken_counter_when_installed() -> None:
    pytest.importorskip("tiktoken")
    try:
        counter = TiktokenCounter()
    except Exception as exc:  # vocabulary download blocked
        pytest.skip(f"tiktoken vocabulary unavailable: {exc}")
    assert counter.count("hello world") == 2
    assert counter.count("bad \udcff byte") > 0


def test_stats_flag_writes_only_to_stderr(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    result = runner.invoke(cli, ["--stats", "--level", "1"], input="hello world\n")
    assert result.exit_code == 0
    assert result.stdout == "hello world\n"
    assert result.stderr.startswith("shrink: stats: 1 document, tokens ~3 -> ~3")


def test_stats_count_the_framing_too() -> None:
    reporter = StatsReporter(ApproxCounter())
    before = Document("x" * 40, head="h" * 40, tail="t" * 40)
    reporter.record(before, before.with_text(""))
    assert (reporter.tokens_before, reporter.tokens_after) == (30, 20)


def test_reporter_sums_steps_over_documents_and_lines_them_up() -> None:
    out = io.StringIO()
    reporter = StatsReporter(ApproxCounter(), stream=out)
    for _ in range(2):
        before = Document("x" * 400)
        steps = [
            Step("whitespace", Document("x" * 400)),
            Step("dedup", Document("x" * 200)),
            Step("markup", Document("x" * 100)),
        ]
        reporter.record(before, steps[-1].after, steps)
    reporter.finish()
    lines = out.getvalue().splitlines()
    assert lines[0] == "shrink: stats: 2 documents, tokens ~200 -> ~50 (-75.0%) [approx]"
    assert lines[1] == "shrink:   whitespace  ~200 -> ~200 (+0.0%, +0.0% of input)"
    assert lines[2] == "shrink:   dedup       ~200 -> ~100 (-50.0%, -50.0% of input)"
    assert lines[3] == "shrink:   markup      ~100 -> ~50 (-50.0%, -25.0% of input)"
    assert reporter.tokens_after == 50


def test_stats_flag_prints_a_line_per_step(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    result = runner.invoke(cli, ["--stats", "--level", "3"], input="a   b\n\n\n<b>x</b>\n")
    assert result.exit_code == 0
    lines = result.stderr.splitlines()
    assert [line.split()[1] for line in lines[1:]] == [
        "whitespace",
        "spaces+duplicates",
        "markup",
    ]
