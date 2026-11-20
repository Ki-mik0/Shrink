from __future__ import annotations

import io

from click.testing import CliRunner

from shrink.cli import cli
from shrink.compressors.base import Step
from shrink.document import Document
from shrink.observability import diff as diff_module
from shrink.observability.diff import DiffReporter
from shrink.observability.stats import ApproxCounter, StatsReporter


def make_reporter() -> tuple[DiffReporter, io.StringIO, io.StringIO]:
    diff_out = io.StringIO()
    stats_out = io.StringIO()
    return DiffReporter(StatsReporter(ApproxCounter(), stats_out), diff_out), diff_out, stats_out


def test_diff_is_a_raw_unified_diff_of_the_text() -> None:
    reporter, out, _ = make_reporter()
    reporter.record(Document("a   b\n\n\nc\n", path="src/a.md"), Document("a b\n\nc\n"))
    assert out.getvalue() == (
        "--- src/a.md (before)\n+++ src/a.md (after)\n@@ -1,3 +1,2 @@\n-a   b\n-\n+a b\n \n"
    )


def test_diff_leaves_the_framing_out() -> None:
    reporter, out, _ = make_reporter()
    before = Document("x  \n", path="a.md", head="a.md\n---\n", tail="---\n")
    reporter.record(before, before.with_text("x\n"))
    assert "a.md\n---" not in out.getvalue().split("@@", 1)[1]
    assert "-x  \n+x\n" in out.getvalue()


def test_unchanged_document_is_reported_as_such() -> None:
    reporter, out, _ = make_reporter()
    reporter.record(Document("same\n", path="a.md"), Document("same\n", path="a.md"))
    assert out.getvalue() == "shrink: a.md: no changes\n"


def test_huge_document_skips_the_diff() -> None:
    reporter, out, _ = make_reporter()
    big = "x\n" * (diff_module.MAX_DIFF_CHARS // 2 + 1)
    reporter.record(Document(big), Document(big[:-2]))
    assert out.getvalue() == (
        f"shrink: <stdin>: diff skipped (larger than {diff_module.MAX_DIFF_CHARS} characters)\n"
    )


def test_reporter_forwards_records_and_finish_to_the_stats() -> None:
    reporter, _, stats_out = make_reporter()
    steps = [Step("only", Document("x" * 40))]
    reporter.record(Document("x" * 80), steps[-1].after, steps)
    reporter.finish()
    lines = stats_out.getvalue().splitlines()
    assert lines[0] == "shrink: stats: 1 document, tokens ~20 -> ~10 (-50.0%) [approx]"
    assert lines[1].startswith("shrink:   only")


def test_diff_flag_prints_diff_then_stats_on_stderr_only(runner: CliRunner) -> None:
    result = runner.invoke(cli, ["--diff", "--level", "1"], input="a   b  \n\n\n\nc\n")
    assert result.exit_code == 0
    assert result.stdout == "a   b\n\nc\n"
    assert "--- <stdin> (before)" in result.stderr
    assert "@@ -1,4 +1,2 @@\n-a   b  \n-\n-\n+a   b\n \n" in result.stderr
    assert result.stderr.index("--- <stdin>") < result.stderr.index("shrink: stats:")


def test_diff_on_packed_input_has_one_diff_per_file(runner: CliRunner) -> None:
    packed = "a.md\n---\nx   y\n\n---\nb.md\n---\nz   w\n\n---\n"
    result = runner.invoke(cli, ["--diff", "--level", "2"], input=packed)
    assert result.exit_code == 0
    assert "--- a.md (before)" in result.stderr
    assert "--- b.md (before)" in result.stderr
    assert "2 documents" in result.stderr
