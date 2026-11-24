from __future__ import annotations

from collections.abc import Iterator, Sequence

from shrink.compressors.base import Step
from shrink.document import Document
from shrink.pipeline import Pipeline
from shrink.splitting.documents import WholeStreamSplitter


class FakeReader:
    def __init__(self, *lines: str) -> None:
        self._lines = lines

    def lines(self) -> Iterator[str]:
        yield from self._lines


class ListWriter:
    def __init__(self) -> None:
        self.docs: list[Document] = []

    def write(self, doc: Document) -> None:
        self.docs.append(doc)


class UpperCompressor:
    name = "upper"
    summary = "upper-case"
    help = ""

    def __init__(self) -> None:
        self.levels: list[int] = []

    def compress(self, doc: Document, level: int) -> Document:
        self.levels.append(level)
        return doc.with_text(doc.text.upper())


class BrokenCompressor:
    name = "broken"
    summary = ""
    help = ""

    def compress(self, doc: Document, level: int) -> Document:
        raise SyntaxError("bad input")


class RecordingReporter:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []
        self.finished = False

    def record(self, before: Document, after: Document, steps: Sequence[Step] = ()) -> None:
        self.pairs.append((before.text, after.text))

    def finish(self) -> None:
        self.finished = True


def test_pipeline_compresses_and_writes() -> None:
    writer = ListWriter()
    compressor = UpperCompressor()
    reporter = RecordingReporter()
    pipeline = Pipeline(
        FakeReader("ab\n", "c"), WholeStreamSplitter(), compressor, writer, reporter
    )

    assert pipeline.run(level=3) == 1
    assert writer.docs == [Document("AB\nC")]
    assert compressor.levels == [3]
    assert reporter.pairs == [("ab\nc", "AB\nC")]
    assert reporter.finished


def test_pipeline_fails_open() -> None:
    writer = ListWriter()
    warnings: list[str] = []
    pipeline = Pipeline(
        FakeReader("x = (\n"),
        WholeStreamSplitter(),
        BrokenCompressor(),
        writer,
        warn=warnings.append,
    )

    pipeline.run(level=2)

    assert writer.docs == [Document("x = (\n")]
    assert len(warnings) == 1
    assert "<stdin>" in warnings[0]
    assert "SyntaxError: bad input" in warnings[0]
    assert "passed through unchanged" in warnings[0]


def test_empty_input_writes_nothing_but_finishes_stats() -> None:
    writer = ListWriter()
    reporter = RecordingReporter()
    pipeline = Pipeline(FakeReader(), WholeStreamSplitter(), UpperCompressor(), writer, reporter)

    assert pipeline.run(level=1) == 0
    assert writer.docs == []
    assert reporter.finished


class StepReporter(RecordingReporter):
    wants_steps = True

    def __init__(self) -> None:
        super().__init__()
        self.steps: list[list[str]] = []

    def record(self, before: Document, after: Document, steps: Sequence[Step] = ()) -> None:
        super().record(before, after)
        self.steps.append([s.name for s in steps])


class TracedUpper(UpperCompressor):
    def __init__(self) -> None:
        super().__init__()
        self.compress_calls = 0

    def compress(self, doc: Document, level: int) -> Document:
        self.compress_calls += 1
        return super().compress(doc, level)

    def trace(self, doc: Document, level: int) -> list[Step]:
        return [Step("upper", doc.with_text(doc.text.upper()))]


def test_pipeline_uses_trace_only_when_the_reporter_wants_steps() -> None:
    compressor = TracedUpper()
    reporter = StepReporter()
    writer = ListWriter()
    Pipeline(FakeReader("ab\n"), WholeStreamSplitter(), compressor, writer, reporter).run(2)

    assert writer.docs == [Document("AB\n")]
    assert reporter.steps == [["upper"]]
    assert compressor.compress_calls == 0

    plain = TracedUpper()
    Pipeline(
        FakeReader("ab\n"), WholeStreamSplitter(), plain, ListWriter(), RecordingReporter()
    ).run(2)
    assert plain.compress_calls == 1


def test_pipeline_fails_open_when_trace_raises() -> None:
    class BrokenTrace(TracedUpper):
        def trace(self, doc: Document, level: int) -> list[Step]:
            raise ValueError("no trace")

    writer = ListWriter()
    warnings: list[str] = []
    Pipeline(
        FakeReader("ab\n"),
        WholeStreamSplitter(),
        BrokenTrace(),
        writer,
        StepReporter(),
        warn=warnings.append,
    ).run(2)
    assert writer.docs == [Document("ab\n")]
    assert "ValueError: no trace" in warnings[0]
