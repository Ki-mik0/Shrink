"""The eval scripts themselves (no network: the model is replaced by a fake)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

import run_quality


@pytest.mark.parametrize(
    ("answer", "expect", "ok"),
    [
        ("70", [["70"]], True),
        ("The default is `70`.", [["70"]], True),
        ("unknown", [["70"]], False),
        ("", [["70"]], False),
        ("re.VERBOSE | re.MULTILINE", [["verbose"], ["multiline"], ["dotall"]], False),
        ("VERBOSE, MULTILINE, DOTALL", [["verbose"], ["multiline"], ["dotall"]], True),
        ('""', [["''"]], True),
    ],
)
def test_correct(answer: str, expect: list[list[str]], ok: bool) -> None:
    assert run_quality.correct(answer, expect) is ok


def test_parse_answers_finds_json_in_a_fence() -> None:
    reply = 'Here you go:\n```json\n{"1": "70", "2": 3}\n```'
    assert run_quality.parse_answers(reply) == {"1": "70", "2": "3"}
    assert run_quality.parse_answers("no json") == {}
    assert run_quality.parse_answers("{broken") == {}


def test_contexts_shrink_and_query_keeps_bodies() -> None:
    corpus = next(c for c in run_quality.load_corpora() if c.name == "textproc")
    sizes = {v: len(run_quality.build_context(corpus, v)) for v in run_quality.VARIANTS}
    assert sizes["original"] > sizes["L1"] > sizes["L2"] > sizes["L3"]
    assert sizes["L3"] < sizes["L3+query"] < sizes["original"]
    assert "normcase" in run_quality.build_context(corpus, "L3+query")
    assert "normcase" not in run_quality.build_context(corpus, "L3")


def test_questions_are_answerable_from_the_original() -> None:
    for corpus in run_quality.load_corpora():
        context = run_quality.build_context(corpus, "original").lower()
        for question in corpus.questions:
            first = question["expect"][0][0].lower()
            assert first.strip("'") in context or question["id"] in {"a4"}, question["id"]


def test_run_saves_answers_resumes_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_quality, "RESULTS", tmp_path)
    monkeypatch.setattr(run_quality, "PAUSE", 0)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    calls: list[str] = []

    def fake_ask(model: str, prompt: str, key: str) -> dict[str, Any]:
        calls.append(prompt)
        answers = {str(i): "70" for i in range(1, 9)}
        return {"choices": [{"message": {"content": json.dumps(answers)}}], "usage": {}}

    monkeypatch.setattr(run_quality, "ask", fake_ask)
    run_quality.run("fake/model", 1, dry_run=False)
    assert len(calls) == 15  # 3 corpora x 5 variants
    run_quality.run("fake/model", 1, dry_run=False)
    assert len(calls) == 15  # everything was saved: nothing asked again
    table = (tmp_path / "quality.md").read_text(encoding="utf-8")
    assert "| original |" in table
    assert "| L3+query |" in table


def test_daily_limit_stops_and_keeps_what_was_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_quality, "RESULTS", tmp_path)
    monkeypatch.setattr(run_quality, "PAUSE", 0)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    answered = []

    def fake_ask(model: str, prompt: str, key: str) -> dict[str, Any]:
        if answered:
            raise run_quality.DailyLimitError("free-models-per-day")
        answered.append(1)
        return {"choices": [{"message": {"content": "{}"}}]}

    monkeypatch.setattr(run_quality, "ask", fake_ask)
    run_quality.run("fake/model", 1, dry_run=False)
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 1


def test_run_needs_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        run_quality.run("fake/model", 1, dry_run=False)


def http_error(code: int, body: str):  # type: ignore[no-untyped-def]
    import io
    import urllib.error

    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(body.encode()))  # type: ignore[arg-type]


def test_busy_model_is_retried_then_stops_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    waits: list[float] = []
    monkeypatch.setattr(run_quality.time, "sleep", waits.append)

    def busy(*args: object, **kwargs: object) -> None:
        raise http_error(429, '{"error": {"message": "temporarily rate-limited upstream"}}')

    monkeypatch.setattr(run_quality.urllib.request, "urlopen", busy)
    with pytest.raises(run_quality.TemporaryLimitError):
        run_quality.ask("m", "p", "k")
    assert waits == run_quality.RETRY_WAITS


def test_daily_limit_and_other_errors_are_told_apart(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_quality.time, "sleep", lambda _: None)
    errors = iter(
        [http_error(429, "Rate limit exceeded: free-models-per-day"), http_error(401, "bad key")]
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise next(errors)

    monkeypatch.setattr(run_quality.urllib.request, "urlopen", fail)
    with pytest.raises(run_quality.DailyLimitError):
        run_quality.ask("m", "p", "k")
    with pytest.raises(RuntimeError, match="HTTP 401"):
        run_quality.ask("m", "p", "k")


def test_busy_model_during_a_run_keeps_answers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(run_quality, "RESULTS", tmp_path)
    monkeypatch.setattr(run_quality, "PAUSE", 0)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    answered: list[int] = []

    def fake_ask(model: str, prompt: str, key: str) -> dict[str, Any]:
        if answered:
            raise run_quality.TemporaryLimitError("busy")
        answered.append(1)
        return {"choices": [{"message": {"content": None}}]}

    monkeypatch.setattr(run_quality, "ask", fake_ask)
    run_quality.run("fake/model", 1, dry_run=False)
    assert len(list((tmp_path / "raw").rglob("*.json"))) == 1
    assert "still busy upstream" in capsys.readouterr().out


def test_main_prints_one_line_for_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*args: object) -> None:
        raise RuntimeError("HTTP 401: bad key")

    monkeypatch.setattr(run_quality, "run", broken)
    monkeypatch.setattr(sys, "argv", ["run_quality.py"])
    with pytest.raises(SystemExit, match="error: HTTP 401"):
        run_quality.main()


def sse(*events: str) -> list[bytes]:
    return [f"{event}\n".encode() for event in events]


def test_read_stream_joins_pieces_and_skips_keep_alives() -> None:
    lines = sse(
        ": OPENROUTER PROCESSING",
        "",
        'data: {"choices": [{"delta": {"content": "{\\"1\\": "}}]}',
        'data: {"choices": [{"delta": {"content": "\\"70\\"}"}, "finish_reason": "stop"}]}',
        'data: {"choices": [], "usage": {"prompt_tokens": 9}}',
        "data: [DONE]",
    )
    reply = run_quality.read_stream(lines)
    assert reply["choices"][0]["message"]["content"] == '{"1": "70"}'
    assert reply["usage"] == {"prompt_tokens": 9}


def test_read_stream_errors_are_temporary() -> None:
    with pytest.raises(run_quality.TemporaryLimitError, match="overloaded"):
        run_quality.read_stream(sse('data: {"error": {"message": "overloaded"}}'))
    with pytest.raises(run_quality.TemporaryLimitError, match="before the answer"):
        run_quality.read_stream(sse('data: {"choices": [{"delta": {"content": "{"}}]}'))


def test_dropped_connection_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    import http.client

    waits: list[float] = []
    monkeypatch.setattr(run_quality.time, "sleep", waits.append)
    outcomes = iter([http.client.IncompleteRead(b"x" * 55), ConnectionResetError("reset")])

    class Response:
        def __enter__(self):  # type: ignore[no-untyped-def]
            return iter(
                sse('data: {"choices": [{"delta": {"content": "{}"}, "finish_reason": "stop"}]}')
            )

        def __exit__(self, *args: object) -> None:
            pass

    def flaky(*args: object, **kwargs: object) -> Response:
        failure = next(outcomes, None)
        if failure is not None:
            raise failure
        return Response()

    monkeypatch.setattr(run_quality.urllib.request, "urlopen", flaky)
    reply = run_quality.ask("m", "p", "k")
    assert reply["choices"][0]["message"]["content"] == "{}"
    assert waits == run_quality.RETRY_WAITS[:2]


def test_redo_moves_a_variants_answers_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_quality, "RESULTS", tmp_path)
    for variant in ("L3", "L3+query"):
        path = run_quality.raw_path("m/x", "json", variant, 1)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    run_quality.archive("m/x", "L3+query")
    folder = run_quality.raw_path("m/x", "x", "x", 0).parent
    assert [p.name for p in folder.glob("*.json")] == ["json--L3--1.json"]
    assert (folder / "previous" / "L3_query" / "json--L3_query--1.json").is_file()
