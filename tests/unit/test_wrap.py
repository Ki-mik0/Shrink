from __future__ import annotations

import http.client
import json
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from shrink.cli import main
from shrink.integrations import commands
from shrink.integrations.adapters import api_of, rewrite_request
from shrink.integrations.gain import GainLog, gain_path, load, report, summary_json
from shrink.integrations.proxy import ShrinkProxy
from shrink.integrations.shrinker import RequestShrinker, Saving

ESC = chr(27)
NOISY = "".join(f"{ESC}[32mPASSED{ESC}[0m test_{i % 3}\n" for i in range(30))


def upper(text: str, kind: str, tool: str | None) -> str:
    return text.upper()


# ------------------------------------------------------------------ adapters


@pytest.mark.parametrize(
    ("path", "api"),
    [
        ("/v1/messages?beta=true", "anthropic"),
        ("/v1/messages/count_tokens", "anthropic"),
        ("/v1/responses", "openai-responses"),
        ("/v1/chat/completions", "openai-chat"),
        ("/v1/models", None),
        ("/api/hello", None),
    ],
)
def test_api_of(path: str, api: str | None) -> None:
    assert api_of(path) == api


def anthropic_body() -> dict[str, Any]:
    return {
        "system": [
            {"type": "text", "text": "system prompt", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                    {"type": "tool_use", "id": "t2", "name": "Read", "input": {"path": "a.py"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "bash out"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "t2",
                        "content": [{"type": "text", "text": "file text"}],
                    },
                    {"type": "text", "text": "next", "cache_control": {"type": "ephemeral"}},
                ],
            },
        ],
    }


def test_anthropic_rewrites_tool_output_and_user_text_only() -> None:
    body = anthropic_body()
    assert rewrite_request("anthropic", body, upper)
    assert body["system"][0]["text"] == "system prompt"
    assert body["messages"][0]["content"] == "HELLO"
    assistant = body["messages"][1]["content"]
    assert assistant[0] == {"type": "thinking", "thinking": "hmm", "signature": "sig"}
    results = body["messages"][2]["content"]
    assert results[0]["content"] == "BASH OUT"
    assert results[1]["content"] == [{"type": "text", "text": "file text"}]  # Read is skipped
    assert results[2] == {"type": "text", "text": "NEXT", "cache_control": {"type": "ephemeral"}}


def test_no_change_reports_false() -> None:
    body = anthropic_body()
    assert not rewrite_request("anthropic", body, lambda text, kind, tool: text)
    assert body == anthropic_body()


def test_openai_responses() -> None:
    body: dict[str, Any] = {
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "out"},
            {"type": "function_call", "call_id": "c2", "name": "read_file", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c2", "output": "file"},
        ]
    }
    assert rewrite_request("openai-responses", body, upper)
    assert body["input"][0]["content"][0]["text"] == "HI"
    assert body["input"][2]["output"] == "OUT"
    assert body["input"][4]["output"] == "file"
    plain: dict[str, Any] = {"input": "question"}
    assert rewrite_request("openai-responses", plain, upper)
    assert plain["input"] == "QUESTION"


def test_openai_chat() -> None:
    body: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "x", "function": {"name": "run"}}]},
            {"role": "tool", "tool_call_id": "x", "content": "out"},
        ]
    }
    assert rewrite_request("openai-chat", body, upper)
    assert [m.get("content") for m in body["messages"]] == ["sys", "HI", None, "OUT"]


@pytest.mark.parametrize("api", ["anthropic", "openai-responses", "openai-chat"])
def test_odd_bodies_are_left_alone(api: str) -> None:
    for body in ({}, {"messages": "x", "input": 3}, {"messages": [1, {"role": "user"}]}):
        assert not rewrite_request(api, dict(body), upper)


# ------------------------------------------------------------------ shrinker


def request(body: dict[str, Any]) -> bytes:
    return json.dumps(body).encode()


def bash_body(output: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Bash"}]},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "t", "content": output}],
            },
        ]
    }


def test_shrinker_compresses_tool_output_and_measures_it() -> None:
    shrinker = RequestShrinker(level=1, count=len)
    out, saving = shrinker.process("/v1/messages", request(bash_body(NOISY)))
    assert saving is not None
    assert saving.blocks == 1
    assert saving.tokens_after < saving.tokens_before
    sent = json.loads(out)["messages"][1]["content"][0]["content"]
    assert ESC not in sent


def test_unchanged_requests_keep_their_exact_bytes() -> None:
    shrinker = RequestShrinker()
    original = b'{"messages": [{"role": "user", "content": "hi"}],  "stream": true}'
    assert shrinker.process("/v1/messages", original) == (original, None)
    assert shrinker.process("/v1/models", b"x") == (b"x", None)
    assert shrinker.process("/v1/messages", b"not json") == (b"not json", None)
    assert shrinker.process("/v1/messages", b"[1]") == (b"[1]", None)
    assert shrinker.process("/v1/messages", request(bash_body(NOISY)), "gzip")[1] is None


def test_repeated_text_is_compressed_identically_and_once() -> None:
    calls: list[str] = []

    def count(text: str) -> int:
        calls.append(text)
        return len(text)

    shrinker = RequestShrinker(count=count)
    first, _ = shrinker.process("/v1/messages", request(bash_body(NOISY)))
    second, _ = shrinker.process("/v1/messages", request(bash_body(NOISY)))
    assert first == second
    assert len(calls) == 2  # before and after, counted once


def test_compression_failure_keeps_the_original(monkeypatch: pytest.MonkeyPatch) -> None:
    from shrink.integrations import shrinker as module

    def boom(text: str, level: int) -> str:
        raise RuntimeError("bad")

    monkeypatch.setattr(module, "_compress_tool_output", boom)
    body = request(bash_body(NOISY))
    assert RequestShrinker().process("/v1/messages", body) == (body, None)


def test_user_text_is_only_translated() -> None:
    class Upper:
        name = "upper"
        summary = ""
        help = ""

        def compress(self, doc, level):  # type: ignore[no-untyped-def]
            return doc.with_text(doc.text.upper())

    body = request({"messages": [{"role": "user", "content": "hello   there"}]})
    assert RequestShrinker().process("/v1/messages", body)[1] is None
    out, saving = RequestShrinker(translate=Upper()).process("/v1/messages", body)
    assert json.loads(out)["messages"][0]["content"] == "HELLO   THERE"
    assert saving is not None


def test_json_tool_output_is_minified() -> None:
    output = json.dumps({"items": list(range(5))}, indent=4)
    out, _ = RequestShrinker().process("/v1/messages", request(bash_body(output)))
    assert json.loads(out)["messages"][1]["content"][0]["content"] == '{"items":[0,1,2,3,4]}'


# ------------------------------------------------------------------ fake upstream + proxy


class Upstream:
    """A local stand-in for the model API that records what it receives."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                owner.requests.append(
                    {"method": "GET", "path": self.path, "headers": dict(self.headers)}
                )
                body = b'{"data": []}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_HEAD(self) -> None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length)
                owner.requests.append(
                    {
                        "method": "POST",
                        "path": self.path,
                        "headers": dict(self.headers),
                        "body": body,
                    }
                )
                if "error" in self.path:
                    payload = b'{"type":"error","error":{"type":"invalid_request_error"}}'
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("x-should-retry", "false")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for event in ("event: ping\ndata: {}\n\n", "event: message_stop\ndata: {}\n\n"):
                    data = event.encode()
                    self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                    self.wfile.flush()
                    time.sleep(0.05)
                self.wfile.write(b"0\r\n\r\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def upstream() -> Iterator[Upstream]:
    server = Upstream()
    yield server
    server.close()


@pytest.fixture
def proxy(upstream: Upstream) -> Iterator[tuple[ShrinkProxy, list[Saving]]]:
    savings: list[Saving] = []
    server = ShrinkProxy(upstream.url, RequestShrinker(count=len), on_saving=savings.append)
    server.start()
    yield server, savings
    server.stop()


def call(
    url: str,
    method: str,
    path: str,
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[http.client.HTTPResponse, bytes]:
    port = int(url.rsplit(":", 1)[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response, data


def test_proxy_forwards_shrunk_body_and_streams_events(
    upstream: Upstream, proxy: tuple[ShrinkProxy, list[Saving]]
) -> None:
    server, savings = proxy
    headers = {
        "x-api-key": "test-key",
        "anthropic-version": "2023-06-01",
        "anthropic-beta": "oauth-2025-04-20,other",
        "content-type": "application/json",
    }
    response, data = call(
        server.url, "POST", "/v1/messages?beta=true", request(bash_body(NOISY)), headers
    )
    assert response.status == 200
    assert response.getheader("Content-Type") == "text/event-stream"
    assert data == b"event: ping\ndata: {}\n\nevent: message_stop\ndata: {}\n\n"
    seen = upstream.requests[-1]
    assert seen["path"] == "/v1/messages?beta=true"
    assert seen["headers"]["x-api-key"] == "test-key"
    assert seen["headers"]["anthropic-beta"] == "oauth-2025-04-20,other"
    assert ESC not in seen["body"].decode()
    assert len(savings) == 1


def test_proxy_passes_other_requests_and_errors_unchanged(
    upstream: Upstream, proxy: tuple[ShrinkProxy, list[Saving]]
) -> None:
    server, savings = proxy
    response, data = call(server.url, "GET", "/v1/models?limit=1000")
    assert (response.status, data) == (200, b'{"data": []}')
    response, data = call(server.url, "POST", "/v1/messages/error", b"{}")
    assert response.status == 400
    assert response.getheader("x-should-retry") == "false"
    assert b"invalid_request_error" in data
    response, _ = call(server.url, "HEAD", "/api/hello")
    assert response.status == 404
    assert savings == []


def test_proxy_accepts_chunked_requests(
    upstream: Upstream, proxy: tuple[ShrinkProxy, list[Saving]]
) -> None:
    server, _ = proxy
    body = request(bash_body(NOISY))
    port = int(server.url.rsplit(":", 1)[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    connection.request(
        "POST", "/v1/messages", body=iter([body[:10], body[10:]]), encode_chunked=True
    )
    assert connection.getresponse().status == 200
    connection.close()
    assert ESC not in upstream.requests[-1]["body"].decode()


def test_proxy_reports_an_unreachable_upstream() -> None:
    server = ShrinkProxy("http://127.0.0.1:9", RequestShrinker())
    server.start()
    try:
        response, data = call(server.url, "POST", "/v1/messages", b"{}")
    finally:
        server.stop()
    assert response.status == 502
    assert json.loads(data)["type"] == "error"


def test_proxy_rejects_a_bad_upstream() -> None:
    with pytest.raises(ValueError, match="http"):
        ShrinkProxy("ftp://example.com", RequestShrinker())


def test_upstream_with_a_path_prefix() -> None:
    server = ShrinkProxy("https://gateway.example.com/anthropic/", RequestShrinker())
    try:
        assert server.target("/v1/messages") == "/anthropic/v1/messages"
        assert server.upstream_host() == "gateway.example.com"
    finally:
        server.stop()


# ------------------------------------------------------------------ gain


def test_gain_log_and_report() -> None:
    log = GainLog(agent="claude")
    log.record(Saving("anthropic", 2, 1000, 400))
    log.record(Saving("anthropic", 1, 200, 100))
    records = load()
    assert len(records) == 2
    assert records[0]["agent"] == "claude"
    assert "text" not in json.dumps(records)
    lines = report(records)
    assert lines[0] == (
        "shrink gain: 2 requests, changed text 1200 -> 500 tokens (-58.3%), saved 700 tokens"
    )
    assert summary_json(records)["tokens_saved"] == 700


def test_gain_skips_damaged_lines_and_handles_no_log() -> None:
    assert report(load()) == [
        "shrink gain: nothing recorded yet; run an agent through `shrink wrap`"
    ]
    path = gain_path()
    path.parent.mkdir(parents=True)
    path.write_text('not json\n[1]\n{"tokens_before": 5, "tokens_after": 1}\n', encoding="utf-8")
    assert len(load()) == 1


def test_gain_command(runner: CliRunner) -> None:
    GainLog(agent="codex").record(Saving("openai-responses", 1, 10, 4))
    result = runner.invoke(commands.gain, ["--json"])
    assert json.loads(result.stdout)["tokens_saved"] == 6
    result = runner.invoke(commands.gain, [])
    assert "codex" in result.stdout
    result = runner.invoke(commands.gain, ["--reset"])
    assert not gain_path().exists()


# ------------------------------------------------------------------ shrink wrap


CHILD = """
import json, os, urllib.request
base = os.environ["ANTHROPIC_BASE_URL"]
noisy = "".join(chr(27) + "[32mPASSED" + chr(27) + "[0m t\\n" for _ in range(30))
body = {"messages": [
    {"role": "assistant", "content": [{"type": "tool_use", "id": "t", "name": "Bash"}]},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": noisy}]},
]}
req = urllib.request.Request(base + "/v1/messages", data=json.dumps(body).encode(),
                             headers={"content-type": "application/json"})
print(urllib.request.urlopen(req).read().decode().count("event:"))
raise SystemExit(3)
"""


def test_wrap_runs_the_agent_through_the_proxy(
    upstream: Upstream, capfd: pytest.CaptureFixture[str]
) -> None:
    code = main(
        ["wrap", "--api", "anthropic", "--upstream", upstream.url, sys.executable, "-c", CHILD]
    )
    assert code == 3  # the agent's exit code
    out, err = capfd.readouterr()
    assert out.strip() == "2"
    assert "session: 1 request(s) shortened" in err
    assert ESC not in upstream.requests[-1]["body"].decode()
    assert load()[0]["agent"] == Path(sys.executable).stem


def test_wrap_needs_to_know_the_api_of_unknown_commands(capfd: pytest.CaptureFixture[str]) -> None:
    assert main(["wrap", sys.executable, "-c", "pass"]) == 2
    assert "--api" in capfd.readouterr().err


def test_wrap_unknown_command(capfd: pytest.CaptureFixture[str]) -> None:
    assert main(["wrap", "--api", "openai", "no-such-agent-xyz"]) == 2
    assert "command not found" in capfd.readouterr().err


def test_agent_presets_and_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    assert commands._agent_api(("claude",), None) == "anthropic"
    assert commands._agent_api(("C:/bin/codex.cmd",), None) == "openai"
    assert commands._agent_api((), None) == "anthropic"
    spec = commands.APIS["openai"]
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    assert commands._upstream(spec, None) == "https://api.openai.com"
    monkeypatch.setenv("OPENAI_BASE_URL", "https://gw.example/v1")
    assert commands._upstream(spec, None) == "https://gw.example"
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:5555/v1")
    assert commands._upstream(spec, None) == "https://api.openai.com"
    assert commands._upstream(spec, "https://x/") == "https://x"


def test_wrap_bad_port_is_reported(capfd: pytest.CaptureFixture[str]) -> None:
    blocker = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
    port = blocker.server_address[1]
    try:
        assert main(["wrap", "--port", str(port), "--api", "anthropic", sys.executable]) == 2
    finally:
        blocker.server_close()
    assert "cannot start the proxy" in capfd.readouterr().err
