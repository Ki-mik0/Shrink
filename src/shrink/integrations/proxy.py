"""A local pass-through proxy between an agent and its model API.

The agent is pointed at ``http://127.0.0.1:<port>`` (``ANTHROPIC_BASE_URL`` and
friends). Every request is forwarded to the real API with the same method,
path, query and headers (``anthropic-*``, ``authorization``, ``x-api-key``
included, so API keys and subscription logins both keep working). Only the
JSON body of model requests may change, through :class:`RequestShrinker`.

Responses are streamed back as they arrive, chunk by chunk, so server-sent
events (including keep-alive pings) reach the agent without delay; status,
headers and error bodies are passed on unchanged. Only the standard library
is used. The proxy listens on the loopback interface only.
"""

from __future__ import annotations

import http.client
import json
import socket
import sys
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from shrink.integrations.shrinker import RequestShrinker, Saving

UPSTREAM_TIMEOUT = 900  # seconds; long thinking pauses are covered by upstream pings
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second program bind a port that is already
    # listening; a proxy that sees API keys must own its port exclusively.
    allow_reuse_address = sys.platform != "win32"

    def server_bind(self) -> None:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class ShrinkProxy:
    def __init__(
        self,
        upstream: str,
        shrinker: RequestShrinker,
        *,
        on_saving: Callable[[Saving], None] | None = None,
        port: int = 0,
    ) -> None:
        parts = urlsplit(upstream)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(f"upstream must be an http(s) URL, got {upstream!r}")
        self.upstream = upstream
        self._scheme = parts.scheme
        self._host = parts.hostname
        self._port = parts.port
        self._prefix = parts.path.rstrip("/")
        self._shrinker = shrinker
        self._on_saving = on_saving
        handler = type("Handler", (_Handler,), {"proxy": self})
        self._server = _Server(("127.0.0.1", port), handler)
        self._thread: threading.Thread | None = None
        self._serving = False

    @property
    def url(self) -> str:
        port = self._server.server_address[1]
        return f"http://127.0.0.1:{port}"

    def start(self) -> str:
        self._serving = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self.url

    def serve_forever(self) -> None:
        self._serving = True
        try:
            self._server.serve_forever()
        finally:
            self._serving = False

    def stop(self) -> None:
        # shutdown() waits for the serving loop, so it would hang if none ever ran.
        if self._serving:
            self._server.shutdown()
            self._serving = False
        self._server.server_close()

    def connect(self) -> http.client.HTTPConnection:
        if self._scheme == "https":
            return http.client.HTTPSConnection(self._host, self._port, timeout=UPSTREAM_TIMEOUT)
        return http.client.HTTPConnection(self._host, self._port, timeout=UPSTREAM_TIMEOUT)

    def target(self, path: str) -> str:
        return self._prefix + path

    def upstream_host(self) -> str:
        default = 443 if self._scheme == "https" else 80
        return self._host if self._port in (None, default) else f"{self._host}:{self._port}"

    def shrink(self, path: str, body: bytes, encoding: str) -> bytes:
        new, saving = self._shrinker.process(path, body, encoding)
        if saving is not None and self._on_saving is not None:
            self._on_saving(saving)
        return new


class _Handler(BaseHTTPRequestHandler):
    proxy: ShrinkProxy
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def do_PUT(self) -> None:
        self._forward()

    def do_PATCH(self) -> None:
        self._forward()

    def do_DELETE(self) -> None:
        self._forward()

    def do_HEAD(self) -> None:
        self._forward()

    def do_OPTIONS(self) -> None:
        self._forward()

    def log_message(self, format: str, *args: object) -> None:
        pass  # never print requests: they may contain prompts and keys

    def _read_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            parts = []
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    break
                parts.append(self.rfile.read(size))
                self.rfile.readline()
            return b"".join(parts)
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def _forward(self) -> None:
        proxy = self.proxy
        body = self._read_body()
        if body:
            body = proxy.shrink(self.path, body, self.headers.get("Content-Encoding", ""))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP_BY_HOP}
        headers["Host"] = proxy.upstream_host()
        if body or self.command in ("POST", "PUT", "PATCH"):
            headers["Content-Length"] = str(len(body))
        connection = proxy.connect()
        try:
            connection.request(self.command, proxy.target(self.path), body=body, headers=headers)
            response = connection.getresponse()
        except OSError as exc:
            connection.close()
            self._bad_gateway(exc)
            return
        try:
            self._relay(response)
        finally:
            connection.close()

    def _relay(self, response: http.client.HTTPResponse) -> None:
        self.send_response_only(response.status, response.reason)
        length = response.getheader("Content-Length")
        chunked = length is None and self.command != "HEAD" and response.status not in (204, 304)
        for name, value in response.getheaders():
            if name.lower() not in _HOP_BY_HOP:
                self.send_header(name, value)
        if length is not None:
            self.send_header("Content-Length", length)
        if chunked:
            self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        if self.command == "HEAD":
            return
        while True:
            piece = response.read1(65536)
            if not piece:
                break
            if chunked:
                self.wfile.write(f"{len(piece):X}\r\n".encode() + piece + b"\r\n")
            else:
                self.wfile.write(piece)
            self.wfile.flush()  # each event goes out at once
        if chunked:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    def _bad_gateway(self, exc: OSError) -> None:
        body = json.dumps(
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": f"shrink proxy could not reach {self.proxy.upstream}: {exc}",
                },
            }
        ).encode()
        self.send_response_only(502, "Bad Gateway")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
