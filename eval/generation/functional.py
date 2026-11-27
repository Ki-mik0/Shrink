"""Does the generated app work? Behaviour checks for ``eval/generation/task.md``.

The keyword checks in ``run_generation.py`` say that the required files and
endpoints are mentioned. These checks run the answer: the starter project plus
the answer's files is written to a temporary folder and the backend is driven
through FastAPI's ``TestClient``, one check per requirement of the task. One
check opens a database created by the *starter* code with the new code, as a
deployed installation would. The frontend is checked statically.

The code under test was written by a model, so it runs in a separate Python
process (``python functional.py <project folder>``) with a minimal environment
(no API keys) and an audit hook that refuses subprocesses, network connections
other than loopback, and writes outside the temporary folder. This guards
against accidents; it is not a security sandbox.

Needs ``pip install -r eval/requirements-generation.txt``; without it
:func:`run_checks` returns ``None``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

STARTER = Path(__file__).resolve().parent / "starter"
TIMEOUT = 120
REQUIRED_MODULES = ("fastapi", "httpx", "pydantic")
PASSWORD = "correct horse 42"
TODO_KEYS = {"id", "title", "done", "created_at", "updated_at"}

Check = Callable[["Project"], None]
CHECKS: list[tuple[str, Check]] = []


def check(name: str) -> Callable[[Check], Check]:
    def register(function: Check) -> Check:
        CHECKS.append((name, function))
        return function

    return register


def available() -> bool:
    return all(importlib.util.find_spec(name) is not None for name in REQUIRED_MODULES)


# ------------------------------------------------------------------ host side


def run_checks(files: dict[str, str]) -> dict[str, str | None] | None:
    """``{check name: None if it passed, else why not}`` for the starter project
    with ``files`` written over it; ``None`` if FastAPI is not installed."""
    if not available():
        return None
    with tempfile.TemporaryDirectory(prefix="shrink-eval-", ignore_cleanup_errors=True) as tmp:
        project = Path(tmp) / "project"
        shutil.copytree(STARTER, project, ignore=shutil.ignore_patterns("__pycache__"))
        for path, text in files.items():
            target = (project / path).resolve()
            if not target.is_relative_to(project.resolve()):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k.upper() in ("SYSTEMROOT", "PATH", "WINDIR")}
        env.update(TEMP=tmp, TMP=tmp, TMPDIR=tmp)
        try:
            done = subprocess.run(
                [sys.executable, "-I", "-B", str(Path(__file__).resolve()), str(project)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIMEOUT,
                env=env,
                cwd=tmp,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {name: f"timed out after {TIMEOUT} s" for name, _ in CHECKS}
    last = done.stdout.strip().rsplit("\n", 1)[-1] if done.stdout.strip() else ""
    try:
        result: dict[str, str | None] = json.loads(last)
    except json.JSONDecodeError:
        why = (done.stderr.strip().splitlines() or ["the checks did not report"])[-1]
        return {name: why for name, _ in CHECKS}
    return result


# ------------------------------------------------------------------ guard (child)


def _guard(root: Path) -> None:
    inside = os.path.normcase(str(root.resolve()))

    def allowed_path(value: Any) -> bool:
        if not isinstance(value, (str, bytes, os.PathLike)):
            return True  # an already open descriptor
        path = os.fsdecode(value)
        if path in (os.devnull, ":memory:"):
            return True
        return os.path.normcase(os.path.abspath(path)).startswith(inside)

    def hook(event: str, args: tuple[Any, ...]) -> None:
        if event.startswith(("subprocess.", "os.system", "os.exec", "os.spawn", "os.startfile",
                             "os.posix_spawn", "winreg.", "ctypes.dlopen")):  # fmt: skip
            raise PermissionError(f"blocked: {event}")
        if event == "socket.connect":
            address = args[1]
            host = address[0] if isinstance(address, tuple) else ""
            if host not in ("127.0.0.1", "::1", "localhost"):
                raise PermissionError(f"blocked: network connection to {host!r}")
        elif event == "open":
            path, mode, flags = args
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)
            )
            if writing and not allowed_path(path):
                raise PermissionError(f"blocked: writing {path!r}")
        elif event in ("os.remove", "os.rename", "os.rmdir", "shutil.rmtree") and not (
            allowed_path(args[0])
        ):
            raise PermissionError(f"blocked: {event} {args[0]!r}")

    sys.addaudithook(hook)


# ------------------------------------------------------------------ the app (child)


class Project:
    """The project under test: builds apps and talks to them."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._databases = 0

    def database_path(self) -> str:
        self._databases += 1
        return str(self.root.parent / f"test-{self._databases}.sqlite3")

    def client(self, db_path: str | None = None) -> Any:
        from app.main import create_app
        from fastapi.testclient import TestClient

        app = create_app(db_path=db_path or self.database_path(), secret="test-secret")
        return TestClient(app, raise_server_exceptions=True)

    def read(self, path: str) -> str:
        target = self.root / path
        return target.read_text(encoding="utf-8") if target.is_file() else ""


def expect(response: Any, status: int | tuple[int, ...], code: str | None = None) -> Any:
    statuses = (status,) if isinstance(status, int) else status
    body = response.json() if response.content else None
    assert response.status_code in statuses, (
        f"{response.request.method} {response.request.url.path}: status "
        f"{response.status_code}, expected {status}: {response.text[:200]}"
    )
    if code is not None:
        got = (body or {}).get("error", {}).get("code")
        assert got == code, f"error code {got!r}, expected {code!r}"
    return body


def sign_up(client: Any, username: str = "alice") -> dict[str, str]:
    """Register and sign in; the headers that authenticate the user."""
    credentials = {"username": username, "password": PASSWORD}
    expect(client.post("/api/auth/register", json=credentials), 201)
    token = expect(client.post("/api/auth/login", json=credentials), 200)["access_token"]
    return {"Authorization": f"Bearer {token}"}


def add_todo(client: Any, headers: dict[str, str], title: str) -> dict[str, Any]:
    body = expect(client.post("/api/todos", json={"title": title}, headers=headers), 201)
    assert set(body) == TODO_KEYS, f"todo keys {sorted(body)}"
    return dict(body)


# ------------------------------------------------------------------ backend checks


@check("app starts, health works")
def _health(project: Project) -> None:
    body = expect(project.client().get("/api/health"), 200)
    assert body["status"] == "ok"


@check("a database of the old version upgrades")
def _upgrade(project: Project) -> None:
    # What the starter code leaves behind: two migrations and user_version = 2.
    path = project.database_path()
    spec = importlib.util.spec_from_file_location("starter_db", STARTER / "backend/app/db.py")
    assert spec is not None and spec.loader is not None
    starter_db = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(starter_db)
    conn = starter_db.connect(path)
    starter_db.init_db(conn)
    conn.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES ('old', 'x', 'then')"
    )
    conn.commit()
    conn.close()
    client = project.client(path)
    expect(client.get("/api/health"), 200)
    add_todo(client, sign_up(client), "after the upgrade")
    # and a second start of the new version must not migrate again
    expect(project.client(path).get("/api/health"), 200)


@check("register: 201 with id, username, created_at")
def _register(project: Project) -> None:
    client = project.client()
    body = expect(
        client.post("/api/auth/register", json={"username": "a_b-1", "password": PASSWORD}), 201
    )
    assert set(body) == {"id", "username", "created_at"}, f"keys {sorted(body)}"
    assert body["username"] == "a_b-1"


@check("register: weak password -> 400 weak_password")
def _weak(project: Project) -> None:
    client = project.client()
    expect(
        client.post("/api/auth/register", json={"username": "alice", "password": "short"}),
        400,
        "weak_password",
    )


@check("register: taken username -> 409 username_taken")
def _taken(project: Project) -> None:
    client = project.client()
    sign_up(client)
    expect(
        client.post("/api/auth/register", json={"username": "alice", "password": PASSWORD}),
        409,
        "username_taken",
    )


@check("register: bad usernames are rejected")
def _bad_names(project: Project) -> None:
    client = project.client()
    for name in ("ab", "x" * 33, "has space", "semi;colon", "ünï"):
        response = client.post("/api/auth/register", json={"username": name, "password": PASSWORD})
        expect(response, (400, 422))
        assert "error" in response.json(), "errors must use the project's error shape"


@check("login: bearer token with expires_in")
def _login(project: Project) -> None:
    client = project.client()
    sign_up(client)
    body = expect(
        client.post("/api/auth/login", json={"username": "alice", "password": PASSWORD}), 200
    )
    assert set(body) == {"access_token", "token_type", "expires_in"}, f"keys {sorted(body)}"
    assert body["token_type"] == "bearer" and body["expires_in"] == 3600


@check("login: same 401 invalid_credentials for unknown user and wrong password")
def _bad_login(project: Project) -> None:
    client = project.client()
    sign_up(client)
    answers = []
    for credentials in (
        {"username": "alice", "password": "wrong password"},
        {"username": "nobody", "password": PASSWORD},
    ):
        answers.append(
            expect(client.post("/api/auth/login", json=credentials), 401, "invalid_credentials")
        )
    assert answers[0] == answers[1], "the two answers differ"


@check("audit log records login and login_failed")
def _audit(project: Project) -> None:
    import sqlite3

    path = project.database_path()
    client = project.client(path)
    sign_up(client)
    client.post("/api/auth/login", json={"username": "alice", "password": "wrong password"})
    client.post("/api/auth/login", json={"username": "nobody", "password": "wrong password"})
    with contextlib.closing(sqlite3.connect(path)) as conn:
        events = [row[0] for row in conn.execute("SELECT event FROM audit_log")]
    assert events.count("login") >= 1, f"events: {events}"
    assert events.count("login_failed") >= 2, f"events: {events}"


@check("me: the signed-in user, 401 without a token")
def _me(project: Project) -> None:
    client = project.client()
    headers = sign_up(client)
    body = expect(client.get("/api/auth/me", headers=headers), 200)
    assert body["username"] == "alice" and "id" in body
    assert "password_hash" not in body
    expect(client.get("/api/auth/me"), 401)


@check("todos: create 201 with a trimmed title")
def _create(project: Project) -> None:
    client = project.client()
    todo = add_todo(client, sign_up(client), "  buy milk  ")
    assert todo["title"] == "buy milk" and todo["done"] is False


@check("todos: title must be 1-200 characters after trimming")
def _titles(project: Project) -> None:
    client = project.client()
    headers = sign_up(client)
    for title in ("", "    ", "x" * 201):
        expect(client.post("/api/todos", json={"title": title}, headers=headers), (400, 422))
    assert add_todo(client, headers, " " + "x" * 200 + " ")["title"] == "x" * 200


@check("todos: newest first, paged, with the paginate envelope")
def _paging(project: Project) -> None:
    client = project.client()
    headers = sign_up(client)
    for number, title in enumerate(("one", "two", "three")):
        if number:
            time.sleep(1.1)  # timestamps have one-second resolution
        add_todo(client, headers, title)
    body = expect(client.get("/api/todos?limit=2", headers=headers), 200)
    assert set(body) == {"items", "total", "limit", "offset"}, f"keys {sorted(body)}"
    assert [t["title"] for t in body["items"]] == ["three", "two"]
    assert (body["total"], body["limit"], body["offset"]) == (3, 2, 0)
    rest = expect(client.get("/api/todos?limit=2&offset=2", headers=headers), 200)
    assert [t["title"] for t in rest["items"]] == ["one"]
    expect(client.get("/api/todos?limit=1000", headers=headers), 400, "page_too_large")


@check("todos: ?done= filter")
def _filter(project: Project) -> None:
    client = project.client()
    headers = sign_up(client)
    first = add_todo(client, headers, "first")
    add_todo(client, headers, "second")
    expect(client.patch(f"/api/todos/{first['id']}", json={"done": True}, headers=headers), 200)
    for value, title in (("true", "first"), ("false", "second")):
        body = expect(client.get(f"/api/todos?done={value}", headers=headers), 200)
        assert [t["title"] for t in body["items"]] == [title], f"done={value}: {body}"
        assert body["total"] == 1
    assert expect(client.get("/api/todos", headers=headers), 200)["total"] == 2


@check("todos: get and patch one")
def _patch(project: Project) -> None:
    client = project.client()
    headers = sign_up(client)
    todo = add_todo(client, headers, "draft")
    url = f"/api/todos/{todo['id']}"
    assert expect(client.get(url, headers=headers), 200)["title"] == "draft"
    done = expect(client.patch(url, json={"done": True}, headers=headers), 200)
    assert done["done"] is True and done["title"] == "draft"
    renamed = expect(client.patch(url, json={"title": "final"}, headers=headers), 200)
    assert renamed["title"] == "final" and renamed["done"] is True
    reopened = expect(client.patch(url, json={"done": False}, headers=headers), 200)
    assert reopened["done"] is False
    assert expect(client.get(url, headers=headers), 200) == reopened


@check("todos: delete 204, then 404 todo_not_found")
def _delete(project: Project) -> None:
    client = project.client()
    headers = sign_up(client)
    url = f"/api/todos/{add_todo(client, headers, 'gone')['id']}"
    response = client.delete(url, headers=headers)
    assert response.status_code == 204 and not response.content
    expect(client.get(url, headers=headers), 404, "todo_not_found")
    expect(client.delete(url, headers=headers), 404, "todo_not_found")


@check("todos: another user's todo is 404 and not listed")
def _isolation(project: Project) -> None:
    client = project.client()
    alice = sign_up(client, "alice")
    bob = sign_up(client, "bob")
    url = f"/api/todos/{add_todo(client, alice, 'private')['id']}"
    expect(client.get(url, headers=bob), 404, "todo_not_found")
    expect(client.patch(url, json={"title": "stolen"}, headers=bob), 404, "todo_not_found")
    expect(client.delete(url, headers=bob), 404, "todo_not_found")
    assert expect(client.get("/api/todos", headers=bob), 200)["total"] == 0
    assert expect(client.get(url, headers=alice), 200)["title"] == "private"


@check("todos: every endpoint needs a token")
def _needs_auth(project: Project) -> None:
    client = project.client()
    expect(client.get("/api/todos"), 401)
    expect(client.post("/api/todos", json={"title": "x"}), 401)
    expect(client.get("/api/todos/1"), 401)
    expect(client.patch("/api/todos/1", json={"done": True}), 401)
    expect(client.delete("/api/todos/1"), 401)


# ------------------------------------------------------------------ frontend checks


def _page(project: Project, name: str, helpers: tuple[str, ...]) -> None:
    html = project.read(f"frontend/{name}.html")
    script = project.read(f"frontend/js/{name}.js")
    assert html.strip(), f"frontend/{name}.html is missing"
    code = re.sub(r"/\*.*?\*/|(?m:^\s*//.*$)", "", script, flags=re.S).strip()
    assert code, f"frontend/js/{name}.js has no code"
    assert f"{name}.js" in html, f"{name}.html does not load js/{name}.js"
    missing = [helper for helper in helpers if helper not in script]
    assert not missing, f"js/{name}.js does not use {', '.join(missing)}"
    assert not re.search(r"(?<![\w.])fetch\(", script), "calls fetch() instead of apiFetch()"
    assert "innerHTML" not in script, "inserts markup with innerHTML"


@check("frontend: sign-in page with its script")
def _login_page(project: Project) -> None:
    _page(project, "login", ("apiFetch", "saveToken"))


@check("frontend: todo page with its script")
def _todo_page(project: Project) -> None:
    _page(project, "todos", ("apiFetch", "clearToken", "showMessage"))


@check("frontend: index links to the new pages")
def _navigation(project: Project) -> None:
    html = project.read("frontend/index.html")
    assert "login.html" in html and "todos.html" in html


# ------------------------------------------------------------------ child entry point


def main(root: Path) -> None:
    sys.path.insert(0, str(root / "backend"))
    # Import the test client before the guard: imports may probe the system.
    from fastapi.testclient import TestClient  # noqa: F401

    _guard(root.parent)
    project = Project(root)
    results: dict[str, str | None] = {}
    for name, function in CHECKS:
        try:
            function(project)
            results[name] = None
        except BaseException as exc:
            last = traceback.extract_tb(exc.__traceback__)[-1]
            message = str(exc) or f"assertion on line {last.lineno}"
            results[name] = f"{type(exc).__name__}: {message}"[:300]
    print(json.dumps(results))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
