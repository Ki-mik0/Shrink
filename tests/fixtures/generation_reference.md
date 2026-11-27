## Patch: backend/app/db.py
```python
MIGRATIONS: list[str] = [
    """
    CREATE TABLE users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
        password_hash TEXT    NOT NULL,
        created_at    TEXT    NOT NULL
    );
    """,
    """
    CREATE TABLE audit_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER REFERENCES users(id) ON DELETE SET NULL,
        event      TEXT    NOT NULL,
        detail     TEXT    NOT NULL DEFAULT '',
        created_at TEXT    NOT NULL
    );
    CREATE INDEX audit_log_user ON audit_log(user_id, created_at);
    """,
    """
    CREATE TABLE todos (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        title      TEXT    NOT NULL,
        done       INTEGER NOT NULL DEFAULT 0,
        created_at TEXT    NOT NULL,
        updated_at TEXT    NOT NULL
    );
    """,
]
```

## File: backend/app/routers/auth.py
```python
import sqlite3

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.db import audit, get_db, utc_now
from app.errors import bad_request, conflict, unauthorized
from app.security import (
    CurrentUser,
    create_access_token,
    get_current_user,
    hash_password,
    password_problem,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Registration(BaseModel):
    username: str = Field(pattern=r"^[A-Za-z0-9_-]{3,32}$")
    password: str


class Credentials(BaseModel):
    username: str
    password: str


@router.post("/register", status_code=201)
def register(body: Registration, conn: sqlite3.Connection = Depends(get_db)):
    problem = password_problem(body.password)
    if problem:
        raise bad_request("weak_password", problem)
    now = utc_now()
    try:
        cursor = conn.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (body.username, hash_password(body.password), now),
        )
    except sqlite3.IntegrityError:
        raise conflict("username_taken", "That username is taken.") from None
    return {"id": cursor.lastrowid, "username": body.username, "created_at": now}


@router.post("/login")
def login(
    body: Credentials,
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
):
    row = conn.execute("SELECT * FROM users WHERE username = ?", (body.username,)).fetchone()
    if row is None or not verify_password(body.password, row["password_hash"]):
        audit(conn, "login_failed", detail=body.username)
        conn.commit()
        raise unauthorized("invalid_credentials", "Wrong username or password.")
    audit(conn, "login", user_id=row["id"])
    token = create_access_token(row["id"], row["username"], settings)
    return {"access_token": token, "token_type": "bearer", "expires_in": settings.token_ttl}


@router.get("/me")
def me(user: CurrentUser = Depends(get_current_user)):
    return {"id": user.id, "username": user.username}
```

## File: backend/app/routers/todos.py
```python
import sqlite3

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from app.db import get_db, utc_now
from app.errors import bad_request, not_found
from app.pagination import PageParams, page_params, paginate
from app.security import CurrentUser, get_current_user

router = APIRouter(prefix="/api/todos", tags=["todos"])


class NewTodo(BaseModel):
    title: str


class Change(BaseModel):
    title: str | None = None
    done: bool | None = None


def clean(title: str) -> str:
    title = title.strip()
    if not 1 <= len(title) <= 200:
        raise bad_request("invalid_title", "Titles are 1-200 characters.")
    return title


def as_json(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "title": row["title"],
        "done": bool(row["done"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def own(conn: sqlite3.Connection, todo_id: int, user: CurrentUser) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM todos WHERE id = ? AND user_id = ?", (todo_id, user.id)
    ).fetchone()
    if row is None:
        raise not_found("todo_not_found", "No such todo.")
    return row


@router.get("")
def list_todos(
    done: bool | None = None,
    page: PageParams = Depends(page_params),
    user: CurrentUser = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    where, values = "user_id = ?", [user.id]
    if done is not None:
        where += " AND done = ?"
        values.append(int(done))
    total = conn.execute(f"SELECT COUNT(*) FROM todos WHERE {where}", values).fetchone()[0]
    rows = conn.execute(
        f"SELECT * FROM todos WHERE {where} ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
        [*values, page.limit, page.offset],
    ).fetchall()
    return paginate([as_json(row) for row in rows], total, page)


@router.post("", status_code=201)
def create(
    body: NewTodo,
    user: CurrentUser = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    now = utc_now()
    cursor = conn.execute(
        "INSERT INTO todos (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (user.id, clean(body.title), now, now),
    )
    return as_json(own(conn, cursor.lastrowid, user))


@router.get("/{todo_id}")
def read(
    todo_id: int,
    user: CurrentUser = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    return as_json(own(conn, todo_id, user))


@router.patch("/{todo_id}")
def update(
    todo_id: int,
    body: Change,
    user: CurrentUser = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    row = own(conn, todo_id, user)
    title = row["title"] if body.title is None else clean(body.title)
    done = row["done"] if body.done is None else int(body.done)
    conn.execute(
        "UPDATE todos SET title = ?, done = ?, updated_at = ? WHERE id = ?",
        (title, done, utc_now(), todo_id),
    )
    return as_json(own(conn, todo_id, user))


@router.delete("/{todo_id}", status_code=204)
def delete(
    todo_id: int,
    user: CurrentUser = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_db),
):
    own(conn, todo_id, user)
    conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
    return Response(status_code=204)
```

## Patch: backend/app/main.py
```python
from app.routers import auth, todos


def create_app(
    settings: Settings | None = None,
    *,
    db_path: str | None = None,
    secret: str | None = None,
) -> FastAPI:
    """Build a configured application."""
    if settings is None:
        settings = Settings(secret=secret or "") if secret is not None else settings_from_env()
    if db_path is not None or secret is not None:
        settings = Settings(
            db_path=db_path or settings.db_path,
            secret=secret or settings.secret,
            token_ttl=settings.token_ttl,
            cors_origins=settings.cors_origins,
            page_size=settings.page_size,
            max_page_size=settings.max_page_size,
        )
    settings.validate()

    app = FastAPI(title="Taskboard", version=__version__)
    app.dependency_overrides[get_settings] = lambda: settings
    configure(Database(settings.db_path))
    errors.install(app)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(todos.router)

    if FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app
```

## Patch: frontend/index.html
```html
<nav id="nav" class="nav">
<a href="login.html">Sign in</a>
<a href="todos.html">My todos</a>
</nav>
```

## File: frontend/login.html
```html<!doctype html>
<html lang="en">
<body>
<form id="login"><input name="username"><input name="password" type="password"><button>Sign in</button></form>
<script type="module" src="js/login.js"></script>
</body>
</html>
```

## File: frontend/js/login.js
```javascript
import { apiFetch, saveToken } from "./api.js";
import { showMessage } from "./ui.js";

document.getElementById("login").addEventListener("submit", async (event) => {
event.preventDefault();
try {
const body = Object.fromEntries(new FormData(event.target));
const answer = await apiFetch("/api/auth/login", { method: "POST", body, auth: false });
saveToken(answer.access_token);
location.href = "todos.html";
} catch (error) {
showMessage(error.message, "error");
}
});
```

## File: frontend/todos.html
```html
<!doctype html>
<html lang="en">
<body>
<ul id="list"></ul>
<script type="module" src="js/todos.js"></script>
</body>
</html>
```

## File: frontend/js/todos.js
```javascript
import { apiFetch, clearToken } from "./api.js";
import { el, clear, showMessage } from "./ui.js";

try {
const page = await apiFetch("/api/todos");
const list = document.getElementById("list");
clear(list);
for (const todo of page.items) list.append(el("li", {}, todo.title));
} catch (error) {
clearToken();
showMessage(error.message, "error");
}
```
