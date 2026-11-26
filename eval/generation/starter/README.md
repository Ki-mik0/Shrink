# Taskboard

A small web application for personal task lists. The backend is a FastAPI
service backed by SQLite; the frontend is plain HTML, CSS and JavaScript with no
build step, served as static files by the same process.

This repository is the starting point. Health checks, configuration, database
access, migrations, password hashing, access tokens, error handling and
pagination are in place. Features are added as routers in
`backend/app/routers/` and pages in `frontend/`.

## Running locally

```bash
cd backend
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export APP_SECRET="change-me"   # Windows: set APP_SECRET=change-me
uvicorn --factory app.main:build --reload
```

Then open http://127.0.0.1:8000/ in a browser. The API lives under `/api`, the
interactive documentation under `/docs`.

## Configuration

All settings come from environment variables, read once at start-up by
`app.config.get_settings()`:

| Variable | Default | Meaning |
| --- | --- | --- |
| `APP_DB_PATH` | `taskboard.sqlite3` | SQLite database file; `:memory:` for tests |
| `APP_SECRET` | none (required) | key used to sign access tokens |
| `APP_TOKEN_TTL` | `3600` | lifetime of an access token, in seconds |
| `APP_CORS_ORIGINS` | empty | comma-separated origins allowed to call the API |
| `APP_PAGE_SIZE` | `20` | default page size for list endpoints |
| `APP_MAX_PAGE_SIZE` | `100` | largest page size a client may ask for |

## Conventions

These rules keep the code base consistent. New code must follow them.

### Backend

* **Routers.** One module per feature in `app/routers/`, each exposing a module
  level `router = APIRouter(prefix="/api/<feature>", tags=["<feature>"])`. The
  router is registered in `app.main.create_app()` with `app.include_router(...)`.
* **Database.** Never open connections yourself. Endpoints receive a connection
  through the `get_db` dependency (`conn: sqlite3.Connection = Depends(get_db)`).
  Rows come back as `sqlite3.Row`, so columns can be read by name. Use `?`
  placeholders, never string formatting, for values in SQL.
* **Schema changes.** Every change to the schema is a new entry appended to
  `app.db.MIGRATIONS`. Existing entries are never edited: a deployed database
  has already run them. `init_db()` applies the missing ones in order.
* **Errors.** Raise `app.errors.ApiError` (or one of its helpers such as
  `not_found()` and `conflict()`) instead of `HTTPException`. Every error leaves
  the API in the same JSON shape: `{"error": {"code": "...", "message": "..."}}`.
  Codes are short snake_case identifiers that clients may rely on.
* **Authentication.** Passwords are hashed with `app.security.hash_password()`
  and checked with `verify_password()`. Tokens are created with
  `create_access_token()`; protected endpoints take the current user from the
  `get_current_user` dependency, which reads `Authorization: Bearer <token>`.
  Never store or log plain passwords or tokens.
* **Lists.** List endpoints accept `limit` and `offset` query parameters through
  the `PageParams` dependency and return `paginate(...)`, which produces
  `{"items": [...], "total": n, "limit": l, "offset": o}`.
* **Time.** Timestamps are stored as ISO 8601 strings in UTC, produced by
  `app.db.utc_now()`.

### Frontend

* Pages are plain HTML files in `frontend/`, scripts in `frontend/js/`, styles
  in `frontend/css/style.css`. No frameworks and no build step.
* All calls to the API go through `apiFetch()` from `js/api.js`. It adds the
  stored token, sends and parses JSON and turns error responses into an
  `ApiError` carrying the server's error code. Do not call `fetch()` directly.
* The token is kept in `localStorage` under the key exported as `TOKEN_KEY`;
  use `saveToken()`, `loadToken()` and `clearToken()` rather than touching
  storage yourself.
* Build DOM nodes with the helpers in `js/ui.js` (`el()`, `clear()`,
  `showMessage()`). Text coming from users is always inserted as text, never as
  HTML, to keep the page safe from script injection.

## Tests

Backend tests use FastAPI's `TestClient` and an in-memory database:

```python
from fastapi.testclient import TestClient
from app.main import create_app

client = TestClient(create_app(db_path=":memory:", secret="test"))
assert client.get("/api/health").json()["status"] == "ok"
```
