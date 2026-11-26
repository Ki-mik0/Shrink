"""SQLite access: connections, the ``get_db`` dependency and schema migrations.

The application uses the standard library's :mod:`sqlite3` directly. There is
no ORM: queries are short and explicit, and every value goes through a ``?``
placeholder so that user input can never change the meaning of a statement.

Connections
-----------
Endpoints never open connections. They declare a dependency::

    @router.get("/things")
    def list_things(conn: sqlite3.Connection = Depends(get_db)):
        rows = conn.execute("SELECT id, name FROM things").fetchall()

``get_db`` hands out a connection whose rows are :class:`sqlite3.Row` objects
(columns readable by name, ``row["name"]``) and with foreign keys enforced. The
work done during one request is committed when the endpoint returns normally
and rolled back if it raises.

Migrations
----------
The schema is defined by :data:`MIGRATIONS`, a list of SQL scripts applied in
order. The number of scripts already applied is stored in SQLite's
``user_version`` pragma, so :func:`init_db` only runs the new ones. To change
the schema, **append** a script; never edit or reorder the existing entries,
because databases in use have already run them.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator
from datetime import datetime, timezone

#: Schema scripts, applied in order. Append only.
MIGRATIONS: list[str] = [
    # 1: users. ``username`` is unique without regard to case, so "Alice" and
    # "alice" cannot both register.
    """
    CREATE TABLE users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT    NOT NULL UNIQUE COLLATE NOCASE,
        password_hash TEXT    NOT NULL,
        created_at    TEXT    NOT NULL
    );
    """,
    # 2: an audit trail of security-relevant events (sign-ins, failed sign-ins).
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
]


def utc_now() -> str:
    """The current time in UTC as an ISO 8601 string, e.g. ``2026-11-24T12:00:00+00:00``.

    All timestamps are stored in this form. Strings in this format sort in the
    same order as the times they represent, so ``ORDER BY created_at`` works.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str) -> sqlite3.Connection:
    """Open a connection configured the way the application expects.

    * rows are :class:`sqlite3.Row`, so columns can be read by name;
    * foreign key constraints are enforced (SQLite leaves them off by default);
    * ``check_same_thread`` is off because FastAPI may run a request's
      dependencies and endpoint in different worker threads. Each connection is
      still used by one request at a time.
    """
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def schema_version(conn: sqlite3.Connection) -> int:
    """How many entries of :data:`MIGRATIONS` have been applied to this database."""
    return int(conn.execute("PRAGMA user_version").fetchone()[0])


def init_db(conn: sqlite3.Connection) -> int:
    """Apply the migrations this database has not run yet; return how many ran.

    Each script runs in its own transaction together with the update of
    ``user_version``, so a failing script leaves the database at the previous
    version instead of half migrated.
    """
    applied = 0
    for number, script in enumerate(MIGRATIONS, start=1):
        if number <= schema_version(conn):
            continue
        with conn:
            conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;")
        applied += 1
    return applied


class Database:
    """Owns the database of one application instance.

    For a file database every request gets its own connection. An in-memory
    database (``":memory:"``) exists only as long as its connection, so in that
    case a single shared connection is kept for the life of the application and
    access to it is serialised with a lock.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self._shared: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        if path == ":memory:":
            self._shared = connect(path)
            init_db(self._shared)
        else:
            with connect(path) as conn:
                init_db(conn)

    def connections(self) -> Iterator[sqlite3.Connection]:
        """Yield one connection for a request, committing or rolling back afterwards."""
        if self._shared is not None:
            with self._lock:
                try:
                    yield self._shared
                    self._shared.commit()
                except BaseException:
                    self._shared.rollback()
                    raise
            return
        conn = connect(self.path)
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


_database: Database | None = None


def configure(database: Database) -> None:
    """Make ``database`` the one :func:`get_db` hands connections from."""
    global _database
    _database = database


def get_db() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency: a connection for the current request.

    Usage::

        def endpoint(conn: sqlite3.Connection = Depends(get_db)): ...

    Raises:
        RuntimeError: if the application was not created through
            :func:`app.main.create_app`, which configures the database.
    """
    if _database is None:
        raise RuntimeError("database not configured: create the app with create_app()")
    yield from _database.connections()


def audit(conn: sqlite3.Connection, event: str, user_id: int | None = None, detail: str = "") -> None:
    """Record a security-relevant event in ``audit_log``.

    Use short event names such as ``"login"``, ``"login_failed"`` or
    ``"register"``. ``detail`` is free text for humans; never put passwords or
    tokens in it.
    """
    conn.execute(
        "INSERT INTO audit_log (user_id, event, detail, created_at) VALUES (?, ?, ?, ?)",
        (user_id, event, detail, utc_now()),
    )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, object] | None:
    """Turn a row into a plain dict (or keep ``None``), e.g. to return it as JSON."""
    return None if row is None else {key: row[key] for key in row.keys()}
