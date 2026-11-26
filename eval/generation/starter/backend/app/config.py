"""Application settings, read from environment variables.

Settings are read once, at start-up, and kept in an immutable object. Code
that needs a value asks :func:`get_settings` for it instead of reading
``os.environ`` directly, so tests can build an application with explicit
settings (see :func:`app.main.create_app`) without touching the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache

DEFAULT_DB_PATH = "taskboard.sqlite3"
DEFAULT_TOKEN_TTL = 3600  # one hour
DEFAULT_PAGE_SIZE = 20
DEFAULT_MAX_PAGE_SIZE = 100


class ConfigError(RuntimeError):
    """A required setting is missing or has a value that cannot be used."""


@dataclass(frozen=True)
class Settings:
    """All tunable values of the application.

    Attributes:
        db_path: Path of the SQLite database file. ``":memory:"`` keeps the
            database in memory, which is what the tests use.
        secret: Key used to sign access tokens. It must stay private: anyone
            who knows it can create valid tokens for any user.
        token_ttl: Lifetime of an access token in seconds.
        cors_origins: Origins allowed to call the API from a browser. Empty
            means same-origin only, which is enough when the frontend is
            served by this process.
        page_size: Number of items a list endpoint returns when the client
            does not ask for a specific ``limit``.
        max_page_size: Largest ``limit`` a client may ask for. Larger values
            are rejected with a validation error rather than silently capped.
    """

    db_path: str = DEFAULT_DB_PATH
    secret: str = ""
    token_ttl: int = DEFAULT_TOKEN_TTL
    cors_origins: tuple[str, ...] = field(default_factory=tuple)
    page_size: int = DEFAULT_PAGE_SIZE
    max_page_size: int = DEFAULT_MAX_PAGE_SIZE

    def validate(self) -> Settings:
        """Check the values and return ``self``, so calls can be chained.

        Raises:
            ConfigError: if the secret is missing or a number is out of range.
        """
        if not self.secret:
            raise ConfigError("APP_SECRET must be set to a non-empty value")
        if self.token_ttl <= 0:
            raise ConfigError("APP_TOKEN_TTL must be a positive number of seconds")
        if not 0 < self.page_size <= self.max_page_size:
            raise ConfigError("APP_PAGE_SIZE must be between 1 and APP_MAX_PAGE_SIZE")
        return self


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, falling back to ``default``.

    A value that is present but not a number is a configuration mistake and is
    reported instead of being ignored.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _list_env(name: str) -> tuple[str, ...]:
    """Read a comma-separated environment variable as a tuple of non-empty items."""
    raw = os.environ.get(name, "")
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def settings_from_env() -> Settings:
    """Build :class:`Settings` from the process environment and validate them."""
    return Settings(
        db_path=os.environ.get("APP_DB_PATH", DEFAULT_DB_PATH),
        secret=os.environ.get("APP_SECRET", ""),
        token_ttl=_int_env("APP_TOKEN_TTL", DEFAULT_TOKEN_TTL),
        cors_origins=_list_env("APP_CORS_ORIGINS"),
        page_size=_int_env("APP_PAGE_SIZE", DEFAULT_PAGE_SIZE),
        max_page_size=_int_env("APP_MAX_PAGE_SIZE", DEFAULT_MAX_PAGE_SIZE),
    ).validate()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The settings of the running process, read on first use and then cached.

    :func:`app.main.create_app` overrides this dependency when it is given
    explicit settings, so endpoints should always depend on ``get_settings``
    (``settings: Settings = Depends(get_settings)``) rather than calling
    :func:`settings_from_env` themselves.
    """
    return settings_from_env()
