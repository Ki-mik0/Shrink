"""The application factory.

``uvicorn --factory app.main:build`` serves an application built from
environment settings. Tests and scripts call :func:`create_app` with explicit
values instead::

    app = create_app(db_path=":memory:", secret="test")

Adding a feature means writing a router module in :mod:`app.routers` and adding
one ``app.include_router(...)`` line below.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__, errors
from app.config import Settings, get_settings, settings_from_env
from app.db import Database, configure
from app.routers import health

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


def create_app(
    settings: Settings | None = None,
    *,
    db_path: str | None = None,
    secret: str | None = None,
) -> FastAPI:
    """Build a configured application.

    Args:
        settings: Complete settings. Read from the environment when omitted.
        db_path: Shortcut that overrides ``settings.db_path`` (tests use
            ``":memory:"``).
        secret: Shortcut that overrides ``settings.secret``.

    The database is opened and migrated here, so the application is ready to
    serve requests as soon as this returns.
    """
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
    # New routers go here, e.g. app.include_router(things.router)

    if FRONTEND_DIR.is_dir():
        # Mounted last so that /api routes take precedence over static files.
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


def build() -> FastAPI:
    """Entry point for ``uvicorn --factory app.main:build``: settings from the environment.

    A factory instead of a module-level ``app`` means importing this module
    never needs ``APP_SECRET``, which keeps imports in tests side-effect free.
    """
    return create_app()
