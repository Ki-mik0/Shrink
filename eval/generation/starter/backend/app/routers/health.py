"""Health and version endpoints.

This module is the reference for how routers are written in this code base:

* a module-level ``router`` with a ``/api/<feature>`` prefix and a tag;
* request and response bodies as Pydantic models next to the endpoints;
* the database through ``Depends(get_db)``, settings through
  ``Depends(get_settings)``, the signed-in user through
  ``Depends(get_current_user)``;
* failures raised as :class:`~app.errors.ApiError`, never ``HTTPException``.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app import __version__
from app.config import Settings, get_settings
from app.db import get_db, schema_version
from app.errors import ApiError

router = APIRouter(prefix="/api/health", tags=["health"])


class Health(BaseModel):
    """What ``GET /api/health`` returns."""

    status: str
    version: str
    schema_version: int
    page_size: int


@router.get("", response_model=Health, summary="Is the service up?")
def health(
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> Health:
    """Answer quickly and touch the database, so a broken database shows up here.

    Load balancers call this every few seconds: keep it cheap and free of side
    effects.
    """
    try:
        version = schema_version(conn)
    except sqlite3.DatabaseError as exc:
        raise ApiError(503, "database_unavailable", "The database cannot be reached.") from exc
    return Health(
        status="ok",
        version=__version__,
        schema_version=version,
        page_size=settings.page_size,
    )
