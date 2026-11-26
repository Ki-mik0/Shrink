"""Paged list responses.

Every list endpoint takes the same two query parameters and returns the same
envelope, so clients can page through any collection the same way::

    GET /api/things?limit=20&offset=40

    {"items": [...], "total": 137, "limit": 20, "offset": 40}

``total`` is the number of items across all pages, which lets a client show
"41-60 of 137". Use :class:`PageParams` as a dependency and :func:`paginate` to
build the response; count with a separate ``SELECT COUNT(*)`` on the same
conditions as the page query.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Query

from app.config import Settings, get_settings
from app.errors import bad_request


@dataclass(frozen=True)
class PageParams:
    """Validated ``limit`` and ``offset`` for one request."""

    limit: int
    offset: int


def page_params(
    limit: int | None = Query(default=None, ge=1, description="Items per page."),
    offset: int = Query(default=0, ge=0, description="Items to skip."),
    settings: Settings = Depends(get_settings),
) -> PageParams:
    """FastAPI dependency: ``limit``/``offset`` from the query string.

    ``limit`` defaults to ``settings.page_size``; a value above
    ``settings.max_page_size`` is rejected with a 400 ``page_too_large`` error
    instead of being capped silently, so a client never believes it received
    everything when it did not.

    Usage::

        def list_things(page: PageParams = Depends(page_params), ...):
            rows = conn.execute("... LIMIT ? OFFSET ?", (..., page.limit, page.offset))
    """
    size = settings.page_size if limit is None else limit
    if size > settings.max_page_size:
        raise bad_request("page_too_large", f"limit may be at most {settings.max_page_size}.")
    return PageParams(limit=size, offset=offset)


def paginate(items: Sequence[Any], total: int, page: PageParams) -> dict[str, Any]:
    """The response envelope for one page of a list.

    Args:
        items: The items of this page, already converted to JSON-friendly values.
        total: How many items match across all pages.
        page: The parameters the page was fetched with.
    """
    return {"items": list(items), "total": total, "limit": page.limit, "offset": page.offset}
