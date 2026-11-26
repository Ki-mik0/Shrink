"""One shape for every error the API returns.

Clients get errors as::

    HTTP/1.1 404 Not Found
    {"error": {"code": "todo_not_found", "message": "Todo 7 does not exist."}}

``code`` is a stable snake_case identifier that client code may branch on;
``message`` is for humans and may change. Validation errors produced by FastAPI
itself (bad JSON, missing fields, wrong types) are converted to the same shape
with the code ``validation_error`` and the list of problems under ``details``.

Raise :class:`ApiError` (or use the helpers below) in endpoints and
dependencies instead of :class:`fastapi.HTTPException`.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


class ApiError(Exception):
    """An error that becomes a JSON response.

    Args:
        status: HTTP status code, e.g. 400, 401, 404, 409.
        code: Stable machine-readable identifier in snake_case.
        message: Human-readable explanation.
        headers: Extra response headers, e.g. ``WWW-Authenticate`` for 401.
    """

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


def bad_request(code: str, message: str) -> ApiError:
    """400: the request is well-formed but cannot be carried out as asked."""
    return ApiError(400, code, message)


def unauthorized(code: str = "not_authenticated", message: str = "Sign in first.") -> ApiError:
    """401: no valid credentials. Adds the ``WWW-Authenticate: Bearer`` header."""
    return ApiError(401, code, message, headers={"WWW-Authenticate": "Bearer"})


def forbidden(code: str = "forbidden", message: str = "Not allowed.") -> ApiError:
    """403: the user is known but may not do this."""
    return ApiError(403, code, message)


def not_found(code: str, message: str) -> ApiError:
    """404: the thing does not exist, *or* it belongs to someone else.

    Answering 404 rather than 403 for other users' resources avoids telling a
    client that an identifier exists.
    """
    return ApiError(404, code, message)


def conflict(code: str, message: str) -> ApiError:
    """409: the request clashes with the current state, e.g. a taken username."""
    return ApiError(409, code, message)


async def _api_error_handler(request: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=exc.status, content=exc.body(), headers=exc.headers)


async def _validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"field": ".".join(str(part) for part in error["loc"][1:]), "problem": error["msg"]}
        for error in exc.errors()
    ]
    body = {
        "error": {
            "code": "validation_error",
            "message": "Some fields are missing or invalid.",
            "details": details,
        }
    }
    return JSONResponse(status_code=422, content=body)


def install(app: FastAPI) -> None:
    """Register the handlers that give every error the shape described above."""
    app.add_exception_handler(ApiError, _api_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(RequestValidationError, _validation_handler)  # type: ignore[arg-type]
