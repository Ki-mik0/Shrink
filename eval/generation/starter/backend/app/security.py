"""Passwords and access tokens.

Passwords
---------
Passwords are never stored. :func:`hash_password` derives a key with
PBKDF2-HMAC-SHA256 and a random salt and returns a self-describing string::

    pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>

:func:`verify_password` reads the parameters back from that string, so the
iteration count can be raised later without invalidating existing hashes.

Tokens
------
After a successful sign-in the API returns an access token. It is a compact,
signed string (the same idea as a JWT, with less ceremony)::

    <base64url(payload JSON)>.<base64url(HMAC-SHA256 signature)>

The payload holds the user id (``sub``), the username (``name``) and the expiry
time (``exp``, a Unix timestamp). The signature uses the application secret,
so a client cannot change the payload without the token becoming invalid.
Tokens cannot be revoked before they expire; keep their lifetime short.

Protected endpoints declare ``user: CurrentUser = Depends(get_current_user)``.
The dependency reads ``Authorization: Bearer <token>``, checks the signature
and the expiry, loads the user and raises a 401 :class:`~app.errors.ApiError`
if anything is wrong.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, Header

from app.config import Settings, get_settings
from app.db import get_db
from app.errors import unauthorized

PBKDF2_ITERATIONS = 240_000
SALT_BYTES = 16
_SCHEME = "pbkdf2_sha256"
MIN_PASSWORD_LENGTH = 8


# ---------------------------------------------------------------- passwords


def hash_password(password: str) -> str:
    """Hash a password for storage; the result includes its own salt and parameters.

    The same password hashes to a different string every time (fresh salt), so
    compare with :func:`verify_password`, never with ``==``.
    """
    salt = secrets.token_bytes(SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return f"{_SCHEME}${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """True if ``password`` matches a hash made by :func:`hash_password`.

    The comparison takes the same time whether or not the password matches,
    so the response time does not leak how much of a guess was right. A
    malformed stored value simply does not match.
    """
    try:
        scheme, iterations, salt_hex, digest_hex = stored.split("$")
        if scheme != _SCHEME:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest.hex(), digest_hex)


def password_problem(password: str) -> str | None:
    """Why a new password is not acceptable, or ``None`` if it is fine.

    The rule is deliberately simple: at least :data:`MIN_PASSWORD_LENGTH`
    characters and not only whitespace. Use it when users register or change
    their password and report the message with a 400 ``weak_password`` error.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Password must be at least {MIN_PASSWORD_LENGTH} characters long."
    if not password.strip():
        return "Password must not be only spaces."
    return None


# ---------------------------------------------------------------- tokens


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: str, secret: str) -> str:
    return _b64encode(hmac.new(secret.encode("utf-8"), payload.encode("ascii"), hashlib.sha256).digest())


def create_access_token(user_id: int, username: str, settings: Settings) -> str:
    """A signed token for this user that expires after ``settings.token_ttl`` seconds."""
    payload = {"sub": user_id, "name": username, "exp": int(time.time()) + settings.token_ttl}
    body = _b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return f"{body}.{_sign(body, settings.secret)}"


class TokenError(ValueError):
    """A token is malformed, has a bad signature or has expired."""


def decode_access_token(token: str, settings: Settings) -> dict[str, Any]:
    """Check a token and return its payload (``sub``, ``name``, ``exp``).

    Raises:
        TokenError: if the token is malformed, tampered with or expired.
    """
    try:
        body, signature = token.split(".")
    except ValueError as exc:
        raise TokenError("malformed token") from exc
    if not hmac.compare_digest(signature, _sign(body, settings.secret)):
        raise TokenError("bad signature")
    try:
        payload: dict[str, Any] = json.loads(_b64decode(body))
    except (ValueError, UnicodeDecodeError) as exc:
        raise TokenError("malformed payload") from exc
    if int(payload.get("exp", 0)) < time.time():
        raise TokenError("token expired")
    return payload


# ---------------------------------------------------------------- current user


@dataclass(frozen=True)
class CurrentUser:
    """The signed-in user, as given to endpoints by :func:`get_current_user`."""

    id: int
    username: str


def get_current_user(
    authorization: str | None = Header(default=None),
    conn: sqlite3.Connection = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    """FastAPI dependency: the user the request's bearer token belongs to.

    Usage::

        @router.get("/mine")
        def mine(user: CurrentUser = Depends(get_current_user)): ...

    Raises:
        ApiError: 401 ``not_authenticated`` without a bearer token, 401
            ``invalid_token`` for a bad or expired token or a deleted user.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise unauthorized()
    token = authorization.split(" ", 1)[1].strip()
    try:
        payload = decode_access_token(token, settings)
    except TokenError as exc:
        raise unauthorized("invalid_token", "Your session is invalid or has expired.") from exc
    row = conn.execute("SELECT id, username FROM users WHERE id = ?", (payload["sub"],)).fetchone()
    if row is None:
        raise unauthorized("invalid_token", "Your session is invalid or has expired.")
    return CurrentUser(id=row["id"], username=row["username"])
