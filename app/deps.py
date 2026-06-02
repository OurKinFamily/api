"""Shared FastAPI dependencies."""
import os
from fastapi import HTTPException, Request

CF_EMAIL_HEADER = "cf-access-authenticated-user-email"
ADMIN_EMAILS = frozenset({"stephenyoung7267@gmail.com"})


def current_email(request: Request) -> str | None:
    return (
        request.headers.get(CF_EMAIL_HEADER)
        or os.environ.get("DEV_USER_EMAIL", "stephenyoung7267@gmail.com")
    )


def is_admin_email(email: str | None) -> bool:
    if os.environ.get("DEV_ADMIN"):
        return True
    return bool(email and email in ADMIN_EMAILS)


async def require_admin(request: Request) -> str:
    email = current_email(request)
    if not is_admin_email(email):
        raise HTTPException(403, "Admin required")
    return email or ""
