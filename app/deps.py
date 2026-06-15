"""Shared FastAPI dependencies."""
import os
from fastapi import HTTPException, Request

CF_EMAIL_HEADER = "cf-access-authenticated-user-email"
ADMIN_EMAILS = frozenset({"stephenyoung7267@gmail.com"})
# Who may see the full photo Gallery (and Search/Albums/Favorites). Broader than
# admin: the owner + Cayce. Everyone else gets the curated family experience
# (People, Biographies, Scrapbook, Family tree) without the firehose gallery.
GALLERY_EMAILS = ADMIN_EMAILS | frozenset({"caycewardyoung@gmail.com"})


def current_email(request: Request) -> str | None:
    return (
        request.headers.get(CF_EMAIL_HEADER)
        or os.environ.get("DEV_USER_EMAIL", "stephenyoung7267@gmail.com")
    )


def is_admin_email(email: str | None) -> bool:
    if os.environ.get("DEV_ADMIN"):
        return True
    return bool(email and email in ADMIN_EMAILS)


def can_see_gallery(email: str | None) -> bool:
    return is_admin_email(email) or bool(email and email in GALLERY_EMAILS)


async def require_admin(request: Request) -> str:
    email = current_email(request)
    if not is_admin_email(email):
        raise HTTPException(403, "Admin required")
    return email or ""


def is_admin_request(request: Request) -> bool:
    """True if the caller is an admin. For read endpoints that show more to the
    owner (e.g. private collections / bios) than to family viewers.

    When the admin is using the "View as <family member>" preview, the frontend
    sends `X-Preview-As-Viewer`; we then report non-admin so the owner sees
    exactly what a logged-in family member would. (Writes are unaffected — the
    write-lock middleware checks the real email, not this.)"""
    if request.headers.get("x-preview-as-viewer"):
        return False
    return is_admin_email(getattr(request.state, "user_email", None) or current_email(request))
