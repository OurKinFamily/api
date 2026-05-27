import os
from pathlib import Path
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from app.db.neo4j import get_session

router = APIRouter(prefix="/admin", tags=["admin"])

# Cloudflare Access injects this header on every authenticated request when the
# app sits behind a Zero-Trust Access policy. In local dev (no Cloudflare in
# front), the middleware falls back to env DEV_USER_EMAIL.
CF_EMAIL_HEADER = "cf-access-authenticated-user-email"


def _current_email(request: Request) -> str | None:
    return (
        request.headers.get(CF_EMAIL_HEADER)
        or os.environ.get("DEV_USER_EMAIL", "stephenyoung7267@gmail.com")
    )

REPORT_PATH = Path(os.environ.get("REPORT_JSON", "/photos/__data/status/report.json"))


@router.get("/report")
async def get_report():
    if not REPORT_PATH.exists():
        raise HTTPException(404, "report.json not found — run the Archive Report job first")
    import json
    return json.loads(REPORT_PATH.read_text())


@router.get("/whoami")
async def whoami(request: Request):
    """Debug: dump every Cloudflare / forwarded header reaching the backend.
    Use this to confirm Cloudflare Access is injecting identity headers."""
    return {
        "resolved_email":  _current_email(request),
        "cf_headers":      {k: v for k, v in request.headers.items() if k.lower().startswith("cf-")},
        "forwarded":       {k: v for k, v in request.headers.items() if "forwarded" in k.lower()},
        "client_host":     request.client.host if request.client else None,
        "all_lower_keys":  sorted(request.headers.keys()),
    }


@router.get("/me")
async def me(request: Request):
    """Current user identity. Resolved from Cloudflare Access header (or dev
    fallback), then matched to a Person by email. Returns null fields if no
    matching Person record exists yet."""
    email = _current_email(request)
    if not email:
        return {"email": None, "person": None}
    async with get_session() as session:
        result = await session.run(
            "MATCH (p:Person {email: $email}) "
            "RETURN p.id AS id, p.name AS name, p.known_as AS known_as, p.avatar AS avatar, p.email AS email",
            email=email,
        )
        row = await result.single()
    return {
        "email":  email,
        "person": dict(row) if row else None,
    }
