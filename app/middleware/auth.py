"""Authorization middleware.

Authentication happens at the edge: Cloudflare Access gates the tunnel, so every
request that reaches the backend already carries a verified family email in the
`Cf-Access-Authenticated-User-Email` header (see app.deps.current_email, which
also has a dev fallback so local runs aren't locked out).

This layer is *authorization*. The archive is a read-only experience for family
members and a read-write one for the owner. So: every mutating request
(POST/PUT/PATCH/DELETE) requires an admin email; reads stay open to anyone who
got through Cloudflare. A single fail-closed chokepoint means a new write
endpoint is locked down by default — no per-route dependency to remember.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.deps import can_see_gallery, current_email, is_admin_email

WRITE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

# POST endpoints that are actually read-only queries (no data is changed), so they
# stay open to any logged-in family member. Matched against the path the app sees
# (the `/api` prefix is stripped before it reaches here). Keep this list tiny and
# obviously-read-only.
READ_ONLY_WRITE_PATHS = {"/search", "/api/search"}

# Write endpoints open to anyone in GALLERY_EMAILS, not just the owner —
# uploading photos is a gallery-tier privilege. Prefix-matched (job id follows).
GALLERY_WRITE_PREFIXES = ("/gallery/upload", "/api/gallery/upload")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        email = current_email(request)
        # Make the resolved email available to handlers/logging.
        request.state.user_email = email

        if (
            request.method in WRITE_METHODS
            and request.url.path not in READ_ONLY_WRITE_PATHS
            and not is_admin_email(email)
        ):
            # Gallery users may upload, even though the rest of the write API is
            # owner-only.
            path = request.url.path
            if any(path.startswith(pfx) for pfx in GALLERY_WRITE_PREFIXES) and can_see_gallery(email):
                return await call_next(request)
            return JSONResponse(
                status_code=403,
                content={"detail": "This archive is read-only for you — only the owner can make changes."},
            )

        return await call_next(request)
