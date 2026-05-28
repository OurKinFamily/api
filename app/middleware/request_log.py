"""Per-request structured logging middleware.

For every HTTP request: assigns a short request-id, captures the user
email from the Cloudflare Access header (falling back to DEV_USER_EMAIL),
times the handler, and emits one `request.completed` log line on the
way out. Downstream handlers can `logger.bind(request_id=..., by=...)`
themselves for finer detail.
"""

import os
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.log import logger

CF_EMAIL_HEADER = "cf-access-authenticated-user-email"


class RequestLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        user = (
            request.headers.get(CF_EMAIL_HEADER)
            or os.environ.get("DEV_USER_EMAIL", "stephenyoung7267@gmail.com")
        )
        request.state.request_id = request_id
        request.state.user_email = user

        started = time.perf_counter()
        log = logger.bind(
            request_id=request_id,
            by=user,
            method=request.method,
            path=request.url.path,
        )
        try:
            response = await call_next(request)
        except Exception:
            latency_ms = round((time.perf_counter() - started) * 1000, 1)
            log.bind(event="request.failed", latency_ms=latency_ms).exception(
                "request.failed"
            )
            raise

        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        log.bind(
            event="request.completed",
            status=response.status_code,
            latency_ms=latency_ms,
        ).info("request.completed")
        return response
