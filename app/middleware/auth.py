from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


class AuthMiddleware(BaseHTTPMiddleware):
    """Placeholder for future authentication. Currently a no-op."""

    async def dispatch(self, request: Request, call_next):
        # TODO: validate token, set request.state.user
        return await call_next(request)
