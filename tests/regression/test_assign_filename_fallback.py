"""Regression: face assign + Similar-Faces silently no-op'd for mis-dated
photos. The face index records a photo by its DATE-bucket path
(archive/2025/12/…), but the file + Media node live at archive/0000/…. The
assign's exact-path `MATCH (photo:Media {path})` matched nothing, so the edge
was never created and the person's count never moved.

Fix: the assign query now falls back to a UNIQUE filename match (and the
search remaps result paths to the real node path). This asserts the endpoint
passes the basename + the query carries the filename fallback, and the honest
count reflects a resolved match.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


def _capturing_session(seen):
    @asynccontextmanager
    async def _session():
        session = AsyncMock()

        def _run(*args, **kwargs):
            cypher = args[0] if args else ""
            seen.append((cypher, kwargs))
            result = AsyncMock()
            # The assign query RETURNs photo.path → resolved real path.
            if "MERGE" in cypher and "APPEARS_IN" in cypher and "RETURN photo.path" in cypher:
                result.single.return_value = {"p": "archive/0000/1752296692516-000.jpg"}
            else:
                result.single.return_value = None
                result.data.return_value = []
            return result

        session.run.side_effect = _run
        yield session

    return _session


@pytest.mark.asyncio
async def test_assign_passes_basename_and_resolves(monkeypatch):
    seen = []
    monkeypatch.setattr("app.routers.faces.get_session", _capturing_session(seen))
    async def _noop(*a, **k): return None
    monkeypatch.setattr("app.services.brain.rebuild_person_brain", _noop)
    monkeypatch.setattr("app.routers.faces._invalidate_brain", lambda: None)
    monkeypatch.setattr("app.routers.faces._get_person_name", AsyncMock(return_value="Dorothy"))

    body = {
        "person_id": "p-dorothy",
        # date-bucket path that won't exact-match; basename is the unique key
        "faces": [{"photo_path": "archive/2025/12/1752296692516-000.jpg", "face_index": 0}],
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/faces/search/assign", json=body)

    assert r.status_code == 200
    data = r.json()
    assert data["assigned"] == 1
    assert data["missing"] == []

    # The MERGE query carried the basename param + a filename fallback.
    assign_calls = [(c, kw) for c, kw in seen if "MERGE" in c and "APPEARS_IN" in c]
    assert assign_calls, "expected an assign query"
    cypher, kwargs = assign_calls[0]
    assert kwargs.get("basename") == "1752296692516-000.jpg"
    assert "filename" in cypher          # the unique-filename fallback
    assert "OPTIONAL MATCH" in cypher     # exact path tried first
