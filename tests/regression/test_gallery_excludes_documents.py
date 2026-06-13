"""Regression: scrapbook / yearbook / journal PAGES (:Document nodes, ~3,400
of them, all null-timestamp) flooded the gallery's undated view as individual
tiles. The gallery grid is photos + videos only — Document pages belong in
Collections (the collection card shows in Scrapbook). Face-crop thumbnails
(__faces) are internal artifacts that also leaked in.

Fix: /gallery list_media always excludes `:Document` and `__faces` paths.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


def _capture_session(seen_cypher: list):
    @asynccontextmanager
    async def _session():
        session = AsyncMock()

        def _run(*args, **kwargs):
            cypher = args[0] if args else ""
            seen_cypher.append(cypher)
            result = AsyncMock()
            result.single.return_value = {"total": 0}
            result.data.return_value = []
            return result

        session.run.side_effect = _run
        yield session

    return _session


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "undated=true", "min_confidence=high"])
async def test_gallery_excludes_documents_and_face_crops(monkeypatch, query):
    seen: list = []
    monkeypatch.setattr("app.routers.gallery.get_session", _capture_session(seen))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(f"/gallery?{query}")

    assert r.status_code == 200
    # Every gallery query (dated, undated, any confidence) must exclude
    # Document pages and face-crop artifacts.
    assert seen, "expected at least one Cypher query"
    for cypher in seen:
        assert "NOT p:Document" in cypher
        assert "__faces" in cypher
