"""Gallery filter query-building: GPS presence, camera model, unassigned faces.

These translate request params into WHERE conditions. Assert the generated
Cypher carries the right predicate (and that defaults add no predicate), so a
future refactor can't silently drop a filter.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


def _capture(seen: list, *, rows=None):
    @asynccontextmanager
    async def _session():
        session = AsyncMock()

        def _run(*args, **kwargs):
            seen.append((args[0] if args else "", kwargs))
            result = AsyncMock()
            result.single.return_value = {"total": 0}
            result.data.return_value = rows or []
            return result

        session.run.side_effect = _run
        yield session

    return _session


async def _call(monkeypatch, query):
    seen: list = []
    monkeypatch.setattr("app.routers.gallery.get_session", _capture(seen))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get(f"/gallery?{query}")
    assert r.status_code == 200
    # The data query (the one with RETURN p) carries the WHERE clause.
    return next(c for c, _ in seen if "RETURN p" in c), seen


@pytest.mark.asyncio
class TestGpsFilter:
    async def test_has_gps_requires_latitude(self, monkeypatch):
        cypher, _ = await _call(monkeypatch, "gps=has")
        assert "p.latitude IS NOT NULL" in cypher

    async def test_no_gps_requires_null_latitude(self, monkeypatch):
        cypher, _ = await _call(monkeypatch, "gps=none")
        assert "p.latitude IS NULL" in cypher

    async def test_both_adds_no_gps_predicate(self, monkeypatch):
        cypher, _ = await _call(monkeypatch, "gps=both")
        assert "latitude" not in cypher


@pytest.mark.asyncio
class TestCameraFilter:
    async def test_camera_model_binds_exact_match(self, monkeypatch):
        cypher, seen = await _call(monkeypatch, "camera_model=iPhone+X")
        assert "p.camera_model = $camera_model" in cypher
        # param threaded through to the driver
        assert any(kw.get("camera_model") == "iPhone X" for _, kw in seen)

    async def test_absent_camera_adds_no_predicate(self, monkeypatch):
        cypher, _ = await _call(monkeypatch, "")
        assert "camera_model" not in cypher


@pytest.mark.asyncio
class TestUnassignedFacesFilter:
    async def test_unassigned_faces_predicate(self, monkeypatch):
        cypher, _ = await _call(monkeypatch, "unassigned_faces=true")
        assert "p.face_count > 0" in cypher
        assert "APPEARS_IN" in cypher
        assert "skipped_faces" in cypher

    async def test_off_by_default(self, monkeypatch):
        cypher, _ = await _call(monkeypatch, "")
        assert "face_count" not in cypher
