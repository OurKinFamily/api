"""Regression: /faces/search/assign returned `{assigned: N}` based on the
input array length even when the Cypher MERGE silently no-op'd (because
`MATCH (photo:Media {path: ...})` didn't bind anything). That made the
ConfirmFacesPage loop forever on the same person.

Fix: count the actual edges created (via RETURN photo.path from the MERGE
clause) + report `missing` so the UI can skip.
"""
import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock
from httpx import AsyncClient, ASGITransport
from app.main import app


def _session_factory(per_call_singles):
    """`per_call_singles` is an iterable; each `session.run(...)` returns
    a result whose `.single()` yields the next value. Lets us simulate
    \"first face matched, second didn't\" with realistic Neo4j semantics."""
    it = iter(per_call_singles)

    @asynccontextmanager
    async def _session():
        session = AsyncMock()

        def _run(*_args, **_kwargs):
            result = AsyncMock()
            result.single.return_value = next(it)
            return result

        session.run.side_effect = _run
        yield session

    return _session


@pytest.mark.asyncio
async def test_assigned_count_reflects_actual_merges_not_input_length(monkeypatch):
    # First face's MATCH resolves → MERGE creates edge → row returned.
    # Second face's MATCH fails (stale path) → row is None.
    singles = [{"p": "archive/0000/x.jpg"}, None]
    monkeypatch.setattr("app.routers.faces.get_session", _session_factory(singles))

    body = {
        "person_id": "person-stephen",
        "faces": [
            {"photo_path": "archive/0000/x.jpg",          "face_index": 0},
            {"photo_path": "archive/2025/12/stale.jpg",   "face_index": 1},
        ],
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/faces/search/assign", json=body)

    assert r.status_code == 200
    data = r.json()
    assert data["assigned"] == 1, "honest count: only one MATCH actually bound"
    assert data["requested"] == 2
    assert len(data["missing"]) == 1
    assert data["missing"][0]["photo_path"] == "archive/2025/12/stale.jpg"


@pytest.mark.asyncio
async def test_all_misses_yields_zero_assigned(monkeypatch):
    # Both faces fail to MATCH (the case that caused the production loop).
    monkeypatch.setattr(
        "app.routers.faces.get_session",
        _session_factory([None, None]),
    )

    body = {
        "person_id": "person-stephen",
        "faces": [
            {"photo_path": "archive/2025/12/a.jpg", "face_index": 0},
            {"photo_path": "archive/2025/12/b.jpg", "face_index": 1},
        ],
    }
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/faces/search/assign", json=body)

    assert r.status_code == 200
    assert r.json() == {
        "assigned": 0,
        "missing": [
            {"photo_path": "archive/2025/12/a.jpg", "face_index": 0},
            {"photo_path": "archive/2025/12/b.jpg", "face_index": 1},
        ],
        "requested": 2,
    }
