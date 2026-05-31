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
    """Each call to `session.run(<MATCH … MERGE … RETURN photo.path>)` returns
    a result whose `.single()` yields the next entry from `per_call_singles`.
    Other `session.run` calls (e.g. the `_get_person_name` lookup or the
    brain-rebuild query) get an inert AsyncMock that won't blow up — we only
    care about the per-face MERGE results here. Extra calls beyond the
    provided list resolve to `single()=None`, not StopIteration."""
    it = iter(per_call_singles)
    # Track which query produced which mock so test stays robust to extra
    # session.run calls from sibling code paths (name lookup, brain rebuild).

    @asynccontextmanager
    async def _session():
        session = AsyncMock()

        def _run(*args, **kwargs):
            result = AsyncMock()
            cypher = args[0] if args else ""
            # Only the per-face MERGE query in bulk_assign_faces consumes our
            # singles list — it RETURNs `photo.path AS p`. Anything else
            # (person-name lookup, brain rebuild) gets harmless defaults.
            if "MERGE" in cypher and "APPEARS_IN" in cypher and "RETURN photo.path" in cypher:
                try:
                    result.single.return_value = next(it)
                except StopIteration:
                    result.single.return_value = None
            else:
                result.single.return_value = None
                result.data.return_value = []
            return result

        session.run.side_effect = _run
        yield session

    return _session


@pytest.fixture
def patched_face_assign(monkeypatch):
    """Common bulk_assign_faces test patches: bypass the post-assign brain
    rebuild + cache invalidation so the test focuses on the assign loop's
    honest-count behavior."""
    async def _noop_rebuild(*_a, **_kw):
        return None
    monkeypatch.setattr("app.services.brain.rebuild_person_brain", _noop_rebuild)
    monkeypatch.setattr("app.routers.faces._invalidate_brain", lambda: None)


@pytest.mark.asyncio
async def test_assigned_count_reflects_actual_merges_not_input_length(monkeypatch, patched_face_assign):
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
async def test_all_misses_yields_zero_assigned(monkeypatch, patched_face_assign):
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
