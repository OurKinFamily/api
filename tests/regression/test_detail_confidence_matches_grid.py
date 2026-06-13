"""Regression: a file in archive/0000 (date-conflict bucket) showed
`confidence: "high"` in the lightbox (/gallery/detail) while the gallery grid
+ filter put it in "low". Same file contradicting itself.

Cause: the grid reads the Neo4j node's `timestamp_confidence` (authoritative —
reflects the 0000 downgrade), but /gallery/detail read the *sidecar's*
`timestamps.primary.confidence` (a naive EXIF read that never saw the
downgrade). The endpoint only preferred the graph value when
`timestamp_source == "manual"`.

Fix: /gallery/detail prefers the graph node's `timestamp_confidence` over the
sidecar whenever the node has one, so the lightbox agrees with the grid.
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


def _session_with_node(media_node: dict):
    """detail runs two queries: the APPEARS_IN people query (`.data()`) then
    the node fetch (`MATCH (p:Media {path}) RETURN p` → `.single()["p"]`)."""
    @asynccontextmanager
    async def _session():
        session = AsyncMock()

        def _run(*args, **kwargs):
            result = AsyncMock()
            cypher = args[0] if args else ""
            # The people query carries APPEARS_IN; the node fetch is the bare
            # `MATCH (p:Media {path}) RETURN p`. Discriminate on APPEARS_IN so
            # the people query (whose `RETURN person.id` also contains the
            # substring "RETURN p") doesn't get mistaken for the node fetch.
            if "APPEARS_IN" in cypher:
                result.data.return_value = []
                result.single.return_value = None
            elif "RETURN p" in cypher:
                result.single.return_value = {"p": media_node}
            else:
                result.data.return_value = []
                result.single.return_value = None
            return result

        session.run.side_effect = _run
        yield session

    return _session


@pytest.mark.asyncio
async def test_detail_confidence_prefers_graph_over_sidecar(tmp_path, monkeypatch):
    # File on disk + a sidecar whose EXIF read claims "high".
    rel = "archive/0000/2025-06-20_14-13-29_001.mov"
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(b"not-a-real-video")  # PIL open fails gracefully (try/except)
    sidecar = tmp_path / (rel + ".json")
    sidecar.write_text(json.dumps({
        "results": [{"metadata": {"timestamps": {"primary": {
            "timestamp": "2022-09-21T02:03:09.000Z",
            "source": "exif",
            "confidence": "high",   # <- naive sidecar read
        }}}}]
    }))

    # Graph node — authoritative — says low (the 0000 date-conflict downgrade).
    media_node = {
        "path": rel,
        "timestamp": "2022-09-21T02:03:09.000Z",
        "timestamp_confidence": "low",
        "timestamp_source": "exif",
    }

    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    monkeypatch.setattr("app.routers.gallery.get_session", _session_with_node(media_node))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/gallery/detail", params={"path": rel})

    assert r.status_code == 200
    ts = r.json()["timestamp"]
    assert ts["confidence"] == "low", "lightbox must match the grid/filter, not the stale sidecar"
    assert ts["value"] == "2022-09-21T02:03:09.000Z"


@pytest.mark.asyncio
async def test_detail_manual_redate_still_wins(tmp_path, monkeypatch):
    # A manual redate must still override the sidecar (pre-existing behavior).
    rel = "archive/2010/05/photo.jpg"
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(b"x")
    sidecar = tmp_path / (rel + ".json")
    sidecar.write_text(json.dumps({
        "results": [{"metadata": {"timestamps": {"primary": {
            "timestamp": "2001-01-01T00:00:00.000Z", "source": "exif", "confidence": "low",
        }}}}]
    }))
    media_node = {
        "path": rel,
        "timestamp": "2010-05-04T00:00:00.000Z",
        "timestamp_confidence": "high",
        "timestamp_source": "manual",
        "timestamp_precision": "day",
    }
    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    monkeypatch.setattr("app.routers.gallery.get_session", _session_with_node(media_node))

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/gallery/detail", params={"path": rel})

    assert r.status_code == 200
    ts = r.json()["timestamp"]
    assert ts["source"] == "manual"
    assert ts["value"] == "2010-05-04T00:00:00.000Z"
    assert ts["confidence"] == "high"
