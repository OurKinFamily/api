"""Manual location assignment — PATCH /gallery/media/location.

Trickles GPS to graph + sidecar + EXIF. Tests focus on the sidecar writer
(which had a present-but-null `location` key bug) and the endpoint wiring.
EXIF (exiftool subprocess) is stubbed.
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.routers.gallery import _write_sidecar_location


class TestSidecarWriter:
    def test_creates_primary_block_when_sidecar_missing(self, tmp_path):
        sc = tmp_path / "x.jpg.json"
        _write_sidecar_location(sc, 42.8, -71.1, "Plaistow")
        meta = json.loads(sc.read_text())["metadata"]
        assert meta["location"]["primary"]["latitude"] == 42.8
        assert meta["location"]["primary"]["source"] == "manual"
        assert meta["location"]["geolocation"]["city"] == "Plaistow"

    def test_handles_present_but_null_location_key(self, tmp_path):
        # regression: a sidecar with "location": null made setdefault return
        # None → TypeError on assignment. Coerce non-dicts to {}.
        sc = tmp_path / "x.jpg.json"
        sc.write_text(json.dumps({"results": [{"metadata": {"location": None}}]}))
        _write_sidecar_location(sc, 1.0, 2.0, None)
        loc = json.loads(sc.read_text())["results"][0]["metadata"]["location"]
        assert loc["primary"]["latitude"] == 1.0

    def test_writes_into_results_wrapper_when_present(self, tmp_path):
        sc = tmp_path / "x.jpg.json"
        sc.write_text(json.dumps({"results": [{"metadata": {"timestamps": {}}}]}))
        _write_sidecar_location(sc, 3.0, 4.0, None)
        assert json.loads(sc.read_text())["results"][0]["metadata"]["location"]["primary"]["longitude"] == 4.0


def _session(found=True):
    @asynccontextmanager
    async def _s():
        session = AsyncMock()

        def _run(*args, **kwargs):
            r = AsyncMock()
            r.single.return_value = {"path": "x"} if found else None
            return r

        session.run.side_effect = _run
        yield session

    return _s


@pytest.mark.asyncio
async def test_set_location_trickles_and_reports(tmp_path, monkeypatch):
    rel = "archive/1989/04/x.jpeg"
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(b"x")
    (tmp_path / (rel + ".json")).write_text(json.dumps({"results": [{"metadata": {}}]}))

    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    monkeypatch.setattr("app.routers.gallery.get_session", _session(found=True))
    # Stub the exiftool subprocess.
    monkeypatch.setattr("app.routers.gallery._write_exif_gps", lambda *a, **k: True)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.patch(
            f"/gallery/media/location?path={rel}",
            json={"latitude": 42.84, "longitude": -71.11, "place_name": "Plaistow"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "manual"
    assert body["sidecar_written"] is True
    assert body["exif_written"] is True
    # sidecar actually updated on disk
    meta = json.loads((tmp_path / (rel + ".json")).read_text())["results"][0]["metadata"]
    assert meta["location"]["primary"]["latitude"] == 42.84


@pytest.mark.asyncio
async def test_set_location_404_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.patch(
            "/gallery/media/location?path=archive/none.jpg",
            json={"latitude": 1, "longitude": 2},
        )
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_bulk_set_location(tmp_path, monkeypatch):
    paths = ["archive/a.jpg", "archive/b.jpg"]
    for p in paths:
        (tmp_path / p).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / p).write_bytes(b"x")
        (tmp_path / (p + ".json")).write_text(json.dumps({"results": [{"metadata": {}}]}))
    # one path with no file on disk → reported as failed, not crashing
    paths.append("archive/missing.jpg")

    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    monkeypatch.setattr("app.routers.gallery.get_session", _session(found=True))
    monkeypatch.setattr("app.routers.gallery._write_exif_gps", lambda *a, **k: True)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.patch(
            "/gallery/media/location/bulk",
            json={"paths": paths, "latitude": 1.0, "longitude": 2.0, "place_name": "Town"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["requested"] == 3
    assert body["updated"] == 2          # the two real files
    assert any(not x["ok"] for x in body["results"])


@pytest.mark.asyncio
async def test_bulk_redate(tmp_path, monkeypatch):
    monkeypatch.setattr("app.routers.gallery.get_session", _session(found=True))
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.patch(
            "/gallery/media/bulk",
            json={"paths": ["a", "b", "c"], "timestamp": "2000-01-01T12:00:00", "precision": "year"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["requested"] == 3
    assert body["updated"] == 3          # session stub reports every node found


@pytest.mark.asyncio
async def test_set_location_rejects_out_of_range(tmp_path, monkeypatch):
    rel = "archive/x.jpg"
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_bytes(b"x")
    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.patch(
            f"/gallery/media/location?path={rel}",
            json={"latitude": 999, "longitude": 0},
        )
    assert r.status_code == 422
