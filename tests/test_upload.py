"""Phone upload pipeline — POST /gallery/upload + GET /gallery/upload/{job_id}.

mpp (subprocess), ffmpeg poster, and the Neo4j session are stubbed; the tests
cover the orchestration: landing path by destination, dedupe on md5, readiness
passthrough, and the job status shape.
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.services import upload as up


# A realistic mpp metadata block for a dated, geotagged phone photo (score 100).
def _meta(md5="abc123", score=100, ts="2023-05-04T13:22:01-04:00", gps=True):
    m = {
        "hashes": {"md5": md5, "perceptual": "ph_" + md5},
        "timestamps": {"primary": {"timestamp": ts, "source": "exif", "confidence": "high", "precision": "day"}},
        "media": {"dimensions": {"width": 4032, "height": 3024}},
        "camera": {"make": "Apple", "model": "iPhone 14"},
        "archiveReadiness": {"score": score, "mediaType": "plain_photo",
                             "checks": {"hasDate": True, "hasGps": gps, "hasMd5": True}, "missing": []},
    }
    if gps:
        m["location"] = {"primary": {"latitude": 41.9, "longitude": -87.6, "source": "exif"}}
    return m


def _session(dup_path=None):
    """Stub get_session. The dedupe query returns dup_path; the MERGE returns a path."""
    @asynccontextmanager
    async def _s():
        session = AsyncMock()

        def _run(query, **kwargs):
            r = AsyncMock()
            if "content_hash" in query and "MERGE" not in query:
                r.single.return_value = {"path": dup_path} if dup_path else None
            else:
                r.single.return_value = {"path": kwargs.get("path", "x")}
            return r

        session.run.side_effect = _run
        yield session

    return _s


def _stub(monkeypatch, tmp_path, meta, dup_path=None):
    monkeypatch.setattr(up.settings, "photos_root", tmp_path)
    monkeypatch.setattr(up, "get_session", _session(dup_path))
    # mpp writes the sidecar next to the input and returns metadata.
    def fake_mpp(path):
        (type(path)(str(path) + ".json")).write_text(json.dumps({"results": [{"metadata": meta}]}))
        return meta
    monkeypatch.setattr(up, "_run_mpp", fake_mpp)
    monkeypatch.setattr(up, "_make_poster", lambda v: None)
    # Never fire a real `docker exec` face worker from the suite; the two
    # face-specific tests install their own spy after calling _stub.
    monkeypatch.setattr(up, "trigger_face_detection", lambda paths: None)


async def _upload(destination="gallery", files=None):
    files = files or [("files", ("beach.jpg", b"\xff\xd8jpegbytes", "image/jpeg"))]
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/gallery/upload", files=files, data={"destination": destination})
    return r


@pytest.mark.asyncio
async def test_upload_lands_in_archive_and_polls_done(tmp_path, monkeypatch):
    _stub(monkeypatch, tmp_path, _meta())
    r = await _upload("gallery")
    assert r.status_code == 202
    job = r.json()
    assert job["status"] == "processing"
    assert len(job["files"]) == 1

    # the endpoint scheduled the real background task; poll until it finishes
    # (awaiting sleep yields the loop so the task can run).
    import asyncio
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        for _ in range(100):
            s = await client.get(f"/gallery/upload/{job['id']}")
            if s.json()["status"] == "done":
                break
            await asyncio.sleep(0.02)
    assert s.status_code == 200
    f = s.json()["files"][0]
    assert f["status"] == "done"
    assert f["path"].startswith("archive/2023/05/")
    assert f["readiness"]["score"] == 100
    assert f["readiness"]["ready"] is True
    # file + sidecar actually landed
    assert (tmp_path / f["path"]).exists()
    assert (tmp_path / (f["path"] + ".json")).exists()


@pytest.mark.asyncio
async def test_staging_destination_flags_staged(tmp_path, monkeypatch):
    captured = {}
    _stub(monkeypatch, tmp_path, _meta())

    async def spy(row):
        captured.update(row)
    monkeypatch.setattr(up, "_create_node", spy)

    job = await up.create_job(_fake_uploads(), "staging", "cayce@x.com")
    await up.process_job(job["id"])
    assert captured["staged"] is True
    assert captured["path"].startswith("staging/uploads/2023/05/")


@pytest.mark.asyncio
async def test_dedupe_drops_upload_and_points_at_existing(tmp_path, monkeypatch):
    _stub(monkeypatch, tmp_path, _meta(md5="dupehash"), dup_path="archive/2019/01/original.jpg")
    job = await up.create_job(_fake_uploads(), "gallery", None)
    await up.process_job(job["id"])
    f = up.load_job(job["id"])["files"][0]
    assert f["status"] == "duplicate"
    assert f["duplicate_of"] == "archive/2019/01/original.jpg"
    # nothing left behind in archive
    assert not list((tmp_path / "archive").rglob("*.jpg")) if (tmp_path / "archive").exists() else True


@pytest.mark.asyncio
async def test_null_metadata_blocks_do_not_crash(tmp_path, monkeypatch):
    # regression: mpp can emit a block present-but-null (e.g. an undated
    # screenshot with "timestamps": null / "location": null). `.get(k, {})`
    # returns None then, so a bare .get() crashed with
    # "'NoneType' object has no attribute 'get'". Coerce every block with `or {}`.
    meta = {
        "hashes": {"md5": "nulls"},
        "timestamps": None,
        "location": None,
        "media": None,
        "camera": None,
        "archiveReadiness": {"score": 20, "checks": {}, "missing": ["date"]},
    }
    _stub(monkeypatch, tmp_path, meta)
    job = await up.create_job(_fake_uploads(), "gallery", None)
    await up.process_job(job["id"])
    f = up.load_job(job["id"])["files"][0]
    assert f["status"] == "done"
    assert f["path"].startswith("archive/0000/00/")   # undated bucket
    assert f["readiness"]["score"] == 20


@pytest.mark.asyncio
async def test_gallery_photos_trigger_face_detection(tmp_path, monkeypatch):
    _stub(monkeypatch, tmp_path, _meta())
    calls = []
    monkeypatch.setattr(up, "trigger_face_detection", lambda paths: calls.append(paths))
    job = await up.create_job(_fake_uploads(), "gallery", None)
    await up.process_job(job["id"])
    assert len(calls) == 1
    assert calls[0][0].startswith("archive/2023/05/")


@pytest.mark.asyncio
async def test_staged_upload_skips_face_detection(tmp_path, monkeypatch):
    _stub(monkeypatch, tmp_path, _meta())
    calls = []
    monkeypatch.setattr(up, "trigger_face_detection", lambda paths: calls.append(paths))
    job = await up.create_job(_fake_uploads(), "staging", None)
    await up.process_job(job["id"])
    assert calls == []   # staged items wait until promoted


@pytest.mark.asyncio
async def test_face_detection_disabled_when_no_worker(tmp_path, monkeypatch):
    monkeypatch.setattr(up, "FACE_WORKER", "")
    # should no-op without touching docker
    await up._run_face_detection(["archive/2023/05/x.jpg"])


@pytest.mark.asyncio
async def test_rejects_bad_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(up.settings, "photos_root", tmp_path)
    r = await _upload("nonsense")
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_status_404_for_unknown_job(tmp_path, monkeypatch):
    monkeypatch.setattr(up.settings, "photos_root", tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/gallery/upload/deadbeef")
    assert r.status_code == 404


def _fake_uploads():
    """A single in-memory UploadFile, for tests that call create_job directly."""
    from io import BytesIO
    from starlette.datastructures import UploadFile
    return [UploadFile(filename="beach.jpg", file=BytesIO(b"\xff\xd8jpegbytes"))]
