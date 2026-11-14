"""Regression: dismissing a face in the lightbox appeared to do nothing.

"Dismiss this face" is for the strangers in the background — a family archive
is full of faces nobody intends to name, and every one of them otherwise sits
in the tagging queue forever, making the real work look endless. It posts to
/faces/skip, which records the index on the Media node as `skipped_faces` so
the decision survives re-clustering.

The queue counts have honoured that list since it was added. /gallery/detail
never did: it built `unidentified` by excluding only the faces somebody had
*assigned*. So the same stranger came back in the panel on the next look, and
there was no way to tell the dismissal had been saved at all.
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


def _session(media_node: dict, people: list[dict]):
    """detail runs the APPEARS_IN people query, then fetches the node."""
    @asynccontextmanager
    async def _s():
        session = AsyncMock()

        def _run(*args, **kwargs):
            result = AsyncMock()
            cypher = args[0] if args else ""
            if "APPEARS_IN" in cypher:
                result.data.return_value = people
                result.single.return_value = None
            elif "RETURN p" in cypher:
                result.single.return_value = {"p": media_node}
            else:
                result.data.return_value = []
                result.single.return_value = None
            return result

        session.run.side_effect = _run
        yield session

    return _s


def _archive(tmp_path, rel, n_faces, crops=True):
    """A file, a sidecar, and the crop files the endpoint checks for."""
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(b"x")
    faces = []
    for i in range(n_faces):
        crop = f"__faces/crops/2026/11/{full.name}_face{i}.jpg"
        if crops:
            cp = tmp_path / crop
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(b"crop")
        faces.append({
            "face_index": i,
            "bbox": [i, i, i + 10, i + 10],
            "confidence": 0.9,
            "crop_path": f"/photos/{crop}",
        })
    (tmp_path / (rel + ".faces.json")).write_text(
        json.dumps({"num_faces": n_faces, "faces": faces}))
    return full


async def _detail(rel):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/gallery/detail", params={"path": rel})
    assert r.status_code == 200
    return r.json()


@pytest.mark.asyncio
async def test_a_dismissed_face_stops_being_offered(tmp_path, monkeypatch):
    rel = "archive/2026/11/party.jpg"
    _archive(tmp_path, rel, 4)
    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    monkeypatch.setattr(
        "app.routers.gallery.get_session",
        _session({"path": rel, "skipped_faces": [1, 2]}, []),
    )

    d = await _detail(rel)
    assert sorted(f["face_index"] for f in d["unidentified"]) == [0, 3]


@pytest.mark.asyncio
async def test_the_detected_count_still_counts_them(tmp_path, monkeypatch):
    # Nothing is deleted and the face stays in the sidecar — "4 faces found"
    # remains true, and the decision is reversible.
    rel = "archive/2026/11/party.jpg"
    _archive(tmp_path, rel, 4)
    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    monkeypatch.setattr(
        "app.routers.gallery.get_session",
        _session({"path": rel, "skipped_faces": [1, 2]}, []),
    )

    assert (await _detail(rel))["face_count"] == 4


@pytest.mark.asyncio
async def test_nothing_dismissed_leaves_every_face_offered(tmp_path, monkeypatch):
    rel = "archive/2026/11/party.jpg"
    _archive(tmp_path, rel, 3)
    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    monkeypatch.setattr(
        "app.routers.gallery.get_session", _session({"path": rel}, []))

    d = await _detail(rel)
    assert sorted(f["face_index"] for f in d["unidentified"]) == [0, 1, 2]


@pytest.mark.asyncio
async def test_dismissed_and_assigned_faces_both_drop_out(tmp_path, monkeypatch):
    rel = "archive/2026/11/party.jpg"
    _archive(tmp_path, rel, 4)
    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    monkeypatch.setattr(
        "app.routers.gallery.get_session",
        _session(
            {"path": rel, "skipped_faces": [2]},
            [{"id": "p1", "name": "Cayce", "known_as": None, "avatar": None,
              "birth_date": None, "birth_date_precision": None,
              "face_index": 0, "crop_path": None}],
        ),
    )

    d = await _detail(rel)
    assert sorted(f["face_index"] for f in d["unidentified"]) == [1, 3]


@pytest.mark.asyncio
async def test_a_rubbish_skip_list_does_not_break_the_panel(tmp_path, monkeypatch):
    # The property is written by several routes over the years. A bad entry
    # should cost that one exclusion, not the whole lightbox.
    rel = "archive/2026/11/party.jpg"
    _archive(tmp_path, rel, 3)
    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    monkeypatch.setattr(
        "app.routers.gallery.get_session",
        _session({"path": rel, "skipped_faces": [1, None, "two", "2"]}, []),
    )

    d = await _detail(rel)
    # "2" parses to 2 and counts; None and "two" are ignored.
    assert sorted(f["face_index"] for f in d["unidentified"]) == [0]
