"""Regression: a dismissed face could never be named from its own photograph.

"Dismiss this face" is written for every face somebody did not tick while
assigning — and the person-specific sweeps (/faces/search/assign, reached
from "Find more" on a person page) write it too. There, not ticking a face
means "not David". It does not mean "nobody".

A family archive is mostly photographs of one relative standing next to
another, so that buried real people. On 2018-11-22_16-08-42_001.jpg, Henry's
face was dismissed while David was being confirmed; the lightbox then showed
David and nothing else, and there was no way to put Henry's name to his own
picture. 32,124 dismissals were in that state.

The dismissal still holds — see test_detail_hides_dismissed_faces.py, which
this must not undo: a stranger stays out of `unidentified` and out of the
queue. But the faces come back in a `dismissed` list of their own, so the
decision is visible and reversible rather than final and silent.
"""
import json
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


def _session(media_node: dict, people: list[dict]):
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


def _archive(tmp_path, rel, n_faces):
    full = tmp_path / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(b"x")
    faces = []
    for i in range(n_faces):
        crop = f"__faces/crops/2026/11/{full.name}_face{i}.jpg"
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


def _setup(tmp_path, monkeypatch, *, n_faces=4, skipped=None, people=None):
    rel = "archive/2026/11/party.jpg"
    _archive(tmp_path, rel, n_faces)
    monkeypatch.setattr("app.routers.gallery.settings.photos_root", tmp_path)
    node = {"path": rel}
    if skipped is not None:
        node["skipped_faces"] = skipped
    monkeypatch.setattr(
        "app.routers.gallery.get_session", _session(node, people or []))
    return rel


@pytest.mark.asyncio
async def test_a_dismissed_face_is_handed_back_separately(tmp_path, monkeypatch):
    rel = _setup(tmp_path, monkeypatch, skipped=[1, 2])
    d = await _detail(rel)
    assert sorted(f["face_index"] for f in d["dismissed"]) == [1, 2]


@pytest.mark.asyncio
async def test_it_is_still_kept_out_of_the_offered_faces(tmp_path, monkeypatch):
    # The earlier fix must survive: a dismissed stranger does not come back
    # into the tagging queue just because it is recoverable.
    rel = _setup(tmp_path, monkeypatch, skipped=[1, 2])
    d = await _detail(rel)
    assert sorted(f["face_index"] for f in d["unidentified"]) == [0, 3]


@pytest.mark.asyncio
async def test_a_dismissed_face_carries_what_it_needs_to_be_named(tmp_path, monkeypatch):
    rel = _setup(tmp_path, monkeypatch, skipped=[1])
    face = (await _detail(rel))["dismissed"][0]
    assert face["bbox"] == [1, 1, 11, 11]
    assert face["crop_url"]
    assert face["crop_path"].endswith("party.jpg_face1.jpg")


@pytest.mark.asyncio
async def test_nothing_dismissed_gives_an_empty_list(tmp_path, monkeypatch):
    rel = _setup(tmp_path, monkeypatch, n_faces=3)
    assert (await _detail(rel))["dismissed"] == []


@pytest.mark.asyncio
async def test_an_assigned_face_is_never_also_offered_as_dismissed(tmp_path, monkeypatch):
    # The two states contradict each other, and 43,138 edges in the live graph
    # held both at once — which also made the "work remaining" sum
    # double-count, so photographs read as finished with a face still unnamed.
    # Assignment wins: a named face is not a dismissed one.
    rel = _setup(
        tmp_path, monkeypatch, skipped=[0, 1],
        people=[{"id": "p1", "name": "David J. Young", "known_as": None,
                 "avatar": None, "birth_date": None,
                 "birth_date_precision": None, "face_index": 1,
                 "crop_path": None}],
    )
    d = await _detail(rel)
    assert [f["face_index"] for f in d["dismissed"]] == [0]
    assert 1 not in [f["face_index"] for f in d["dismissed"]]
    assert 1 not in [f["face_index"] for f in d["unidentified"]]
