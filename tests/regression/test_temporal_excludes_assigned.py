"""Regression: a just-assigned mis-dated face kept reappearing as a Similar-
Faces candidate. The in-loop exclusion keys off the face index's DATE-bucket
path, but the exclude set (person faces + assigned) is REAL-path keyed — so a
freshly-assigned archive/0000 face slipped the loop and showed up again.

Fix: after remapping candidate paths to real node paths, re-apply the exclude
set so assigned faces are dropped.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
import app.routers.faces as faces


# Person tagged on 4 dated photos; one extra "candidate" in the index whose
# date-bucket path remaps to a real path that's ALREADY assigned to her.
ROWS = [
    {"photo_path": "archive/1990/01/a.jpg", "face_index": 0, "ts": "1990-01-01T00:00:00Z"},
    {"photo_path": "archive/1992/01/b.jpg", "face_index": 0, "ts": "1992-01-01T00:00:00Z"},
    {"photo_path": "archive/1994/01/c.jpg", "face_index": 0, "ts": "1994-01-01T00:00:00Z"},
    {"photo_path": "archive/1996/01/d.jpg", "face_index": 0, "ts": "1996-01-01T00:00:00Z"},
]
CAND_BUCKET = "archive/2025/12/cand.jpg"   # what the face index records
CAND_REAL   = "archive/0000/cand.jpg"      # the real node/file — already assigned


def _index():
    # All-ones embeddings → every cosine sim is 1.0 ≥ threshold, so the
    # candidate is a strong "hit" that the exclusion must remove.
    matrix = np.ones((5, 8), dtype=np.float32)
    meta = [
        {"photo_path": "archive/1990/01/a.jpg", "face_index": 0, "crop_path": "crops/a.jpg"},
        {"photo_path": "archive/1992/01/b.jpg", "face_index": 0, "crop_path": "crops/b.jpg"},
        {"photo_path": "archive/1994/01/c.jpg", "face_index": 0, "crop_path": "crops/c.jpg"},
        {"photo_path": "archive/1996/01/d.jpg", "face_index": 0, "crop_path": "crops/d.jpg"},
        {"photo_path": CAND_BUCKET, "face_index": 0, "crop_path": "crops/cand.jpg"},
    ]
    lookup = {(m["photo_path"], m["face_index"]): i for i, m in enumerate(meta)}
    return matrix, meta, lookup


def _session_returning(rows):
    @asynccontextmanager
    async def _s():
        session = AsyncMock()

        def _run(*a, **k):
            r = AsyncMock(); r.data.return_value = rows; return r

        session.run.side_effect = _run
        yield session

    return _s


@pytest.mark.asyncio
async def test_assigned_real_path_is_excluded_after_remap(monkeypatch):
    monkeypatch.setattr(faces, "_load_embedding_index", lambda: _index())
    monkeypatch.setattr(faces, "get_session", _session_returning(ROWS))
    # candidate's date-bucket path resolves to its real (already-assigned) path
    async def _resolve(paths): return {CAND_BUCKET: CAND_REAL}
    monkeypatch.setattr(faces, "_resolve_real_paths", _resolve)
    # that real path is already assigned to her
    async def _assigned(): return {(CAND_REAL, 0)}
    async def _empty_set(*a, **k): return set()
    async def _empty_dict(*a, **k): return {}
    monkeypatch.setattr(faces, "_get_neo4j_assigned", _assigned)
    monkeypatch.setattr(faces, "_get_connection_ids", _empty_set)
    monkeypatch.setattr(faces, "_get_photo_meta", _empty_dict)
    monkeypatch.setattr(faces, "_get_cooccurrence", _empty_dict)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/faces/search/by_person_temporal", json={
            "person_id": "p1", "threshold": 0.5, "unassigned_only": True,
        })

    assert r.status_code == 200, r.text
    paths = [x["photo_path"] for x in r.json()["results"]]
    # Pre-fix: CAND_REAL leaked back in. Now it's excluded.
    assert CAND_REAL not in paths
    assert CAND_BUCKET not in paths


@pytest.mark.asyncio
async def test_candidate_with_no_media_node_is_dropped(monkeypatch):
    # A face extracted from a bio crop / video poster has no backing Media node,
    # so it's unassignable — it must NOT appear as a candidate (else assigning
    # it silently no-ops and it reappears forever).
    monkeypatch.setattr(faces, "_load_embedding_index", lambda: _index())
    monkeypatch.setattr(faces, "get_session", _session_returning(ROWS))
    async def _resolve(paths): return {}    # candidate resolves to NO node
    monkeypatch.setattr(faces, "_resolve_real_paths", _resolve)
    async def _empty_set(*a, **k): return set()
    async def _empty_dict(*a, **k): return {}
    monkeypatch.setattr(faces, "_get_neo4j_assigned", _empty_set)
    monkeypatch.setattr(faces, "_get_connection_ids", _empty_set)
    monkeypatch.setattr(faces, "_get_photo_meta", _empty_dict)
    monkeypatch.setattr(faces, "_get_cooccurrence", _empty_dict)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/faces/search/by_person_temporal", json={
            "person_id": "p1", "threshold": 0.5, "unassigned_only": True,
        })

    assert r.status_code == 200, r.text
    assert r.json()["results"] == []   # nothing assignable → empty
