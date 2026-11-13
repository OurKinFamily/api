"""Regression: "Find more faces" returned an EMPTY page for heavily-tagged
people — the ones you most want more faces for.

The scan kept the best ENRICH_CAP (1500) rows by similarity and excluded the
already-assigned ones afterwards. Those two steps look commutative and are
not. Someone with sixteen thousand confirmed faces IS the nearest neighbour
of their own bucket mean sixteen thousand times over, so all 1500 survivors
of the cap were already theirs, every one got excluded, and the endpoint
returned nothing at all. The more completely a person was tagged, the more
completely the page failed them.

Fix: build an `ineligible` mask from the exclude set BEFORE ranking, so the
cap spends its 1500 slots on rows that can actually be returned.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
import app.routers.faces as faces


# More tagged faces than the cap, so capping-before-excluding wipes the page.
N_TAGGED = faces.ENRICH_CAP + 100
DIM = 8

CAND_PATH = "archive/2001/07/candidate.jpg"


def _rows():
    """The person's own faces, spread over >5 years so temporal bucketing runs."""
    return [
        {
            "photo_path": f"archive/{1990 + i % 20}/01/tagged{i}.jpg",
            "face_index": 0,
            "ts": f"{1990 + i % 20}-01-01T00:00:00Z",
        }
        for i in range(N_TAGGED)
    ]


def _index(rows):
    """Tagged faces sit exactly on the bucket mean (sim 1.0); the one genuine
    candidate is nearby but strictly lower (sim 0.9) — so it is precisely what
    a similarity-ranked cap throws away first."""
    matrix = np.zeros((N_TAGGED + 1, DIM), dtype=np.float32)
    matrix[:N_TAGGED, 0] = 1.0
    matrix[N_TAGGED, 0] = 0.9
    matrix[N_TAGGED, 1] = float(np.sqrt(1.0 - 0.81))

    meta = [
        {"photo_path": r["photo_path"], "face_index": 0, "crop_path": f"crops/{i}.jpg"}
        for i, r in enumerate(rows)
    ]
    meta.append({"photo_path": CAND_PATH, "face_index": 0, "crop_path": "crops/cand.jpg"})

    lookup = {(m["photo_path"], m["face_index"]): i for i, m in enumerate(meta)}
    return matrix, meta, lookup


def _session_returning(rows):
    @asynccontextmanager
    async def _s():
        session = AsyncMock()

        def _run(*a, **k):
            r = AsyncMock()
            r.data.return_value = rows
            return r

        session.run.side_effect = _run
        yield session

    return _s


def _patch_enrichment(monkeypatch):
    async def _resolve(paths):
        return {p: p for p in paths}

    async def _empty_set(*a, **k):
        return set()

    async def _empty_dict(*a, **k):
        return {}

    monkeypatch.setattr(faces, "_resolve_real_paths", _resolve)
    monkeypatch.setattr(faces, "_get_neo4j_assigned", _empty_set)
    monkeypatch.setattr(faces, "_get_connection_ids", _empty_set)
    monkeypatch.setattr(faces, "_get_photo_meta", _empty_dict)
    monkeypatch.setattr(faces, "_get_cooccurrence", _empty_dict)


@pytest.mark.asyncio
async def test_candidate_survives_when_tagged_faces_exceed_cap(monkeypatch):
    rows = _rows()
    monkeypatch.setattr(faces, "_load_embedding_index", lambda: _index(rows))
    monkeypatch.setattr(faces, "get_session", _session_returning(rows))
    _patch_enrichment(monkeypatch)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/faces/search/by_person_temporal",
            json={"person_id": "p1", "threshold": 0.5, "unassigned_only": True},
        )

    assert r.status_code == 200, r.text
    paths = [x["photo_path"] for x in r.json()["results"]]

    # Pre-fix this list was empty: the cap kept 1500 sim-1.0 rows that were all
    # the person's own, and the exclusion then removed every one of them.
    assert CAND_PATH in paths


@pytest.mark.asyncio
async def test_own_faces_never_returned_as_candidates(monkeypatch):
    """The companion guarantee — the mask must not leak the person's own faces
    back in. Excluding before the cap has to exclude just as hard."""
    rows = _rows()
    monkeypatch.setattr(faces, "_load_embedding_index", lambda: _index(rows))
    monkeypatch.setattr(faces, "get_session", _session_returning(rows))
    _patch_enrichment(monkeypatch)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post(
            "/faces/search/by_person_temporal",
            json={"person_id": "p1", "threshold": 0.5, "unassigned_only": True},
        )

    assert r.status_code == 200, r.text
    returned = {x["photo_path"] for x in r.json()["results"]}
    tagged = {row["photo_path"] for row in rows}
    assert not (returned & tagged)
