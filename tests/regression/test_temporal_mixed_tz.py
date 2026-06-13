"""Regression: /faces/search/by_person_temporal 500'd with
`TypeError: can't compare offset-naive and offset-aware datetimes`.

A person tagged across the archive has photo timestamps in BOTH shapes — some
tz-aware (…Z → +00:00) and some naive (no tz). Once the dated set was big
enough + spanned ≥5 years, the temporal-bucketing path ran
`dated.sort(key=lambda x: x[1])` over the mixed datetimes and crashed.

Fix: normalise every parsed timestamp to UTC-aware before sorting.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
import app.routers.faces as faces


# Four dated faces, MIXED tz, spanning 1958→1975 (>5y) so the temporal path runs.
ROWS = [
    {"photo_path": "archive/1958/08/a.jpg", "face_index": 0, "ts": "1958-08-18T19:10:27Z"},          # aware
    {"photo_path": "archive/1965/01/b.jpg", "face_index": 0, "ts": "1965-01-01T00:00:00"},            # naive
    {"photo_path": "archive/1970/06/c.jpg", "face_index": 0, "ts": "1970-06-01T12:00:00+00:00"},      # aware
    {"photo_path": "archive/1975/03/d.jpg", "face_index": 0, "ts": "1975-03-03T03:03:03"},            # naive
]


def _index():
    # 5 embeddings: the 4 person faces + 1 candidate. lookup keys are (rel, fi).
    matrix = np.eye(5, dtype=np.float32)
    meta = [
        {"photo_path": "archive/1958/08/a.jpg", "face_index": 0, "crop_path": "crops/a.jpg"},
        {"photo_path": "archive/1965/01/b.jpg", "face_index": 0, "crop_path": "crops/b.jpg"},
        {"photo_path": "archive/1970/06/c.jpg", "face_index": 0, "crop_path": "crops/c.jpg"},
        {"photo_path": "archive/1975/03/d.jpg", "face_index": 0, "crop_path": "crops/d.jpg"},
        {"photo_path": "archive/2000/01/cand.jpg", "face_index": 0, "crop_path": "crops/cand.jpg"},
    ]
    lookup = {(m["photo_path"], m["face_index"]): i for i, m in enumerate(meta)}
    return matrix, meta, lookup


def _session_returning(rows):
    @asynccontextmanager
    async def _s():
        session = AsyncMock()

        def _run(*args, **kwargs):
            r = AsyncMock()
            r.data.return_value = rows
            return r

        session.run.side_effect = _run
        yield session

    return _s


@pytest.mark.asyncio
async def test_temporal_search_handles_mixed_tz_timestamps(monkeypatch):
    monkeypatch.setattr(faces, "_load_embedding_index", lambda: _index())
    monkeypatch.setattr(faces, "get_session", _session_returning(ROWS))

    async def _empty_set(*a, **k): return set()
    async def _empty_dict(*a, **k): return {}
    monkeypatch.setattr(faces, "_get_neo4j_assigned", _empty_set)
    monkeypatch.setattr(faces, "_get_connection_ids", _empty_set)
    monkeypatch.setattr(faces, "_get_photo_meta", _empty_dict)
    monkeypatch.setattr(faces, "_get_cooccurrence", _empty_dict)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/faces/search/by_person_temporal", json={
            "person_id": "p-mixed-tz",
            "threshold": 0.5,
            "min_bucket_faces": 2,   # 4 dated ≥ 2*2 → temporal bucketing path runs
            "unassigned_only": False,
        })

    # Pre-fix this raised TypeError → 500. Now it sorts fine and returns buckets.
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["faces_used"] == 4
    assert body["buckets"], "temporal bucketing should have engaged (span > 5y)"
