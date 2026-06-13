"""Life-stage strip: age buckets when a birth date is known, and a decade
fallback ('through the years') when it isn't — so people with unknown DOBs
still get a strip."""
import pytest
from datetime import datetime
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.routers.people import _decade_label, _decade_buckets, _pick_chosen


# ── pure helpers ────────────────────────────────────────────────────────────

def _still(path, iso, **kw):
    return {"path": path, "ts_parsed": datetime.fromisoformat(iso),
            "favorited": kw.get("favorited", False), "solo": kw.get("solo", False)}


class TestDecadeLabel:
    def test_floors_year_to_its_decade(self):
        assert _decade_label(datetime(1994, 6, 1)) == "1990s"
        assert _decade_label(datetime(2000, 1, 1)) == "2000s"
        assert _decade_label(datetime(1969, 12, 31)) == "1960s"


class TestDecadeBuckets:
    def test_groups_by_decade_oldest_first(self):
        stills = [_still("a", "2001-01-01T00:00:00"),
                  _still("b", "1985-01-01T00:00:00"),
                  _still("c", "1988-01-01T00:00:00")]
        out = _decade_buckets(stills)
        assert [label for label, _ in out] == ["1980s", "2000s"]
        assert len(dict(out)["1980s"]) == 2


class TestPickChosen:
    def test_auto_picks_first_when_no_lock(self):
        bucket = [_still("a", "1990-01-01T00:00:00"), _still("b", "1991-01-01T00:00:00")]
        chosen, locked = _pick_chosen(bucket, None)
        assert chosen["path"] == "a"
        assert locked is False

    def test_honors_lock_when_locked_photo_in_pool(self):
        bucket = [_still("a", "1990-01-01T00:00:00"), _still("b", "1991-01-01T00:00:00")]
        chosen, locked = _pick_chosen(bucket, "b")
        assert chosen["path"] == "b"
        assert locked is True

    def test_falls_back_to_auto_when_locked_photo_gone(self):
        bucket = [_still("a", "1990-01-01T00:00:00")]
        chosen, locked = _pick_chosen(bucket, "vanished")
        assert chosen["path"] == "a"
        assert locked is False


# ── endpoints (mocked Neo4j) ────────────────────────────────────────────────

def mock_pool_session(rec):
    @asynccontextmanager
    async def _session():
        session = AsyncMock()
        result = AsyncMock()
        result.single.return_value = rec
        result.data.return_value = []
        session.run.return_value = result
        yield session
    return _session


def _photo(path, ts, **kw):
    return {"path": path, "ts": ts, "is_video": kw.get("is_video", False),
            "poster_path": None, "crop_path": kw.get("crop_path"),
            "favorited": kw.get("favorited", False), "solo": kw.get("solo", True)}


async def _get(url, rec):
    with patch("app.routers.people.get_session", mock_pool_session(rec)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            return await client.get(url)


class TestLifeStagesEndpoint:
    @pytest.mark.asyncio
    async def test_falls_back_to_decade_buckets_without_birth_date(self):
        rec = {"birth": None, "death": None, "locks": [], "photos": [
            _photo("p/x.jpg", "1994-06-01T12:00:00"),
            _photo("p/y.jpg", "2003-01-01T00:00:00"),
            _photo("p/z.jpg", "2005-01-01T00:00:00"),
        ]}
        res = await _get("/people/pid/life-stages", rec)
        assert res.status_code == 200
        buckets = res.json()["buckets"]
        assert [b["bucket"] for b in buckets] == ["1990s", "2000s"]
        # caption is the decade label, not an age phrase
        assert all(b["age_text"] == b["bucket"] for b in buckets)
        assert next(b for b in buckets if b["bucket"] == "2000s")["count"] == 2

    @pytest.mark.asyncio
    async def test_uses_age_buckets_when_birth_date_present(self):
        rec = {"birth": "1986-04-15", "death": None, "locks": [], "photos": [
            _photo("p/baby.jpg", "1986-09-01T00:00:00"),   # ~0
            _photo("p/teen.jpg", "2002-06-01T00:00:00"),   # ~16
        ]}
        res = await _get("/people/pid/life-stages", rec)
        assert res.status_code == 200
        labels = [b["bucket"] for b in res.json()["buckets"]]
        assert "baby" in labels and "teen" in labels
        # not decade labels
        assert not any(b.endswith("0s") for b in labels)

    @pytest.mark.asyncio
    async def test_empty_without_any_dated_photos(self):
        rec = {"birth": None, "death": None, "locks": [], "photos": []}
        res = await _get("/people/pid/life-stages", rec)
        assert res.status_code == 200
        assert res.json()["buckets"] == []


class TestLifeStageCandidatesEndpoint:
    @pytest.mark.asyncio
    async def test_decade_bucket_candidates_without_birth_date(self):
        rec = {"birth": None, "death": None, "locks": [], "photos": [
            _photo("p/x.jpg", "1994-06-01T12:00:00"),
            _photo("p/w.jpg", "1996-01-01T00:00:00"),
        ]}
        res = await _get("/people/pid/life-stages/1990s/candidates", rec)
        assert res.status_code == 200
        cands = res.json()["candidates"]
        assert len(cands) == 2
        assert all(c["age_text"] == "1990s" for c in cands)

    @pytest.mark.asyncio
    async def test_non_decade_bucket_is_404_without_birth_date(self):
        rec = {"birth": None, "death": None, "locks": [], "photos": [
            _photo("p/x.jpg", "1994-06-01T12:00:00"),
        ]}
        res = await _get("/people/pid/life-stages/twenties/candidates", rec)
        assert res.status_code == 404
