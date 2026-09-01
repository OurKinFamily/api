"""Life-stage strip: age buckets when a birth date is known, and a decade
fallback ('through the years') when it isn't — so people with unknown DOBs
still get a strip."""
import pytest
from datetime import datetime
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.routers.people import (
    _decade_label, _decade_buckets, _pick_chosen,
    _life_buckets_for, _resolve_locks, _bucket_bounds,
)


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
        # Age buckets are keyed "y<lo>" or "y<lo>-<hi>", never decade labels.
        assert all(b.startswith("y") for b in labels)
        assert not any(b.endswith("0s") for b in labels)
        # Infancy and the teen photo land in different buckets.
        assert len(labels) == 2

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


class TestAdaptiveAgeBuckets:
    """The strip should stay roughly _TARGET_TILES long whatever the age."""

    def test_gives_a_young_child_one_bucket_per_year(self):
        # regression: a fixed named ladder (baby/toddler/kid/...) gave an
        # 8-year-old only three buckets, so 15,180 photos of Margaret rendered
        # as three tiles.
        buckets = _life_buckets_for(0, 8)
        assert len(buckets) == 9
        assert all(lo == hi for lo, hi, _ in buckets)
        assert [label for _, _, label in buckets][:3] == ["y0", "y1", "y2"]

    def test_widens_the_stride_as_the_span_grows(self):
        # A 20-year-old gets two-year strides, a 40-year-old four.
        assert _life_buckets_for(0, 20)[0][:2] == (0, 1)
        assert _life_buckets_for(0, 40)[0][:2] == (0, 3)

    @pytest.mark.parametrize("hi_age", [2, 8, 10, 20, 30, 40, 60, 80, 100])
    def test_stays_near_the_target_length_at_any_age(self, hi_age):
        assert len(_life_buckets_for(0, hi_age)) <= 12

    def test_strides_the_photographed_range_not_the_whole_life(self):
        # Someone photographed only between 20 and 70 should get a dense strip
        # of those years, not ten empty brackets covering an unphotographed
        # childhood.
        buckets = _life_buckets_for(20, 70)
        assert buckets[0][0] == 20
        assert 10 <= len(buckets) <= 12


class TestLockResolution:
    def _still(self, path, age):
        return {"path": path, "age_years": age}

    def test_rekeys_a_legacy_named_lock_onto_the_current_bucket(self):
        # regression: locks were matched on the label they were saved under.
        # Replacing the named ladder would have orphaned every existing lock
        # (Margaret had 'toddler' and 'kid' locked) the moment labels changed.
        stills = [self._still("p/two.jpg", 2.4)]
        buckets = _life_buckets_for(0, 8)
        assert _resolve_locks({"toddler": "p/two.jpg"}, stills, buckets) == {"y2": "p/two.jpg"}

    def test_places_a_mid_year_photo_in_its_own_year(self):
        # regression(2026-08-31): buckets are whole years but age_years is
        # fractional, so an inclusive upper bound (2 <= 2.4 <= 2) matched
        # nothing once the stride narrowed to one year — most photos would have
        # fallen into no bucket at all.
        stills = [self._still("p/mid.jpg", 2.9)]
        out = _resolve_locks({"y2": "p/mid.jpg"}, stills, _life_buckets_for(0, 8))
        assert out == {"y2": "p/mid.jpg"}

    def test_drops_a_lock_whose_photo_left_the_pool(self):
        out = _resolve_locks({"kid": "p/gone.jpg"}, [], _life_buckets_for(0, 8))
        assert out == {}

    def test_survives_a_widening_stride(self):
        # Same lock, same photo, but the person has aged into 2-year strides.
        stills = [self._still("p/two.jpg", 2.4)]
        out = _resolve_locks({"y2": "p/two.jpg"}, stills, _life_buckets_for(0, 20))
        assert out == {"y2-3": "p/two.jpg"}


class TestBucketBounds:
    @pytest.mark.parametrize("label,expected", [
        ("y5", (5, 5)),
        ("y4-7", (4, 7)),
        ("y0", (0, 0)),
    ])
    def test_parses_age_bucket_keys(self, label, expected):
        assert _bucket_bounds(label) == expected

    @pytest.mark.parametrize("label", ["1990s", "toddler", "", "yy", "y-3"])
    def test_rejects_anything_else(self, label):
        assert _bucket_bounds(label) is None
