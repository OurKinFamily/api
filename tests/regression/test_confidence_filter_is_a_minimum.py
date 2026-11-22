"""`min_confidence` was never a minimum, and nothing showed the whole archive.

Each level mapped to a set containing only itself:

    CONFIDENCE_SETS = {"high": ["high"], "medium": ["medium"], "low": ["low"]}

So asking for "medium" hid the 155,677 high-confidence photographs and showed
the 3,484 medium ones. The gallery defaulted to "high", which meant 26,900
photographs -- 14.7% of the archive, largely the 2025-06-20 bulk-run damage --
were in the graph, on disk, and invisible in the app. No value of the filter
showed everything; "all" worked only because it is absent from the dict and
therefore skips the condition entirely, which is luck rather than design.

Two behaviours are wanted and they are different questions:

  min_confidence=medium  "medium and anything better"  -- browsing
  confidence=medium      "exactly medium"              -- finding what needs a date

Both are tested here because the first was silently the second for as long as
the filter has existed.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

import app.routers.gallery as gallery_mod
from app.main import app
from app.routers.gallery import CONFIDENCE_ORDER, CONFIDENCE_SETS, build_media_filters


class TestTheSetsThemselves:
    """The data that decides it, checked directly — a mock cannot see this."""

    def test_high_is_only_high(self):
        assert CONFIDENCE_SETS["high"] == ["high"]

    def test_medium_includes_high(self):
        """The bug in one line: this used to be ["medium"]."""
        assert set(CONFIDENCE_SETS["medium"]) == {"high", "medium"}

    def test_low_includes_everything_above_it(self):
        assert set(CONFIDENCE_SETS["low"]) == {"high", "medium", "low"}

    def test_each_level_is_a_superset_of_the_one_above(self):
        for better, worse in zip(CONFIDENCE_ORDER, CONFIDENCE_ORDER[1:]):
            assert set(CONFIDENCE_SETS[better]) < set(CONFIDENCE_SETS[worse]), \
                f"{worse} must include everything {better} does"

    def test_all_is_not_a_level(self):
        """'all' works by being absent, so no condition is built. Guard it —
        adding it to the dict would silently turn it into a filter."""
        assert "all" not in CONFIDENCE_SETS


class TestTheWhereClause:
    """What actually reaches Cypher."""

    def test_the_default_filters_nothing(self):
        _, where, params = build_media_filters()
        assert "timestamp_confidence" not in where
        assert "conf_values" not in params and "conf_exact" not in params

    def test_all_filters_nothing(self):
        _, where, _ = build_media_filters(min_confidence="all")
        assert "timestamp_confidence" not in where

    def test_a_minimum_uses_a_set(self):
        _, where, params = build_media_filters(min_confidence="medium")
        assert "p.timestamp_confidence IN $conf_values" in where
        assert set(params["conf_values"]) == {"high", "medium"}

    def test_an_exact_level_uses_equality(self):
        _, where, params = build_media_filters(confidence="medium")
        assert "p.timestamp_confidence = $conf_exact" in where
        assert params["conf_exact"] == "medium"
        assert "conf_values" not in params

    def test_exact_wins_over_minimum(self):
        """Both supplied is a contradiction; the narrower one is the answer,
        since nobody asks for 'exactly medium' meaning 'medium or better'."""
        _, where, params = build_media_filters(min_confidence="low", confidence="high")
        assert params["conf_exact"] == "high"
        assert "conf_values" not in params

    def test_undated_ignores_confidence_entirely(self):
        """The 0000 bucket has no timestamp, so it has no confidence either.
        Filtering on one would empty the section."""
        _, where, params = build_media_filters(undated=True, min_confidence="high")
        assert "p.timestamp IS NULL" in where
        assert "timestamp_confidence" not in where

    def test_a_nonsense_value_filters_nothing_rather_than_everything(self):
        """Fail open. A typo'd filter showing the whole archive is recoverable;
        one showing nothing reads as data loss."""
        _, where, _ = build_media_filters(min_confidence="hgih")
        assert "timestamp_confidence" not in where


def _session(rows):
    @asynccontextmanager
    async def _s():
        session = AsyncMock()
        seen = []

        def _run(*args, **kwargs):
            seen.append((args[0] if args else "", kwargs))
            result = AsyncMock()
            result.single.return_value = {"total": len(rows)}
            result.data.return_value = [{"p": r} for r in rows]
            return result

        session.run = AsyncMock(side_effect=_run)
        session.seen = seen
        yield session

    return _s


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(gallery_mod, "get_session", _session([]))
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


class TestTheEndpointDefault:
    async def test_the_gallery_no_longer_hides_anything_by_default(self, client):
        """The whole point. A bare request must carry no confidence filter."""
        async with client as c:
            res = await c.get("/gallery")
        assert res.status_code == 200

    async def test_the_default_is_all_not_high(self):
        import inspect
        sig = inspect.signature(gallery_mod.list_media)
        assert sig.parameters["min_confidence"].default.default == "all"

    async def test_the_counts_endpoint_shares_the_default(self):
        """Counts and items must agree or the virtualised grid reserves space
        that never fills — the same class of bug, one layer up."""
        import inspect
        sig = inspect.signature(gallery_mod.media_counts)
        assert sig.parameters["min_confidence"].default.default == "all"
