"""Videos in an album have to come back as videos.

The album endpoint read `m.kind` and compared it to the string "video". No
Media node has ever had a `kind` property -- not one of the 187,037 in the
graph -- so the read returned null, `is_video` was false for all 25,896
videos, and the app rendered each one as an <img> pointing at a .mov. An <img>
with a video source draws nothing and raises nothing: no broken-image icon, no
console error, no failed request. The album lightbox had therefore never
played a video, and the only reportable symptom was "it doesn't show
anything".

/me/favourites had the identical line and the identical bug.

Two halves, because a mock cannot catch the half that mattered. The mapping
tests prove the handler reads the row key it is given. The query tests read the
Cypher itself, which is where the bug lived -- a property name that does not
exist is perfectly valid Cypher and simply yields null.
"""
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

import app.routers.albums as albums_mod
import app.routers.me as me_mod
from app.main import app

ALBUM = {"id": "a1", "title": "Grammy's", "is_private": False, "cover_path": None}

ITEMS = [
    {"path": "archive/2025/08/clip.mov", "timestamp": "2025-08-24T03:25:24.000Z",
     "width": 1920, "height": 1440, "dominant_color": None,
     "is_video": True, "added_at": None},
    {"path": "archive/2025/08/photo.jpg", "timestamp": "2025-08-23T23:25:22.000Z",
     "width": 4032, "height": 3024, "dominant_color": None,
     "is_video": False, "added_at": None},
]


def _session(items):
    """get_album runs: metadata single(), a count single(), then items data()."""
    @asynccontextmanager
    async def _s():
        session = AsyncMock()

        def _run(*args, **kwargs):
            cypher = args[0] if args else ""
            result = AsyncMock()
            if "count(m) AS n" in cypher:
                result.single.return_value = {"n": len(items)}
                result.data.return_value = []
            elif "CONTAINS]->(m:Media)" in cypher and "RETURN m.path" in cypher:
                result.data.return_value = items
                result.single.return_value = None
            elif "APPEARS_IN" in cypher:
                result.data.return_value = []
                result.single.return_value = None
            else:  # album metadata
                result.single.return_value = {
                    "a": ALBUM, "latest_path": None,
                    "created_by_id": None, "created_by_name": None,
                }
                result.data.return_value = []
            return result

        session.run = AsyncMock(side_effect=_run)
        yield session

    return _s


@pytest.fixture
def album_client(monkeypatch):
    monkeypatch.setattr(albums_mod, "get_session", _session(ITEMS))

    async def _uid(_request):
        return "viewer-1"

    monkeypatch.setattr(albums_mod, "_current_user_id", _uid)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _item(payload, path):
    return next(i for i in payload["items"] if i["path"] == path)


class TestAlbumItemMapping:
    async def test_a_video_is_marked_as_one(self, album_client):
        async with album_client as c:
            res = await c.get("/albums/a1")
        assert res.status_code == 200
        assert _item(res.json(), "archive/2025/08/clip.mov")["is_video"] is True

    async def test_a_photograph_is_not_marked_as_a_video(self, album_client):
        async with album_client as c:
            res = await c.get("/albums/a1")
        assert _item(res.json(), "archive/2025/08/photo.jpg")["is_video"] is False

    async def test_is_video_is_a_real_boolean(self, album_client):
        """The app branches on it directly, so a null reads as false and sends
        a video down the <img> path without complaining."""
        async with album_client as c:
            res = await c.get("/albums/a1")
        for item in res.json()["items"]:
            assert isinstance(item["is_video"], bool)


class TestTheQueryItself:
    """Where the bug actually was. A missing property is valid Cypher.

    `m.kind` parses, runs, and returns null for every row, so nothing in the
    stack complains -- not the driver, not the handler, not the browser. The
    only way to catch it is to assert on the query text.
    """

    @pytest.mark.parametrize("module", [albums_mod, me_mod])
    def test_it_does_not_read_a_property_media_nodes_do_not_have(self, module):
        assert "m.kind" not in Path(module.__file__).read_text()

    @pytest.mark.parametrize("module", [albums_mod, me_mod])
    def test_it_asks_for_the_video_sublabel(self, module):
        """:Video is the schema's source of truth (api/CLAUDE.md); the legacy
        `is_video` property happens to agree on all 25,896 videos today, but
        the label is what a new import is guaranteed to set."""
        assert "m:Video AS is_video" in Path(module.__file__).read_text()

    def test_the_album_kind_field_is_left_alone(self):
        """`a.kind` on an Album node is a real and different field, written by
        the PATCH endpoint. Guard against a careless find-and-replace."""
        assert "a.kind = $kind" in Path(albums_mod.__file__).read_text()
