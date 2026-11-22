"""Regression: an album whose cover is a video showed a blank cover.

Four albums had no cover at all. Every one of them was an album whose newest
item — the implicit cover, when nobody has chosen one — happened to be a
video.

/media/thumb has always known to reach for the sibling "<video>.poster.jpg".
/media/medium never did: it handed the .mov straight to PIL, which said
`cannot identify image file`, and the endpoint turned that into a 500. The
albums list asks for covers at medium size, so the cover came back empty while
the same video's small thumbnail elsewhere on the page was fine. Nothing in
the interface said why, and the only trace was a row of 500s in the api log.

The poster is resized rather than served whole. It is a full-resolution frame
and the entire point of this endpoint is to not send that down the wire.
"""
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from app.main import app
from app.routers import media as media_router


@pytest.fixture
def archive(tmp_path, monkeypatch):
    """A photos root with its own caches, so nothing leaks between tests."""
    monkeypatch.setattr(media_router.settings, "photos_root", tmp_path)
    monkeypatch.setattr(media_router, "MEDIUM_ROOT", tmp_path / "_medium")
    monkeypatch.setattr(media_router, "THUMBS_ROOT", tmp_path / "__thumbs")
    monkeypatch.setattr(media_router, "THUMB_FALLBACK_ROOT", tmp_path / "_thumbs")
    return tmp_path


def _video(root: Path, rel: str, *, poster=True, size=(2400, 1600)):
    full = root / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_bytes(b"not an image, because it is a film")
    if poster:
        Image.new("RGB", size, (90, 140, 200)).save(
            full.parent / (full.name + ".poster.jpg"))
    return full


async def _get(url):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(url)


@pytest.mark.asyncio
async def test_a_video_cover_renders_instead_of_failing(archive):
    _video(archive, "archive/2025/12/newyear.mov")
    r = await _get("/media/medium/archive/2025/12/newyear.mov")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"


@pytest.mark.asyncio
async def test_the_frame_is_resized_not_sent_whole(archive):
    # 2400x1600 in, MEDIUM_MAX out. Serving the poster untouched would be a
    # fix that quietly made every album page heavier.
    _video(archive, "archive/2025/12/newyear.mov", size=(2400, 1600))
    r = await _get("/media/medium/archive/2025/12/newyear.mov")
    assert r.status_code == 200
    import io
    img = Image.open(io.BytesIO(r.content))
    assert max(img.size) == media_router.MEDIUM_MAX


@pytest.mark.asyncio
@pytest.mark.parametrize("ext", [".mov", ".mp4", ".m4v", ".avi"])
async def test_every_video_kind_the_archive_holds(archive, ext):
    _video(archive, f"archive/2025/12/newyear{ext}")
    r = await _get(f"/media/medium/archive/2025/12/newyear{ext}")
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_a_video_with_no_poster_says_so_rather_than_500ing(archive):
    # 404 is the honest answer: there is nothing to show. A 500 says the
    # server broke, which sent me looking in the wrong place for an afternoon.
    _video(archive, "archive/2025/12/silent.mov", poster=False)
    r = await _get("/media/medium/archive/2025/12/silent.mov")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_a_photograph_is_untouched_by_any_of_this(archive):
    rel = "archive/2025/12/tree.jpg"
    (archive / rel).parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2000, 1200), (200, 80, 80)).save(archive / rel)
    r = await _get(f"/media/medium/{rel}")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/webp"


@pytest.mark.asyncio
async def test_the_small_thumbnail_still_works_too(archive):
    # It always did — this is the behaviour /media/medium was missing, and it
    # should keep working after the extension set moved up the file.
    _video(archive, "archive/2025/12/newyear.mov")
    r = await _get("/media/thumb/archive/2025/12/newyear.mov")
    assert r.status_code == 200
