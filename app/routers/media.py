import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from app.config import settings

router = APIRouter(prefix="/media", tags=["media"])

# Bump frontend MEDIA_VERSION (app/src/lib/media.js) to invalidate browser caches
# after a thumb regen. URLs are immutable for 7d unless ?v= changes.
CACHE_HEADERS = {"Cache-Control": "public, max-age=604800, immutable"}

THUMBS_ROOT = settings.photos_root / "__thumbs"
# /photos is mounted read-only in container, so on-demand caches live in
# writable temp dirs. Lost on container restart — acceptable.
MEDIUM_ROOT = Path("/tmp/medium")
MEDIUM_MAX  = 1600  # longest edge in px
THUMB_FALLBACK_ROOT = Path("/tmp/thumbs")
THUMB_MAX = 400  # longest edge in px (matches mpp thumbnail size)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif", ".tif", ".tiff", ".bmp"}


def _srt_to_vtt(srt: str) -> str:
    # WebVTT = SRT with a header and '.' decimal separators in timestamps
    body = re.sub(r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", srt)
    return "WEBVTT\n\n" + body


@router.get("/thumb/{path:path}")
async def serve_thumb(path: str):
    # Photo paths in DB are "archive/YYYY/MM/..." but thumbs omit the "archive/" prefix
    thumb_path = path.removeprefix("archive/")
    thumb = (THUMBS_ROOT / (thumb_path + ".webp")).resolve()
    try:
        thumb.relative_to(THUMBS_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden")
    if thumb.exists():
        return FileResponse(thumb, media_type="image/webp", headers=CACHE_HEADERS)

    # Resolve source under photos_root. Archive paths arrive with the
    # "archive/" prefix stripped (gallery convention); other trees like
    # "heritage/..." are passed through. Probe both.
    candidates = [settings.photos_root / path]
    if not path.startswith("archive/"):
        candidates.append(settings.photos_root / "archive" / path)
    src = next((c.resolve() for c in candidates if c.exists()), None)
    if src is not None:
        try:
            src.relative_to(settings.photos_root.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Forbidden")

    # Fallback 1: videos use the sibling "<video>.poster.jpg".
    if src is not None:
        poster = src.parent / (src.name + ".poster.jpg")
        if poster.exists():
            return FileResponse(poster, media_type="image/jpeg", headers=CACHE_HEADERS)

    # Fallback 2: image missing a webp thumb — generate on-demand, cache to /tmp.
    if src is not None and src.suffix.lower() in IMAGE_EXTS:
        cached = (THUMB_FALLBACK_ROOT / (thumb_path + ".webp")).resolve()
        try:
            cached.relative_to(THUMB_FALLBACK_ROOT.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Forbidden")
        if cached.exists():
            return FileResponse(cached, media_type="image/webp", headers=CACHE_HEADERS)
        from PIL import Image, ImageOps
        try:
            img = Image.open(src)
            img = ImageOps.exif_transpose(img)
            img.thumbnail((THUMB_MAX, THUMB_MAX), Image.LANCZOS)
            cached.parent.mkdir(parents=True, exist_ok=True)
            img.convert("RGB").save(cached, "WEBP", quality=80)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"thumb gen failed: {e}")
        return FileResponse(cached, media_type="image/webp", headers=CACHE_HEADERS)

    raise HTTPException(status_code=404, detail="Thumbnail not found")


@router.get("/vtt/{path:path}")
async def serve_vtt(path: str):
    # path is the video path; the .srt sits next to it (video stem + .srt)
    video = (settings.photos_root / path).resolve()
    try:
        video.relative_to(settings.photos_root.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden")
    srt = video.with_suffix(".srt")
    if not srt.exists():
        raise HTTPException(status_code=404, detail="Subtitles not found")
    return Response(content=_srt_to_vtt(srt.read_text(encoding="utf-8")),
                    media_type="text/vtt")


@router.get("/medium/{path:path}")
async def serve_medium(path: str):
    """Medium-res webp (longest edge MEDIUM_MAX). Generated on first request, cached on disk."""
    src_rel = path.removeprefix("archive/")
    cached = (MEDIUM_ROOT / (src_rel + ".webp")).resolve()
    try:
        cached.relative_to(MEDIUM_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden")
    if cached.exists():
        return FileResponse(cached, media_type="image/webp", headers=CACHE_HEADERS)

    src = (settings.photos_root / path).resolve()
    try:
        src.relative_to(settings.photos_root.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not src.exists():
        raise HTTPException(status_code=404, detail="File not found")

    from PIL import Image, ImageOps
    try:
        img = Image.open(src)
        img = ImageOps.exif_transpose(img)
        img.thumbnail((MEDIUM_MAX, MEDIUM_MAX), Image.LANCZOS)
        cached.parent.mkdir(parents=True, exist_ok=True)
        img.convert("RGB").save(cached, "WEBP", quality=85)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"resize failed: {e}")
    return FileResponse(cached, media_type="image/webp", headers=CACHE_HEADERS)


VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi"}


@router.get("/{path:path}")
async def serve_media(path: str):
    full_path = settings.photos_root / path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        full_path.resolve().relative_to(settings.photos_root.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden")
    # Prefer a browser-playable H.264 proxy if one exists. HEVC/ProRes originals
    # don't play in most browsers. Proxies live in a parallel /photos/__web tree
    # (mirroring the original's path) so archive scanners / extractors never see
    # them. Generated by workshop/tools/make_web_proxies.py.
    if full_path.suffix.lower() in VIDEO_EXTS:
        proxy = settings.photos_root / "__web" / Path(path).with_suffix(".web.mp4")
        if proxy.exists():
            full_path = proxy
    return FileResponse(full_path, headers=CACHE_HEADERS)
