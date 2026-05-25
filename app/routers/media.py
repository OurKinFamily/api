import re
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from app.config import settings

router = APIRouter(prefix="/media", tags=["media"])

THUMBS_ROOT = settings.photos_root / "__thumbs"


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
    if not thumb.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(thumb, media_type="image/webp")


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


@router.get("/{path:path}")
async def serve_media(path: str):
    full_path = settings.photos_root / path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    try:
        full_path.resolve().relative_to(settings.photos_root.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden")
    return FileResponse(full_path)
