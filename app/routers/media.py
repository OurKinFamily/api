from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter(prefix="/media", tags=["media"])

THUMBS_ROOT = settings.photos_root / "__thumbs"


@router.get("/thumb/{path:path}")
async def serve_thumb(path: str):
    thumb = (THUMBS_ROOT / (path + ".webp")).resolve()
    try:
        thumb.relative_to(THUMBS_ROOT.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not thumb.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    return FileResponse(thumb, media_type="image/webp")


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
