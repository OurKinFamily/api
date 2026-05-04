from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import settings

router = APIRouter(prefix="/media", tags=["media"])


@router.get("/{path:path}")
async def serve_media(path: str):
    full_path = settings.photos_root / path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    # Prevent path traversal outside photos_root
    try:
        full_path.resolve().relative_to(settings.photos_root.resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden")
    return FileResponse(full_path)
