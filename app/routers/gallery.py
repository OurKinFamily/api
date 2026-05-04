"""
Gallery API — paginated media list sorted by most recent first.

Sorts by archive directory structure (year/month desc) which matches
capture date for the vast majority of photos. The 0000/ directory
(undated/unclassified) is placed last.
"""

import json
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Query
from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/gallery", tags=["gallery"])

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.webp', '.tiff', '.tif', '.bmp', '.gif'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mpg', '.mpeg'}
MEDIA_EXTS = IMAGE_EXTS | VIDEO_EXTS

_photo_index: list[Path] | None = None


def _build_index() -> list[Path]:
    global _photo_index
    if _photo_index is not None:
        return _photo_index

    archive = settings.photos_root / "archive"
    if not archive.exists():
        _photo_index = []
        return _photo_index

    dated: list[Path] = []
    undated: list[Path] = []

    for year_dir in sorted(archive.iterdir(), reverse=True):
        if not year_dir.is_dir() or year_dir.name.startswith('_'):
            continue
        is_undated = year_dir.name == "0000"
        for month_dir in sorted(year_dir.iterdir(), reverse=True):
            if not month_dir.is_dir():
                continue
            files = sorted(
                (f for f in month_dir.iterdir()
                 if f.is_file() and f.suffix.lower() in MEDIA_EXTS),
                reverse=True,
            )
            if is_undated:
                undated.extend(files)
            else:
                dated.extend(files)

    _photo_index = dated + undated
    log.info(f"Gallery index built: {len(_photo_index):,} media files")
    return _photo_index


def _read_sidecar(photo_path: Path) -> dict:
    sidecar = Path(str(photo_path) + ".json")
    if not sidecar.exists():
        return {}
    try:
        data = json.loads(sidecar.read_text())
        results = data.get("results") or []
        meta = (results[0].get("metadata", {}) if results else data.get("metadata", {}))
        ts_block = meta.get("timestamps", {})
        primary  = ts_block.get("primary", {})
        media    = meta.get("media", {})
        return {
            "timestamp":       primary.get("timestamp"),
            "confidence":      primary.get("confidence"),
            "dominant_color":  media.get("dominantColor"),
            "is_video":        photo_path.suffix.lower() in VIDEO_EXTS,
        }
    except Exception:
        return {}


@router.get("")
async def list_media(
    limit:  int = Query(default=48, le=100),
    offset: int = Query(default=0, ge=0),
):
    index = _build_index()
    total = len(index)
    page  = index[offset: offset + limit]

    photos = []
    for p in page:
        rel  = p.relative_to(settings.photos_root)
        info = _read_sidecar(p)
        photos.append({
            "path":           str(rel),
            "url":            f"/api/media/{rel}",
            "filename":       p.name,
            "timestamp":      info.get("timestamp"),
            "confidence":     info.get("confidence"),
            "dominant_color": info.get("dominant_color"),
            "is_video":       info.get("is_video", p.suffix.lower() in VIDEO_EXTS),
        })

    return {
        "photos":   photos,
        "total":    total,
        "offset":   offset,
        "has_more": offset + limit < total,
    }


@router.post("/reindex")
async def reindex():
    global _photo_index
    _photo_index = None
    count = len(_build_index())
    return {"count": count}
