"""
Gallery API — paginated media list sorted by most recent first.

Sorts by archive directory structure (year/month desc) which matches
capture date for the vast majority of photos. The 0000/ directory
(undated/unclassified) is placed last.
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
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


@router.get("/detail")
async def media_detail(path: str = Query(...)):
    full_path = settings.photos_root / path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Media not found")

    sidecar = Path(str(full_path) + ".json")
    base = {"path": path, "filename": full_path.name}
    if not sidecar.exists():
        return base

    try:
        data    = json.loads(sidecar.read_text())
        results = data.get("results") or []
        meta    = results[0].get("metadata", {}) if results else data.get("metadata", {})

        file_m   = meta.get("file", {})
        media    = meta.get("media", {})
        dims     = media.get("dimensions", {})
        ts       = meta.get("timestamps", {}).get("primary", {})
        loc_block = meta.get("location", {})
        primary_loc = loc_block.get("primary") or {}
        geoloc   = loc_block.get("geolocation") or {}
        camera   = meta.get("camera", {})
        expos    = meta.get("settings", {})
        proc     = meta.get("processing", {})

        return {
            **base,
            "file": {
                "size":     file_m.get("size"),
                "mimeType": file_m.get("mimeType"),
            },
            "media": {
                "type":        media.get("type"),
                "format":      media.get("format"),
                "width":       dims.get("width"),
                "height":      dims.get("height"),
                "megapixels":  dims.get("megapixels"),
                "orientation": dims.get("orientation"),
                "dominantColor": media.get("dominantColor"),
                "meanColor":     media.get("meanColor"),
                "salientColor":  media.get("salientColor"),
            },
            "timestamp": {
                "value":      ts.get("timestamp"),
                "source":     ts.get("source"),
                "confidence": ts.get("confidence"),
            },
            "location": {
                "latitude":  primary_loc.get("latitude"),
                "longitude": primary_loc.get("longitude"),
                "source":    primary_loc.get("source"),
                "city":      geoloc.get("city"),
                "state":     geoloc.get("state_code"),
                "county":    geoloc.get("county_name"),
            } if primary_loc.get("latitude") else None,
            "camera": {
                "make":   camera.get("make"),
                "model":  camera.get("model"),
                "lens":   camera.get("lens"),
            } if camera else None,
            "settings": {
                "iso":         expos.get("iso"),
                "aperture":    expos.get("aperture"),
                "shutterSpeed": expos.get("shutterSpeed"),
                "focalLength": expos.get("focalLength"),
                "flash":       expos.get("flash"),
            } if expos else None,
            "processing": {
                "processor":   proc.get("processor"),
                "extractedAt": proc.get("extractedAt"),
            },
        }
    except Exception as e:
        log.warning(f"Failed to parse sidecar for {path}: {e}")
        return base


@router.post("/reindex")
async def reindex():
    global _photo_index
    _photo_index = None
    count = len(_build_index())
    return {"count": count}
