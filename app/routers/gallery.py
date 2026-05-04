"""
Gallery API — paginated media list backed by Neo4j Photo nodes.

Filters: min_confidence (default: high), year_from, year_to, media_type, person_id
Sort:    actual capture timestamp DESC
"""

import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from app.config import settings
from app.db.neo4j import get_session

log = logging.getLogger(__name__)
router = APIRouter(prefix="/gallery", tags=["gallery"])

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mpg', '.mpeg'}

CONFIDENCE_SETS = {
    "high":   ["high"],
    "medium": ["high", "medium"],
    "low":    ["high", "medium", "low"],
}


@router.get("")
async def list_media(
    limit:          int           = Query(default=48, le=200),
    offset:         int           = Query(default=0, ge=0),
    min_confidence: str           = Query(default="high"),
    year_from:      Optional[int] = Query(default=None),
    year_to:        Optional[int] = Query(default=None),
    media_type:     str           = Query(default="all"),   # photo | video | all
    person_id:      Optional[str] = Query(default=None),
):
    conditions = []
    params: dict = {"offset": offset, "limit": limit}

    if min_confidence in CONFIDENCE_SETS:
        conditions.append("p.timestamp_confidence IN $conf_values")
        params["conf_values"] = CONFIDENCE_SETS[min_confidence]

    if year_from is not None:
        conditions.append("p.timestamp >= $ts_from")
        params["ts_from"] = f"{year_from}-01-01"

    if year_to is not None:
        conditions.append("p.timestamp < $ts_to")
        params["ts_to"] = f"{year_to + 1}-01-01"

    if media_type == "photo":
        conditions.append("p.is_video = false")
    elif media_type == "video":
        conditions.append("p.is_video = true")

    if person_id:
        match = "MATCH (:Person {id: $person_id})-[:APPEARS_IN]->(p:Photo)"
        params["person_id"] = person_id
    else:
        match = "MATCH (p:Photo)"

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    count_q = f"{match} {where} RETURN count(p) AS total"
    data_q  = f"""
        {match} {where}
        RETURN p
        ORDER BY p.timestamp DESC, p.path DESC
        SKIP $offset LIMIT $limit
    """

    async with get_session() as session:
        count_res  = await session.run(count_q, **params)
        count_rec  = await count_res.single()
        total      = count_rec["total"] if count_rec else 0

        data_res   = await session.run(data_q, **params)
        records    = await data_res.data()

    photos = []
    for r in records:
        p    = r["p"]
        path = p.get("path", "")
        photos.append({
            "path":           path,
            "url":            f"/api/media/{path}",
            "filename":       p.get("filename") or Path(path).name,
            "timestamp":      p.get("timestamp"),
            "confidence":     p.get("timestamp_confidence"),
            "dominant_color": p.get("dominant_color"),
            "is_video":       p.get("is_video", False),
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

        file_m      = meta.get("file", {})
        media       = meta.get("media", {})
        dims        = media.get("dimensions", {})
        ts          = meta.get("timestamps", {}).get("primary", {})
        loc_block   = meta.get("location", {})
        primary_loc = loc_block.get("primary") or {}
        geoloc      = loc_block.get("geolocation") or {}
        camera      = meta.get("camera", {})
        expos       = meta.get("settings", {})
        proc        = meta.get("processing", {})

        return {
            **base,
            "file": {
                "size":     file_m.get("size"),
                "mimeType": file_m.get("mimeType"),
            },
            "media": {
                "type":          media.get("type"),
                "format":        media.get("format"),
                "width":         dims.get("width"),
                "height":        dims.get("height"),
                "megapixels":    dims.get("megapixels"),
                "orientation":   dims.get("orientation"),
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
                "make":  camera.get("make"),
                "model": camera.get("model"),
                "lens":  camera.get("lens"),
            } if camera else None,
            "settings": {
                "iso":          expos.get("iso"),
                "aperture":     expos.get("aperture"),
                "shutterSpeed": expos.get("shutterSpeed"),
                "focalLength":  expos.get("focalLength"),
                "flash":        expos.get("flash"),
            } if expos else None,
            "processing": {
                "processor":   proc.get("processor"),
                "extractedAt": proc.get("extractedAt"),
            },
        }
    except Exception as e:
        log.warning(f"Failed to parse sidecar for {path}: {e}")
        return base
