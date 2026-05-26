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
    ts_from:        Optional[str] = Query(default=None),       # ISO timestamp, takes precedence over year_from
    ts_to:          Optional[str] = Query(default=None),       # ISO timestamp, takes precedence over year_to
    sort:           str           = Query(default="desc"),     # desc | asc
    media_type:     str           = Query(default="all"),      # photo | video | all
    person_ids:     Optional[str] = Query(default=None),       # comma-separated person IDs
):
    conditions = []
    params: dict = {"offset": offset, "limit": limit}

    if min_confidence in CONFIDENCE_SETS:
        conditions.append("p.timestamp_confidence IN $conf_values")
        params["conf_values"] = CONFIDENCE_SETS[min_confidence]

    # ts_from / ts_to (ISO timestamps) override year_from / year_to when present.
    effective_from = ts_from or (f"{year_from}-01-01" if year_from is not None else None)
    effective_to   = ts_to   or (f"{year_to + 1}-01-01" if year_to is not None else None)

    if effective_from is not None:
        conditions.append("p.timestamp >= $ts_from")
        params["ts_from"] = effective_from

    if effective_to is not None:
        conditions.append("p.timestamp < $ts_to")
        params["ts_to"] = effective_to

    if media_type == "photo":
        conditions.append("p:Photo")
    elif media_type == "video":
        conditions.append("p:Video")

    ids = [i.strip() for i in person_ids.split(",") if i.strip()] if person_ids else []
    if ids:
        conditions.append("ALL(pid IN $person_ids WHERE EXISTS { (:Person {id: pid})-[:APPEARS_IN]->(p) })")
        params["person_ids"] = ids
        match = "MATCH (p:Media)"
    else:
        match = "MATCH (p:Media)"

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    order = "ASC" if sort == "asc" else "DESC"
    count_q = f"{match} {where} RETURN count(p) AS total"
    data_q  = f"""
        {match} {where}
        RETURN p
        ORDER BY p.timestamp {order}, p.path {order}
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
        # heritage videos carry a poster jpg; regular media use the webp thumb route
        poster = p.get("poster_path")
        thumbnail_url = (
            f"/api/media/{poster}" if poster
            else f"/api/media/thumb/{path.removeprefix('archive/')}"
        )
        photos.append({
            "path":           path,
            "url":            f"/api/media/{path}",
            "thumbnail_url":  thumbnail_url,
            "filename":       p.get("filename") or Path(path).name,
            "timestamp":      p.get("timestamp"),
            "confidence":     p.get("timestamp_confidence"),
            "dominant_color": p.get("dominant_color"),
            "is_video":       p.get("is_video", False),
            "width":          p.get("width"),
            "height":         p.get("height"),
            "place_name":     p.get("place_name"),
            "city":           p.get("city"),
        })

    return {
        "media":    photos,
        "total":    total,
        "offset":   offset,
        "has_more": offset + limit < total,
    }


@router.get("/years")
async def list_years(min_confidence: str = Query(default="high")):
    """Distinct years that have media, with counts. Used by the date scrubber."""
    conf = CONFIDENCE_SETS.get(min_confidence, CONFIDENCE_SETS["high"])
    q = """
        MATCH (p:Media)
        WHERE p.timestamp_confidence IN $conf AND p.timestamp IS NOT NULL
        WITH substring(toString(p.timestamp), 0, 4) AS year
        RETURN year, count(*) AS count
        ORDER BY year DESC
    """
    async with get_session() as session:
        res = await session.run(q, conf=conf)
        rows = await res.data()
    return [{"year": int(r["year"]), "count": r["count"]} for r in rows if r["year"].isdigit()]


@router.get("/detail")
async def media_detail(path: str = Query(...)):
    full_path = settings.photos_root / path
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="Media not found")

    sidecar = Path(str(full_path) + ".json")
    base = {"path": path, "filename": full_path.name}

    # Query Neo4j for people (with face_index + crop_path from relationship)
    # and the Media node itself (for heritage fields).
    people = []
    face_count = None
    media_node = {}
    try:
        async with get_session() as session:
            result = await session.run(
                """
                MATCH (person:Person)-[rel:APPEARS_IN]->(p:Media {path: $path})
                RETURN person.id AS id, person.name AS name,
                       person.known_as AS known_as, person.avatar AS avatar,
                       rel.face_index AS face_index, rel.crop_path AS crop_path
                ORDER BY person.name
                """,
                path=path,
            )
            people = await result.data()

            node_res = await session.run(
                "MATCH (p:Media {path: $path}) RETURN p",
                path=path,
            )
            node_rec = await node_res.single()
            if node_rec:
                media_node = dict(node_rec["p"])
    except Exception as e:
        log.warning(f"Neo4j query failed for {path}: {e}")

    # Heritage fields live on the Media node (set when the item was imported
    # via mpp's heritage flags). Surface them so MediaDetail can render the
    # same scrapbook context whether the user got here from the main gallery
    # or from a collection.
    audio_url = None
    if media_node.get("audio_file"):
        audio_rel = f"{Path(path).parent}/{media_node['audio_file']}"
        if (settings.photos_root / audio_rel).exists():
            audio_url = f"/api/media/{audio_rel}"
    subtitle_url = f"/api/media/vtt/{path}" if media_node.get("has_subtitles") else None
    heritage = {
        "content_date":             media_node.get("content_date"),
        "content_date_precision":   media_node.get("content_date_precision"),
        "content_date_explanation": media_node.get("content_date_explanation"),
        "context_type":             media_node.get("context_type"),
        "context_subject":          media_node.get("title"),
        "context_notes":            media_node.get("notes"),
        "description":              media_node.get("description"),
        "transcription":            media_node.get("transcription"),
        "place_name":               media_node.get("place_name"),
        "physical_status":          media_node.get("physical_status"),
        "physical_condition":       media_node.get("physical_condition"),
        "page_number":              media_node.get("page_number"),
        "audio_url":                audio_url,
        "audio_description":        media_node.get("audio_description"),
        "subtitle_url":             subtitle_url,
    }
    # Drop empty so the frontend can check `if (detail.heritage)` cleanly.
    if not any(v not in (None, "", []) for v in heritage.values()):
        heritage = None

    # Read faces sidecar — face count + bbox lookup by face_index
    unidentified = []
    faces_sidecar = Path(str(full_path) + ".faces.json")
    if faces_sidecar.exists():
        try:
            fd = json.loads(faces_sidecar.read_text())
            face_count = fd.get("num_faces")
            bbox_by_index = {f["face_index"]: f.get("bbox") for f in fd.get("faces", [])}
            for p in people:
                p["bbox"]     = bbox_by_index.get(p.get("face_index"))
                p["crop_url"] = f"/api/media/{p['crop_path']}" if p.get("crop_path") else None

            # Derive crop paths for faces not yet assigned to anyone
            assigned_indexes = {p.get("face_index") for p in people}
            parts = Path(path).parts  # ('archive', '2026', '04', 'file.jpg')
            year, month, fname = parts[1], parts[2], parts[3]
            for face in fd.get("faces", []):
                fi = face["face_index"]
                if fi in assigned_indexes:
                    continue
                crop_path = f"__faces/crops/{year}/{month}/{fname}_face{fi}.jpg"
                crop_full = settings.photos_root / crop_path
                if crop_full.exists():
                    unidentified.append({
                        "face_index": fi,
                        "bbox":       face.get("bbox"),
                        "crop_path":  crop_path,
                        "crop_url":   f"/api/media/{crop_path}",
                        "confidence": face.get("confidence"),
                    })
        except Exception as e:
            log.warning(f"Failed to parse faces sidecar for {path}: {e}")

    # Read objects sidecar for detected labels
    objects = []
    objects_sidecar = Path(str(full_path) + ".objects.json")
    if objects_sidecar.exists():
        try:
            od = json.loads(objects_sidecar.read_text())
            counts: dict[str, int] = {}
            for det in od.get("detections", []):
                name = det.get("class_name", "")
                counts[name] = counts.get(name, 0) + 1
            objects = [{"label": k, "count": v} for k, v in sorted(counts.items())]
        except Exception:
            pass

    if not sidecar.exists():
        return {
            **base,
            "people": people, "face_count": face_count,
            "unidentified": unidentified, "objects": objects,
            "heritage": heritage,
        }

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
        landmarks   = loc_block.get("landmarks") or []
        top_landmark = next(
            (l for l in landmarks if l.get("confidence", 0) >= 0.8),
            None
        )
        camera      = meta.get("camera", {})
        expos       = meta.get("settings", {})
        proc        = meta.get("processing", {})

        return {
            **base,
            "people":      people,
            "face_count":  face_count,
            "unidentified": unidentified,
            "objects":     objects,
            "heritage":    heritage,
            "sidecar":     meta,
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
                "latitude":             primary_loc.get("latitude"),
                "longitude":            primary_loc.get("longitude"),
                "source":               primary_loc.get("source"),
                "city":                 geoloc.get("city"),
                "state":                geoloc.get("state_code"),
                "county":               geoloc.get("county_name"),
                "city_confidence":      geoloc.get("confidence"),
                "landmark":             top_landmark["landmark"]["name"] if top_landmark else None,
                "landmark_category":    top_landmark["landmark"]["category"] if top_landmark else None,
                "landmark_distance_m":  top_landmark["distance"] if top_landmark else None,
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
        return {
            **base,
            "people": people, "face_count": face_count,
            "unidentified": unidentified, "objects": objects,
            "heritage": heritage,
        }
