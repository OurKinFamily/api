"""
Gallery API — paginated media list backed by Neo4j Photo nodes.

Filters: min_confidence (default: high), year_from, year_to, media_type, person_id
Sort:    actual capture timestamp DESC
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

import asyncio

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, field_validator
from app.config import settings, with_v
from app.db.neo4j import get_session
from app.log import logger
from app.services import upload as upload_svc

router = APIRouter(prefix="/gallery", tags=["gallery"])

# Local alias so the diff-against-old `log.warning(...)` calls below stay
# minimal; new code should `from app.log import logger` directly and use
# `logger.bind(...).info(...)` for structured events.
log = logger

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.mpg', '.mpeg'}

# Exclusive levels: each filter shows ONLY that confidence (not a floor), so
# "Low" surfaces only the low-confidence items to review/re-date. "all" = no filter.
CONFIDENCE_SETS = {
    "high":   ["high"],
    "medium": ["medium"],
    "low":    ["low"],
}



def build_media_filters(
    *, min_confidence="high", year_from=None, year_to=None, ts_from=None, ts_to=None,
    media_type="all", person_ids=None, undated=False, gps="both",
    camera_model=None, unassigned_faces=False,
):
    """The gallery's WHERE clause, shared by every endpoint that lists media.

    Extracted so the counts endpoint cannot drift from the items endpoint. The
    virtualised grid sizes itself from counts and then fetches items for a date
    range; if the two disagree by even one photo, the layout reserves space that
    never fills or drops photos that exist — the "hidden photos" failure again,
    with a subtler cause.
    """
    conditions = []
    params: dict = {}

    # The gallery grid is photos + videos only. Scrapbook / yearbook / journal
    # PAGES are :Document nodes — they belong in Collections (the collection
    # card shows in Scrapbook), not as ~3,400 individual tiles flooding the
    # grid. Face-crop thumbnails (__faces) are internal artifacts, never items.
    conditions.append("NOT p:Document")
    conditions.append("NOT p.path CONTAINS '__faces'")
    # Phone uploads sent to "staging" get a node but stay out of the grid until
    # a future promote step flips the flag.
    conditions.append("(p.staged IS NULL OR p.staged = false)")
    if undated:
        # The "0000" / no-date bucket — scanned albums and the like with no
        # timestamp at all. Confidence + year filters don't apply here.
        conditions.append("p.timestamp IS NULL")
    elif min_confidence in CONFIDENCE_SETS:
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
    # GPS presence — latitude/longitude are set together at extraction time.
    if gps == "has":
        conditions.append("p.latitude IS NOT NULL")
    elif gps == "none":
        conditions.append("p.latitude IS NULL")
    # "both" (or anything else) = no GPS condition.
    if camera_model:
        conditions.append("p.camera_model = $camera_model")
        params["camera_model"] = camera_model
    # Detected-but-unassigned faces. `face_count` was backfilled from the
    # .faces.json sidecars (num_faces). A face is "handled" once it's either
    # assigned (an APPEARS_IN edge carrying its face_index) or skipped
    # (m.skipped_faces). So unassigned work remains when the detected count
    # exceeds assigned + skipped.
    if unassigned_faces:
        conditions.append(
            "p.face_count > 0 AND p.face_count > "
            "size(coalesce(p.skipped_faces, [])) + "
            "COUNT { (p)<-[r:APPEARS_IN]-(:Person) WHERE r.face_index IS NOT NULL }"
        )
    ids = [i.strip() for i in person_ids.split(",") if i.strip()] if person_ids else []
    if ids:
        conditions.append("ALL(pid IN $person_ids WHERE EXISTS { (:Person {id: pid})-[:APPEARS_IN]->(p) })")
        params["person_ids"] = ids
        match = "MATCH (p:Media)"
    else:
        match = "MATCH (p:Media)"

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    return "MATCH (p:Media)", where, params


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
    undated:        bool          = Query(default=False),      # only media with NO timestamp ("0000" bucket)
    gps:            str           = Query(default="both"),     # has | none | both — GPS coordinates present?
    camera_model:   Optional[str] = Query(default=None),       # exact camera_model match
    unassigned_faces: bool        = Query(default=False),      # only media with detected-but-unassigned faces
):
    match, where, filter_params = build_media_filters(
        min_confidence=min_confidence, year_from=year_from, year_to=year_to,
        ts_from=ts_from, ts_to=ts_to, media_type=media_type,
        person_ids=person_ids, undated=undated, gps=gps,
        camera_model=camera_model, unassigned_faces=unassigned_faces,
    )
    params: dict = {"offset": offset, "limit": limit, **filter_params}


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
        thumbnail_url = with_v(
            f"/api/media/{poster}" if poster
            else f"/api/media/thumb/{path.removeprefix('archive/')}"
        )
        photos.append({
            "path":           path,
            "url":            with_v(f"/api/media/{path}"),
            "thumbnail_url":  thumbnail_url,
            "filename":       p.get("filename") or Path(path).name,
            "timestamp":      p.get("timestamp"),
            "confidence":     p.get("timestamp_confidence"),
            "dominant_color": p.get("dominant_color"),
            "is_video":       p.get("is_video", False),
            # Seconds. Lets a video tile show its length without the client
            # having to fetch the file to find out.
            "duration":       p.get("duration"),
            "width":          p.get("width"),
            "height":         p.get("height"),
            "place_name":     p.get("place_name"),
            # The importer and the location PATCH endpoint both write
            # location_city; reading "city" meant the gallery never showed a
            # place, despite 155k photos carrying a reverse-geocoded one.
            "city":           p.get("location_city") or p.get("city"),
            "state":          p.get("location_state"),
        })

    return {
        "media":    photos,
        "total":    total,
        "offset":   offset,
        "has_more": offset + limit < total,
    }



@router.get("/counts")
async def media_counts(
    bucket:         str           = Query(default="month"),   # month | day
    min_confidence: str           = Query(default="high"),
    media_type:     str           = Query(default="all"),
    person_ids:     Optional[str] = Query(default=None),
    gps:            str           = Query(default="both"),
    camera_model:   Optional[str] = Query(default=None),
):
    """How many media items fall in each date bucket, newest first.

    The virtualised grid needs to know how tall the whole timeline is before it
    has any photographs. Fetching 150k rows to find out would defeat the point,
    so this returns one row per month (or day) and the grid estimates the rest:
    photos per bucket, divided by how many fit a row at the current width,
    gives a height it can reserve. Real dimensions replace the estimate as
    ranges load.

    It's also what makes the date scrubber instant — jumping to June 2015 is a
    lookup in this list rather than paging backwards through a decade.

    Filters mirror /gallery exactly (shared builder), because a count that
    disagrees with the items reserves space that never fills.
    """
    if bucket not in ("month", "day"):
        raise HTTPException(400, "bucket must be 'month' or 'day'")
    width = 7 if bucket == "month" else 10

    match, where, params = build_media_filters(
        min_confidence=min_confidence, media_type=media_type,
        person_ids=person_ids, gps=gps, camera_model=camera_model,
    )
    # Undated media has no bucket to live in; the grid shows it in its own
    # section, so exclude it here rather than inventing a date.
    where = (where + " AND " if where else "WHERE ") + "p.timestamp IS NOT NULL"
    params["width"] = width

    q = f"""
    {match} {where}
    WITH substring(toString(p.timestamp), 0, $width) AS bucket, count(p) AS n
    RETURN bucket, n ORDER BY bucket DESC
    """

    async with get_session() as session:
        res = await session.run(q, **params)
        rows = [{"bucket": r["bucket"], "count": r["n"]} async for r in res]

    return {
        "bucket": bucket,
        "buckets": rows,
        "total": sum(r["count"] for r in rows),
    }


@router.post("/upload", status_code=202)
async def upload_media(
    request: Request,
    files: list[UploadFile] = File(...),
    destination: str = Form("staging"),
):
    """Accept N phone photos/videos, kick off background mpp processing, return a
    job id to poll. `destination` = "gallery" (into archive/, shows in the grid)
    or "staging" (parked out of the grid for later review). Open to the gallery
    tier — see the auth middleware's gallery-write exception."""
    if destination not in ("gallery", "staging"):
        raise HTTPException(422, "destination must be 'gallery' or 'staging'")
    if not files:
        raise HTTPException(422, "no files uploaded")

    email = getattr(request.state, "user_email", None)
    job = await upload_svc.create_job(files, destination, email)
    asyncio.create_task(upload_svc.process_job(job["id"]))

    logger.bind(
        event="media.upload_started",
        job_id=job["id"],
        count=len(job["files"]),
        destination=destination,
        by=email,
        request_id=getattr(request.state, "request_id", None),
    ).info("media upload started")
    return upload_svc.public_job(job)


@router.get("/upload/{job_id}")
async def upload_status(job_id: str):
    """Poll a running/finished upload job for per-file status + readiness."""
    job = upload_svc.load_job(job_id)
    if not job:
        raise HTTPException(404, "upload job not found")
    return upload_svc.public_job(job)


class RedateBody(BaseModel):
    """Manual override of a Media's capture date.

    `timestamp` is a full ISO datetime (we always store the full shape so
    sort works); `precision` tells the UI how much of it to display + how
    much to truth-claim. A year-precision edit of "2000" is stored as
    `2000-01-01T00:00:00-04:00` with `precision='year'` so the UI knows
    not to render "Jan 1, 2000". TZ is always America/New_York per the
    project's single-tz policy.

    Re-ingest tooling (`mpp`) MUST honor `timestamp_source = 'manual'`
    and skip overwriting `m.timestamp` / `m.original_timestamp` — that
    keeps the user override across imports. (Cross-Claude contract.)
    """
    timestamp: str
    precision: Literal['day', 'month', 'year'] = 'day'

    @field_validator('timestamp')
    @classmethod
    def must_be_iso(cls, v: str) -> str:
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
        except ValueError as exc:
            raise ValueError(f"timestamp must be ISO-8601: {exc}") from exc
        return v


@router.patch("/media")
async def redate_media(request: Request, path: str = Query(...), body: RedateBody = ...):
    """Soft redate: overwrite Media.timestamp with a human-curated value,
    capturing the original on first edit. EXIF + sidecar untouched."""
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (m:Media {path: $path})
            SET m.original_timestamp   = coalesce(m.original_timestamp, m.timestamp),
                m.timestamp            = $timestamp,
                m.timestamp_source     = 'manual',
                m.timestamp_confidence = 'high',
                m.timestamp_precision  = $precision
            RETURN m { .path, .timestamp, .original_timestamp,
                       .timestamp_source, .timestamp_confidence,
                       .timestamp_precision } AS media
            """,
            path=path,
            timestamp=body.timestamp,
            precision=body.precision,
        )
        rec = await result.single()
        if not rec:
            raise HTTPException(404, "Media not found")
    media = rec["media"]
    logger.bind(
        event="media.redated",
        path=path,
        old_ts=media.get("original_timestamp"),
        new_ts=media["timestamp"],
        precision=body.precision,
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("media redated")
    return media


class LocationBody(BaseModel):
    """Manually assign GPS coordinates to a media file. Trickles to all three
    stores: the Neo4j node (so the gallery's GPS filter sees it), the .json
    sidecar (so /gallery/detail renders it), and the file's own EXIF/XMP GPS
    tags (so it survives re-extraction + travels with the file).
    """
    latitude:   float
    longitude:  float
    place_name: Optional[str] = None

    @field_validator('latitude')
    @classmethod
    def lat_range(cls, v: float) -> float:
        if not -90 <= v <= 90:
            raise ValueError("latitude must be between -90 and 90")
        return v

    @field_validator('longitude')
    @classmethod
    def lon_range(cls, v: float) -> float:
        if not -180 <= v <= 180:
            raise ValueError("longitude must be between -180 and 180")
        return v


def _write_sidecar_location(sidecar: Path, lat: float, lon: float, place_name: Optional[str]) -> None:
    """Set location.primary (+ optional geolocation.city) in the .json sidecar,
    matching the shape /gallery/detail reads (results[0].metadata or metadata)."""
    data = json.loads(sidecar.read_text()) if sidecar.exists() else {}

    # Pre-existing keys can be present-but-null (e.g. "location": null), so
    # setdefault won't help — coerce each level to a dict explicitly.
    def _dict_at(parent: dict, key: str) -> dict:
        v = parent.get(key)
        if not isinstance(v, dict):
            v = {}
            parent[key] = v
        return v

    results = data.get("results")
    container = results[0] if isinstance(results, list) and results and isinstance(results[0], dict) else data
    meta = _dict_at(container, "metadata")
    loc = _dict_at(meta, "location")
    loc["primary"] = {
        "latitude": lat,
        "longitude": lon,
        "source": "manual",
        "sourceDetails": "Manually set in app",
    }
    if place_name:
        _dict_at(loc, "geolocation")["city"] = place_name
    sidecar.write_text(json.dumps(data, indent=2))


def _write_exif_gps(full_path: Path, lat: float, lon: float) -> bool:
    """Write GPS into the file. Images use the standard EXIF GPS block; videos
    use XMP-exif (QuickTime EXIF GPS is unreliable). Refs come from the sign;
    values are absolute."""
    import subprocess
    lat_ref = "N" if lat >= 0 else "S"
    lon_ref = "E" if lon >= 0 else "W"
    prefix = "XMP-exif:" if full_path.suffix.lower() in VIDEO_EXTS else ""
    tags = [
        f"-{prefix}GPSLatitude={abs(lat)}",  f"-{prefix}GPSLatitudeRef={lat_ref}",
        f"-{prefix}GPSLongitude={abs(lon)}", f"-{prefix}GPSLongitudeRef={lon_ref}",
    ]
    try:
        r = subprocess.run(["exiftool", "-overwrite_original", *tags, str(full_path)],
                           capture_output=True, text=True, timeout=60)
        return r.returncode == 0
    except Exception as e:
        log.warning(f"exiftool GPS write failed for {full_path}: {e}")
        return False


async def _apply_location(session, path: str, lat: float, lon: float, place_name: Optional[str]) -> dict:
    """Trickle one file's GPS to graph + sidecar + EXIF. Shared by the single
    and bulk endpoints. Returns a per-item result dict (never raises)."""
    full_path = settings.photos_root / path
    if not full_path.exists():
        return {"path": path, "ok": False, "error": "file not found"}
    result = await session.run(
        """
        MATCH (m:Media {path: $path})
        SET m.original_latitude  = coalesce(m.original_latitude, m.latitude),
            m.original_longitude = coalesce(m.original_longitude, m.longitude),
            m.latitude        = $lat,
            m.longitude       = $lon,
            m.location_source = 'manual',
            m.location_city   = coalesce($place_name, m.location_city)
        RETURN m.path AS path
        """,
        path=path, lat=lat, lon=lon, place_name=place_name,
    )
    if not await result.single():
        return {"path": path, "ok": False, "error": "node not found"}

    # Sidecar + EXIF — best-effort. Graph is the filter's source of truth, so a
    # sidecar/EXIF hiccup doesn't fail the item.
    sidecar_ok = False
    try:
        _write_sidecar_location(Path(str(full_path) + ".json"), lat, lon, place_name)
        sidecar_ok = True
    except Exception as e:
        log.warning(f"sidecar location write failed for {path}: {e}")
    exif_ok = _write_exif_gps(full_path, lat, lon)
    return {"path": path, "ok": True, "sidecar_written": sidecar_ok, "exif_written": exif_ok}


@router.patch("/media/location")
async def set_media_location(request: Request, path: str = Query(...), body: LocationBody = ...):
    """Manually set a media file's GPS. Writes graph + sidecar + EXIF."""
    full_path = settings.photos_root / path
    if not full_path.exists():
        raise HTTPException(404, "Media file not found")
    async with get_session() as session:
        res = await _apply_location(session, path, body.latitude, body.longitude, body.place_name)
    if not res["ok"]:
        raise HTTPException(404, "Media node not found")

    logger.bind(
        event="media.located",
        path=path,
        latitude=body.latitude,
        longitude=body.longitude,
        place_name=body.place_name,
        sidecar_ok=res["sidecar_written"],
        exif_ok=res["exif_written"],
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("media located")
    return {
        "path": path,
        "latitude": body.latitude,
        "longitude": body.longitude,
        "place_name": body.place_name,
        "source": "manual",
        "sidecar_written": res["sidecar_written"],
        "exif_written": res["exif_written"],
    }


class BulkLocationBody(LocationBody):
    paths: list[str]


class BulkRedateBody(RedateBody):
    paths: list[str]


@router.patch("/media/location/bulk")
async def bulk_set_location(request: Request, body: BulkLocationBody):
    """Set the same GPS on many files at once — graph + sidecar + EXIF each,
    exactly like the single endpoint. Returns per-item ok/failed."""
    import uuid
    bulk_id = uuid.uuid4().hex[:12]
    results = []
    async with get_session() as session:
        for p in body.paths:
            results.append(await _apply_location(session, p, body.latitude, body.longitude, body.place_name))
    ok = [r for r in results if r["ok"]]
    logger.bind(
        event="media.located_bulk",
        bulk_id=bulk_id,
        count=len(ok),
        requested=len(body.paths),
        latitude=body.latitude, longitude=body.longitude, place_name=body.place_name,
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("media located (bulk)")
    return {"updated": len(ok), "requested": len(body.paths), "results": results}


async def _apply_redate(session, path: str, timestamp: str, precision: str) -> bool:
    """Graph-only soft redate (EXIF + sidecar untouched, matching the single
    redate's design). Returns True if the node was found + updated."""
    result = await session.run(
        """
        MATCH (m:Media {path: $path})
        SET m.original_timestamp   = coalesce(m.original_timestamp, m.timestamp),
            m.timestamp            = $timestamp,
            m.timestamp_source     = 'manual',
            m.timestamp_confidence = 'high',
            m.timestamp_precision  = $precision
        RETURN m.path AS path
        """,
        path=path, timestamp=timestamp, precision=precision,
    )
    return bool(await result.single())


@router.patch("/media/bulk")
async def bulk_redate(request: Request, body: BulkRedateBody):
    """Set the same capture date on many files at once (graph-only, like the
    single redate). Returns per-item ok/failed."""
    import uuid
    bulk_id = uuid.uuid4().hex[:12]
    results = []
    async with get_session() as session:
        for p in body.paths:
            ok = await _apply_redate(session, p, body.timestamp, body.precision)
            results.append({"path": p, "ok": ok})
    ok = [r for r in results if r["ok"]]
    logger.bind(
        event="media.redated_bulk",
        bulk_id=bulk_id,
        count=len(ok),
        requested=len(body.paths),
        timestamp=body.timestamp, precision=body.precision,
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("media redated (bulk)")
    return {"updated": len(ok), "requested": len(body.paths), "results": results}


@router.get("/place-shortcuts")
async def place_shortcuts():
    """Known place → coords from the mmp config (Stephen's quick-picks). Empty
    when the config isn't present (e.g. deployed container)."""
    cfg = Path.home() / ".config" / "mmp" / "config.json"
    out = []
    try:
        locs = json.loads(cfg.read_text()).get("shortcuts", {}).get("locations", {})
        for name, coords in locs.items():
            try:
                lat, lon = (float(x) for x in str(coords).split(","))
                out.append({"name": name, "latitude": lat, "longitude": lon})
            except (ValueError, TypeError):
                continue
    except Exception:
        pass

    # Several shortcuts are the same spot under two names — "didi-house" and
    # "didis", "seacoast" and "seacoast-classical". Both stay valid for anything
    # that already refers to them by name; the picker just shows one, or the
    # same place appears twice and neither is obviously right.
    #
    # The longer name wins: it is the more specific one, and "Seacoast
    # Classical" tells you more than "Seacoast".
    by_coord: dict = {}
    for place in out:
        key = (round(place["latitude"], 5), round(place["longitude"], 5))
        kept = by_coord.get(key)
        if kept is None or len(place["name"]) > len(kept["name"]):
            by_coord[key] = place

    return sorted(by_coord.values(), key=lambda p: p["name"])


# The archive is overwhelmingly a New Hampshire family. Searching "Haverhill"
# should offer the one twenty minutes away before the one in Suffolk, England.
_NEW_ENGLAND = ("New Hampshire", "Massachusetts", "Maine", "Vermont",
                "Rhode Island", "Connecticut")
# Roughly New England, for Nominatim's soft viewbox bias.
_NE_VIEWBOX = "-73.8,47.5,-66.9,40.9"


def _place_rank(display_name: str) -> int:
    """Lower sorts first. Home state, then the region, then anywhere."""
    if "New Hampshire" in display_name:
        return 0
    if any(state in display_name for state in _NEW_ENGLAND):
        return 1
    if "United States" in display_name:
        return 2
    return 3


@router.get("/geocode")
async def geocode(
    q: str = Query(..., min_length=2),
    anywhere: bool = Query(default=False, description="Drop the US/New England bias"),
):
    """Free-text place → candidate coordinates via OpenStreetMap Nominatim, so
    the location editor accepts "Haverhill, MA" instead of raw lat/lon.

    Results are biased toward home. Nominatim's own ordering is by prominence,
    which puts the English Haverhill above the one this family actually lives
    near — so the request carries a New England viewbox and the results are then
    re-ranked by state. `anywhere=true` turns that off, for the holiday photos.
    """
    import httpx
    params = {
        "q": q,
        "format": "json",
        # Ask for more than we show: the re-rank below needs candidates to
        # choose between, and Nominatim returns them by prominence.
        "limit": 15,
    }
    if not anywhere:
        params["countrycodes"] = "us"
        params["viewbox"] = _NE_VIEWBOX
        # bounded=0: a hint, not a fence. Austin still needs to be findable.
        params["bounded"] = 0
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(
                "https://nominatim.openstreetmap.org/search",
                params=params,
                headers={"User-Agent": "ourkin-family-archive/1.0 (personal, non-commercial)"},
            )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        log.warning(f"geocode failed for {q!r}: {e}")
        raise HTTPException(502, "Geocoding service unavailable")

    out = []
    for d in data:
        try:
            out.append({
                "display_name": d.get("display_name"),
                "latitude": float(d["lat"]),
                "longitude": float(d["lon"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    if not anywhere:
        # Stable sort: within a rank, Nominatim's prominence ordering stands.
        out.sort(key=lambda o: _place_rank(o["display_name"] or ""))
    return out[:8]


@router.post("/media/rotate")
async def rotate_media_endpoint(
    request: Request,
    path: str = Query(...),
    degrees: int = Query(default=90, description="90, 180 or 270, clockwise"),
):
    """Turn a photograph, and everything that has to turn with it.

    JPEGs go through jpegtran, which rearranges DCT blocks rather than decoding
    and re-encoding — lossless, because an archive should not shed a little
    quality every time somebody straightens a scan.

    Face bounding boxes are transformed and the face crops rotated to match.
    Leave those and every face lands boxed on the wrong side of the picture,
    which is worse than not offering rotation at all.
    """
    from app.services.rotate import rotate_media

    try:
        result = rotate_media(
            settings.photos_root, path, degrees,
            thumbs_root=settings.photos_root / "__thumbs",
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except FileNotFoundError:
        raise HTTPException(404, "Media not found")
    except Exception as e:
        log.warning(f"rotate failed for {path!r}: {e}")
        raise HTTPException(500, f"could not rotate: {e}")

    # The graph carries display dimensions, which have just swapped.
    async with get_session() as session:
        await session.run(
            "MATCH (m:Media {path: $path}) SET m.width = $w, m.height = $h",
            path=path, w=result["width"], h=result["height"],
        )

    logger.bind(
        event="media.rotated",
        path=path, degrees=degrees,
        lossless=result["lossless"],
        faces_moved=result["faces"],
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("media rotated")

    return result


@router.delete("/media", status_code=204)
async def delete_media(request: Request, path: str = Query(...)):
    """Soft-delete a Media node + its file + all sidecar siblings. The
    file and its `.json` / `.faces.json` / `.objects.json` / `.poster.jpg`
    / `.srt` / `.txt` / `_plain.txt` companions all move to
    `/photos/trash/<original-subpath>/` so the disk-report's drift
    detection stops flagging them as "unindexed", and they can be
    restored by moving them back. The Neo4j node is detached + deleted.
    """
    from pathlib import Path
    src = settings.photos_root / path
    trash_root = settings.photos_root / "trash"

    moved: list[str] = []
    file_existed = src.exists()
    if file_existed:
        dst = trash_root / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            src.rename(dst)
            moved.append(path)
        except OSError as e:
            raise HTTPException(500, f"could not move file to trash: {e}")
        # All known sidecar siblings live next to the main file with the
        # same stem + a known suffix. Move whichever exist.
        SIDECARS = [
            '.json', '.objects.json', '.clip.json', '.scenes.json',
            '.faces.json', '.srt', '.txt', '.poster.jpg', '_plain.txt',
        ]
        for suf in SIDECARS:
            side_src = Path(str(src) + suf)
            if side_src.exists():
                side_dst = Path(str(dst) + suf)
                try:
                    side_src.rename(side_dst)
                    moved.append(str(side_src.relative_to(settings.photos_root)))
                except OSError:
                    pass  # don't fail the whole delete on a stray sidecar

    async with get_session() as session:
        res = await session.run(
            "MATCH (m:Media {path: $path}) DETACH DELETE m RETURN 1 AS ok",
            path=path,
        )
        node_existed = bool(await res.single())

    if not node_existed and not file_existed:
        raise HTTPException(404, "Media not found in graph or on disk")

    logger.bind(
        event="media.deleted",
        path=path,
        moved_to_trash=moved,
        node_existed=node_existed,
        file_existed=file_existed,
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("media deleted")


@router.get("/years")
async def list_years(min_confidence: str = Query(default="high")):
    """Distinct years that have media, with counts. Used by the date scrubber."""
    # min_confidence not in the set (e.g. "all") = no confidence filter, so
    # undated / NULL-confidence media (scanned albums etc.) become visible too.
    conf = CONFIDENCE_SETS.get(min_confidence)
    where = "p.timestamp IS NOT NULL"
    qparams: dict = {}
    if conf is not None:
        where = "p.timestamp_confidence IN $conf AND " + where
        qparams["conf"] = conf
    q = f"""
        MATCH (p:Media)
        WHERE {where}
        WITH substring(toString(p.timestamp), 0, 4) AS year
        RETURN year, count(*) AS count
        ORDER BY year DESC
    """
    async with get_session() as session:
        res = await session.run(q, **qparams)
        rows = await res.data()
    return [{"year": int(r["year"]), "count": r["count"]} for r in rows if r["year"].isdigit()]


@router.get("/cameras")
async def list_cameras():
    """Distinct camera models with counts — populates the gallery filter
    dropdown. Documents / face-crops excluded (consistent with the grid)."""
    q = """
        MATCH (p:Media)
        WHERE p.camera_model IS NOT NULL
          AND NOT p:Document AND NOT p.path CONTAINS '__faces'
        RETURN p.camera_make AS make, p.camera_model AS model, count(*) AS count
        ORDER BY count DESC
    """
    async with get_session() as session:
        res = await session.run(q)
        rows = await res.data()
    return [{"make": r["make"], "model": r["model"], "count": r["count"]} for r in rows]


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
                       person.birth_date AS birth_date,
                       person.birth_date_precision AS birth_date_precision,
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
            audio_url = with_v(f"/api/media/{audio_rel}")
    subtitle_url = with_v(f"/api/media/vtt/{path}") if media_node.get("has_subtitles") else None
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

    # Face extractor uses cv2.imread which (since OpenCV 4.7) applies EXIF
    # orientation automatically — so bboxes in the .faces.json sidecar are
    # already in DISPLAY coordinates. The /media/medium/ endpoint also serves
    # display orientation (via PIL ImageOps.exif_transpose). All we need to do
    # is make sure the media.width/height we report matches that display
    # orientation. mpp's dimension fields are unreliable for this — use PIL
    # directly.
    display_w, display_h = 0, 0
    try:
        from PIL import Image, ImageOps
        with Image.open(full_path) as _im:
            _im = ImageOps.exif_transpose(_im)
            display_w, display_h = _im.size
    except Exception:
        pass

    # Read faces sidecar — face count + bbox lookup by face_index
    unidentified = []
    faces_sidecar = Path(str(full_path) + ".faces.json")
    if faces_sidecar.exists():
        try:
            fd = json.loads(faces_sidecar.read_text())
            face_count = fd.get("num_faces")
            bbox_by_index = {f["face_index"]: f.get("bbox") for f in fd.get("faces", [])}
            for p in people:
                p["bbox"] = bbox_by_index.get(p.get("face_index"))
                # Only hand back a crop URL if the crop is actually on disk.
                # ~0.4% of APPEARS_IN edges carry a crop_path whose file has
                # since gone, and an assigned face used to get a URL regardless
                # — 404ing in the lightbox. Unassigned faces below already
                # checked; this makes the two consistent.
                cp = p.get("crop_path")
                p["crop_url"] = (
                    with_v(f"/api/media/{cp}")
                    if cp and (settings.photos_root / cp).exists()
                    else None
                )

            # Faces not yet assigned to anyone. Use the crop_path the worker
            # recorded in the sidecar — it stores crops by the photo's date,
            # which can differ from the file's archive/YYYY/MM path (e.g.
            # archive/0000/ junk-date photos), and that path may have only 3
            # parts. Reconstructing from the file path crashed / mismatched.
            assigned_indexes = {p.get("face_index") for p in people}
            root = str(settings.photos_root)
            for face in fd.get("faces", []):
                fi = face.get("face_index")
                if fi in assigned_indexes:
                    continue
                cp = face.get("crop_path")
                if not cp:
                    continue
                crop_rel = cp.removeprefix(root + "/").removeprefix("/photos/").lstrip("/")
                if (settings.photos_root / crop_rel).exists():
                    unidentified.append({
                        "face_index": fi,
                        "bbox":       face.get("bbox"),
                        "crop_path":  crop_rel,
                        "crop_url":   with_v(f"/api/media/{crop_rel}"),
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

        # Soft-redate override: when Neo4j flags the timestamp as user-edited,
        # it wins over the sidecar (sidecar is untouched by redate by design).
        if media_node.get("timestamp_source") == "manual":
            ts = {
                "timestamp":  media_node["timestamp"],
                "source":     "manual",
                "confidence": media_node.get("timestamp_confidence", "high"),
                "precision":  media_node.get("timestamp_precision", "day"),
            }
            meta.setdefault("timestamps", {})["primary"] = ts
        # Otherwise the graph node's timestamp_confidence is still authoritative
        # over the sidecar's: it reflects archive/0000 date-conflict downgrades
        # the sidecar (a naive EXIF read) never saw. Without this the lightbox
        # shows "high" for a file the gallery filtered as "low" — the same file
        # contradicting itself. Surface the graph value so they agree.
        elif media_node.get("timestamp_confidence"):
            ts = {
                "timestamp":  media_node.get("timestamp") or ts.get("timestamp"),
                "source":     media_node.get("timestamp_source") or ts.get("source"),
                "confidence": media_node.get("timestamp_confidence"),
            }
            meta.setdefault("timestamps", {})["primary"] = ts
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
                # Report display dimensions (PIL .size after exif_transpose).
                # Bboxes are already in display coords (cv2.imread applies EXIF
                # orientation since OpenCV 4.7), so the frontend normalizes
                # directly against these values.
                "width":         display_w or dims.get("width"),
                "height":        display_h or dims.get("height"),
                "megapixels":    dims.get("megapixels"),
                "orientation":   dims.get("orientation"),
                "dominantColor": media.get("dominantColor"),
                "meanColor":     media.get("meanColor"),
                "salientColor":  media.get("salientColor"),
                # A Live Photo is a different object from a still — it has motion,
                # and the viewer had no way to know.
                "isLivePhoto":   media.get("isLivePhoto"),
                "liveDuration":  (media.get("livePhotoInfo") or {}).get("duration"),
            },
            "timestamp": {
                "value":      ts.get("timestamp"),
                "source":     ts.get("source"),
                "confidence": ts.get("confidence"),
                # The zone the photograph was taken in, not the one the reader
                # happens to be in. A 1998 holiday shown in the reader's current
                # timezone is quietly wrong.
                "timezone":   geoloc.get("timezone"),
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
                # Nearly every photograph has several nearby named features. The
                # single nearest is often a stream nobody has heard of while the
                # third is the lake they were actually at.
                "landmarks": [
                    {
                        "name":       (l.get("landmark") or {}).get("name"),
                        "category":   (l.get("landmark") or {}).get("category"),
                        "distance_m": l.get("distance"),
                        "confidence": l.get("confidence"),
                    }
                    for l in landmarks[:8]
                    if (l.get("landmark") or {}).get("name")
                ],
                "altitude_m":  primary_loc.get("altitude"),
                "accuracy":    primary_loc.get("accuracy"),
                "postal_code": geoloc.get("postal_code"),
                "timezone":    geoloc.get("timezone"),
            } if primary_loc.get("latitude") else None,
            "camera": {
                "make":  camera.get("make"),
                "model": camera.get("model"),
                "lens":  camera.get("lens"),
                # The OS build that wrote the file — sometimes the only clue to
                # when a scan was actually digitised.
                "software": camera.get("software"),
            } if camera else None,
            "settings": {
                "iso":          expos.get("iso"),
                "aperture":     expos.get("aperture"),
                "shutterSpeed": expos.get("shutterSpeed"),
                "focalLength":  expos.get("focalLength"),
                "flash":        expos.get("flash"),
                "focalLength35mm": expos.get("focalLength35mm"),
            } if expos else None,
            "processing": {
                "processor":   proc.get("processor"),
                "extractedAt": proc.get("extractedAt"),
                },
                # mpp's own verdict on whether this file is fully described, with
                # the specific gaps named. A per-photograph to-do list — what turns
                # the archive into something you can improve rather than only
                # browse.
                "readiness": {
                    "score":     (meta.get("archiveReadiness") or {}).get("score"),
                    "mediaType": (meta.get("archiveReadiness") or {}).get("mediaType"),
                    "missing":   (meta.get("archiveReadiness") or {}).get("missing") or [],
                    "checks":    (meta.get("archiveReadiness") or {}).get("checks") or {},
                } if meta.get("archiveReadiness") else None,
        }
    except Exception as e:
        log.warning(f"Failed to parse sidecar for {path}: {e}")
        return {
            **base,
            "people": people, "face_count": face_count,
            "unidentified": unidentified, "objects": objects,
            "heritage": heritage,
        }
