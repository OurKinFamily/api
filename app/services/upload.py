"""Phone upload pipeline — accept multipart files, run them through mpp, land them.

The archive is normally fed offline by the workshop `mpp` pipeline; this is the
in-API path so a family member can log in on their phone and add photos/videos
directly. It reuses `mpp` (the same Media Processing Pipeline the offline scan
uses) so an uploaded file gets the IDENTICAL sidecar every other archive item
has — hashes (md5 for dedupe + perceptual), `archiveReadiness.score`, timestamps,
GPS, dimensions, camera. No hand-rolled extraction, no second source of truth.

Flow per upload batch:
  1. bytes saved to <photos>/__data/uploads/pending/<job_id>/
  2. a job record (JSON on disk, like the /jobs runner) tracks per-file status
  3. a background task runs `mpp` on each file, reads the sidecar, dedupes on the
     md5 hash, moves the file to its final home (with its sidecar) and MERGEs the
     :Media node.

Destination is the caller's choice:
  - "gallery"  → archive/YYYY/MM/…          (shows in the gallery grid)
  - "staging"  → staging/uploads/YYYY/MM/…  (node flagged staged=true, kept out
                 of the grid until a future "promote" step)

Readiness = mpp's own `archiveReadiness` block, surfaced verbatim; it drives the
frontend's confirm-before-gallery UX. A phone photo with a real capture date +
GPS scores 100.

DEPLOY NOTE: `mpp` must be on PATH in the API container (it's a Node CLI, not a
pip dep). See the Dockerfile / deploy stack.
"""
import asyncio
import json
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.config import settings, with_v
from app.db.neo4j import get_session
from app.log import logger

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mpg", ".mpeg", ".webm", ".3gp"}
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")

# Reuse the already-running face worker (container `ourkin-worker-face`) to run
# detection on JUST the freshly-uploaded photos — `python -m face extract <paths>`
# loads the model once, skips already-processed files, and writes to the same DB
# the rest of the archive uses. Set FACE_WORKER_CONTAINER="" to disable (e.g. an
# environment without the worker). The worker mounts the photos volume at /photos.
FACE_WORKER = os.environ.get("FACE_WORKER_CONTAINER", "ourkin-worker-face")
WORKER_PHOTOS_ROOT = "/photos"

# Hold references to fire-and-forget background tasks so they aren't GC'd mid-run.
_bg_tasks: set = set()


# ── job store (JSON on disk, mirrors the /jobs runner) ──────────────────────

def _uploads_dir() -> Path:
    return settings.photos_root / "__data" / "uploads"


def _job_path(job_id: str) -> Path:
    return _uploads_dir() / f"{job_id}.json"


def load_job(job_id: str) -> dict | None:
    p = _job_path(job_id)
    return json.loads(p.read_text()) if p.exists() else None


def save_job(job: dict) -> None:
    d = _uploads_dir()
    d.mkdir(parents=True, exist_ok=True)
    _job_path(job["id"]).write_text(json.dumps(job, indent=2))


def public_job(job: dict) -> dict:
    """Strip internal bookkeeping (pending paths) before returning to the client."""
    return {
        "id": job["id"],
        "created_at": job["created_at"],
        "destination": job["destination"],
        "status": job["status"],
        "files": [{k: v for k, v in f.items() if not k.startswith("_")} for f in job["files"]],
    }


# ── mpp extraction ──────────────────────────────────────────────────────────

def _is_video(name: str) -> bool:
    return Path(name).suffix.lower() in VIDEO_EXTS


def _run_mpp(path: Path) -> dict:
    """Run mpp on a file → its sidecar's `metadata` block. `-o <path>` makes mpp
    write `<path>.json` (the archive-standard sidecar name)."""
    subprocess.run(
        ["mpp", "-f", str(path), "-o", str(path), "--overwrite", "--quiet"],
        capture_output=True, text=True, timeout=300, check=True,
    )
    sidecar = Path(str(path) + ".json")
    data = json.loads(sidecar.read_text())
    results = data.get("results") or []
    meta = (results[0].get("metadata") if results else data.get("metadata"))
    return meta or {}


def _readiness(meta: dict) -> dict:
    """mpp's archiveReadiness block, surfaced verbatim (+ a convenience `ready`
    flag). `ready` = a fully archive-ready item (score 100), which the frontend
    uses to auto-confirm gallery uploads; anything short prompts a confirm."""
    ar = meta.get("archiveReadiness") or {}
    score = ar.get("score", 0)
    return {
        "score": score,
        "checks": ar.get("checks", {}),
        "missing": ar.get("missing", []),
        "media_type": ar.get("mediaType"),
        "ready": score >= 100,
    }


async def _find_duplicate(md5: str | None) -> str | None:
    """Return the path of an existing Media node with this md5, if any."""
    if not md5:
        return None
    async with get_session() as session:
        res = await session.run(
            "MATCH (m:Media {content_hash: $h}) RETURN m.path AS path LIMIT 1",
            h=md5,
        )
        rec = await res.single()
    return rec["path"] if rec else None


# ── filesystem landing ──────────────────────────────────────────────────────

def _safe_name(name: str) -> str:
    stem = _SAFE.sub("_", Path(name).stem).strip("._") or "upload"
    suffix = Path(name).suffix.lower() or ".bin"
    return f"{stem}{suffix}"


def _final_path(destination: str, year: str, month: str, filename: str) -> Path:
    base = "archive" if destination == "gallery" else "staging/uploads"
    dest_dir = settings.photos_root / base / year / month
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / filename
    if dst.exists():  # never clobber an existing archive file
        dst = dest_dir / f"{dst.stem}_{uuid.uuid4().hex[:8]}{dst.suffix}"
    return dst


def _make_poster(video: Path) -> Path | None:
    """Midpoint frame → <video>.poster.jpg (thumbnail source for the grid)."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)],
            capture_output=True, text=True, timeout=60,
        )
        dur = float(out.stdout.strip() or 0)
    except Exception:
        dur = 0.0
    mid = max(0, int(dur / 2))
    h, rem = divmod(mid, 3600)
    mm, ss = divmod(rem, 60)
    ts = f"{h}:{mm:02d}:{ss:02d}" if h else f"{mm}:{ss:02d}"
    poster = Path(str(video) + ".poster.jpg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", ts, "-i", str(video), "-frames:v", "1", "-q:v", "2", str(poster)],
            capture_output=True, text=True, timeout=120,
        )
        return poster if poster.exists() else None
    except Exception as e:
        logger.warning(f"poster generation failed for {video.name}: {e}")
        return None


async def _create_node(row: dict) -> None:
    label = "Video" if row["is_video"] else "Photo"
    async with get_session() as session:
        await session.run(
            f"""
            MERGE (m:Media {{path: $path}})
            SET m:{label},
                m.filename = $filename, m.is_video = $is_video, m.source = 'phone',
                m.content_hash = $md5, m.perceptual_hash = $phash,
                m.timestamp = $timestamp, m.timestamp_source = $ts_source,
                m.timestamp_confidence = $ts_conf, m.timestamp_precision = $ts_prec,
                m.latitude = $lat, m.longitude = $lon, m.location_source = $loc_source,
                m.width = $width, m.height = $height,
                m.camera_make = $make, m.camera_model = $model,
                m.poster_path = $poster_path,
                m.staged = $staged, m.uploaded_by = $by, m.uploaded_at = $uploaded_at,
                m.readiness_score = $readiness
            RETURN m.path AS path
            """,
            **row,
        )


# ── orchestration ───────────────────────────────────────────────────────────

async def create_job(files: list[UploadFile], destination: str, by: str | None) -> dict:
    """Save the uploaded bytes to a pending dir + build the job record."""
    job_id = uuid.uuid4().hex[:12]
    pending = _uploads_dir() / "pending" / job_id
    pending.mkdir(parents=True, exist_ok=True)

    entries = []
    for f in files:
        name = _safe_name(f.filename or "upload.bin")
        dst = pending / name
        if dst.exists():
            dst = pending / f"{dst.stem}_{uuid.uuid4().hex[:6]}{dst.suffix}"
        dst.write_bytes(await f.read())
        entries.append({
            "filename": name,
            "status": "queued",
            "is_video": _is_video(name),
            "path": None,
            "url": None,
            "thumbnail_url": None,
            "timestamp": None,
            "readiness": None,
            "duplicate_of": None,
            "error": None,
            "_pending": str(dst),
        })

    job = {
        "id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "by": by,
        "destination": destination,
        "status": "processing",
        "files": entries,
    }
    save_job(job)
    return job


def _cleanup(*paths: Path) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


async def _process_file(job: dict, f: dict) -> None:
    pending = Path(f["_pending"])
    pending_sidecar = Path(str(pending) + ".json")

    meta = await asyncio.to_thread(_run_mpp, pending)

    # Every block is coerced with `or {}` — mpp can emit a key present-but-null
    # (e.g. "timestamps": null for an undated screenshot), and `.get(k, {})`
    # returns None in that case, so a bare .get() on it would crash.
    hashes = meta.get("hashes") or {}
    md5 = hashes.get("md5")
    phash = hashes.get("perceptual")
    ts = (meta.get("timestamps") or {}).get("primary") or {}
    ts_iso = ts.get("timestamp")
    loc = (meta.get("location") or {}).get("primary") or {}
    lat, lon = loc.get("latitude"), loc.get("longitude")
    dims = (meta.get("media") or {}).get("dimensions") or {}
    width, height = dims.get("width"), dims.get("height")
    camera = meta.get("camera") or {}
    readiness = _readiness(meta)

    # Dedupe on content hash — if this exact file is already in the archive, drop
    # the upload (file + its fresh sidecar) and point at the existing node.
    dup = await _find_duplicate(md5)
    if dup:
        await asyncio.to_thread(_cleanup, pending, pending_sidecar)
        f.update(status="duplicate", duplicate_of=dup, readiness=readiness)
        return

    year = ts_iso[:4] if ts_iso else "0000"
    month = ts_iso[5:7] if ts_iso else "00"
    final = _final_path(job["destination"], year, month, f["filename"])

    await asyncio.to_thread(shutil.move, str(pending), str(final))
    await asyncio.to_thread(shutil.move, str(pending_sidecar), str(final) + ".json")

    poster_path = None
    if f["is_video"]:
        poster = await asyncio.to_thread(_make_poster, final)
        if poster:
            poster_path = str(poster.relative_to(settings.photos_root))

    rel = str(final.relative_to(settings.photos_root))
    await _create_node({
        "path": rel, "filename": final.name, "is_video": f["is_video"],
        "md5": md5, "phash": phash,
        "timestamp": ts_iso, "ts_source": ts.get("source"),
        "ts_conf": ts.get("confidence"), "ts_prec": ts.get("precision", "day"),
        "lat": lat, "lon": lon, "loc_source": loc.get("source") if lat is not None else None,
        "width": width, "height": height,
        "make": camera.get("make"), "model": camera.get("model"),
        "poster_path": poster_path,
        "staged": job["destination"] == "staging", "by": job["by"],
        "uploaded_at": job["created_at"], "readiness": readiness["score"],
    })

    f.update(
        status="done",
        path=rel,
        url=with_v(f"/api/media/{rel}"),
        thumbnail_url=with_v(
            f"/api/media/{poster_path}" if poster_path
            else f"/api/media/thumb/{rel.removeprefix('archive/')}"
        ),
        timestamp=ts_iso,
        readiness=readiness,
    )


async def _run_face_detection(paths: list[str]) -> None:
    """docker exec the running face worker on the given (photos-root-relative)
    paths. Best-effort: any failure is logged, never surfaced to the upload."""
    if not FACE_WORKER or not paths:
        return
    targets = [f"{WORKER_PHOTOS_ROOT}/{p}" for p in paths]
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "exec", FACE_WORKER, "python", "-m", "face", "extract", *targets,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0:
            logger.warning(f"face detection exited {proc.returncode}: {(err or b'').decode()[:200]}")
        else:
            logger.bind(event="media.faces_detected", count=len(targets)).info("face detection ran")
    except FileNotFoundError:
        logger.warning("face detection skipped: docker CLI not available")
    except Exception as e:
        logger.warning(f"face detection failed: {e}")


def trigger_face_detection(paths: list[str]) -> None:
    """Fire-and-forget face detection on freshly-uploaded photos."""
    t = asyncio.create_task(_run_face_detection(paths))
    _bg_tasks.add(t)
    t.add_done_callback(_bg_tasks.discard)


async def process_job(job_id: str) -> None:
    job = load_job(job_id)
    if not job:
        return
    for f in job["files"]:
        f["status"] = "processing"
        save_job(job)
        try:
            await _process_file(job, f)
        except Exception as e:
            f["status"] = "error"
            f["error"] = str(e)
            logger.warning(f"upload processing failed for {f['filename']}: {e}")
        save_job(job)
    job["status"] = "done"
    save_job(job)

    # Run face detection on the new gallery photos (images only — this worker
    # doesn't sample video frames). Staged uploads wait until they're promoted.
    photos = [
        f["path"] for f in job["files"]
        if f["status"] == "done" and not f["is_video"]
        and f["path"] and f["path"].startswith("archive/")
    ]
    if photos:
        trigger_face_detection(photos)
