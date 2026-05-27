"""
Face cluster management — reads/writes /photos/__faces/clusters/*.json
and syncs assignments to Neo4j APPEARS_IN relationships.
"""

import json
import logging
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.config import settings
from app.db.neo4j import get_session

log = logging.getLogger(__name__)
router = APIRouter(prefix="/faces", tags=["faces"])

CLUSTERS_DIR            = settings.photos_root / "__faces" / "clusters"
CLUSTERS_FILE           = CLUSTERS_DIR / "clusters.json"
SKIPPED_FILE            = CLUSTERS_DIR / "skipped.json"
EMBEDDINGS_NPY          = CLUSTERS_DIR / "embeddings.npy"
FACES_META_FILE         = CLUSTERS_DIR / "faces_meta.json"

# In-memory embedding index (lazy-loaded)
_emb_matrix   = None
_emb_meta     = None
_emb_lookup: dict[tuple[str, int], int] | None = None  # (photo_path, face_index) → row index

# Neo4j assigned-faces cache (all faces with any APPEARS_IN edge)
_neo4j_assigned: set[tuple[str, int]] | None = None
_neo4j_assigned_ts: float = 0.0
_NEO4J_ASSIGNED_TTL = 60.0  # seconds

_neo4j_skipped: set[tuple[str, int]] | None = None
_neo4j_skipped_ts: float = 0.0
_NEO4J_SKIPPED_TTL = 60.0  # seconds
_emb_lock     = threading.Lock()


def _load_embedding_index():
    global _emb_matrix, _emb_meta, _emb_lookup
    with _emb_lock:
        if _emb_matrix is not None:
            return _emb_matrix, _emb_meta, _emb_lookup
        if not EMBEDDINGS_NPY.exists() or not FACES_META_FILE.exists():
            return None, None, None
        import numpy as np
        _emb_matrix = np.load(str(EMBEDDINGS_NPY))
        _emb_meta   = json.loads(FACES_META_FILE.read_text())
        _emb_lookup = {}
        for idx, m in enumerate(_emb_meta):
            pp = m["photo_path"].replace("/photos/", "", 1) if m["photo_path"].startswith("/photos/") else m["photo_path"]
            _emb_lookup[(pp, int(m["face_index"]))] = idx
        return _emb_matrix, _emb_meta, _emb_lookup

SAMPLE_CROPS = 6  # how many crop previews to return per cluster

# Reverse index: (rel_photo_path, face_index) -> cluster_id
_cluster_idx: dict[tuple[str, int], str] | None = None
_cluster_sizes: dict[str, int] = {}
_cluster_idx_mtime: float = 0.0
_cluster_idx_lock = threading.Lock()


def _build_cluster_index() -> None:
    global _cluster_idx, _cluster_sizes, _cluster_idx_mtime
    try:
        mtime = CLUSTERS_FILE.stat().st_mtime
    except FileNotFoundError:
        _cluster_idx = {}
        _cluster_sizes = {}
        return
    with _cluster_idx_lock:
        if _cluster_idx is not None and mtime == _cluster_idx_mtime:
            return
        clusters = _load(CLUSTERS_FILE, {})
        idx: dict[tuple[str, int], str] = {}
        sizes: dict[str, int] = {}
        for cid, faces in clusters.items():
            if str(cid) == "-1":
                continue
            cid_str = str(cid)
            sizes[cid_str] = len(faces)
            for f in faces:
                pp = f.get("photo_path", "").replace("/photos/", "", 1)
                fi = f.get("face_index")
                if pp and fi is not None:
                    idx[(pp, int(fi))] = cid_str
        _cluster_idx = idx
        _cluster_sizes = sizes
        _cluster_idx_mtime = mtime


def _load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _crop_url(crop_path: str) -> str:
    rel = crop_path.replace("/photos/", "", 1) if crop_path.startswith("/photos/") else crop_path
    return f"/api/media/{rel}"


async def _get_neo4j_assigned() -> set[tuple[str, int]]:
    """All (photo_path, face_index) pairs with any APPEARS_IN in Neo4j. 60s cache."""
    import time
    global _neo4j_assigned, _neo4j_assigned_ts
    if _neo4j_assigned is not None and (time.time() - _neo4j_assigned_ts) < _NEO4J_ASSIGNED_TTL:
        return _neo4j_assigned
    async with get_session() as session:
        result = await session.run(
            "MATCH ()-[r:APPEARS_IN]->(photo:Media) WHERE r.face_index IS NOT NULL "
            "RETURN photo.path AS pp, r.face_index AS fi"
        )
        rows = await result.data()
    assigned: set[tuple[str, int]] = set()
    for row in rows:
        pp = row["pp"].replace("/photos/", "", 1) if row["pp"].startswith("/photos/") else row["pp"]
        assigned.add((pp, int(row["fi"])))
    _neo4j_assigned = assigned
    _neo4j_assigned_ts = time.time()
    return assigned


async def _get_neo4j_skipped() -> set[tuple[str, int]]:
    """All (photo_path, face_index) pairs marked as skipped in Neo4j (Media.skipped_faces). 60s cache."""
    import time
    global _neo4j_skipped, _neo4j_skipped_ts
    if _neo4j_skipped is not None and (time.time() - _neo4j_skipped_ts) < _NEO4J_SKIPPED_TTL:
        return _neo4j_skipped
    async with get_session() as session:
        result = await session.run(
            "MATCH (m:Media) WHERE m.skipped_faces IS NOT NULL "
            "RETURN m.path AS p, m.skipped_faces AS fis"
        )
        rows = await result.data()
    skipped: set[tuple[str, int]] = set()
    for row in rows:
        p = row["p"].replace("/photos/", "", 1) if row["p"].startswith("/photos/") else row["p"]
        for fi in (row["fis"] or []):
            skipped.add((p, int(fi)))
    _neo4j_skipped = skipped
    _neo4j_skipped_ts = time.time()
    return skipped


async def _add_skipped_faces(items: list[tuple[str, int]]):
    """Append (path, face_index) to Media.skipped_faces in Neo4j."""
    by_media: dict[str, list[int]] = {}
    for p, fi in items:
        if not p or fi is None: continue
        pp = p.replace("/photos/", "", 1) if p.startswith("/photos/") else p
        by_media.setdefault(pp, []).append(int(fi))
    if not by_media: return
    async with get_session() as session:
        await session.run(
            """
            UNWIND $rows AS row
            MATCH (m:Media {path: row.p})
            SET m.skipped_faces = [x IN coalesce(m.skipped_faces, []) WHERE NOT x IN row.fis] + row.fis
            """,
            rows=[{"p": p, "fis": fis} for p, fis in by_media.items()],
        )
    global _neo4j_skipped
    _neo4j_skipped = None


async def _remove_skipped_faces(items: list[tuple[str, int]]):
    """Remove (path, face_index) from Media.skipped_faces in Neo4j."""
    by_media: dict[str, list[int]] = {}
    for p, fi in items:
        if not p or fi is None: continue
        pp = p.replace("/photos/", "", 1) if p.startswith("/photos/") else p
        by_media.setdefault(pp, []).append(int(fi))
    if not by_media: return
    async with get_session() as session:
        await session.run(
            """
            UNWIND $rows AS row
            MATCH (m:Media {path: row.p})
            SET m.skipped_faces = [x IN coalesce(m.skipped_faces, []) WHERE NOT x IN row.fis]
            """,
            rows=[{"p": p, "fis": fis} for p, fis in by_media.items()],
        )
    global _neo4j_skipped
    _neo4j_skipped = None


class SearchBody(BaseModel):
    photo_path: str
    face_index: int
    threshold: float = 0.5
    limit: int = 100000
    unassigned_only: bool = True


class SearchByPersonBody(BaseModel):
    person_id: str
    threshold: float = 0.5
    limit: int = 100000
    unassigned_only: bool = True


def _run_similarity_search(matrix, meta, q, threshold, limit, assigned, exclude: set[tuple[str, int]]):
    """Core cosine similarity search. q must be L2-normalised."""
    import numpy as np
    sims  = matrix @ q
    order = np.argsort(-sims)
    results = []
    for i in order:
        sim = float(sims[i])
        if sim < (1.0 - threshold):
            break
        m = meta[i]
        rel_path = m["photo_path"].replace("/photos/", "", 1) if m["photo_path"].startswith("/photos/") else m["photo_path"]
        key = (rel_path, int(m["face_index"]))
        if key in exclude:
            continue
        if assigned and key in assigned:
            continue
        crop = m.get("crop_path", "")
        crop_url = f"/api/media/{crop.replace('/photos/', '', 1)}" if crop.startswith("/photos/") else f"/api/media/{crop}"
        results.append({
            "photo_path":  rel_path,
            "face_index":  m["face_index"],
            "crop_url":    crop_url,
            "similarity":  round(sim, 4),
            "distance":    round(1.0 - sim, 4),
        })
        if len(results) >= limit:
            break
    return results


@router.post("/search")
async def search_similar_faces(body: SearchBody):
    """Find faces similar to a given face using the prebuilt embedding index."""
    import asyncio, numpy as np

    matrix, meta, _ = await asyncio.to_thread(_load_embedding_index)
    if matrix is None:
        raise HTTPException(503, "Embedding index not built yet — run build_embedding_index.py first")

    # Load query embedding from sidecar
    photo_abs = settings.photos_root / body.photo_path
    sidecar   = Path(str(photo_abs) + ".faces.json")
    if not sidecar.exists():
        raise HTTPException(404, f"No faces sidecar for {body.photo_path}")

    data = json.loads(sidecar.read_text())
    query_emb = next(
        (f["embedding"] for f in data.get("faces", []) if f["face_index"] == body.face_index),
        None,
    )
    if query_emb is None:
        raise HTTPException(404, f"face_index {body.face_index} not found in sidecar")

    q = np.array(query_emb, dtype=np.float32)
    q /= np.linalg.norm(q) or 1.0

    assigned = await _get_neo4j_assigned() if body.unassigned_only else set()
    exclude  = {(body.photo_path, body.face_index)}
    results  = _run_similarity_search(matrix, meta, q, body.threshold, body.limit, assigned, exclude)
    return {"results": results, "total": len(results)}


@router.post("/search/by_person")
async def search_similar_faces_by_person(body: SearchByPersonBody):
    """Find untagged faces similar to a known person using the mean of all their face embeddings."""
    import asyncio, numpy as np

    matrix, meta, lookup = await asyncio.to_thread(_load_embedding_index)
    if matrix is None:
        raise HTTPException(503, "Embedding index not built yet")

    # Fetch all person's APPEARS_IN faces from Neo4j
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $person_id})-[r:APPEARS_IN]->(photo:Media)
            WHERE r.face_index IS NOT NULL
            RETURN photo.path AS photo_path, r.face_index AS face_index
            """,
            person_id=body.person_id,
        )
        rows = await result.data()

    if not rows:
        raise HTTPException(404, f"No tagged faces found for person {body.person_id}")

    # Collect embedding rows for this person
    person_faces: set[tuple[str, int]] = set()
    row_indices: list[int] = []
    for row in rows:
        pp = row["photo_path"].replace("/photos/", "", 1) if row["photo_path"].startswith("/photos/") else row["photo_path"]
        fi = int(row["face_index"])
        person_faces.add((pp, fi))
        idx = lookup.get((pp, fi))
        if idx is not None:
            row_indices.append(idx)

    if not row_indices:
        raise HTTPException(404, "Person's faces not found in embedding index — index may be stale")

    # Mean embedding, L2-normalise
    person_matrix = matrix[row_indices]
    q = person_matrix.mean(axis=0).astype(np.float32)
    q /= np.linalg.norm(q) or 1.0

    assigned = await _get_neo4j_assigned() if body.unassigned_only else set()
    # Exclude all of this person's already-tagged faces from results
    results  = _run_similarity_search(matrix, meta, q, body.threshold, body.limit, assigned, person_faces)
    return {"results": results, "total": len(results), "faces_used": len(row_indices)}


class SearchByPersonTemporalBody(BaseModel):
    person_id: str
    threshold: float = 0.5
    limit: int = 100000
    unassigned_only: bool = True
    n_buckets: int = 4
    min_bucket_faces: int = 20


async def _get_connection_ids(person_id: str) -> set[str]:
    """Return IDs of people connected to person_id (family + KNOWS, 1-2 hops)."""
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})-[:KNOWS|MARRIED_TO|PARENT_OF*1..2]-(c:Person)
            RETURN DISTINCT c.id AS cid
            """,
            id=person_id,
        )
        return {r["cid"] for r in await result.data()}


async def _get_photo_meta(paths: list[str]) -> dict[str, dict]:
    """Fetch timestamp + GPS for a list of photo paths."""
    if not paths:
        return {}
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (photo:Media) WHERE photo.path IN $paths
            RETURN photo.path AS path, photo.timestamp AS ts,
                   photo.latitude AS lat, photo.longitude AS lon
            """,
            paths=paths,
        )
        return {r["path"]: r for r in await result.data()}


async def _get_cooccurrence(paths: list[str], connection_ids: set[str]) -> dict[str, int]:
    """Count how many of a person's connections appear in each candidate photo."""
    if not paths or not connection_ids:
        return {}
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (person:Person)-[:APPEARS_IN]->(photo:Media)
            WHERE photo.path IN $paths AND person.id IN $cids
            RETURN photo.path AS path, count(person) AS n
            """,
            paths=paths,
            cids=list(connection_ids),
        )
        return {r["path"]: r["n"] for r in await result.data()}


def _event_key(meta: dict):
    """Derive a ~1km-day event bucket key from photo metadata, or None."""
    ts  = meta.get("ts")
    lat = meta.get("lat")
    lon = meta.get("lon")
    if not ts or str(ts).startswith("1700"):
        return None
    date = str(ts)[:10]
    if lat is not None and lon is not None:
        return (date, round(float(lat), 2), round(float(lon), 2))
    return (date,)  # date-only bucket if no GPS


@router.post("/search/by_person_temporal")
async def search_similar_faces_temporal(body: SearchByPersonTemporalBody):
    """
    Multi-centroid temporal search. Splits a person's tagged faces into N equal-count
    time buckets, computes a mean per bucket, searches with each, and merges by max similarity.
    Falls back to single mean if date spread < 5 years or not enough dated faces.
    """
    import asyncio, numpy as np
    from datetime import datetime, timezone

    matrix, meta, lookup = await asyncio.to_thread(_load_embedding_index)
    if matrix is None:
        raise HTTPException(503, "Embedding index not built yet")

    # Fetch all APPEARS_IN faces with photo timestamps
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $person_id})-[r:APPEARS_IN]->(photo:Media)
            WHERE r.face_index IS NOT NULL
            RETURN photo.path AS photo_path, r.face_index AS face_index,
                   photo.timestamp AS ts
            """,
            person_id=body.person_id,
        )
        rows = await result.data()

    if not rows:
        raise HTTPException(404, f"No tagged faces found for person {body.person_id}")

    person_faces: set[tuple[str, int]] = set()

    # Normalise paths and split into dated / undated
    dated: list[tuple[str, datetime, int]] = []   # (pp, dt, matrix_idx)
    undated_indices: list[int] = []

    for row in rows:
        pp = row["photo_path"].replace("/photos/", "", 1) if row["photo_path"].startswith("/photos/") else row["photo_path"]
        fi = int(row["face_index"])
        person_faces.add((pp, fi))
        idx = lookup.get((pp, fi))
        if idx is None:
            continue
        ts = row.get("ts")
        if ts and not str(ts).startswith("1700"):
            try:
                dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                dated.append((pp, dt, idx))
            except ValueError:
                undated_indices.append(idx)
        else:
            undated_indices.append(idx)

    all_indices = [x[2] for x in dated] + undated_indices
    if not all_indices:
        raise HTTPException(404, "Person's faces not found in embedding index")

    # Decide whether to use temporal bucketing
    use_temporal = False
    bucket_means: list[np.ndarray] = []
    bucket_info: list[dict] = []

    if len(dated) >= body.min_bucket_faces * 2:
        dated.sort(key=lambda x: x[1])
        earliest = dated[0][1]
        latest   = dated[-1][1]
        span_years = (latest - earliest).days / 365.25

        if span_years >= 5:
            use_temporal = True
            n = min(body.n_buckets, len(dated) // body.min_bucket_faces)
            bucket_size = len(dated) // n
            for i in range(n):
                start = i * bucket_size
                end   = start + bucket_size if i < n - 1 else len(dated)
                chunk = dated[start:end]
                indices = [x[2] for x in chunk]
                m = matrix[indices].mean(axis=0).astype(np.float32)
                m /= np.linalg.norm(m) or 1.0
                bucket_means.append(m)
                bucket_info.append({
                    "from": chunk[0][1].year,
                    "to":   chunk[-1][1].year,
                    "faces": len(chunk),
                })

    if not use_temporal:
        # Fall back: single mean over all faces
        q = matrix[all_indices].mean(axis=0).astype(np.float32)
        q /= np.linalg.norm(q) or 1.0
        bucket_means = [q]

    async def _empty_set(): return set()
    async def _empty_dict(): return {}

    # Fetch exclusion set + social connections in parallel
    neo4j_assigned, connection_ids = await asyncio.gather(
        _get_neo4j_assigned() if body.unassigned_only else _empty_set(),
        _get_connection_ids(body.person_id),
    )
    exclude = person_faces | (neo4j_assigned if body.unassigned_only else set())

    # Search with each bucket mean, keep max similarity per face
    best: dict[tuple[str, int], dict] = {}
    for q in bucket_means:
        sims  = matrix @ q
        order = np.argsort(-sims)
        for i in order:
            sim = float(sims[i])
            if sim < (1.0 - body.threshold):
                break
            m   = meta[i]
            rel = m["photo_path"].replace("/photos/", "", 1) if m["photo_path"].startswith("/photos/") else m["photo_path"]
            key = (rel, int(m["face_index"]))
            if key in exclude:
                continue
            if key not in best or sim > best[key]["similarity"]:
                crop = m.get("crop_path", "")
                crop_url = f"/api/media/{crop.replace('/photos/', '', 1)}" if crop.startswith("/photos/") else f"/api/media/{crop}"
                best[key] = {
                    "photo_path": rel,
                    "face_index": m["face_index"],
                    "crop_url":   crop_url,
                    "similarity": round(sim, 4),
                    "distance":   round(1.0 - sim, 4),
                }

    if not best:
        return {"results": [], "total": 0, "faces_used": len(all_indices), "buckets": bucket_info if use_temporal else None}

    # ── Post-search enrichment ────────────────────────────────────────────────
    candidate_paths = list({r["photo_path"] for r in best.values()})

    # Fetch photo metadata (timestamp, GPS) + co-occurrence in parallel
    photo_meta, cooc_counts = await asyncio.gather(
        _get_photo_meta(candidate_paths),
        _get_cooccurrence(candidate_paths, connection_ids) if connection_ids else _empty_dict(),
    )

    # Event clustering: group by (date, ~1km grid), count hits >= base threshold
    from collections import defaultdict
    EVENT_BASE = max(0.55, 1.0 - body.threshold - 0.1)
    event_hits: dict = defaultdict(int)
    for r in best.values():
        if r["similarity"] >= EVENT_BASE:
            key = _event_key(photo_meta.get(r["photo_path"], {}))
            if key:
                event_hits[key] += 1

    # Apply boosts and build final list
    results = []
    for r in best.values():
        sim = r["similarity"]
        meta_entry = photo_meta.get(r["photo_path"], {})

        # Co-occurrence boost: +0.04 per connected person in photo, cap +0.12
        cooc = cooc_counts.get(r["photo_path"], 0)
        if cooc:
            sim = min(1.0, sim + min(cooc * 0.04, 0.12))

        # Event boost: +0.03 per additional hit in same event, cap +0.10
        ekey = _event_key(meta_entry)
        if ekey and event_hits.get(ekey, 0) >= 2:
            sim = min(1.0, sim + min((event_hits[ekey] - 1) * 0.03, 0.10))

        results.append({**r, "similarity": round(sim, 4), "distance": round(1.0 - sim, 4)})

    results.sort(key=lambda r: -r["similarity"])
    results = results[:body.limit]

    return {
        "results":    results,
        "total":      len(results),
        "faces_used": len(all_indices),
        "buckets":    bucket_info if use_temporal else None,
    }


@router.post("/search/assign")
async def bulk_assign_faces(body: dict):
    """Assign multiple individual faces to a person.

    Frontend often doesn't know the crop path — derive it server-side from
    `photo_path + face_index` so every edge gets a usable crop_path. This
    matches the canonical layout face extraction writes:
    `__faces/crops/<year>/<month>/<filename>_face<N>.jpg`."""
    person_id = body.get("person_id")
    faces     = body.get("faces", [])  # list of {photo_path, face_index, crop_path?}
    if not person_id or not faces:
        raise HTTPException(400, "person_id and faces required")

    def _canonical_crop_path(photo_path: str, face_index) -> str:
        if not photo_path or face_index is None: return ""
        if photo_path.startswith("archive/"):
            stripped = photo_path[len("archive/"):]
            candidate = f"__faces/crops/{stripped}_face{face_index}.jpg"
            full = settings.photos_root / candidate
            return candidate if full.exists() else ""
        return ""

    async with get_session() as session:
        for f in faces:
            photo_path = f.get("photo_path", "").replace("/photos/", "", 1)
            crop_path  = f.get("crop_path",  "").replace("/photos/", "", 1) or _canonical_crop_path(photo_path, f.get("face_index"))
            await session.run(
                """
                MATCH (person:Person {id: $person_id})
                MATCH (photo:Media {path: $photo_path})
                MERGE (person)-[r:APPEARS_IN {face_index: $face_index}]->(photo)
                SET r.crop_path  = $crop_path,
                    r.photo_path = $photo_path
                """,
                person_id=person_id,
                photo_path=photo_path,
                face_index=f.get("face_index"),
                crop_path=crop_path,
            )

    # Invalidate Neo4j assigned-faces cache
    global _neo4j_assigned
    _neo4j_assigned = None

    return {"assigned": len(faces)}


@router.delete("/assignment", status_code=204)
async def unassign_face(person_id: str, photo_path: str, face_index: int):
    """Remove the APPEARS_IN edge for a specific (person, photo, face_index)
    so a misidentified face returns to the unassigned pool."""
    pp = photo_path.replace("/photos/", "", 1) if photo_path.startswith("/photos/") else photo_path
    async with get_session() as session:
        res = await session.run(
            """
            MATCH (p:Person {id: $person_id})-[r:APPEARS_IN]->(m:Media {path: $pp})
            WHERE r.face_index = $face_index
            DELETE r RETURN 1 AS ok
            """,
            person_id=person_id, pp=pp, face_index=face_index,
        )
        if not await res.single():
            raise HTTPException(404, "Assignment not found")

    global _neo4j_assigned
    _neo4j_assigned = None


@router.get("/clusters/lookup")
async def lookup_face_cluster(photo_path: str, face_index: int):
    """Return the cluster_id (if any) for a specific face in a photo."""
    import asyncio
    await asyncio.to_thread(_build_cluster_index)
    cid = (_cluster_idx or {}).get((photo_path, face_index))
    if not cid:
        return {"cluster_id": None}
    return {
        "cluster_id": cid,
        "size": _cluster_sizes.get(cid, 0),
    }


@router.get("/clusters")
async def list_clusters(status: str = "unassigned", limit: int = 50, offset: int = 0):
    # Assigned view is sourced directly from Neo4j APPEARS_IN edges.
    # Returns one row per Person, with the cluster_id set to
    # `person:<uuid>` so the detail endpoint knows to fetch by person.
    if status == "assigned":
        async with get_session() as session:
            tot_res = await session.run(
                """
                MATCH (p:Person)-[r:APPEARS_IN]->(:Media)
                WHERE r.face_index IS NOT NULL
                  AND r.crop_path IS NOT NULL AND r.crop_path <> ''
                RETURN count(DISTINCT p) AS total
                """
            )
            tot_row = await tot_res.single()
            total   = tot_row["total"] if tot_row else 0

            data_res = await session.run(
                """
                MATCH (p:Person)-[r:APPEARS_IN]->(:Media)
                WHERE r.face_index IS NOT NULL
                  AND r.crop_path IS NOT NULL AND r.crop_path <> ''
                WITH p, count(r) AS face_count, collect(r.crop_path)[..4] AS sample_paths
                ORDER BY face_count DESC, p.name ASC
                SKIP $offset LIMIT $limit
                RETURN p.id AS id, p.name AS name, p.known_as AS known_as, p.avatar AS avatar,
                       face_count, sample_paths
                """,
                offset=offset, limit=limit,
            )
            rows = await data_res.data()

        result = []
        for r in rows:
            samples = [_crop_url(cp) for cp in (r["sample_paths"] or []) if cp]
            result.append({
                "id":              f"person:{r['id']}",
                "size":            r["face_count"],
                "samples":         samples,
                "person_id":       r["id"],
                "person_name":     r["name"] or r["known_as"],
                "person_known_as": r["known_as"] if r["known_as"] != r["name"] else None,
                "person_avatar":   r["avatar"],
            })
        return {"clusters": result, "total": total, "offset": offset}

    clusters = _load(CLUSTERS_FILE, {})
    skipped  = set(str(s) for s in _load(SKIPPED_FILE, []))

    if status == "unassigned":
        # Face-level "already done" set: Neo4j edges + face-level skips.
        # We filter each cluster's faces by this so the user only sees work
        # that's actually remaining.
        done = await _get_neo4j_assigned()
        done = set(done) | await _get_neo4j_skipped()

        remaining_by_id = {}
        for k, faces in clusters.items():
            if k == "-1" or k in skipped: continue
            rem = []
            for f in faces:
                pp = f.get("photo_path", "")
                pp = pp.replace("/photos/", "", 1) if pp.startswith("/photos/") else pp
                fi = f.get("face_index")
                if pp and fi is not None and (pp, int(fi)) not in done:
                    rem.append(f)
            if rem:
                remaining_by_id[k] = rem

        ids = sorted(remaining_by_id.keys(), key=lambda k: len(remaining_by_id[k]), reverse=True)
        total = len(ids)
        page  = ids[offset:offset + limit]
        result = []
        for cid in page:
            faces = remaining_by_id[cid]
            samples = [_crop_url(f["crop_path"]) for f in faces[:SAMPLE_CROPS] if f.get("crop_path")]
            result.append({
                "id":          cid,
                "size":        len(faces),
                "samples":     samples,
                "person_id":   None,
            })
        return {"clusters": result, "total": total, "offset": offset}
    else:
        ids = [k for k in clusters if k != "-1"]

    # Sort by cluster size desc
    ids.sort(key=lambda k: len(clusters[k]), reverse=True)
    total = len(ids)
    ids   = ids[offset:offset + limit]

    result = []
    person_ids_to_lookup = set()
    for cid in ids:
        faces = clusters[cid]
        samples = [_crop_url(f["crop_path"]) for f in faces[:SAMPLE_CROPS] if f.get("crop_path")]
        pid = assignments.get(cid)
        if pid:
            person_ids_to_lookup.add(pid)
        result.append({
            "id":          cid,
            "size":        len(faces),
            "samples":     samples,
            "person_id":   pid,
        })

    # Enrich assigned clusters with the person's display name + avatar so the UI
    # doesn't have to render UUIDs.
    if person_ids_to_lookup:
        people = {}
        try:
            async with get_session() as session:
                res = await session.run(
                    "MATCH (p:Person) WHERE p.id IN $ids "
                    "RETURN p.id AS id, p.name AS name, p.known_as AS known_as, p.avatar AS avatar",
                    ids=list(person_ids_to_lookup),
                )
                for row in await res.data():
                    people[row["id"]] = row
        except Exception as e:
            log.warning(f"person lookup failed for cluster list: {e}")
        for r in result:
            p = people.get(r["person_id"])
            if p:
                r["person_name"]   = p.get("name") or p.get("known_as")
                r["person_known_as"] = p.get("known_as") if p.get("known_as") != p.get("name") else None
                r["person_avatar"] = p.get("avatar")

    return {"clusters": result, "total": total, "offset": offset}


@router.get("/clusters/{cluster_id}")
async def get_cluster(cluster_id: str):
    # Person-scoped detail (from the Neo4j-backed assigned view).
    if cluster_id.startswith("person:"):
        person_id = cluster_id[len("person:"):]
        async with get_session() as session:
            res = await session.run(
                """
                MATCH (p:Person {id: $pid})-[r:APPEARS_IN]->(m:Media)
                WHERE r.face_index IS NOT NULL
                  AND r.crop_path IS NOT NULL AND r.crop_path <> ''
                RETURN m.path AS photo_path, r.face_index AS face_index, r.crop_path AS crop_path
                ORDER BY m.timestamp DESC
                """,
                pid=person_id,
            )
            rows = await res.data()
        faces = [
            {
                "photo_path": r["photo_path"],
                "face_index": r["face_index"],
                "crop_url":   _crop_url(r["crop_path"]) if r["crop_path"] else None,
            }
            for r in rows
        ]
        return {"id": cluster_id, "size": len(faces), "faces": faces, "person_id": person_id}

    clusters = _load(CLUSTERS_FILE, {})

    if cluster_id not in clusters:
        raise HTTPException(404, "Cluster not found")

    faces = clusters[cluster_id]
    # Hide faces already done (Neo4j edges or face-level skips).
    done = await _get_neo4j_assigned()
    done = set(done) | await _get_neo4j_skipped()
    faces = [
        f for f in faces
        if (
            (f.get("photo_path", "").replace("/photos/", "", 1)
              if f.get("photo_path", "").startswith("/photos/") else f.get("photo_path", "")),
            int(f.get("face_index", -1))
        ) not in done
    ]

    return {
        "id":        cluster_id,
        "size":      len(faces),
        "faces":     [
            {
                "photo_path": f.get("photo_path"),
                "face_index": f.get("face_index"),
                "crop_url":   _crop_url(f["crop_path"]) if f.get("crop_path") else None,
                "confidence": f.get("confidence"),
            }
            for f in faces
        ],
        "person_id": None,
    }


class AssignBody(BaseModel):
    person_id: str
    # Optional faces to exclude (permanent skip — written to Media.skipped_faces in Neo4j).
    exclude: list[list] | None = None
    # Optional whitelist: only these faces are assigned. The cluster as a
    # whole is never marked "done" — Neo4j edges per face are the only
    # signal, so remaining faces stay in the cluster for later sessions.
    include: list[list] | None = None


@router.post("/clusters/{cluster_id}/assign", status_code=204)
async def assign_cluster(cluster_id: str, body: AssignBody):
    clusters = _load(CLUSTERS_FILE, {})
    if cluster_id not in clusters:
        raise HTTPException(404, "Cluster not found")

    faces = clusters[cluster_id]

    # Exclude (permanent face-level skip)
    if body.exclude:
        exclude_set = {(row[0], row[1]) for row in body.exclude if len(row) >= 2}
        faces = [
            f for f in faces
            if (f.get("photo_path"), f.get("face_index")) not in exclude_set
        ]
        await _add_skipped_faces([(p, fi) for p, fi in exclude_set if p])

    # Optional whitelist: only these faces get assigned.
    if body.include is not None:
        include_set = {(row[0], row[1]) for row in body.include if len(row) >= 2}
        faces = [
            f for f in faces
            if (f.get("photo_path"), f.get("face_index")) in include_set
        ]
    # Neo4j sync below writes APPEARS_IN edges for `faces`; those edges
    # are what _get_neo4j_assigned uses to hide already-done faces.

    # Bust the Neo4j-assigned cache so the next list call sees the new edges.
    global _neo4j_assigned
    _neo4j_assigned = None

    # Write Neo4j APPEARS_IN in background
    threading.Thread(
        target=_sync_to_neo4j,
        args=(body.person_id, faces),
        daemon=True,
    ).start()


@router.post("/clusters/{cluster_id}/skip", status_code=204)
async def skip_cluster(cluster_id: str):
    clusters = _load(CLUSTERS_FILE, {})
    if cluster_id not in clusters:
        raise HTTPException(404, "Cluster not found")
    skipped = _load(SKIPPED_FILE, [])
    if cluster_id not in skipped:
        skipped.append(cluster_id)
        _save(SKIPPED_FILE, skipped)

    # Persist face-level skip so it survives reclustering
    items = [
        (f.get("photo_path"), f.get("face_index"))
        for f in clusters[cluster_id]
        if f.get("photo_path") and f.get("face_index") is not None
    ]
    await _add_skipped_faces(items)


@router.post("/clusters/{cluster_id}/unskip", status_code=204)
async def unskip_cluster(cluster_id: str):
    clusters = _load(CLUSTERS_FILE, {})
    skipped = _load(SKIPPED_FILE, [])
    skipped = [s for s in skipped if str(s) != str(cluster_id)]
    _save(SKIPPED_FILE, skipped)

    # Remove these faces from the stable skip list
    if cluster_id in clusters:
        items = [
            (f.get("photo_path"), f.get("face_index"))
            for f in clusters[cluster_id]
            if f.get("photo_path") and f.get("face_index") is not None
        ]
        await _remove_skipped_faces(items)


def _sync_to_neo4j(person_id: str, faces: list):
    import asyncio
    from neo4j import GraphDatabase
    from app.config import settings as s

    try:
        driver = GraphDatabase.driver(s.neo4j_uri, auth=(s.neo4j_user, s.neo4j_password))
        with driver.session() as session:
            batch = []
            for f in faces:
                photo_path = f.get("photo_path", "").replace("/photos/", "", 1)
                crop_path  = f.get("crop_path", "").replace("/photos/", "", 1)
                batch.append({
                    "photo_path": photo_path,
                    "face_index": f.get("face_index"),
                    "crop_path":  crop_path,
                })
            for i in range(0, len(batch), 200):
                session.run(
                    """
                    UNWIND $batch AS row
                    MATCH (person:Person {id: $person_id})
                    MATCH (photo:Media {path: row.photo_path})
                    MERGE (person)-[r:APPEARS_IN {face_index: row.face_index}]->(photo)
                    SET r.crop_path  = row.crop_path,
                        r.photo_path = row.photo_path
                    """,
                    person_id=person_id,
                    batch=batch[i:i+200],
                )
        driver.close()
        log.info(f"Synced {len(faces)} faces for person {person_id}")
    except Exception as e:
        log.error(f"Neo4j sync failed for person {person_id}: {e}")
