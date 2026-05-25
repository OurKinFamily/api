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
ASSIGNMENTS_FILE        = CLUSTERS_DIR / "assignments.json"
SKIPPED_FILE            = CLUSTERS_DIR / "skipped.json"
SKIPPED_FACES_FILE      = CLUSTERS_DIR / "skipped_faces.json"
INDIVIDUAL_ASSIGNS_FILE = CLUSTERS_DIR / "individual_assignments.json"  # [(photo_path, face_index), ...]
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
            "MATCH ()-[r:APPEARS_IN]->(photo:Photo) WHERE r.face_index IS NOT NULL "
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


def _build_assigned_face_set() -> set[tuple[str, int]]:
    """Return set of (rel_photo_path, face_index) for all assigned faces (cluster + individual)."""
    assigned: set[tuple[str, int]] = set()

    # Cluster-level assignments
    assignments = _load(ASSIGNMENTS_FILE, {})
    if assignments:
        clusters = _load(CLUSTERS_FILE, {})
        for cid in assignments:
            for f in clusters.get(cid, []):
                pp = f.get("photo_path", "").replace("/photos/", "", 1)
                fi = f.get("face_index")
                if pp and fi is not None:
                    assigned.add((pp, int(fi)))

    # Individual assignments (from search/assign)
    for row in _load(INDIVIDUAL_ASSIGNS_FILE, []):
        assigned.add((row[0], int(row[1])))

    return assigned


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

    assigned = _build_assigned_face_set() if body.unassigned_only else set()
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
            MATCH (p:Person {id: $person_id})-[r:APPEARS_IN]->(photo:Photo)
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

    assigned = _build_assigned_face_set() if body.unassigned_only else set()
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
            MATCH (photo:Photo) WHERE photo.path IN $paths
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
            MATCH (person:Person)-[:APPEARS_IN]->(photo:Photo)
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
            MATCH (p:Person {id: $person_id})-[r:APPEARS_IN]->(photo:Photo)
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
    # Combine all exclusion sources: person's own faces + Neo4j APPEARS_IN + file-based cluster assignments
    file_assigned = _build_assigned_face_set() if body.unassigned_only else set()
    exclude = person_faces | file_assigned | (neo4j_assigned if body.unassigned_only else set())

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
    """Assign multiple individual faces to a person."""
    person_id = body.get("person_id")
    faces     = body.get("faces", [])  # list of {photo_path, face_index, crop_path}
    if not person_id or not faces:
        raise HTTPException(400, "person_id and faces required")

    async with get_session() as session:
        for f in faces:
            photo_path = f.get("photo_path", "").replace("/photos/", "", 1)
            crop_path  = f.get("crop_path",  "").replace("/photos/", "", 1)
            await session.run(
                """
                MATCH (person:Person {id: $person_id})
                MATCH (photo:Photo {path: $photo_path})
                MERGE (person)-[r:APPEARS_IN]->(photo)
                SET r.face_index = $face_index,
                    r.crop_path  = $crop_path,
                    r.photo_path = $photo_path
                """,
                person_id=person_id,
                photo_path=photo_path,
                face_index=f.get("face_index"),
                crop_path=crop_path,
            )

    # Track individually assigned faces so similarity search excludes them
    existing = {tuple(row) for row in _load(INDIVIDUAL_ASSIGNS_FILE, [])}
    for f in faces:
        pp = f.get("photo_path", "").replace("/photos/", "", 1)
        fi = f.get("face_index")
        if pp and fi is not None:
            existing.add((pp, int(fi)))
    _save(INDIVIDUAL_ASSIGNS_FILE, [list(row) for row in existing])

    # Invalidate Neo4j assigned-faces cache
    global _neo4j_assigned
    _neo4j_assigned = None

    return {"assigned": len(faces)}


@router.get("/clusters/lookup")
async def lookup_face_cluster(photo_path: str, face_index: int):
    """Return the cluster_id (if any) for a specific face in a photo."""
    import asyncio
    await asyncio.to_thread(_build_cluster_index)
    cid = (_cluster_idx or {}).get((photo_path, face_index))
    if not cid:
        return {"cluster_id": None}
    assignments = _load(ASSIGNMENTS_FILE, {})
    return {
        "cluster_id": cid,
        "size": _cluster_sizes.get(cid, 0),
        "already_assigned": cid in assignments,
    }


@router.get("/clusters")
async def list_clusters(status: str = "unassigned", limit: int = 50, offset: int = 0):
    clusters    = _load(CLUSTERS_FILE, {})
    assignments = _load(ASSIGNMENTS_FILE, {})
    skipped     = set(str(s) for s in _load(SKIPPED_FILE, []))

    if status == "unassigned":
        ids = [k for k in clusters if k not in assignments and k not in skipped and k != "-1"]
    elif status == "assigned":
        ids = [k for k in clusters if k in assignments and k != "-1"]
    else:
        ids = [k for k in clusters if k != "-1"]

    # Sort by cluster size desc
    ids.sort(key=lambda k: len(clusters[k]), reverse=True)
    total = len(ids)
    ids   = ids[offset:offset + limit]

    result = []
    for cid in ids:
        faces = clusters[cid]
        samples = [_crop_url(f["crop_path"]) for f in faces[:SAMPLE_CROPS] if f.get("crop_path")]
        result.append({
            "id":          cid,
            "size":        len(faces),
            "samples":     samples,
            "person_id":   assignments.get(cid),
        })

    return {"clusters": result, "total": total, "offset": offset}


@router.get("/clusters/{cluster_id}")
async def get_cluster(cluster_id: str):
    clusters    = _load(CLUSTERS_FILE, {})
    assignments = _load(ASSIGNMENTS_FILE, {})

    if cluster_id not in clusters:
        raise HTTPException(404, "Cluster not found")

    faces = clusters[cluster_id]
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
        "person_id": assignments.get(cluster_id),
    }


class AssignBody(BaseModel):
    person_id: str


@router.post("/clusters/{cluster_id}/assign", status_code=204)
async def assign_cluster(cluster_id: str, body: AssignBody):
    clusters    = _load(CLUSTERS_FILE, {})
    assignments = _load(ASSIGNMENTS_FILE, {})

    if cluster_id not in clusters:
        raise HTTPException(404, "Cluster not found")

    assignments[cluster_id] = body.person_id
    _save(ASSIGNMENTS_FILE, assignments)

    # Write Neo4j APPEARS_IN in background
    faces = clusters[cluster_id]
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
    existing = {(row[0], row[1]) for row in _load(SKIPPED_FACES_FILE, [])}
    new_entries = []
    for f in clusters[cluster_id]:
        key = (f.get("photo_path"), f.get("face_index"))
        if key[0] and key[1] is not None and key not in existing:
            new_entries.append(list(key))
            existing.add(key)
    if new_entries:
        all_entries = _load(SKIPPED_FACES_FILE, []) + new_entries
        _save(SKIPPED_FACES_FILE, all_entries)


@router.post("/clusters/{cluster_id}/unskip", status_code=204)
async def unskip_cluster(cluster_id: str):
    clusters = _load(CLUSTERS_FILE, {})
    skipped = _load(SKIPPED_FILE, [])
    skipped = [s for s in skipped if str(s) != str(cluster_id)]
    _save(SKIPPED_FILE, skipped)

    # Remove these faces from the stable skip list
    if cluster_id in clusters:
        remove = {(f.get("photo_path"), f.get("face_index")) for f in clusters[cluster_id]}
        remaining = [row for row in _load(SKIPPED_FACES_FILE, []) if tuple(row) not in remove]
        _save(SKIPPED_FACES_FILE, remaining)


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
                    MATCH (photo:Photo {path: row.photo_path})
                    MERGE (person)-[r:APPEARS_IN]->(photo)
                    SET r.face_index = row.face_index,
                        r.crop_path  = row.crop_path,
                        r.photo_path = row.photo_path
                    """,
                    person_id=person_id,
                    batch=batch[i:i+200],
                )
        driver.close()
        log.info(f"Synced {len(faces)} faces for person {person_id}")
    except Exception as e:
        log.error(f"Neo4j sync failed for person {person_id}: {e}")
