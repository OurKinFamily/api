"""
Face cluster management — reads/writes /photos/__faces/clusters/*.json
and syncs assignments to Neo4j APPEARS_IN relationships.
"""

import json
import threading
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from app.config import settings, with_v
from app.db.neo4j import get_session
from app.deps import require_admin
from app.log import logger as log

router = APIRouter(prefix="/faces", tags=["faces"], dependencies=[Depends(require_admin)])

CLUSTERS_DIR            = settings.photos_root / "__faces" / "clusters"
CLUSTERS_FILE           = CLUSTERS_DIR / "clusters.json"
SKIPPED_FILE            = CLUSTERS_DIR / "skipped.json"
EMBEDDINGS_NPY          = CLUSTERS_DIR / "embeddings.npy"
FACES_META_FILE         = CLUSTERS_DIR / "faces_meta.json"
BRAIN_DIR               = settings.photos_root / "__faces" / "brain"

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

# Brain (per-person multi-centroid identity model) — lazy-loaded from
# /photos/__faces/brain/<uuid>.json. Rebuilt by scripts/build_brain.py.
_brain_centroids = None         # numpy [n_buckets, 512] L2-normalized
_brain_bucket_person_ix = None  # numpy int [n_buckets] → index into _brain_persons_order
_brain_persons_order: list[str] | None = None  # person_id at each persons-index
_brain_persons: dict[str, dict] | None = None  # person_id → {name, dob_year, n_total, negatives}
_brain_bucket_meta: list[dict] | None = None   # {key, n, radius} per bucket
_brain_seg_starts = None         # numpy int — np.maximum.reduceat segment starts
_brain_seg_persons = None        # numpy int — person ix at each segment start
_brain_sort_idx = None           # numpy int — sort buckets by person ix
_brain_loaded_at: float = 0.0
_brain_lock = threading.Lock()


def _load_brain():
    """Lazy-load all per-person brain files into a single numpy structure for
    fast cluster-vs-brain scoring. Reloads if any brain file is newer than
    last load. Returns None tuple if no brain files exist yet."""
    import numpy as np
    global _brain_centroids, _brain_bucket_person_ix, _brain_persons_order, _brain_persons
    global _brain_bucket_meta, _brain_seg_starts, _brain_seg_persons, _brain_sort_idx, _brain_loaded_at
    with _brain_lock:
        if not BRAIN_DIR.exists():
            return None
        files = list(BRAIN_DIR.glob("*.json"))
        if not files:
            return None
        if _brain_centroids is not None:
            try:
                newest = max(f.stat().st_mtime for f in files)
                if newest <= _brain_loaded_at:
                    return (_brain_centroids, _brain_bucket_person_ix, _brain_persons_order,
                            _brain_persons, _brain_bucket_meta, _brain_seg_starts,
                            _brain_seg_persons, _brain_sort_idx)
            except (FileNotFoundError, ValueError):
                pass
        centroids: list[list[float]] = []
        bucket_person_ix: list[int] = []
        bucket_meta: list[dict] = []
        persons_order: list[str] = []
        persons: dict[str, dict] = {}
        for f in files:
            try:
                d = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            pid = d.get("person_id")
            if not pid or not d.get("buckets"):
                continue
            person_ix = len(persons_order)
            persons_order.append(pid)
            persons[pid] = {
                "name":       d.get("person_name"),
                "dob_year":   d.get("dob_year"),
                "n_total":    d.get("n_confirmed_total", 0),
                "negatives":  {(x[0], int(x[1])) for x in d.get("negative_examples", []) if x},
            }
            for b in d["buckets"]:
                centroids.append(b["centroid"])
                bucket_person_ix.append(person_ix)
                bucket_meta.append({"key": b["key"], "n": b["n"], "radius": b["radius"]})
        if not centroids:
            return None
        _brain_centroids = np.asarray(centroids, dtype=np.float32)
        _brain_bucket_person_ix = np.asarray(bucket_person_ix, dtype=np.int32)
        _brain_persons_order = persons_order
        _brain_persons = persons
        _brain_bucket_meta = bucket_meta
        # Pre-compute reduceat segment info for fast per-person max
        _brain_sort_idx = np.argsort(_brain_bucket_person_ix, kind="stable")
        sorted_persons = _brain_bucket_person_ix[_brain_sort_idx]
        _brain_seg_starts = np.concatenate(([0], np.where(np.diff(sorted_persons) != 0)[0] + 1)).astype(np.int64)
        _brain_seg_persons = sorted_persons[_brain_seg_starts]
        import time
        _brain_loaded_at = time.time()
        log.info(
            f"brain loaded: {len(persons_order)} people, {len(centroids)} buckets"
        )
        return (_brain_centroids, _brain_bucket_person_ix, _brain_persons_order,
                _brain_persons, _brain_bucket_meta, _brain_seg_starts,
                _brain_seg_persons, _brain_sort_idx)


def _invalidate_brain():
    """Force next _load_brain() to re-read all files."""
    global _brain_centroids
    with _brain_lock:
        _brain_centroids = None


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


# Parsed-JSON cache keyed by path → (mtime, data). clusters.json is now ~96MB
# (181k clusters at min_samples=1); re-parsing it on every request — and it's
# touched by grouped/count/leftover/assign/get_cluster — blocked the async event
# loop for seconds each call, starving thumbnail serving. Cache by mtime so the
# 96MB parse happens once per file change. The cluster job / purge script bump
# the mtime when they rewrite, so the cache self-invalidates.
_load_cache: dict[str, tuple[float, object]] = {}


def _load(path: Path, default):
    if not path.exists():
        return default
    try:
        mtime = path.stat().st_mtime
        key = str(path)
        hit = _load_cache.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
        data = json.loads(path.read_text())
        _load_cache[key] = (mtime, data)
        return data
    except Exception:
        return default


def _save(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
    # Write through the parsed-JSON cache so the next _load sees fresh data
    # without re-parsing, and never returns a stale pre-save copy.
    try:
        _load_cache[str(path)] = (path.stat().st_mtime, data)
    except OSError:
        _load_cache.pop(str(path), None)


def _crop_url(crop_path: str) -> str:
    rel = crop_path.replace("/photos/", "", 1) if crop_path.startswith("/photos/") else crop_path
    return with_v(f"/api/media/{rel}")


# Small in-process cache for person_id → display name, used to enrich log
# events so Grafana panels can show real names instead of UUIDs.
_person_name_cache: dict[str, str] = {}


async def _get_person_name(person_id: str, session=None) -> str:
    """Look up Person.name by id, with a tiny in-process cache. Falls back
    to the id itself if not found. `session` is optional — pass an open one
    to reuse it, otherwise a fresh session is created."""
    if not person_id:
        return ""
    if person_id in _person_name_cache:
        return _person_name_cache[person_id]
    try:
        if session is None:
            async with get_session() as s:
                res = await s.run(
                    "MATCH (p:Person {id: $id}) RETURN p.name AS name LIMIT 1",
                    id=person_id,
                )
                row = await res.single()
        else:
            res = await session.run(
                "MATCH (p:Person {id: $id}) RETURN p.name AS name LIMIT 1",
                id=person_id,
            )
            row = await res.single()
        name = (row and row["name"]) or person_id
    except Exception:
        name = person_id
    _person_name_cache[person_id] = name
    return name


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
        crop_url = with_v(f"/api/media/{crop.replace('/photos/', '', 1)}") if crop.startswith("/photos/") else with_v(f"/api/media/{crop}")
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


async def _resolve_real_paths(paths: list[str]) -> dict[str, str]:
    """Map face-index photo_paths → real Media-node paths.

    The face index records a photo by its DATE-bucket path (archive/2025/12/…),
    but mis-dated files live at archive/0000/…, so those paths match no node and
    don't exist on disk (broken lightbox image, empty photo-meta). This maps each
    given path to the actual node path: identity when it already matches, else by
    unique filename. Paths with no/ambiguous match are omitted.
    """
    if not paths:
        return {}
    bases = {p: p.rsplit("/", 1)[-1] for p in paths}
    async with get_session() as session:
        result = await session.run(
            """
            UNWIND $rows AS row
            OPTIONAL MATCH (exact:Media {path: row.path})
            CALL {
                WITH row
                MATCH (alt:Media {filename: row.base})
                RETURN collect(alt.path) AS alts
            }
            WITH row, coalesce(exact.path, CASE WHEN size(alts) = 1 THEN alts[0] END) AS real
            WHERE real IS NOT NULL
            RETURN row.path AS given, real
            """,
            rows=[{"path": p, "base": b} for p, b in bases.items()],
        )
        return {r["given"]: r["real"] for r in await result.data()}


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
                # Timestamps are a mix of tz-aware (…Z) and naive (no tz) across
                # the archive; normalise to UTC-aware so the sort + span math
                # below don't blow up comparing naive vs aware datetimes.
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
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
                crop_url = with_v(f"/api/media/{crop.replace('/photos/', '', 1)}") if crop.startswith("/photos/") else with_v(f"/api/media/{crop}")
                best[key] = {
                    "photo_path": rel,
                    "face_index": m["face_index"],
                    "crop_url":   crop_url,
                    "similarity": round(sim, 4),
                    "distance":   round(1.0 - sim, 4),
                }

    if not best:
        return {"results": [], "total": 0, "faces_used": len(all_indices), "buckets": bucket_info if use_temporal else None}

    # Remap face-index date-bucket paths → real Media-node paths so the
    # lightbox can load the original image and photo-meta enrichment resolves.
    # crop_url is unaffected (crops are keyed by their own path). Candidates with
    # NO resolvable Media node (bio crops, video poster frames — faces were
    # extracted but there's nothing to attach an APPEARS_IN edge to) are dropped:
    # they're unassignable, so showing them just produces silent no-op assigns
    # that reappear forever.
    real_paths = await _resolve_real_paths(list({r["photo_path"] for r in best.values()}))
    resolved_best: dict[tuple[str, int], dict] = {}
    for k, v in best.items():
        real = real_paths.get(v["photo_path"])
        if real is None:
            continue
        v["photo_path"] = real
        # Re-apply the exclusion set on the REAL path. The in-loop exclusion
        # keys off the date-bucket path, but `exclude` (person's faces +
        # assigned) is real-path keyed — so a just-assigned mis-dated face
        # slips the loop and would reappear. Drop it now that paths resolved.
        if body.unassigned_only and v.get("face_index") is not None \
                and (real, int(v["face_index"])) in exclude:
            continue
        resolved_best[k] = v
    best = resolved_best
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
async def bulk_assign_faces(body: dict, request: Request):
    """Assign multiple individual faces to a person.

    Frontend often doesn't know the crop path — derive it server-side from
    `photo_path + face_index` so every edge gets a usable crop_path. This
    matches the canonical layout face extraction writes:
    `__faces/crops/<year>/<month>/<filename>_face<N>.jpg`.

    Emits one structured log line per face so Loki/Grafana can answer
    "every face Stephen has assigned to this person, ever."
    """
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

    by = getattr(request.state, "user_email", None)
    request_id = getattr(request.state, "request_id", None)
    bulk_id = uuid.uuid4().hex[:12]
    person_name = await _get_person_name(person_id)
    log.bind(
        event="face.bulk_assign.started",
        person_id=person_id,
        person_name=person_name,
        count=len(faces),
        bulk_id=bulk_id,
        by=by,
        request_id=request_id,
    ).info("face bulk assign started")

    assigned_count = 0
    missing: list[dict] = []
    async with get_session() as session:
        for f in faces:
            photo_path = f.get("photo_path", "").replace("/photos/", "", 1)
            crop_path  = f.get("crop_path",  "").replace("/photos/", "", 1) or _canonical_crop_path(photo_path, f.get("face_index"))
            face_index = f.get("face_index")
            # The face index records a photo by its DATE-bucket path
            # (e.g. archive/2025/12/…), but mis-dated files live under
            # archive/0000/… — so an exact path MATCH silently misses and the
            # assign no-ops. Fall back to the (unique) filename when the exact
            # path doesn't resolve; bail to `missing` if the filename is
            # ambiguous so we never assign to the wrong photo.
            basename = photo_path.rsplit("/", 1)[-1]
            res = await session.run(
                """
                MATCH (person:Person {id: $person_id})
                OPTIONAL MATCH (exact:Media {path: $photo_path})
                CALL {
                    MATCH (alt:Media {filename: $basename})
                    RETURN collect(alt) AS alts
                }
                WITH person, coalesce(exact, CASE WHEN size(alts) = 1 THEN alts[0] END) AS photo
                WHERE photo IS NOT NULL
                MERGE (person)-[r:APPEARS_IN {face_index: $face_index}]->(photo)
                SET r.crop_path  = $crop_path,
                    r.photo_path = photo.path
                RETURN photo.path AS p
                """,
                person_id=person_id,
                photo_path=photo_path,
                basename=basename,
                face_index=face_index,
                crop_path=crop_path,
            )
            row = await res.single()
            if row is None:
                # Either person doesn't exist or media path doesn't match
                # anything in Neo4j (common when embedding index has stale
                # paths). Log it loudly so the UI sees the count drop +
                # we get a Loki signal.
                missing.append({"photo_path": photo_path, "face_index": face_index})
                log.bind(
                    event="face.assign.skipped",
                    person_id=person_id,
                    photo_path=photo_path,
                    face_index=face_index,
                    reason="media_or_person_not_found",
                    bulk_id=bulk_id,
                    by=by,
                    request_id=request_id,
                ).warning("face assign skipped")
                continue
            assigned_count += 1
            log.bind(
                event="face.assigned",
                person_id=person_id,
                person_name=person_name,
                photo_path=photo_path,
                face_index=face_index,
                crop_path=crop_path,
                source="search_assign",
                bulk_id=bulk_id,
                by=by,
                request_id=request_id,
            ).info("face assigned")

    # Invalidate Neo4j assigned-faces cache
    global _neo4j_assigned
    _neo4j_assigned = None

    # Rebuild this person's brain so the suggestion engine sees the new faces.
    if assigned_count:
        try:
            from app.services.brain import rebuild_person_brain
            async with get_session() as session:
                await rebuild_person_brain(person_id, session)
            _invalidate_brain()
        except Exception as e:
            log.warning(f"brain rebuild failed for {person_id}: {e}")

    return {"assigned": assigned_count, "missing": missing, "requested": len(faces)}


@router.delete("/assignment", status_code=204)
async def unassign_face(request: Request, person_id: str, photo_path: str, face_index: int):
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

        # Rebuild brain in the same session.
        try:
            from app.services.brain import rebuild_person_brain
            await rebuild_person_brain(person_id, session)
            _invalidate_brain()
        except Exception as e:
            log.warning(f"brain rebuild failed for {person_id}: {e}")

    global _neo4j_assigned
    _neo4j_assigned = None

    person_name = await _get_person_name(person_id)
    log.bind(
        event="face.unassigned",
        person_id=person_id,
        person_name=person_name,
        photo_path=pp,
        face_index=face_index,
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("face unassigned")


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


class GroupedSuggestionsBody(BaseModel):
    threshold: float = 0.75
    margin: float = 0.05            # gap between #1 and #2 to count as "group" vs "cascade"
    limit: int = 50                 # max groups (or ambiguous clusters) to return per list
    person_ids: list[str] | None = None  # if set, only score against these brain people
    min_cluster_size: int = 3       # skip remaining clusters smaller than this
    inter_threshold: float = 0.80   # pairwise sim to call 2 unmatched clusters the same unknown
    min_unknown_clusters: int = 2   # min clusters in an unknown-person candidate group
    min_unknown_faces: int = 6      # min total faces in an unknown-person candidate group
    maybe_threshold: float = 0.65   # combined-centroid sim floor for "Maybe X?" hints on unknown candidates


@router.get("/unassigned/count")
async def unassigned_count():
    """Lightweight tally for the Face Suggestions header: how many detected faces
    are still unassigned (and not skipped). Excludes noise (-1), skipped clusters,
    and every face already assigned via APPEARS_IN or face-level skipped. Counts
    only faces present in the embedding index (the ones actually actionable)."""
    import asyncio
    _matrix, _meta, lookup = await asyncio.to_thread(_load_embedding_index)
    clusters = _load(CLUSTERS_FILE, {})
    skipped = set(str(s) for s in _load(SKIPPED_FILE, []))
    done = set(await _get_neo4j_assigned()) | set(await _get_neo4j_skipped())

    remaining = 0
    for cid, faces in clusters.items():
        if cid == "-1" or cid in skipped:
            continue
        for f in faces:
            pp = f.get("photo_path", "")
            pp = pp.replace("/photos/", "", 1) if pp.startswith("/photos/") else pp
            fi = f.get("face_index")
            if pp and fi is not None and (pp, int(fi)) not in done:
                if lookup is None or (pp, int(fi)) in lookup:
                    remaining += 1
    return {"remaining": remaining, "assigned": len(await _get_neo4j_assigned())}


@router.post("/unassigned/grouped")
async def grouped_suggestions(body: GroupedSuggestionsBody):
    """For every remaining-unassigned cluster, score its centroid against every
    person's brain (best-bucket per person), then group clusters by inferred
    identity. Returns:
      - `groups`: high-confidence groupings ("Likely Stephen — 8 clusters, 240 faces")
      - `ambiguous`: clusters whose top-2 candidates are within `margin` (cascade fodder)
      - `unmatched_count`: clusters with no candidate above `threshold`

    No writes. UI calls this to render the "Likely matches" section.
    """
    import asyncio, numpy as np

    matrix, meta, lookup = await asyncio.to_thread(_load_embedding_index)
    if matrix is None:
        raise HTTPException(503, "Embedding index not built yet")

    brain = await asyncio.to_thread(_load_brain)
    if brain is None:
        raise HTTPException(503, "Brain not built — run scripts/build_brain.py")
    (centroids, bucket_person_ix, persons_order, persons, bucket_meta,
     seg_starts, seg_persons, sort_idx) = brain

    # Optional person-subset filter — rebuild seg structures over the filtered slice
    if body.person_ids:
        wanted = set(body.person_ids)
        keep_person_ix = {pi for pi, pid in enumerate(persons_order) if pid in wanted}
        if not keep_person_ix:
            return {"groups": [], "ambiguous": [], "unmatched_count": 0}
        mask = np.isin(bucket_person_ix, np.array(sorted(keep_person_ix), dtype=np.int32))
        centroids = centroids[mask]
        sub_person_ix = bucket_person_ix[mask]
        sort_idx = np.argsort(sub_person_ix, kind="stable")
        sorted_p = sub_person_ix[sort_idx]
        seg_starts = np.concatenate(([0], np.where(np.diff(sorted_p) != 0)[0] + 1)).astype(np.int64)
        seg_persons = sorted_p[seg_starts]

    clusters_data = _load(CLUSTERS_FILE, {})
    skipped = set(str(s) for s in _load(SKIPPED_FILE, []))
    done = await _get_neo4j_assigned()
    done = set(done) | await _get_neo4j_skipped()

    # Walk every unskipped cluster, drop already-handled faces, keep remaining + their embedding rows
    cluster_summary: list[dict] = []
    for cid, faces in clusters_data.items():
        if cid == "-1" or cid in skipped:
            continue
        remaining = []
        emb_indices: list[int] = []
        for f in faces:
            pp = f.get("photo_path", "")
            pp = pp.replace("/photos/", "", 1) if pp.startswith("/photos/") else pp
            fi = f.get("face_index")
            if pp and fi is not None and (pp, int(fi)) not in done:
                remaining.append(f)
                idx = lookup.get((pp, int(fi)))
                if idx is not None:
                    emb_indices.append(idx)
        if len(remaining) < body.min_cluster_size or not emb_indices:
            continue
        cluster_summary.append({
            "cluster_id": str(cid),
            "remaining": remaining,
            "emb_indices": emb_indices,
        })

    if not cluster_summary:
        return {"groups": [], "ambiguous": [], "unmatched_count": 0}

    # Build per-cluster centroids
    n_clusters = len(cluster_summary)
    cluster_centroids = np.zeros((n_clusters, matrix.shape[1]), dtype=np.float32)
    for i, cs in enumerate(cluster_summary):
        c = matrix[cs["emb_indices"]].mean(axis=0)
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        cluster_centroids[i] = c

    # [n_clusters, n_buckets] cosine sims. Score each cluster by the MAX over its
    # individual faces (not the cluster mean): one clean face that strongly matches
    # a centroid is strong evidence, whereas averaging the cluster's faces dilutes
    # that signal when the cluster also holds blurry / off-angle crops. The mean
    # `cluster_centroids` (built above) is still used downstream for unknown-person
    # pairwise grouping.
    #
    # Vectorised: stack every cluster's faces into one matrix, do a single matmul
    # against all centroids, then segment-max per cluster via reduceat. Avoids a
    # Python loop of ~thousands of small matmuls (which was ~20s at 150 centroids).
    all_face_idx = []
    face_seg_starts = []          # row in the stacked matrix where each cluster begins
    cursor = 0
    for cs in cluster_summary:
        face_seg_starts.append(cursor)
        all_face_idx.extend(cs["emb_indices"])
        cursor += len(cs["emb_indices"])
    face_sims_all = matrix[all_face_idx] @ centroids.T         # [total_faces, n_buckets]
    sims = np.maximum.reduceat(face_sims_all, face_seg_starts, axis=0)  # [n_clusters, n_buckets]
    sorted_sims = sims[:, sort_idx]
    per_person_sims = np.maximum.reduceat(sorted_sims, seg_starts, axis=1)
    # per_person_sims[ci, k] = best sim of cluster ci to person at seg_persons[k]

    # Per cluster: top-K candidates (sorted by sim desc) → decide group/cascade/unmatched
    groups_by_person: dict[str, list[dict]] = {}
    ambiguous: list[dict] = []
    unmatched_indices: list[int] = []   # cluster_summary indices for unknown-person grouping

    # Pre-sort each row's person scores descending. Top-N per cluster is cheap.
    TOP_K = 3
    for ci, cs in enumerate(cluster_summary):
        row = per_person_sims[ci]
        # Sort high-to-low
        order = np.argsort(-row)[:TOP_K]
        top_pid = persons_order[seg_persons[order[0]]]
        top_sim = float(row[order[0]])

        if top_sim < body.threshold:
            unmatched_indices.append(ci)
            continue

        samples = [
            {
                "crop_url":   _crop_url(f["crop_path"]),
                "photo_path": (f.get("photo_path", "") or "").replace("/photos/", "", 1)
                              if (f.get("photo_path", "") or "").startswith("/photos/")
                              else (f.get("photo_path", "") or ""),
            }
            for f in cs["remaining"][:SAMPLE_CROPS] if f.get("crop_path")
        ]
        cluster_entry = {
            "cluster_id":  cs["cluster_id"],
            "n_faces":     len(cs["remaining"]),
            "similarity":  round(top_sim, 4),
            "samples":     samples,
        }

        second_sim = float(row[order[1]]) if len(order) > 1 else 0.0
        gap = top_sim - second_sim

        if gap >= body.margin:
            groups_by_person.setdefault(top_pid, []).append(cluster_entry)
        else:
            candidates = []
            for oi in order:
                sim = float(row[oi])
                if sim < body.threshold:
                    break
                pid = persons_order[seg_persons[oi]]
                p = persons.get(pid, {})
                candidates.append({
                    "person_id":   pid,
                    "person_name": p.get("name"),
                    "n_total":     p.get("n_total", 0),
                    "similarity":  round(sim, 4),
                })
            cluster_entry["candidates"] = candidates
            ambiguous.append(cluster_entry)

    groups = []
    for pid, cl_list in groups_by_person.items():
        p = persons.get(pid, {})
        n_faces = sum(c["n_faces"] for c in cl_list)
        avg_sim = sum(c["similarity"] for c in cl_list) / len(cl_list)
        groups.append({
            "person_id":      pid,
            "person_name":    p.get("name"),
            "person_n_total": p.get("n_total", 0),
            "n_clusters":     len(cl_list),
            "n_faces":        n_faces,
            "avg_similarity": round(avg_sim, 4),
            "clusters":       sorted(cl_list, key=lambda c: -c["n_faces"]),
        })
    groups.sort(key=lambda g: -g["n_faces"])

    # Enrich groups with avatar from Neo4j (one query)
    if groups:
        try:
            async with get_session() as session:
                res = await session.run(
                    "MATCH (p:Person) WHERE p.id IN $ids RETURN p.id AS id, p.avatar AS avatar, p.known_as AS known_as",
                    ids=[g["person_id"] for g in groups],
                )
                meta_rows = {row["id"]: row for row in await res.data()}
            for g in groups:
                m = meta_rows.get(g["person_id"])
                if m:
                    g["person_avatar"]   = m.get("avatar")
                    g["person_known_as"] = m.get("known_as") if m.get("known_as") != g["person_name"] else None
        except Exception as e:
            log.warning(f"grouped_suggestions avatar lookup failed: {e}")

    # Unknown-person candidates: pairwise sim on UNMATCHED cluster centroids → transitive closure
    unknown_candidates: list[dict] = []
    grouped_unmatched = set()  # cluster_summary indices absorbed into an unknown candidate
    n_unmatched = len(unmatched_indices)

    # The pairwise step builds an N×N matrix. At min_cluster_size=1 the unmatched
    # set is dominated by singletons (one stranger face each) — 140k of them →
    # 140k² float32 ≈ 73 GiB and an instant OOM. A single face can't be a
    # "recurring unknown" anyway, so only clusters with ≥2 faces are eligible;
    # MAX_UNKNOWN_PAIRWISE is a hard backstop (keep the largest, most-likely-real
    # clusters) so N² stays bounded no matter the input.
    MAX_UNKNOWN_PAIRWISE = 6000
    pair_pool = [i for i in unmatched_indices if len(cluster_summary[i]["remaining"]) >= 2]
    if len(pair_pool) > MAX_UNKNOWN_PAIRWISE:
        pair_pool.sort(key=lambda i: -len(cluster_summary[i]["remaining"]))
        pair_pool = pair_pool[:MAX_UNKNOWN_PAIRWISE]
        log.warning(f"grouped_suggestions: capped unknown-pairwise pool to {MAX_UNKNOWN_PAIRWISE} (of {n_unmatched} unmatched)")
    n_pool = len(pair_pool)

    if n_pool >= body.min_unknown_clusters:
        sub = cluster_centroids[np.asarray(pair_pool)]
        pair_sims = sub @ sub.T                          # symmetric, diag=1
        # Union-find over pairs above threshold (i < j to avoid double-edges + diagonal)
        parent = list(range(n_pool))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        iu, ju = np.where(np.triu(pair_sims >= body.inter_threshold, k=1))
        for a, b in zip(iu, ju):
            union(int(a), int(b))
        # Bucket by root
        groups_local: dict[int, list[int]] = {}
        for i in range(n_pool):
            groups_local.setdefault(find(i), []).append(i)
        for members in groups_local.values():
            if len(members) < body.min_unknown_clusters:
                continue
            # Map back to cluster_summary indices
            cs_indices = [pair_pool[m] for m in members]
            cluster_entries = []
            total_faces = 0
            for cs_ix in cs_indices:
                cs = cluster_summary[cs_ix]
                cluster_entries.append({
                    "cluster_id": cs["cluster_id"],
                    "n_faces":    len(cs["remaining"]),
                    "samples":    [
                        {
                            "crop_url":   _crop_url(f["crop_path"]),
                            "photo_path": (f.get("photo_path", "") or "").replace("/photos/", "", 1)
                                          if (f.get("photo_path", "") or "").startswith("/photos/")
                                          else (f.get("photo_path", "") or ""),
                        }
                        for f in cs["remaining"][:SAMPLE_CROPS] if f.get("crop_path")
                    ],
                })
                total_faces += len(cs["remaining"])
            if total_faces < body.min_unknown_faces:
                continue
            # Avg pairwise sim within the group (upper triangle only)
            m_idx = np.asarray(members)
            sub_pairs = pair_sims[np.ix_(m_idx, m_idx)]
            triu = sub_pairs[np.triu_indices_from(sub_pairs, k=1)]
            avg_inter = float(triu.mean()) if triu.size else 1.0

            # "Maybe X?" hints — score the COMBINED candidate centroid against brain.
            # Take mean of member-cluster per_person_sims as approximation. Surface
            # top-3 known people whose mean-sim across all member clusters >= maybe_threshold.
            maybe_candidates = []
            cs_ix_arr = np.asarray(cs_indices)
            mean_per_person = per_person_sims[cs_ix_arr].mean(axis=0)
            maybe_order = np.argsort(-mean_per_person)[:3]
            for oi in maybe_order:
                msim = float(mean_per_person[oi])
                if msim < body.maybe_threshold:
                    break
                pid = persons_order[seg_persons[oi]]
                p = persons.get(pid, {})
                maybe_candidates.append({
                    "person_id":   pid,
                    "person_name": p.get("name"),
                    "n_total":     p.get("n_total", 0),
                    "similarity":  round(msim, 4),
                })

            unknown_candidates.append({
                # Deterministic id from sorted cluster_ids so the same candidate is stable across requests
                "candidate_id":      "uc:" + ",".join(sorted(c["cluster_id"] for c in cluster_entries)),
                "n_clusters":        len(cluster_entries),
                "n_faces":           total_faces,
                "avg_inter_similarity": round(avg_inter, 4),
                "clusters":          sorted(cluster_entries, key=lambda c: -c["n_faces"]),
                "maybe_candidates":  maybe_candidates,
            })
            for cs_ix in cs_indices:
                grouped_unmatched.add(cs_ix)
        # Sort: candidates with "Maybe X?" hints first (more actionable), then by face count desc
        unknown_candidates.sort(key=lambda c: (0 if c.get("maybe_candidates") else 1, -c["n_faces"]))

    unmatched_count = n_unmatched - len(grouped_unmatched)

    return {
        "groups":             groups[:body.limit],
        "ambiguous":          ambiguous[:body.limit],
        "unknown_candidates": unknown_candidates[:body.limit],
        "unmatched_count":    unmatched_count,
    }


class LeftoverBody(BaseModel):
    threshold: float = 0.75
    inter_threshold: float = 0.80
    min_unknown_clusters: int = 2
    min_unknown_faces: int = 6
    min_cluster_size: int = 3
    maybe_threshold: float = 0.55     # surface best-guess person even below main threshold
    offset: int = 0
    limit: int = 50


@router.post("/unassigned/leftover")
async def leftover_unassigned(body: LeftoverBody):
    """Paginated brain-unmatched clusters — the long tail that didn't make it into
    `groups`, `ambiguous`, or `unknown_candidates` on `/unassigned/grouped`. UI
    renders these so every cluster is reachable from `/suggestions` (no need to
    bounce out to `/unassigned`)."""
    import asyncio, numpy as np

    matrix, meta, lookup = await asyncio.to_thread(_load_embedding_index)
    if matrix is None:
        raise HTTPException(503, "Embedding index not built yet")
    brain = await asyncio.to_thread(_load_brain)
    if brain is None:
        raise HTTPException(503, "Brain not built — run scripts/build_brain.py")
    (centroids, bucket_person_ix, persons_order, persons, bucket_meta,
     seg_starts, seg_persons, sort_idx) = brain

    clusters_data = _load(CLUSTERS_FILE, {})
    skipped = set(str(s) for s in _load(SKIPPED_FILE, []))
    done = await _get_neo4j_assigned()
    done = set(done) | await _get_neo4j_skipped()

    cluster_summary: list[dict] = []
    for cid, faces in clusters_data.items():
        if cid == "-1" or cid in skipped:
            continue
        remaining = []
        emb_indices: list[int] = []
        for f in faces:
            pp = f.get("photo_path", "")
            pp = pp.replace("/photos/", "", 1) if pp.startswith("/photos/") else pp
            fi = f.get("face_index")
            if pp and fi is not None and (pp, int(fi)) not in done:
                remaining.append(f)
                idx = lookup.get((pp, int(fi)))
                if idx is not None:
                    emb_indices.append(idx)
        if len(remaining) < body.min_cluster_size or not emb_indices:
            continue
        cluster_summary.append({
            "cluster_id":  str(cid),
            "remaining":   remaining,
            "emb_indices": emb_indices,
        })
    if not cluster_summary:
        return {"leftover": [], "total": 0, "offset": body.offset, "limit": body.limit}

    n_clusters = len(cluster_summary)
    cluster_centroids = np.zeros((n_clusters, matrix.shape[1]), dtype=np.float32)
    for i, cs in enumerate(cluster_summary):
        c = matrix[cs["emb_indices"]].mean(axis=0)
        norm = np.linalg.norm(c)
        if norm > 0:
            c = c / norm
        cluster_centroids[i] = c

    sims = cluster_centroids @ centroids.T
    sorted_sims = sims[:, sort_idx]
    per_person_sims = np.maximum.reduceat(sorted_sims, seg_starts, axis=1)

    unmatched_indices = [ci for ci in range(n_clusters)
                         if float(per_person_sims[ci].max()) < body.threshold]

    # Strip clusters that would join an unknown-person candidate (those are surfaced separately)
    grouped_unmatched = set()
    n_unmatched = len(unmatched_indices)
    if n_unmatched >= body.min_unknown_clusters:
        sub = cluster_centroids[np.asarray(unmatched_indices)]
        pair_sims = sub @ sub.T
        parent = list(range(n_unmatched))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
        iu, ju = np.where(np.triu(pair_sims >= body.inter_threshold, k=1))
        for a, b in zip(iu, ju):
            union(int(a), int(b))
        groups_local: dict[int, list[int]] = {}
        for i in range(n_unmatched):
            groups_local.setdefault(find(i), []).append(i)
        for members in groups_local.values():
            if len(members) < body.min_unknown_clusters:
                continue
            total = sum(len(cluster_summary[unmatched_indices[m]]["remaining"]) for m in members)
            if total < body.min_unknown_faces:
                continue
            for m in members:
                grouped_unmatched.add(unmatched_indices[m])

    leftover_indices = [ci for ci in unmatched_indices if ci not in grouped_unmatched]
    # Sort: clusters where brain has any "best guess" (>= maybe_threshold) come first,
    # then by face count desc. Surfaces the actionable ones at the top of the long tail.
    leftover_indices.sort(key=lambda ci: (
        0 if float(per_person_sims[ci].max()) >= body.maybe_threshold else 1,
        -len(cluster_summary[ci]["remaining"]),
    ))

    total = len(leftover_indices)
    page = leftover_indices[body.offset : body.offset + body.limit]

    leftover = []
    for ci in page:
        cs = cluster_summary[ci]
        row = per_person_sims[ci]
        top_ix = int(np.argmax(row))
        top_sim = float(row[top_ix])
        best_candidate = None
        if top_sim >= body.maybe_threshold:
            pid = persons_order[seg_persons[top_ix]]
            p = persons.get(pid, {})
            best_candidate = {
                "person_id":   pid,
                "person_name": p.get("name"),
                "n_total":     p.get("n_total", 0),
                "similarity":  round(top_sim, 4),
            }
        leftover.append({
            "cluster_id": cs["cluster_id"],
            "n_faces":    len(cs["remaining"]),
            "samples":    [
                {
                    "crop_url":   _crop_url(f["crop_path"]),
                    "photo_path": (f.get("photo_path", "") or "").replace("/photos/", "", 1)
                                  if (f.get("photo_path", "") or "").startswith("/photos/")
                                  else (f.get("photo_path", "") or ""),
                }
                for f in cs["remaining"][:SAMPLE_CROPS] if f.get("crop_path")
            ],
            "best_candidate": best_candidate,
        })

    # Enrich best_candidate entries with known_as so UI can render "Full Name (Nickname)"
    bc_ids = {c["best_candidate"]["person_id"] for c in leftover if c.get("best_candidate")}
    if bc_ids:
        try:
            async with get_session() as session:
                res = await session.run(
                    "MATCH (p:Person) WHERE p.id IN $ids "
                    "RETURN p.id AS id, p.known_as AS known_as",
                    ids=list(bc_ids),
                )
                kn_map = {row["id"]: row["known_as"] for row in await res.data()}
            for c in leftover:
                bc = c.get("best_candidate")
                if bc:
                    known_as = kn_map.get(bc["person_id"])
                    bc["person_known_as"] = known_as if known_as and known_as != bc.get("person_name") else None
        except Exception as e:
            log.warning(f"leftover known_as enrichment failed: {e}")

    return {
        "leftover": leftover,
        "total":    total,
        "offset":   body.offset,
        "limit":    body.limit,
    }


class RejectBody(BaseModel):
    person_id: str
    faces: list[dict]   # each: {photo_path, face_index}


@router.post("/reject")
async def reject_faces(body: RejectBody, request: Request):
    """Tell the brain that none of these faces are this person. Appends to the
    person's `negative_examples` list so they never get suggested again.

    Used by cascade mode ("No, that's not Patty") and group-level rejection.
    Idempotent — re-rejecting an already-negative face is a no-op.
    """
    from app.services.brain import append_negatives

    items: list[tuple[str, int]] = []
    for f in body.faces:
        pp = f.get("photo_path", "")
        fi = f.get("face_index")
        if pp and fi is not None:
            items.append((pp, int(fi)))
    if not items:
        return {"added": 0, "requested": 0}

    added = append_negatives(body.person_id, items)
    if added:
        _invalidate_brain()

    person_name = await _get_person_name(body.person_id)
    log.bind(
        event="face.rejected",
        person_id=body.person_id,
        person_name=person_name,
        count=added,
        requested=len(items),
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("faces rejected for person (added to negative_examples)")

    return {"added": added, "requested": len(items)}


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


class BulkAssignBody(BaseModel):
    person_id: str
    cluster_ids: list[str]
    # Optional faces to exclude across the whole batch (permanent skip).
    exclude: list[list] | None = None


@router.post("/clusters/assign-bulk", status_code=200)
async def assign_clusters_bulk(body: BulkAssignBody, request: Request):
    """Assign many clusters to one person in a SINGLE Neo4j sync (one batched
    write + one brain rebuild). Confirming a min_cluster_size=1 group can span
    hundreds of singleton clusters — doing one sync per cluster stampedes the
    transaction-memory pool and rebuilds the same brain hundreds of times. This
    collapses it to one."""
    clusters = _load(CLUSTERS_FILE, {})
    exclude_set = {(row[0], row[1]) for row in (body.exclude or []) if len(row) >= 2}

    # Honor the same "done" set the suggestions UI uses to compute each cluster's
    # remaining count: faces already assigned (to anyone) or skipped. Without this
    # the endpoint re-merges the WHOLE cluster — the card shows "10 remaining" but
    # we'd write all 291, re-touching done faces and, worse, re-assigning ones the
    # user had deliberately skipped. Match grouped_suggestions' path-stripping.
    done = set(await _get_neo4j_assigned()) | set(await _get_neo4j_skipped())

    faces: list = []
    missing: list[str] = []
    skipped_done = 0
    for cid in body.cluster_ids:
        cl = clusters.get(cid)
        if cl is None:
            missing.append(cid)
            continue
        for f in cl:
            if (f.get("photo_path"), f.get("face_index")) in exclude_set:
                continue
            pp = f.get("photo_path", "") or ""
            pp = pp.replace("/photos/", "", 1) if pp.startswith("/photos/") else pp
            fi = f.get("face_index")
            if not pp or fi is None or (pp, int(fi)) in done:
                skipped_done += 1
                continue
            faces.append(f)

    if exclude_set:
        await _add_skipped_faces([(p, fi) for p, fi in exclude_set if p])

    # Bust the Neo4j-assigned cache so the next list call sees the new edges.
    global _neo4j_assigned
    _neo4j_assigned = None

    # One background sync for the whole group → one batched write, one rebuild.
    threading.Thread(
        target=_sync_to_neo4j,
        args=(body.person_id, faces),
        daemon=True,
    ).start()

    person_name = await _get_person_name(body.person_id)
    log.bind(
        event="cluster.assigned.bulk",
        person_id=body.person_id,
        person_name=person_name,
        cluster_count=len(body.cluster_ids) - len(missing),
        count=len(faces),
        excluded=len(exclude_set),
        already_done=skipped_done,
        missing_clusters=len(missing),
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("clusters bulk-assigned to person")

    return {"assigned": len(faces), "clusters": len(body.cluster_ids) - len(missing), "missing": missing}


@router.post("/clusters/{cluster_id}/assign", status_code=204)
async def assign_cluster(cluster_id: str, body: AssignBody, request: Request):
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

    person_name = await _get_person_name(body.person_id)
    log.bind(
        event="cluster.assigned",
        cluster_id=cluster_id,
        person_id=body.person_id,
        person_name=person_name,
        count=len(faces),
        excluded=len(body.exclude or []),
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("cluster assigned to person")


@router.post("/clusters/{cluster_id}/skip", status_code=204)
async def skip_cluster(cluster_id: str, request: Request):
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

    log.bind(
        event="cluster.skipped",
        cluster_id=cluster_id,
        count=len(items),
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("cluster skipped")


@router.post("/clusters/{cluster_id}/unskip", status_code=204)
async def unskip_cluster(cluster_id: str, request: Request):
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

    log.bind(
        event="cluster.unskipped",
        cluster_id=cluster_id,
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    ).info("cluster unskipped")


# Serialize Neo4j sync threads. Each assign spawns a background sync that
# opens its own transactions and rebuilds the person's brain; firing hundreds
# at once (e.g. confirming a big min_cluster_size=1 group) stacks concurrent
# transactions past dbms.memory.transaction.total.max (~358 MiB on the small
# staging heap) → MemoryPoolOutOfMemoryError. One at a time keeps each sync
# comfortably under the pool. Prefer the bulk endpoint so a whole group is one
# sync, not one-per-cluster.
_NEO4J_SYNC_LOCK = threading.Lock()


def _sync_to_neo4j(person_id: str, faces: list):
    from neo4j import GraphDatabase
    from app.config import settings as s
    from app.services.brain import rebuild_person_brain_sync

    try:
      with _NEO4J_SYNC_LOCK:
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
            # Pre-check Media existence so we can loudly warn about ghost paths.
            # Silent MERGE-skips on missing Media nodes is what tricked us into
            # thinking the brain was broken when really the cluster index was
            # pointing at deleted files.
            distinct_paths = list({b["photo_path"] for b in batch if b["photo_path"]})
            present_paths: set[str] = set()
            for i in range(0, len(distinct_paths), 1000):
                res = session.run(
                    "MATCH (m:Media) WHERE m.path IN $paths RETURN m.path AS p",
                    paths=distinct_paths[i:i+1000],
                )
                for row in res:
                    present_paths.add(row["p"])
            ghost_batch = [b for b in batch if b["photo_path"] not in present_paths]
            if ghost_batch:
                log.bind(
                    event="face.assign.ghost_paths",
                    person_id=person_id,
                    ghost_count=len(ghost_batch),
                    total_requested=len(batch),
                    sample=[b["photo_path"] for b in ghost_batch[:5]],
                ).warning(
                    f"{len(ghost_batch)} of {len(batch)} cluster faces point at Media nodes "
                    f"that don't exist in Neo4j — skipping. Run "
                    f"scripts/purge_ghost_cluster_faces.py to clean the cluster index."
                )
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
            # Rebuild brain in the same session so the file reflects the new edges.
            try:
                brain = rebuild_person_brain_sync(person_id, session)
                _invalidate_brain()
                log.info(
                    f"brain rebuilt for {person_id}: "
                    f"n_total={brain['n_confirmed_total'] if brain else 'below-floor'}"
                )
            except Exception as e:
                log.warning(f"brain rebuild failed for {person_id}: {e}")
        driver.close()
        log.info(f"Synced {len(faces)} faces for person {person_id}")
    except Exception as e:
        log.error(f"Neo4j sync failed for person {person_id}: {e}")
