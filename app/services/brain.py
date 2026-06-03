"""Per-person face brain — multi-centroid identity model.

Write-side companion to the read-side cache in `app.routers.faces`. Use this
module to rebuild one person's brain after their APPEARS_IN edges change, or
to append/clear negative_examples after a user rejects a suggestion.

Brain files live at /photos/__faces/brain/<uuid>.json. They are disposable —
Neo4j APPEARS_IN edges are the source of truth. `scripts/build_brain.py`
rebuilds every brain from scratch by walking those edges.
"""
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from app.config import settings

COLD_START_MIN = 1
BUCKET_YEARS = 5
APPEARANCE_K_MAX = 150        # ceiling on appearance centroids per person
APPEARANCE_PER_CENTROID = 50  # ~one centroid per this many clean faces
APPEARANCE_MIN_FACES = 6      # skip appearance pass if fewer clean faces
OUTLIER_SIGMA = 2.0           # drop confirmed faces > N sigma from global mean
BRAIN_DIR = settings.photos_root / "__faces" / "brain"


def kmeans_centroids(X: np.ndarray, K: int, n_iter: int = 25) -> np.ndarray:
    """Lloyd K-means on L2-normalised embeddings. Returns K normalised centroids."""
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), K, replace=False)
    centroids = X[idx].copy()
    for _ in range(n_iter):
        labels = (X @ centroids.T).argmax(axis=1)
        new = np.zeros_like(centroids)
        for k in range(K):
            m = labels == k
            if m.any():
                c = X[m].mean(axis=0)
                n = np.linalg.norm(c)
                new[k] = c / n if n > 0 else centroids[k]
            else:
                new[k] = centroids[k]
        if np.allclose(centroids, new, atol=1e-4):
            break
        centroids = new
    return centroids


def _strip_root(p: str) -> str:
    return p.replace("/photos/", "", 1) if p.startswith("/photos/") else p


def _parse_year(ts) -> int | None:
    if ts is None:
        return None
    s = str(ts)
    if s.startswith("1700"):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).year
    except ValueError:
        pass
    try:
        return int(s[:4])
    except (ValueError, TypeError):
        return None


def _parse_dob_year(dob) -> int | None:
    if not dob:
        return None
    try:
        return int(str(dob)[:4])
    except (ValueError, TypeError):
        return None


def bucket_key(dob_year: int | None, photo_year: int | None) -> str:
    if dob_year is not None and photo_year is not None:
        age = photo_year - dob_year
        if 0 <= age <= 120:
            lo = (age // BUCKET_YEARS) * BUCKET_YEARS
            return f"age:{lo}-{lo + BUCKET_YEARS - 1}"
    if photo_year is not None:
        dec = (photo_year // 10) * 10
        return f"decade:{dec}s"
    return "undated"


def _brain_path(person_id: str) -> Path:
    return BRAIN_DIR / f"{person_id}.json"


def read_negatives(person_id: str) -> list:
    p = _brain_path(person_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("negative_examples", [])
    except (json.JSONDecodeError, OSError):
        return []


def build_brain_dict(
    person_id: str,
    person_name: str | None,
    dob,
    rows: list[dict],
    matrix: np.ndarray,
    lookup: dict[tuple[str, int], int],
    prior_negatives: list | None = None,
) -> dict | None:
    """Compute the brain JSON for one person. rows are Neo4j-shaped dicts with
    keys: pp (photo path), fi (face_index), ts (photo timestamp). Returns
    None if the person has fewer than COLD_START_MIN embedded faces.
    """
    dob_year = _parse_dob_year(dob)
    bucket_rows: dict[str, list[int]] = defaultdict(list)
    missing = 0
    for r in rows:
        pp = _strip_root(r["pp"])
        fi = int(r["fi"])
        idx = lookup.get((pp, fi))
        if idx is None:
            missing += 1
            continue
        bucket_rows[bucket_key(dob_year, _parse_year(r.get("ts")))].append(idx)

    n_total = sum(len(v) for v in bucket_rows.values())
    if n_total < COLD_START_MIN:
        return None

    # Outlier removal: drop faces > OUTLIER_SIGMA std below the global mean
    # similarity (blurry / side-angle / misidentified crops dilute centroids).
    all_indices_raw = [i for v in bucket_rows.values() for i in v]
    if len(all_indices_raw) >= 4:
        all_embs = matrix[all_indices_raw].astype(np.float32)
        g = all_embs.mean(axis=0)
        g /= (np.linalg.norm(g) + 1e-8)
        sims_g = all_embs @ g
        cutoff = sims_g.mean() - OUTLIER_SIGMA * sims_g.std()
        keep = {all_indices_raw[i] for i, s in enumerate(sims_g) if s >= cutoff}
        bucket_rows = {k: [i for i in v if i in keep] for k, v in bucket_rows.items()}
        bucket_rows = {k: v for k, v in bucket_rows.items() if v}
    n_outliers = n_total - sum(len(v) for v in bucket_rows.values())

    # Time-bucketed centroids
    buckets = []
    for key, indices in sorted(bucket_rows.items()):
        sub = matrix[indices].astype(np.float32)
        centroid = sub.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        radius = float(1.0 - (sub @ centroid).mean()) if len(indices) >= 2 else 0.0
        buckets.append({
            "key":      key,
            "n":        len(indices),
            "radius":   radius,
            "centroid": centroid.tolist(),
        })

    # Appearance centroids: K-means over all cleaned faces, capturing visual
    # modes that cross time boundaries. Added as "appearance:N" bucket keys;
    # the read-side picks them up automatically.
    all_clean = [i for v in bucket_rows.values() for i in v]
    if len(all_clean) >= APPEARANCE_MIN_FACES:
        K = min(APPEARANCE_K_MAX, max(2, len(all_clean) // APPEARANCE_PER_CENTROID))
        X = matrix[all_clean].astype(np.float32)
        app_centroids = kmeans_centroids(X, K)
        labels = (X @ app_centroids.T).argmax(axis=1)
        for k, c in enumerate(app_centroids):
            n_k = int((labels == k).sum())
            if n_k == 0:
                continue
            radius_k = float(1.0 - (X[labels == k] @ c).mean()) if n_k >= 2 else 0.0
            buckets.append({
                "key":      f"appearance:{k}",
                "n":        n_k,
                "radius":   radius_k,
                "centroid": c.tolist(),
            })

    n_app = sum(1 for b in buckets if b["key"].startswith("appearance:"))
    return {
        "person_id":         person_id,
        "person_name":       person_name,
        "dob_year":          dob_year,
        "updated_at":        datetime.utcnow().isoformat() + "Z",
        "n_confirmed_total": n_total,
        "n_outliers_removed": n_outliers,
        "missing_embedding": missing,
        "buckets":           buckets,
        "n_appearance_centroids": n_app,
        "negative_examples": prior_negatives or [],
    }


def write_brain(brain: dict) -> None:
    BRAIN_DIR.mkdir(parents=True, exist_ok=True)
    _brain_path(brain["person_id"]).write_text(json.dumps(brain, indent=2))


def delete_brain(person_id: str) -> bool:
    p = _brain_path(person_id)
    if p.exists():
        p.unlink()
        return True
    return False


_ROWS_CYPHER = """
MATCH (p:Person {id: $pid})-[r:APPEARS_IN]->(m:Media)
WHERE r.face_index IS NOT NULL
RETURN p.name AS pname, p.birth_date AS dob,
       m.path AS pp, r.face_index AS fi, m.timestamp AS ts
"""


def _build_and_write(person_id: str, rows: list[dict]) -> dict | None:
    if not rows:
        delete_brain(person_id)
        return None
    from app.routers.faces import _load_embedding_index
    matrix, _, lookup = _load_embedding_index()
    if matrix is None:
        return None
    brain = build_brain_dict(
        person_id,
        rows[0].get("pname"),
        rows[0].get("dob"),
        rows,
        matrix,
        lookup,
        read_negatives(person_id),
    )
    if brain is None:
        delete_brain(person_id)
        return None
    write_brain(brain)
    return brain


async def rebuild_person_brain(person_id: str, session) -> dict | None:
    """Re-read this person's APPEARS_IN edges from Neo4j (async session),
    recompute their multi-centroid brain, and write the file. Returns the
    new brain dict, or None if the person dropped below the cold-start floor
    (file is deleted in that case).

    Caller is responsible for calling `app.routers.faces._invalidate_brain()`
    after this returns, so the cluster-scoring endpoint reloads the new file.
    """
    res = await session.run(_ROWS_CYPHER, pid=person_id)
    return _build_and_write(person_id, await res.data())


def rebuild_person_brain_sync(person_id: str, session) -> dict | None:
    """Sync sibling for callers running in a background thread with a
    blocking Neo4j driver (e.g. `_sync_to_neo4j`). Same contract as the async
    version including the cache-invalidation requirement on the caller."""
    res = session.run(_ROWS_CYPHER, pid=person_id)
    rows = [dict(r) for r in res]
    return _build_and_write(person_id, rows)


def append_negatives(person_id: str, items: Iterable[tuple[str, int]]) -> int:
    """Append (photo_path, face_index) pairs to this person's
    negative_examples list. Idempotent — already-present pairs are skipped.
    Returns the number newly added. Returns 0 if no brain file exists (person
    is below the cold-start floor; nothing to remember against).

    Caller must call `_invalidate_brain()` after if any returned > 0.
    """
    p = _brain_path(person_id)
    if not p.exists():
        return 0
    try:
        brain = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return 0
    existing = {(x[0], int(x[1])) for x in brain.get("negative_examples", []) if x}
    added = 0
    negs = brain.setdefault("negative_examples", [])
    for pp, fi in items:
        pp = _strip_root(pp)
        key = (pp, int(fi))
        if key not in existing:
            negs.append([pp, int(fi)])
            existing.add(key)
            added += 1
    if added:
        brain["updated_at"] = datetime.utcnow().isoformat() + "Z"
        p.write_text(json.dumps(brain, indent=2))
    return added


def remove_negatives(person_id: str, items: Iterable[tuple[str, int]]) -> int:
    """Inverse of append_negatives — pull pairs out of the list. Returns the
    count removed. Used when a user reverses a mis-reject."""
    p = _brain_path(person_id)
    if not p.exists():
        return 0
    try:
        brain = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return 0
    drop = {(_strip_root(pp), int(fi)) for pp, fi in items}
    before = brain.get("negative_examples", [])
    after = [x for x in before if x and (x[0], int(x[1])) not in drop]
    removed = len(before) - len(after)
    if removed:
        brain["negative_examples"] = after
        brain["updated_at"] = datetime.utcnow().isoformat() + "Z"
        p.write_text(json.dumps(brain, indent=2))
    return removed
