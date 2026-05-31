"""build_brain.py — Phase 1 of #57 active-learning face identity model.

Walks Neo4j APPEARS_IN edges, computes per-person multi-centroid identity
model, writes /photos/__faces/brain/<uuid>.json.

Buckets:
  - 5y age windows when person.birth_date is known
  - Photo-decade windows otherwise
  - Faces with no usable photo timestamp + no DOB land in "undated"

Cold start: persons with fewer than COLD_START_MIN confirmed (and embedded)
faces are skipped. Brain only wakes up for people we have enough signal on.

Brain is disposable. Neo4j APPEARS_IN edges are source of truth — rerun this
script any time to rebuild from scratch.

Usage:
    python scripts/build_brain.py
    python scripts/build_brain.py --dry-run        # log, don't write
    python scripts/build_brain.py --person-id UUID # only one person
"""
import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import AsyncGraphDatabase
from app.config import settings

COLD_START_MIN = 1
BUCKET_YEARS = 5

PHOTOS_ROOT = settings.photos_root
EMB_NPY = PHOTOS_ROOT / "__faces" / "clusters" / "embeddings.npy"
META_JSON = PHOTOS_ROOT / "__faces" / "clusters" / "faces_meta.json"
BRAIN_DIR = PHOTOS_ROOT / "__faces" / "brain"


def log(msg: str) -> None:
    print(msg, flush=True)


def strip_root(p: str) -> str:
    return p.replace("/photos/", "", 1) if p.startswith("/photos/") else p


def parse_year(ts) -> int | None:
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


def parse_dob_year(dob) -> int | None:
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


def load_embedding_index() -> tuple[np.ndarray, dict[tuple[str, int], int]]:
    if not EMB_NPY.exists() or not META_JSON.exists():
        raise SystemExit(f"missing {EMB_NPY} or {META_JSON} — run face extraction first")
    matrix = np.load(str(EMB_NPY))
    meta = json.loads(META_JSON.read_text())
    lookup: dict[tuple[str, int], int] = {}
    for idx, m in enumerate(meta):
        pp = strip_root(m["photo_path"])
        lookup[(pp, int(m["face_index"]))] = idx
    return matrix, lookup


async def fetch_assignments(person_id: str | None) -> list[dict]:
    drv = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    try:
        async with drv.session() as session:
            cypher = """
                MATCH (p:Person)-[r:APPEARS_IN]->(m:Media)
                WHERE r.face_index IS NOT NULL
                {person_filter}
                RETURN p.id AS pid, p.name AS pname, p.birth_date AS dob,
                       m.path AS pp, r.face_index AS fi, m.timestamp AS ts
            """.format(person_filter="AND p.id = $person_id" if person_id else "")
            params = {"person_id": person_id} if person_id else {}
            result = await session.run(cypher, **params)
            return await result.data()
    finally:
        await drv.close()


def build_brain_for_person(
    pid: str,
    pname: str | None,
    dob: str | None,
    rows: list[dict],
    matrix: np.ndarray,
    lookup: dict[tuple[str, int], int],
) -> dict | None:
    dob_year = parse_dob_year(dob)

    bucket_rows: dict[str, list[int]] = defaultdict(list)
    missing_embedding = 0
    for r in rows:
        pp = strip_root(r["pp"])
        fi = int(r["fi"])
        idx = lookup.get((pp, fi))
        if idx is None:
            missing_embedding += 1
            continue
        bucket_rows[bucket_key(dob_year, parse_year(r.get("ts")))].append(idx)

    n_total = sum(len(v) for v in bucket_rows.values())
    if n_total < COLD_START_MIN:
        return {"_skip": True, "n_total": n_total, "missing_embedding": missing_embedding}

    buckets = []
    for key, indices in sorted(bucket_rows.items()):
        sub = matrix[indices]
        centroid = sub.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        if len(indices) >= 2:
            sims = sub @ centroid
            radius = float(1.0 - sims.mean())
        else:
            radius = 0.0
        buckets.append({
            "key": key,
            "n": len(indices),
            "radius": radius,
            "centroid": centroid.tolist(),
        })

    return {
        "person_id": pid,
        "person_name": pname,
        "dob_year": dob_year,
        "updated_at": datetime.utcnow().isoformat() + "Z",
        "n_confirmed_total": n_total,
        "missing_embedding": missing_embedding,
        "buckets": buckets,
        "negative_examples": [],
    }


def merge_negatives(brain: dict, brain_path: Path) -> dict:
    """Preserve negative_examples from existing file (only mutated by reject API)."""
    if not brain_path.exists():
        return brain
    try:
        prior = json.loads(brain_path.read_text())
        brain["negative_examples"] = prior.get("negative_examples", [])
    except (json.JSONDecodeError, OSError):
        pass
    return brain


async def main() -> int:
    parser = argparse.ArgumentParser(description="Build per-person face brain files")
    parser.add_argument("--dry-run", action="store_true", help="log but don't write files")
    parser.add_argument("--person-id", help="only rebuild for this person UUID")
    args = parser.parse_args()

    t0 = time.perf_counter()
    log(f"loading embeddings + meta from {EMB_NPY.parent}...")
    matrix, lookup = load_embedding_index()
    log(f"  {len(matrix):,} embeddings, {matrix.shape[1]} dims, {len(lookup):,} indexable")

    log("querying Neo4j for APPEARS_IN edges...")
    rows = await fetch_assignments(args.person_id)
    log(f"  {len(rows):,} confirmed face assignments")

    by_person: dict[str, list[dict]] = defaultdict(list)
    person_meta: dict[str, dict] = {}
    for r in rows:
        pid = r["pid"]
        if not pid:
            continue
        by_person[pid].append(r)
        if pid not in person_meta:
            person_meta[pid] = {"pname": r.get("pname"), "dob": r.get("dob")}

    log(f"  {len(by_person):,} distinct people with at least one assignment")

    if not args.dry_run:
        BRAIN_DIR.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_below_floor = 0
    for pid, faces in by_person.items():
        meta = person_meta[pid]
        brain = build_brain_for_person(
            pid, meta["pname"], meta["dob"], faces, matrix, lookup
        )
        if brain is None:
            n_skipped += 1
            continue
        if brain.get("_skip"):
            n_below_floor += 1
            log(
                f"  skip {pid[:8]}.. {meta['pname'] or '?':24} "
                f"only {brain['n_total']} confirmed (floor {COLD_START_MIN})"
                + (f"  [+{brain['missing_embedding']} missing emb]" if brain["missing_embedding"] else "")
            )
            continue

        path = BRAIN_DIR / f"{pid}.json"
        brain = merge_negatives(brain, path)
        log(
            f"  brain {pid[:8]}.. {meta['pname'] or '?':24} "
            f"n={brain['n_confirmed_total']:>5} buckets={len(brain['buckets'])} "
            f"keys={','.join(b['key'] for b in brain['buckets'][:4])}"
            + ("..." if len(brain["buckets"]) > 4 else "")
        )
        if not args.dry_run:
            path.write_text(json.dumps(brain, indent=2))
        n_written += 1

    log("")
    log(f"  written       : {n_written}")
    log(f"  below floor   : {n_below_floor}")
    log(f"  no usable rows: {n_skipped}")
    log(f"  total time    : {time.perf_counter() - t0:.1f}s")
    log(f"  output dir    : {BRAIN_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
