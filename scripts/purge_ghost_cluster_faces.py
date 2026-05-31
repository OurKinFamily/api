"""purge_ghost_cluster_faces.py — drop faces from clusters.json whose
photo_path has no Media node in Neo4j.

Background: face extraction can leave behind "ghost" entries in
clusters.json + embeddings.npy when the source photo gets deleted or moved
out of the archive. The face_index still has an embedding, the cluster still
references it, but the photo's Media node doesn't exist in Neo4j — so
APPEARS_IN edges can't be written. /unassigned/grouped keeps re-surfacing
those clusters because the corresponding photo never gets "done."

This script:
  1. Reads /photos/__faces/clusters/clusters.json
  2. Collects every distinct photo_path referenced by any cluster
  3. Queries Neo4j in batches for Media nodes with those paths
  4. Drops faces from each cluster whose photo_path isn't in Neo4j
  5. Drops clusters that end up empty
  6. Backs up the original to clusters.json.bak (overwrites prior bak)
  7. Writes the cleaned clusters.json

Embeddings.npy + faces_meta.json are NOT touched — they're indexed by row
position, so removing rows would invalidate every downstream lookup. The
ghost rows stay in those files but become unreferenced.

Usage:
    python scripts/purge_ghost_cluster_faces.py            # write changes
    python scripts/purge_ghost_cluster_faces.py --dry-run  # report only
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import GraphDatabase
from app.config import settings

CLUSTERS_FILE = settings.photos_root / "__faces" / "clusters" / "clusters.json"
BACKUP_FILE   = CLUSTERS_FILE.with_suffix(CLUSTERS_FILE.suffix + ".bak")
BATCH = 5000


def strip_root(p: str) -> str:
    return p.replace("/photos/", "", 1) if p.startswith("/photos/") else p


def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Drop ghost cluster faces (photo_path has no Media node)")
    parser.add_argument("--dry-run", action="store_true", help="report only, don't write")
    args = parser.parse_args()

    if not CLUSTERS_FILE.exists():
        log(f"missing {CLUSTERS_FILE}")
        return 1

    log(f"loading {CLUSTERS_FILE}...")
    clusters = json.loads(CLUSTERS_FILE.read_text())
    n_clusters_in = sum(1 for k in clusters if k != "-1")
    n_faces_in = sum(len(v) for k, v in clusters.items() if k != "-1")
    log(f"  {n_clusters_in:,} clusters, {n_faces_in:,} faces (excluding noise -1)")

    paths = set()
    for cid, faces in clusters.items():
        if cid == "-1":
            continue
        for f in faces:
            pp = strip_root(f.get("photo_path", ""))
            if pp:
                paths.add(pp)
    log(f"  {len(paths):,} distinct photo paths")

    log("checking Media nodes in Neo4j...")
    drv = GraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    present = set()
    plist = list(paths)
    try:
        with drv.session() as session:
            for i in range(0, len(plist), BATCH):
                res = session.run(
                    "MATCH (m:Media) WHERE m.path IN $paths RETURN m.path AS p",
                    paths=plist[i:i + BATCH],
                )
                for r in res:
                    present.add(r["p"])
    finally:
        drv.close()

    missing = paths - present
    log(f"  Media nodes present : {len(present):,}")
    log(f"  Media nodes missing : {len(missing):,}")

    if not missing:
        log("nothing to purge.")
        return 0

    by_dir = defaultdict(int)
    for p in missing:
        parts = p.split("/")
        if len(parts) >= 3 and parts[0] == "archive":
            by_dir[f"{parts[1]}/{parts[2]}"] += 1
        else:
            by_dir["other"] += 1
    log("  missing distribution (top 10):")
    for k, v in sorted(by_dir.items(), key=lambda x: -x[1])[:10]:
        log(f"    {k:12} {v:>6,}")

    cleaned: dict[str, list] = {}
    dropped_faces = 0
    dropped_clusters = 0
    shrunk_clusters = 0
    for cid, faces in clusters.items():
        if cid == "-1":
            cleaned[cid] = faces
            continue
        kept = []
        for f in faces:
            pp = strip_root(f.get("photo_path", ""))
            if pp and pp not in missing:
                kept.append(f)
            else:
                dropped_faces += 1
        if not kept:
            dropped_clusters += 1
            continue
        if len(kept) != len(faces):
            shrunk_clusters += 1
        cleaned[cid] = kept

    n_clusters_out = sum(1 for k in cleaned if k != "-1")
    n_faces_out = sum(len(v) for k, v in cleaned.items() if k != "-1")
    log("")
    log(f"  faces dropped       : {dropped_faces:,}")
    log(f"  clusters removed    : {dropped_clusters:,}")
    log(f"  clusters shrunk     : {shrunk_clusters:,}")
    log(f"  clusters out        : {n_clusters_out:,}")
    log(f"  faces out           : {n_faces_out:,}")

    if args.dry_run:
        log("--dry-run: not writing.")
        return 0

    log(f"backing up to {BACKUP_FILE}")
    BACKUP_FILE.write_text(CLUSTERS_FILE.read_text())
    log(f"writing cleaned {CLUSTERS_FILE}")
    CLUSTERS_FILE.write_text(json.dumps(cleaned, indent=2))
    log("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
