"""Walk every Media node's sidecar JSON and copy salientColor →
Neo4j Media.salient_color. One-off backfill. Idempotent.
"""
import asyncio
import json
import os
import sys
from pathlib import Path

# Make app importable when running as a standalone script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import AsyncGraphDatabase
from app.config import settings

BATCH = 1000


async def main():
    drv = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    print("fetching Media paths...")
    async with drv.session() as s:
        res = await s.run("MATCH (m:Media) WHERE m.dominant_color IS NOT NULL RETURN m.path AS path")
        paths = [r["path"] async for r in res]
    print(f"  {len(paths):,} media nodes")

    print("reading sidecars...")
    updates = []
    misses = 0
    parse_err = 0
    no_color = 0
    for i, path in enumerate(paths):
        if i and i % 10000 == 0:
            print(f"  scanned {i:,} / {len(paths):,} — kept {len(updates):,}")
        sc = settings.photos_root / (path + ".json")
        if not sc.exists():
            misses += 1
            continue
        try:
            data = json.loads(sc.read_text())
            media = (data.get("results") or [{}])[0].get("metadata", {}).get("media", {})
            sal = media.get("salientColor")
        except Exception:
            parse_err += 1
            continue
        if not (isinstance(sal, str) and sal.startswith("#") and len(sal) == 7):
            no_color += 1
            continue
        updates.append({"path": path, "sal": sal.lower()})

    print(
        f"  scanned {len(paths):,} — kept {len(updates):,} "
        f"(missing sidecars: {misses:,}, parse errors: {parse_err:,}, no salientColor: {no_color:,})"
    )

    print("writing to Neo4j...")
    async with drv.session() as s:
        for i in range(0, len(updates), BATCH):
            batch = updates[i:i + BATCH]
            await s.run(
                """
                UNWIND $rows AS row
                MATCH (m:Media {path: row.path})
                SET m.salient_color = row.sal
                """,
                rows=batch,
            )
            print(f"  wrote {i + len(batch):,} / {len(updates):,}")

    await drv.close()
    print("done.")


if __name__ == "__main__":
    asyncio.run(main())
