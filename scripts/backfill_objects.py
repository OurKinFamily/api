"""Walk every Media node's `.objects.json` sidecar and copy unique
detected class names (where confidence >= 0.40) → Media.objects array.
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import AsyncGraphDatabase
from app.config import settings

BATCH = 1000
CONF_FLOOR = 0.40


async def main():
    drv = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    print("fetching Media paths...")
    async with drv.session() as s:
        res = await s.run("MATCH (m:Media) RETURN m.path AS path")
        paths = [r["path"] async for r in res]
    print(f"  {len(paths):,} media nodes")

    print("reading object sidecars...")
    updates = []
    misses = 0
    parse_err = 0
    no_obj = 0
    for i, path in enumerate(paths):
        if i and i % 10000 == 0:
            print(f"  scanned {i:,} / {len(paths):,} — kept {len(updates):,}")
        sc = settings.photos_root / (path + ".objects.json")
        if not sc.exists():
            misses += 1
            continue
        try:
            data = json.loads(sc.read_text())
            dets = data.get("detections") or []
        except Exception:
            parse_err += 1
            continue
        classes = sorted({
            d["class_name"]
            for d in dets
            if isinstance(d, dict)
            and d.get("class_name")
            and (d.get("confidence") or 0) >= CONF_FLOOR
        })
        if not classes:
            no_obj += 1
            updates.append({"path": path, "objs": []})  # explicit empty to wipe stale
            continue
        updates.append({"path": path, "objs": classes})

    print(
        f"  scanned {len(paths):,} — kept {len(updates):,} "
        f"(missing sidecars: {misses:,}, parse errors: {parse_err:,}, empty: {no_obj:,})"
    )

    print("writing to Neo4j...")
    async with drv.session() as s:
        for i in range(0, len(updates), BATCH):
            batch = updates[i:i + BATCH]
            await s.run(
                """
                UNWIND $rows AS row
                MATCH (m:Media {path: row.path})
                SET m.objects = row.objs
                """,
                rows=batch,
            )
            print(f"  wrote {i + len(batch):,} / {len(updates):,}")

    await drv.close()
    print("done.")


if __name__ == "__main__":
    asyncio.run(main())
