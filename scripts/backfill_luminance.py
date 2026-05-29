"""Walk every Media node's webp thumbnail, compute the luminance standard
deviation (a proxy for internal contrast / "how busy" the photo is), and
write it to `m.luminance_stddev`. Used by the mosaic builder's edge-aware
matching so high-contrast cells in the source pick tiles that themselves
have rich internal detail.
"""
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import AsyncGraphDatabase
from app.config import settings

WRITE_BATCH = 1000
PROGRESS_EVERY = 1000
THUMBS_ROOT = settings.photos_root / "__thumbs"


def thumb_path(media_path: str) -> Path:
    rel = media_path.removeprefix("archive/")
    return THUMBS_ROOT / (rel + ".webp")


def log(msg: str) -> None:
    print(msg, flush=True)


async def main():
    drv = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    t0 = time.perf_counter()
    log("opening Media stream from Neo4j...")

    pending_writes: list[dict] = []
    pending_flush_count = 0
    no_thumb = 0
    err = 0
    seen = 0
    last_log = time.perf_counter()

    async def flush(session, force=False):
        nonlocal pending_writes
        if not pending_writes:
            return
        if not force and len(pending_writes) < WRITE_BATCH:
            return
        batch = pending_writes
        pending_writes = []
        await session.run(
            """
            UNWIND $rows AS row
            MATCH (m:Media {path: row.path})
            SET m.luminance_stddev = row.std
            """,
            rows=batch,
        )

    async with drv.session() as read_s, drv.session() as write_s:
        # Stream paths instead of buffering all 156k in memory before we start
        # — first thumb gets processed immediately, progress logs flow.
        res = await read_s.run("MATCH (m:Media) RETURN m.path AS path")
        async for r in res:
            seen += 1
            path = r["path"]
            tp = thumb_path(path)
            if not tp.exists():
                no_thumb += 1
            else:
                try:
                    img = Image.open(tp).convert("L")
                    arr = np.asarray(img, dtype=np.float32)
                    std = float(arr.std())
                    pending_writes.append({"path": path, "std": std})
                except Exception:
                    err += 1

            if len(pending_writes) >= WRITE_BATCH:
                await flush(write_s)

            if seen % PROGRESS_EVERY == 0:
                now = time.perf_counter()
                rate = PROGRESS_EVERY / max(1e-9, now - last_log)
                last_log = now
                log(
                    f"  seen {seen:>7,}  "
                    f"writes_queued {len(pending_writes):>4}  "
                    f"no_thumb {no_thumb:>5,}  err {err:>4,}  "
                    f"{rate:>5.0f} /s"
                )

        await flush(write_s, force=True)

    await drv.close()
    log(
        f"done in {time.perf_counter() - t0:,.1f}s — "
        f"seen {seen:,}, no_thumb {no_thumb:,}, errors {err:,}"
    )


if __name__ == "__main__":
    asyncio.run(main())
