"""Compute 3×3 sub-region mean colors per Media thumbnail and store as
`m.subcolors` (list of 9 lowercase hex strings, raster order TL→BR).

Used by the mosaic builder's heavy color-matching mode: each photo
contributes 9 color anchors to the candidate pool instead of one. KD-tree
queries can then find sub-region matches the whole-image mean would miss,
and the matching sub-rect gets cropped as the tile.
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


def sub_means(arr: np.ndarray) -> list[str]:
    """Split an HxWx3 RGB array into a 3×3 grid, return mean color of each
    sub-region as a lowercase hex string, raster-order (TL, TM, TR, ML,
    MM, MR, BL, BM, BR).
    """
    h, w, _ = arr.shape
    # row / col split boundaries
    ys = [0, h // 3, 2 * h // 3, h]
    xs = [0, w // 3, 2 * w // 3, w]
    out: list[str] = []
    for r in range(3):
        for c in range(3):
            patch = arr[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            mean = patch.reshape(-1, 3).mean(axis=0)
            r8, g8, b8 = int(mean[0]), int(mean[1]), int(mean[2])
            out.append(f"#{r8:02x}{g8:02x}{b8:02x}")
    return out


async def main():
    drv = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )

    t0 = time.perf_counter()
    log("opening Media stream from Neo4j...")

    pending: list[dict] = []
    no_thumb = 0
    err = 0
    seen = 0
    last_log = time.perf_counter()

    async def flush(session, force=False):
        nonlocal pending
        if not pending:
            return
        if not force and len(pending) < WRITE_BATCH:
            return
        batch = pending
        pending = []
        await session.run(
            """
            UNWIND $rows AS row
            MATCH (m:Media {path: row.path})
            SET m.subcolors = row.cs
            """,
            rows=batch,
        )

    async with drv.session() as read_s, drv.session() as write_s:
        res = await read_s.run("MATCH (m:Media) RETURN m.path AS path")
        async for r in res:
            seen += 1
            path = r["path"]
            tp = thumb_path(path)
            if not tp.exists():
                no_thumb += 1
            else:
                try:
                    img = Image.open(tp).convert("RGB")
                    arr = np.asarray(img, dtype=np.float32)
                    cs = sub_means(arr)
                    pending.append({"path": path, "cs": cs})
                except Exception:
                    err += 1

            if len(pending) >= WRITE_BATCH:
                await flush(write_s)

            if seen % PROGRESS_EVERY == 0:
                now = time.perf_counter()
                rate = PROGRESS_EVERY / max(1e-9, now - last_log)
                last_log = now
                log(
                    f"  seen {seen:>7,}  "
                    f"writes_queued {len(pending):>4}  "
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
