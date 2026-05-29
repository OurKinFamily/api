"""Compute the most-detailed quadrant of each Media node's thumbnail and
store it as `m.best_crop_quadrant` (0=TL, 1=TR, 2=BL, 3=BR). Used by the
mosaic builder's saliency-crop mode: instead of center-cropping each tile
(which can chop a face in half), crop the quadrant with the highest local
luminance stddev — densest detail, most likely to hold the subject.
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

# Quadrant index encoding:
#   0 = top-left,  1 = top-right
#   2 = bot-left,  3 = bot-right


def thumb_path(media_path: str) -> Path:
    rel = media_path.removeprefix("archive/")
    return THUMBS_ROOT / (rel + ".webp")


def log(msg: str) -> None:
    print(msg, flush=True)


def best_quadrant(arr: np.ndarray) -> int:
    h, w = arr.shape
    my, mx = h // 2, w // 2
    quads = [
        arr[:my, :mx],   # TL
        arr[:my, mx:],   # TR
        arr[my:, :mx],   # BL
        arr[my:, mx:],   # BR
    ]
    stds = [float(q.std()) for q in quads]
    return int(np.argmax(stds))


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
    quad_counts = [0, 0, 0, 0]  # sanity: distribution of chosen quadrant

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
            SET m.best_crop_quadrant = row.q
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
                    img = Image.open(tp).convert("L")
                    arr = np.asarray(img, dtype=np.float32)
                    q = best_quadrant(arr)
                    quad_counts[q] += 1
                    pending.append({"path": path, "q": q})
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
                    f"{rate:>5.0f} /s  "
                    f"TL/TR/BL/BR {quad_counts}"
                )

        await flush(write_s, force=True)

    await drv.close()
    total = sum(quad_counts) or 1
    pcts = [f"{100 * c / total:.1f}%" for c in quad_counts]
    log(
        f"done in {time.perf_counter() - t0:,.1f}s — "
        f"seen {seen:,}, no_thumb {no_thumb:,}, errors {err:,}"
    )
    log(f"  quadrant distribution: TL {pcts[0]}  TR {pcts[1]}  BL {pcts[2]}  BR {pcts[3]}")


if __name__ == "__main__":
    asyncio.run(main())
