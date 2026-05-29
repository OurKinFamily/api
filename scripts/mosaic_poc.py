"""POC: build a photo-mosaic of a source image using archive thumbs as tiles.

Usage:
    venv/bin/python3 scripts/mosaic_poc.py \\
        --source /path/to/portrait.jpg \\
        --output /tmp/mosaic.jpg \\
        --grid 60x80 \\
        --tile 48 \\
        --color mean

Notes:
    - `--color` is which Media property to match against: dominant, mean, salient.
    - `--max-reuse` caps how many cells one tile can fill (avoids same photo
      everywhere). Default 8.
    - Skips tiles whose thumb isn't on disk.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import AsyncGraphDatabase
from app.config import settings

THUMBS_ROOT = settings.photos_root / "__thumbs"


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


PROP = {"dominant": "dominant_color", "mean": "mean_color", "salient": "salient_color"}


async def fetch_pool(color_kind: str) -> list[dict]:
    prop = PROP[color_kind]
    drv = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_user, settings.neo4j_password),
    )
    async with drv.session() as s:
        res = await s.run(
            f"MATCH (m:Media) WHERE m.{prop} IS NOT NULL AND size(m.{prop}) = 7 "
            f"RETURN m.path AS path, m.{prop} AS color"
        )
        pool = [{"path": r["path"], "color": r["color"]} async for r in res]
    await drv.close()
    return pool


def thumb_path(media_path: str) -> Path:
    rel = media_path.removeprefix("archive/")
    return THUMBS_ROOT / (rel + ".webp")


def parse_grid(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def build_mosaic(args):
    print(f"loading pool ({args.color}) from Neo4j...")
    pool = asyncio.run(fetch_pool(args.color))
    print(f"  {len(pool):,} candidate tiles")

    # Drop ones with no thumb on disk so we don't fail mid-render.
    print("verifying thumbs exist on disk...")
    pool = [p for p in pool if thumb_path(p["path"]).exists()]
    print(f"  {len(pool):,} after thumb check")
    if not pool:
        print("no tiles available — bailing")
        return

    pts = np.array([hex_to_rgb(p["color"]) for p in pool], dtype=np.float32)
    tree = cKDTree(pts)

    print(f"loading source {args.source}")
    src = Image.open(args.source).convert("RGB")
    gw, gh = parse_grid(args.grid)
    print(f"  resizing to {gw}x{gh} grid")
    src_small = src.resize((gw, gh), Image.LANCZOS)
    arr = np.array(src_small, dtype=np.float32)

    out = Image.new("RGB", (gw * args.tile, gh * args.tile))
    use_count: dict[str, int] = {}

    K = max(args.max_reuse * 4, 20)
    for y in range(gh):
        if y and y % 10 == 0:
            print(f"  row {y}/{gh}")
        for x in range(gw):
            rgb = arr[y, x]
            _, idxs = tree.query(rgb, k=K)
            choice = None
            for idx in np.atleast_1d(idxs):
                p = pool[int(idx)]
                if use_count.get(p["path"], 0) < args.max_reuse:
                    choice = p
                    break
            if choice is None:
                choice = pool[int(np.atleast_1d(idxs)[0])]
            use_count[choice["path"]] = use_count.get(choice["path"], 0) + 1
            tp = thumb_path(choice["path"])
            try:
                t = Image.open(tp).convert("RGB")
                t = center_crop_square(t).resize((args.tile, args.tile), Image.LANCZOS)
                out.paste(t, (x * args.tile, y * args.tile))
            except Exception as e:
                print(f"  skip {tp}: {e}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=92)
    print(f"wrote {out_path}  ({out.size[0]}x{out.size[1]} px)")
    print(f"unique tiles used: {len(use_count):,} / {len(pool):,} pool")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",    required=True, help="path to source image")
    ap.add_argument("--output",    required=True, help="path to write mosaic JPG")
    ap.add_argument("--grid",      default="60x80", help="grid WxH (e.g. 60x80)")
    ap.add_argument("--tile",      type=int, default=48, help="tile size in px")
    ap.add_argument("--color",     choices=["dominant", "mean", "salient"], default="mean")
    ap.add_argument("--max-reuse", type=int, default=8, dest="max_reuse")
    build_mosaic(ap.parse_args())


if __name__ == "__main__":
    main()
