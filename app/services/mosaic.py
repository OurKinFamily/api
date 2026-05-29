"""Photo-mosaic builder. Match each grid cell of a source image to an archive
photo whose color (dominant / mean / salient) is closest in RGB space, then
paste a thumbnail of that photo into the cell.
"""
import io
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.spatial import cKDTree

from app.config import settings
from app.db.neo4j import get_session

THUMBS_ROOT = settings.photos_root / "__thumbs"
PROP_BY_SOURCE = {
    "dominant": "dominant_color",
    "mean":     "mean_color",
    "salient":  "salient_color",
}


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# CIE D65 white point (Y normalized to 1)
_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)

# sRGB → linear XYZ matrix (D65, IEC 61966-2-1)
_SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
], dtype=np.float32)


def _rgb255_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB array (int or float 0-255) to CIE Lab (D65). Works on
    shape (..., 3). Distances in Lab are roughly perceptually uniform, so
    Euclidean distance there matches \"these colors look similar\" better
    than RGB distance does — fewer weird greens-near-grays mismatches.
    """
    arr = rgb.astype(np.float32) / 255.0
    # sRGB → linear sRGB
    mask = arr > 0.04045
    lin = np.where(mask, ((arr + 0.055) / 1.055) ** 2.4, arr / 12.92)
    # linear sRGB → XYZ
    xyz = lin @ _SRGB_TO_XYZ.T
    # normalize by D65
    xyz = xyz / _D65
    # XYZ → Lab
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


def _thumb_path(media_path: str) -> Path:
    rel = media_path.removeprefix("archive/")
    return THUMBS_ROOT / (rel + ".webp")


def _center_crop_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def _quadrant_crop(img: Image.Image, q: int) -> Image.Image:
    """Crop one of the four quadrants by index: 0=TL 1=TR 2=BL 3=BR.
    Falls back to center crop if `q` is None / out of range.
    """
    if q is None or q < 0 or q > 3:
        return _center_crop_square(img)
    w, h = img.size
    mx, my = w // 2, h // 2
    boxes = {
        0: (0,  0,  mx, my),
        1: (mx, 0,  w,  my),
        2: (0,  my, mx, h),
        3: (mx, my, w,  h),
    }
    return img.crop(boxes[q])


def _make_dot_mask(size: int, margin: int) -> Image.Image:
    """Circle inscribed in a square cell with a transparent margin around it,
    giving the newsprint / Lichtenstein halftone look."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    r = max(1, size // 2 - margin)
    cx = size // 2
    draw.ellipse((cx - r, cx - r, cx + r, cx + r), fill=255)
    return mask


def _make_hex_mask(w: int, h: int) -> Image.Image:
    """Pointy-top hexagon mask. Origin top-left, full opacity inside the hex."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    pts = [
        (w / 2, 0),
        (w,     h / 4),
        (w,     3 * h / 4),
        (w / 2, h),
        (0,     3 * h / 4),
        (0,     h / 4),
    ]
    draw.polygon(pts, fill=255)
    return mask


async def fetch_pool(
    color_source: str,
    person_ids: list[str] | None = None,
    exclude_person_ids: list[str] | None = None,
) -> list[dict]:
    """Pull candidate tiles.

    `person_ids` = whitelist. Only Media where AT LEAST ONE of these people
    appears (logical OR). Empty/None means "everyone".

    `exclude_person_ids` = blacklist. Media where ANY excluded person
    appears is dropped (logical AND). Lets the caller say "only photos
    of Cayce and Stephen, but not if Henry is also in frame".
    """
    prop = PROP_BY_SOURCE[color_source]
    async with get_session() as s:
        if person_ids:
            res = await s.run(
                f"""
                MATCH (p:Person)-[:APPEARS_IN]->(m:Media)
                WHERE p.id IN $ids
                  AND m.{prop} IS NOT NULL AND size(m.{prop}) = 7
                  AND ($exclude IS NULL OR size($exclude) = 0
                       OR NOT EXISTS {{
                         MATCH (px:Person)-[:APPEARS_IN]->(m)
                         WHERE px.id IN $exclude
                       }})
                RETURN DISTINCT m.path AS path, m.{prop} AS color,
                       m.luminance_stddev AS contrast,
                       m.best_crop_quadrant AS quad
                """,
                ids=person_ids,
                exclude=exclude_person_ids or [],
            )
        else:
            res = await s.run(
                f"""
                MATCH (m:Media) WHERE m.{prop} IS NOT NULL AND size(m.{prop}) = 7
                  AND ($exclude IS NULL OR size($exclude) = 0
                       OR NOT EXISTS {{
                         MATCH (px:Person)-[:APPEARS_IN]->(m)
                         WHERE px.id IN $exclude
                       }})
                RETURN m.path AS path, m.{prop} AS color,
                       m.luminance_stddev AS contrast,
                       m.best_crop_quadrant AS quad
                """,
                exclude=exclude_person_ids or [],
            )
        return [
            {
                "path":     r["path"],
                "color":    r["color"],
                "contrast": r["contrast"],
                "quad":     r["quad"],
            }
            async for r in res
        ]


def build_mosaic(
    source_bytes: bytes,
    pool: list[dict],
    grid_w: int,
    grid_h: int,
    tile_size: int,
    max_reuse: int,
    shape: str = "square",
    color_distance: str = "lab",
    edge_aware: bool = False,
    source_smooth: int = 1,
    crop: str = "center",
) -> tuple[bytes, dict]:
    """Render a mosaic and return (jpeg_bytes, metadata).

    Filters pool down to entries whose thumb exists on disk, picks tiles by
    nearest RGB distance, and respects `max_reuse` so one striking photo
    doesn't carpet the output.
    """
    t0 = time.perf_counter()

    pool = [p for p in pool if _thumb_path(p["path"]).exists()]
    if not pool:
        raise ValueError("no candidate tiles found (no thumbs on disk)")

    rgb_pts = np.array([_hex_to_rgb(p["color"]) for p in pool], dtype=np.float32)
    if color_distance == "lab":
        pts = _rgb255_to_lab(rgb_pts)
    else:
        pts = rgb_pts
    tree = cKDTree(pts)

    # Pool contrast vector aligned to pool indices. Missing values get the
    # median so they neither bias high nor low.
    pool_contrast = np.array(
        [(p.get("contrast") if p.get("contrast") is not None else float("nan")) for p in pool],
        dtype=np.float32,
    )
    if edge_aware and np.isnan(pool_contrast).any():
        med = float(np.nanmedian(pool_contrast)) if not np.isnan(pool_contrast).all() else 30.0
        pool_contrast = np.where(np.isnan(pool_contrast), med, pool_contrast)

    src = Image.open(io.BytesIO(source_bytes))
    src = ImageOps.exif_transpose(src).convert("RGB")

    # Fit user's grid box to source aspect ratio — no stretching. We shrink
    # whichever dim would otherwise distort. Output ≤ user's chosen W×H.
    src_w, src_h = src.size
    src_aspect = src_w / src_h
    box_aspect = grid_w / grid_h
    if src_aspect > box_aspect:
        grid_w = grid_w
        grid_h = max(1, round(grid_w / src_aspect))
    else:
        grid_h = grid_h
        grid_w = max(1, round(grid_h * src_aspect))

    if source_smooth and source_smooth > 1:
        # Oversample then block-average. Each cell becomes the mean of an
        # N×N patch of the source, where N = source_smooth. Removes
        # single-pixel noise from the source and gives smoother color
        # transitions in the final mosaic. Costs ~10 ms extra.
        N = int(source_smooth)
        big = src.resize((grid_w * N, grid_h * N), Image.LANCZOS)
        big_arr = np.array(big, dtype=np.float32)
        # Reshape (gh*N, gw*N, 3) → (gh, N, gw, N, 3) → mean over both N axes
        arr_rgb = big_arr.reshape(grid_h, N, grid_w, N, 3).mean(axis=(1, 3))
    else:
        src_small = src.resize((grid_w, grid_h), Image.LANCZOS)
        arr_rgb = np.array(src_small, dtype=np.float32)
    arr_query = _rgb255_to_lab(arr_rgb) if color_distance == "lab" else arr_rgb
    # arr still holds RGB for the solid-fallback fill colour.
    arr = arr_rgb

    # Per-cell target contrast derived from a Sobel edge magnitude on the
    # source's luminance. High-edge cells "want" tiles whose own thumbnails
    # are busy (high luminance_stddev); flat cells want smoother tiles.
    target_contrast = None
    if edge_aware:
        # Luminance via Rec.709, then Sobel via convolution.
        lum = 0.2126 * arr_rgb[..., 0] + 0.7152 * arr_rgb[..., 1] + 0.0722 * arr_rgb[..., 2]
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = kx.T
        pad = np.pad(lum, 1, mode="edge")
        gx = (
            -pad[:-2, :-2] + pad[:-2, 2:]
            - 2 * pad[1:-1, :-2] + 2 * pad[1:-1, 2:]
            -    pad[2:,  :-2] +     pad[2:,  2:]
        )
        gy = (
            -pad[:-2, :-2] - 2 * pad[:-2, 1:-1] - pad[:-2, 2:]
            +    pad[2:,  :-2] + 2 * pad[2:,  1:-1] + pad[2:,  2:]
        )
        mag = np.hypot(gx, gy)
        if mag.max() > 0:
            mag /= mag.max()
        # Stretch to the pool's contrast range so percentile comparisons
        # are meaningful even when sources differ.
        c_lo, c_hi = float(np.percentile(pool_contrast, 5)), float(np.percentile(pool_contrast, 95))
        target_contrast = c_lo + mag * (c_hi - c_lo)

    if shape == "hex":
        hex_w = tile_size
        hex_h = int(round(tile_size * 2.0 / math.sqrt(3)))
        v_step = int(round(hex_h * 0.75))
        out_w = grid_w * hex_w + hex_w // 2
        out_h = (grid_h - 1) * v_step + hex_h
        out = Image.new("RGB", (out_w, out_h), (0, 0, 0))
        hex_mask = _make_hex_mask(hex_w, hex_h)
    elif shape == "dot":
        # Square cells with a circle inscribed inside. Background white so the
        # gaps around each dot read as paper, like newsprint or halftone.
        out = Image.new("RGB", (grid_w * tile_size, grid_h * tile_size), (255, 255, 255))
        dot_mask = _make_dot_mask(tile_size, margin=max(1, tile_size // 20))
    else:
        out = Image.new("RGB", (grid_w * tile_size, grid_h * tile_size))
    use_count: dict[str, int] = {}

    # Search a wide neighborhood so we can honour `max_reuse` even when many
    # cells target the same color (e.g. skin tones over a face). When every
    # candidate in K is already maxed out, pick the *least-used* of them so
    # repeats are at least spread out evenly instead of clumping onto the
    # nearest tile.
    K = min(max(max_reuse * 50, 200), len(pool))
    solid_fallbacks = 0  # cells where no unused tile remained — painted solid hex
    placed = [[None] * grid_w for _ in range(grid_h)]  # path per cell, for neighbor dedup
    # Rolling history of the last N placed tiles. Wider than just the four
    # adjacent cells — kills the ABAB / ABCABC repeating-tile patterns that
    # form when only 2-3 tiles match a color region's needs.
    LOOKBACK = max(24, grid_w // 4)
    history: deque[str] = deque(maxlen=LOOKBACK)
    # When the top few candidates are basically tied on color distance,
    # shuffle them so the same cell-pair doesn't always get the same
    # arrangement (extra entropy against patterns).
    TIE_BAND = 3
    for y in range(grid_h):
        for x in range(grid_w):
            rgb = arr[y, x]
            dists, idxs = tree.query(arr_query[y, x], k=K)
            idxs  = np.atleast_1d(idxs)
            dists = np.atleast_1d(dists)
            if not edge_aware and len(idxs) >= TIE_BAND:
                # Shuffle the top TIE_BAND candidates among themselves.
                head = list(range(TIE_BAND))
                random.shuffle(head)
                idxs = np.concatenate([idxs[head], idxs[TIE_BAND:]])
            if edge_aware:
                # Re-rank: keep the top K by color, then sort by how close
                # each candidate's own contrast is to the per-cell target.
                # This nudges busy source regions toward busy tiles without
                # abandoning the color match.
                tgt = float(target_contrast[y, x])
                cand_contrast = pool_contrast[idxs]
                order = np.argsort(np.abs(cand_contrast - tgt))
                idxs = idxs[order]
            # Collect immediate neighbors that already have a tile placed.
            # For square: left + top. For hex with staggered odd rows: top-left
            # + top-right too. We just include all eight, harmless when None.
            forbidden = set(history)
            forbidden.update(
                p for p in (
                    placed[y][x - 1] if x > 0 else None,
                    placed[y - 1][x] if y > 0 else None,
                    placed[y - 1][x - 1] if y > 0 and x > 0 else None,
                    placed[y - 1][x + 1] if y > 0 and x + 1 < grid_w else None,
                ) if p is not None
            )
            choice = None
            for idx in idxs:
                p = pool[int(idx)]
                if p["path"] in forbidden:
                    continue
                if use_count.get(p["path"], 0) < max_reuse:
                    choice = p
                    break
            # Last-chance relaxation: if the history was too aggressive and
            # no candidate passed, drop the wide-history rule but still keep
            # the immediate neighbors blocked — same as old behaviour.
            if choice is None:
                neighbors_only = forbidden - set(history)
                for idx in idxs:
                    p = pool[int(idx)]
                    if p["path"] in neighbors_only:
                        continue
                    if use_count.get(p["path"], 0) < max_reuse:
                        choice = p
                        break
            if shape == "hex":
                paste_x = x * hex_w + (y % 2) * (hex_w // 2)
                paste_y = y * v_step
                paste_size = (hex_w, hex_h)
                mask = hex_mask
                fill_box = (paste_x, paste_y, paste_x + hex_w, paste_y + hex_h)
            elif shape == "dot":
                paste_x = x * tile_size
                paste_y = y * tile_size
                paste_size = (tile_size, tile_size)
                mask = dot_mask
                fill_box = (paste_x, paste_y, paste_x + tile_size, paste_y + tile_size)
            else:
                paste_x = x * tile_size
                paste_y = y * tile_size
                paste_size = (tile_size, tile_size)
                mask = None
                fill_box = (paste_x, paste_y, paste_x + tile_size, paste_y + tile_size)

            if choice is None:
                fill = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
                if mask is not None:
                    patch = Image.new("RGB", paste_size, fill)
                    out.paste(patch, (paste_x, paste_y), mask)
                else:
                    out.paste(fill, fill_box)
                solid_fallbacks += 1
                continue
            use_count[choice["path"]] = use_count.get(choice["path"], 0) + 1
            placed[y][x] = choice["path"]
            history.append(choice["path"])
            try:
                t = Image.open(_thumb_path(choice["path"])).convert("RGB")
                if crop == "saliency":
                    t = _quadrant_crop(t, choice.get("quad"))
                    # Saliency quadrant is half-w × half-h. Square it before
                    # resizing so the tile stays proportional.
                    t = _center_crop_square(t)
                else:
                    t = _center_crop_square(t)
                t = t.resize(paste_size, Image.LANCZOS)
                out.paste(t, (paste_x, paste_y), mask)
            except Exception:
                continue

    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=88, optimize=True)
    payload = buf.getvalue()
    meta = {
        "grid_w":           grid_w,
        "grid_h":           grid_h,
        "tile_size":        tile_size,
        "shape":            shape,
        "color_distance":   color_distance,
        "edge_aware":       edge_aware,
        "source_smooth":    source_smooth,
        "crop":             crop,
        "pool_size":        len(pool),
        "unique_tiles":     len(use_count),
        "cells":            grid_w * grid_h,
        "max_reuse":        max_reuse,
        "solid_fallbacks":  solid_fallbacks,
        "file_size_bytes":  len(payload),
        "out_w":            out.size[0],
        "out_h":            out.size[1],
        "render_ms":        int((time.perf_counter() - t0) * 1000),
    }
    return payload, meta
