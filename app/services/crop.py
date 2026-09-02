"""Cropping a photograph.

Destructive, deliberately. Rotation is reversible — four quarter-turns and the
file is bit-for-bit where it started — but a crop throws pixels away and there
is no copy kept. An earlier version preserved the uncropped file under
originals/, and that was dropped: at 1.3TB the archive cannot carry a second
copy of everything anybody trims, and a store of originals nobody intends to
restore from is just a slow leak.

What this means for a caller: check the rectangle before sending it. There is
no undo.

JPEGs are cropped with jpegtran, which rearranges DCT blocks rather than
decoding and re-encoding. The catch is that it can only cut on the MCU grid, so
it rounds the rectangle OUTWARD to the nearest block: ask for 300x200 at
+101+53 and you get 305x205. That is the right way to be wrong — a few pixels
of extra picture, never a few pixels missing — but it means the result is
reported back rather than assumed.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from app.services.rotate import (
    JPEG_SUFFIXES,
    MIRRORED_ORIENTATIONS,
    _exif_orientation,
    clear_derived_caches,
)

# Nothing smaller is a crop; it is a mistake.
MIN_SIDE = 32

# jpegtran cuts on the MCU grid. 16 is the block size for 4:2:0 chroma
# subsampling, which is what a phone or a scanner writes; 4:4:4 files use 8, so
# assuming 16 is the safe direction — it is always a multiple of the real one.
MCU = 16

# Undoing an orientation: pure rotations swap with their opposite, mirrors and
# transposes are their own inverse.
INVERSE_ORIENTATION = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8, 7: 7, 8: 6}


def stored_rect(rect, width: int, height: int, orientation: int):
    """Where a rectangle drawn on the DISPLAYED photograph sits in the file.

    jpegtran cuts the stored pixels and leaves the EXIF orientation tag alone,
    so a viewer still turns the result afterwards. Writing displayed = O(stored),
    the rectangle has to be mapped back through O — the same reasoning that
    rotation needed, and the same trap: a portrait phone photograph is stored
    landscape, so cropping "the top" without this takes a strip off one side.

    `width` and `height` describe the DISPLAYED image.
    """
    x, y, w, h = rect
    o = orientation
    if o == 1:
        return (x, y, w, h)
    if o == 3:                        # 180
        return (width - x - w, height - y - h, w, h)
    if o == 6:                        # displayed is the file turned 90 CW
        return (y, width - x - w, h, w)
    if o == 8:                        # 270
        return (height - y - h, x, h, w)
    if o == 2:                        # mirrored horizontally
        return (width - x - w, y, w, h)
    if o == 4:                        # mirrored vertically
        return (x, height - y - h, w, h)
    if o == 5:                        # transpose
        return (y, x, h, w)
    if o == 7:                        # transverse
        return (height - y - h, width - x - w, h, w)
    return (x, y, w, h)


def crop_bbox(bbox, x: int, y: int, w: int, h: int):
    """Move a face box into the cropped image, or say it no longer fits.

    Returns None when the face lies outside the rectangle entirely — a face
    that has been cropped away should stop being drawn, not be clamped to the
    edge where it would sit over somebody else.
    """
    if not bbox or len(bbox) < 4:
        return bbox
    x1, y1, x2, y2 = bbox[:4]
    nx1, ny1 = max(x1 - x, 0), max(y1 - y, 0)
    nx2, ny2 = min(x2 - x, w), min(y2 - y, h)
    if nx2 - nx1 < 1 or ny2 - ny1 < 1:
        return None
    # Mostly outside counts as outside: a sliver of somebody's ear is not a
    # face, and boxing it makes the panel look broken.
    if (nx2 - nx1) * (ny2 - ny1) < 0.25 * max((x2 - x1) * (y2 - y1), 1):
        return None
    return [nx1, ny1, nx2, ny2]


def align_inward(rect) -> tuple[int, int, int, int]:
    """Move the crop origin FORWARD to the next block boundary.

    jpegtran snaps an unaligned origin BACKWARD, keeps the requested extent,
    and so returns the pixels you asked it to remove. Trim nine pixels off an
    edge and you get all nine back — the crop reports success and the
    photograph is unchanged, which is exactly what it looks like from the
    outside: a button that does nothing.

    Snapping forward instead means cutting up to fifteen pixels MORE than
    asked. For trimming a border — which is what small crops are for — erring
    towards removing it is plainly the right direction.
    """
    x, y, w, h = rect
    nx, ny = -(-x // MCU) * MCU, -(-y // MCU) * MCU     # ceil to the grid
    return nx, ny, w - (nx - x), h - (ny - y)


def _crop_jpeg_lossless(path: Path, rect) -> bool:
    """jpegtran, which cuts DCT blocks rather than re-encoding."""
    if not shutil.which("jpegtran"):
        return False
    x, y, w, h = rect
    tmp = path.with_suffix(path.suffix + ".cropping")
    try:
        subprocess.run(
            ["jpegtran", "-crop", f"{w}x{h}+{x}+{y}",
             "-copy", "all", "-outfile", str(tmp), str(path)],
            check=True, capture_output=True,
        )
        tmp.replace(path)
        return True
    except (subprocess.CalledProcessError, OSError):
        tmp.unlink(missing_ok=True)
        return False


def rotate_bbox_free(bbox, w: int, h: int, angle: float, new_w: int, new_h: int):
    """Where a face box lands after the picture is turned by an arbitrary angle.

    A box is four numbers describing an upright rectangle, and a rotated
    rectangle is not one. So its corners are rotated and the smallest upright
    box containing them is taken — slightly generous, which is the right
    direction to be wrong for something drawn around a face.
    """
    import math
    if not bbox or len(bbox) < 4:
        return bbox
    x1, y1, x2, y2 = bbox[:4]
    # Screen coordinates have y pointing DOWN, so a clockwise turn is the
    # POSITIVE direction here — the opposite of the maths convention, and the
    # reason the naive sign put every face on the wrong side of the picture.
    rad = math.radians(angle)
    cos, sin = math.cos(rad), math.sin(rad)
    cx, cy = w / 2, h / 2
    ncx, ncy = new_w / 2, new_h / 2
    xs, ys = [], []
    for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        dx, dy = px - cx, py - cy
        xs.append(ncx + dx * cos - dy * sin)
        ys.append(ncy + dx * sin + dy * cos)
    return [min(xs), min(ys), max(xs), max(ys)]


def _straighten(path: Path, angle: float) -> tuple[int, int]:
    """Turn a photograph by a fraction of a degree, and say how big it got.

    This is the one operation here that cannot be lossless: jpegtran works in
    whole MCU blocks and whole quarter-turns, so anything else means decoding,
    rotating and re-encoding. Quality 97 keeps the loss well below what a
    scanner introduced in the first place, and straightening a crooked scan is
    worth one generation.

    `expand` keeps the corners, so the result is larger than the original and
    padded with white — the crop that follows is expected to cut that away.
    """
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        exif = im.info.get("exif")
        icc = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im)
        # Pillow turns anticlockwise; every caller here means clockwise.
        im = im.rotate(-angle, expand=True, resample=Image.BICUBIC, fillcolor=(255, 255, 255))
        params = {}
        if path.suffix.lower() in JPEG_SUFFIXES:
            params.update(quality=97, optimize=True)
        if exif:
            from app.services.rotate import _clear_orientation
            params["exif"] = _clear_orientation(exif)
        if icc:
            params["icc_profile"] = icc
        im.save(path, **params)
        return im.size


def _crop_with_pillow(path: Path, rect) -> None:
    """Re-encode fallback, for anything jpegtran will not touch."""
    from PIL import Image, ImageOps
    x, y, w, h = rect
    with Image.open(path) as im:
        exif = im.info.get("exif")
        icc = im.info.get("icc_profile")
        im = ImageOps.exif_transpose(im)
        im = im.crop((x, y, x + w, y + h))
        params = {}
        if path.suffix.lower() in JPEG_SUFFIXES:
            params.update(quality=97, optimize=True)
        if exif:
            from app.services.rotate import _clear_orientation
            params["exif"] = _clear_orientation(exif)
        if icc:
            params["icc_profile"] = icc
        im.save(path, **params)


def crop_media(
    photos_root: Path,
    rel_path: str,
    rect,
    *,
    angle: float = 0.0,
    thumbs_root: Path | None = None,
    cache_roots: tuple[Path, ...] | None = None,
) -> dict:
    """Crop a photograph to `rect` — (x, y, w, h) on the DISPLAYED image.

    `angle` straightens first, clockwise in degrees, and the rectangle is then
    read against the STRAIGHTENED picture: the caller drew it on a preview that
    was already turned. Any angle that is not a multiple of 90 forces a
    re-encode, so this is the one path here that costs a generation.

    Everything describing the picture moves with it: face boxes shift, faces
    cropped away are dropped, the sidecar's dimensions change, and every cached
    rendition is deleted. The pixels outside the rectangle are gone for good.
    """
    src = (photos_root / rel_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(rel_path)

    from PIL import Image, ImageOps
    with Image.open(src) as im:
        old_w, old_h = ImageOps.exif_transpose(im).size

    was_w, was_h = old_w, old_h
    straightened = None
    if angle:
        straightened = _straighten(src, angle)
        # The rectangle was drawn on the turned picture, so everything from
        # here — bounds, face boxes, the crop itself — works in its size.
        old_w, old_h = straightened

    x, y, w, h = (int(round(v)) for v in rect)
    x, y = max(0, x), max(0, y)
    w, h = min(w, old_w - x), min(h, old_h - y)
    if w < MIN_SIDE or h < MIN_SIDE:
        raise ValueError(f"crop must be at least {MIN_SIDE}x{MIN_SIDE}")
    if (x, y, w, h) == (0, 0, old_w, old_h):
        raise ValueError("crop matches the whole photograph")

    orientation = _exif_orientation(src)
    with Image.open(src) as im:
        stored_w, stored_h = im.size
    sx, sy, sw, sh = stored_rect((x, y, w, h), old_w, old_h, orientation)
    lossless = False
    # Straightening already re-encoded and cleared the orientation tag, so the
    # file is now upright and jpegtran can cut it straight.
    if angle:
        orientation = 1
    if src.suffix.lower() in JPEG_SUFFIXES:
        ax, ay, aw, ah = align_inward((sx, sy, sw, sh))
        # If honouring the grid would eat the crop, re-encode instead.
        if aw >= MIN_SIDE and ah >= MIN_SIDE:
            lossless = _crop_jpeg_lossless(src, (ax, ay, aw, ah))
            if lossless:
                sx, sy, sw, sh = ax, ay, aw, ah
    if not lossless:
        _crop_with_pillow(src, (x, y, w, h))

    with Image.open(src) as im:
        new_w, new_h = ImageOps.exif_transpose(im).size

    # Where the crop ACTUALLY started.
    #
    # jpegtran snaps the origin back to the MCU grid and keeps the requested
    # extent, so the picture comes out a few pixels bigger and starts a few
    # pixels earlier than asked. Face boxes have to move by what happened, not
    # by what was requested, or every one of them sits slightly off.
    #
    # The snapping happens on the STORED axes, and for an oriented file those
    # are not the displayed ones — so the shift is measured in stored space and
    # mapped back, rather than guessed at in display space.
    actual_x, actual_y = x, y
    if lossless:
        with Image.open(src) as im:
            new_sw, new_sh = im.size
        shifted = (
            sx - (new_sw - sw), sy - (new_sh - sh), new_sw, new_sh,
        )
        actual_x, actual_y = stored_rect(
            shifted, stored_w, stored_h, INVERSE_ORIENTATION[orientation],
        )[:2]
    actual_x, actual_y = max(0, actual_x), max(0, actual_y)

    touched = {"lossless": lossless, "faces_kept": 0, "faces_dropped": 0}

    faces_file = Path(str(src) + ".faces.json")
    if faces_file.exists():
        try:
            data = json.loads(faces_file.read_text())
            kept = []
            for face in data.get("faces", []):
                box = face.get("bbox")
                if angle and straightened:
                    box = rotate_bbox_free(
                        box, was_w, was_h, angle, straightened[0], straightened[1],
                    )
                moved = crop_bbox(box, actual_x, actual_y, new_w, new_h)
                if moved is None:
                    touched["faces_dropped"] += 1
                    continue
                face["bbox"] = moved
                kept.append(face)
                touched["faces_kept"] += 1
            data["faces"] = kept
            faces_file.write_text(json.dumps(data, indent=2))
        except (json.JSONDecodeError, OSError):
            pass

    sidecar = Path(str(src) + ".json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
            results = data.get("results") or []
            meta = results[0].get("metadata") if results else data.get("metadata")
            if meta:
                dims = (meta.setdefault("media", {})).setdefault("dimensions", {})
                dims["width"], dims["height"] = new_w, new_h
                sidecar.write_text(json.dumps(data, indent=2))
        except (json.JSONDecodeError, OSError):
            pass

    clear_derived_caches(rel_path, thumbs_root, cache_roots)

    try:
        version = int(src.stat().st_mtime)
    except OSError:
        version = 0

    return {
        "width": new_w, "height": new_h, "version": version,
        "angle": angle, **touched,
    }
