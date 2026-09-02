"""Cropping a photograph, keeping the original.

Rotation is reversible: turn a photograph four times and it is exactly where it
started, bit for bit. Cropping is not — the pixels outside the rectangle are
gone. So the original is moved aside before anything is written, the way
deleting already works, and a crop can be undone by putting it back.

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


def preserve_original(photos_root: Path, rel_path: str) -> str:
    """Put the uncropped photograph somewhere it can be found again.

    Beside the trash rather than in it: a cropped original has not been
    deleted, and mixing the two would make "empty the trash" destroy the only
    copy of something still in the archive. Same disk, so restic already backs
    it up.
    """
    src = photos_root / rel_path
    dst = photos_root / "originals" / "pre-crop" / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite: the first original is the one worth keeping. A second
    # crop of an already-cropped file must not replace the true original.
    if not dst.exists():
        shutil.copy2(src, dst)
    return str(dst.relative_to(photos_root))


def crop_media(
    photos_root: Path,
    rel_path: str,
    rect,
    *,
    thumbs_root: Path | None = None,
    cache_roots: tuple[Path, ...] | None = None,
) -> dict:
    """Crop a photograph to `rect` — (x, y, w, h) on the DISPLAYED image.

    The original is copied aside first. Everything that describes the picture
    moves with it: face boxes shift, faces cropped away are dropped, the
    sidecar's dimensions change, and every cached rendition is deleted.
    """
    src = (photos_root / rel_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(rel_path)

    from PIL import Image, ImageOps
    with Image.open(src) as im:
        old_w, old_h = ImageOps.exif_transpose(im).size

    x, y, w, h = (int(round(v)) for v in rect)
    x, y = max(0, x), max(0, y)
    w, h = min(w, old_w - x), min(h, old_h - y)
    if w < MIN_SIDE or h < MIN_SIDE:
        raise ValueError(f"crop must be at least {MIN_SIDE}x{MIN_SIDE}")
    if (x, y, w, h) == (0, 0, old_w, old_h):
        raise ValueError("crop matches the whole photograph")

    original = preserve_original(photos_root, rel_path)

    orientation = _exif_orientation(src)
    lossless = False
    if src.suffix.lower() in JPEG_SUFFIXES:
        lossless = _crop_jpeg_lossless(
            src, stored_rect((x, y, w, h), old_w, old_h, orientation)
        )
    if not lossless:
        _crop_with_pillow(src, (x, y, w, h))

    with Image.open(src) as im:
        new_w, new_h = ImageOps.exif_transpose(im).size

    # jpegtran rounds outward to the MCU grid, so the picture can be a few
    # pixels wider than asked. The face boxes have to move by what actually
    # happened, not by what was requested.
    actual_x = x - (new_w - w) if orientation in MIRRORED_ORIENTATIONS else x
    actual_x = max(0, min(x, actual_x))
    actual_y = max(0, y - (new_h - h))

    touched = {"lossless": lossless, "faces_kept": 0, "faces_dropped": 0}

    faces_file = Path(str(src) + ".faces.json")
    if faces_file.exists():
        try:
            data = json.loads(faces_file.read_text())
            kept = []
            for face in data.get("faces", []):
                moved = crop_bbox(face.get("bbox"), actual_x, actual_y, new_w, new_h)
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
        "original": original, **touched,
    }
