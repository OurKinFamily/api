"""Rotating a photograph, and everything that has to move with it.

Rotation is not one operation. A photograph in this archive is a file plus a
metadata sidecar, a faces sidecar carrying bounding boxes, a set of cropped
face images, a cached thumbnail, and a node in the graph recording its
dimensions. Turn the pixels and leave the rest, and the face boxes land
sideways — which is worse than not offering rotation at all.

JPEGs are rotated with jpegtran, which transforms the DCT coefficients rather
than decoding and re-encoding. That is lossless: an archive should not lose a
little quality every time somebody straightens a photograph. Anything else
falls back to Pillow.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

JPEG_SUFFIXES = {".jpg", ".jpeg"}

# EXIF orientations that include a mirror. The other four are pure rotations.
MIRRORED_ORIENTATIONS = {2, 4, 5, 7}

# ExifTool's wording, which is what the sidecars already use.
ORIENTATION_NAMES = {
    1: "Horizontal (normal)",
    2: "Mirror horizontal",
    3: "Rotate 180",
    4: "Mirror vertical",
    5: "Mirror horizontal and rotate 270 CW",
    6: "Rotate 90 CW",
    7: "Mirror horizontal and rotate 90 CW",
    8: "Rotate 270 CW",
}


def _exif_orientation(path: Path) -> int:
    """The file's own orientation tag, or 1 when it has none."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return (im.getexif() or {}).get(274) or 1
    except Exception:
        return 1


def stored_rotation(degrees: int, orientation: int) -> int:
    """How far to turn the STORED pixels to turn the DISPLAYED image by `degrees`.

    jpegtran rewrites the pixels and leaves the EXIF orientation tag alone, so
    a viewer still applies that tag afterwards. Writing `displayed = O(stored)`,
    what the pixels need is O-inverse . R . O — and in the dihedral group that
    composes down to a plain rotation: by `degrees` when O is a pure rotation,
    and by MINUS `degrees` when O contains a mirror, because a reflection
    reverses the sense of rotation.

    So the common case (orientation 6, every portrait phone photo) is correct
    only because rotations commute, and the mirrored cases are correct only if
    we turn the other way. Turning the same way for a mirrored file lands it
    180 degrees out.
    """
    d = degrees % 360
    return (-d) % 360 if orientation in MIRRORED_ORIENTATIONS else d


def rotate_bbox(bbox, width: int, height: int, degrees: int):
    """Move a bounding box to where it lands after the image is rotated.

    Boxes are [x1, y1, x2, y2] in the image's display pixels, and `width` and
    `height` describe the image BEFORE the rotation.

    Rotating clockwise by 90 sends a point (x, y) to (height - y, x), so the
    box's x-range comes from its old y-range reversed, and its y-range from its
    old x-range. Getting this the wrong way round is the classic way to end up
    with every face boxed on the opposite side of the picture.
    """
    if not bbox or len(bbox) < 4:
        return bbox
    x1, y1, x2, y2 = bbox[:4]
    d = degrees % 360
    if d == 90:
        return [height - y2, x1, height - y1, x2]
    if d == 180:
        return [width - x2, height - y2, width - x1, height - y1]
    if d == 270:
        return [y1, width - x2, y2, width - x1]
    return [x1, y1, x2, y2]


def _rotate_jpeg_lossless(path: Path, degrees: int) -> bool:
    """jpegtran, which rearranges DCT blocks rather than re-encoding.

    Only exact multiples of 90 work, and only when the dimensions are multiples
    of the MCU block size — `-perfect` refuses rather than silently discarding
    edge pixels, and being refused is the right outcome for an archive.
    """
    if not shutil.which("jpegtran"):
        return False
    tmp = path.with_suffix(path.suffix + ".rotating")
    try:
        subprocess.run(
            ["jpegtran", "-rotate", str(degrees), "-perfect",
             "-copy", "all", "-outfile", str(tmp), str(path)],
            check=True, capture_output=True,
        )
        tmp.replace(path)
        return True
    except (subprocess.CalledProcessError, OSError):
        tmp.unlink(missing_ok=True)
        return False


def _rotate_with_pillow(path: Path, degrees: int, *, keep_exif: bool = True) -> None:
    """Re-encode fallback, for anything jpegtran will not touch losslessly.

    Carries the EXIF across explicitly. Pillow's save() writes none by default,
    so the obvious version of this function quietly destroys the date, the GPS
    and the camera — the three things that make a photograph findable — and
    does it to the file in place.
    """
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        exif = im.info.get("exif") if keep_exif else None
        icc = im.info.get("icc_profile")
        # Apply any existing EXIF orientation first, so the result is what a
        # viewer actually sees rather than what the sensor recorded.
        im = ImageOps.exif_transpose(im)
        # Pillow rotates anticlockwise; every caller here means clockwise.
        im = im.rotate(-degrees, expand=True)

        params = {}
        if path.suffix.lower() in JPEG_SUFFIXES:
            # 97 rather than 95: this is the copy the archive keeps.
            # No subsampling="keep": once rotate() has run the image is no
            # longer the original JPEG object and Pillow refuses.
            params.update(quality=97, optimize=True)
        if exif:
            # exif_transpose() already applied the old orientation, so the tag
            # has to be cleared or every viewer will rotate it a second time.
            exif = _clear_orientation(exif)
            params["exif"] = exif
        if icc:
            params["icc_profile"] = icc
        im.save(path, **params)


def _clear_orientation(exif_bytes: bytes) -> bytes:
    """Set the EXIF orientation tag to 1 (upright), leaving everything else."""
    try:
        from PIL import Image
        img_exif = Image.Exif()
        img_exif.load(exif_bytes)
        if 274 in img_exif:
            img_exif[274] = 1
        return img_exif.tobytes()
    except Exception:
        return exif_bytes


def clear_derived_caches(rel_path: str, thumbs_root=None, cache_roots=None) -> bool:
    """Drop every rendition of a photograph. Returns whether any existed.

    They are all regenerated on demand, so deleting is the whole fix. Missing
    one is worse than it sounds: the grid went on showing the old thumbnail
    after a rotate, and a reader with no reason to open the photograph would
    never have seen it change.

    There are three caches, not one — the webp tree beside the archive, and two
    on-demand caches under /tmp that the media routes fall back to. They share
    a key: the path with "archive/" stripped, plus ".webp".
    """
    key = rel_path.removeprefix("archive/") + ".webp"
    cleared = False
    for root in (thumbs_root, *(cache_roots or ())):
        if not root:
            continue
        cached = root / key
        if cached.is_file():
            cached.unlink(missing_ok=True)
            cleared = True
    return cleared


def rotate_media(
    photos_root: Path,
    rel_path: str,
    degrees: int,
    *,
    thumbs_root: Path | None = None,
    cache_roots: tuple[Path, ...] | None = None,
) -> dict:
    """Rotate a photograph and bring its metadata with it.

    Returns the new width and height, and what was touched.
    """
    if degrees % 90 or degrees % 360 == 0:
        raise ValueError("degrees must be 90, 180 or 270")
    degrees %= 360

    src = (photos_root / rel_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(rel_path)

    from PIL import Image, ImageOps
    with Image.open(src) as im:
        before = ImageOps.exif_transpose(im).size
    old_w, old_h = before

    orientation = _exif_orientation(src)
    lossless = src.suffix.lower() in JPEG_SUFFIXES and _rotate_jpeg_lossless(
        src, stored_rotation(degrees, orientation)
    )
    if not lossless:
        _rotate_with_pillow(src, degrees)

    with Image.open(src) as im:
        new_w, new_h = ImageOps.exif_transpose(im).size

    touched = {"lossless": lossless, "faces": 0, "crops": 0, "thumb": False}

    # ── faces: boxes move, and so do the crops cut from them ────────────────
    faces_file = Path(str(src) + ".faces.json")
    if faces_file.exists():
        try:
            fd = json.loads(faces_file.read_text())
            for face in fd.get("faces", []):
                face["bbox"] = rotate_bbox(face.get("bbox"), old_w, old_h, degrees)
                touched["faces"] += 1
                # The crop is a still of the face at the old rotation. Cheap to
                # turn, and leaving it means a sidebar full of faces lying on
                # their side.
                cp = face.get("crop_path")
                if cp:
                    crop = photos_root / cp
                    if crop.is_file():
                        try:
                            _rotate_with_pillow(crop, degrees)
                            touched["crops"] += 1
                        except Exception:
                            pass
            faces_file.write_text(json.dumps(fd, indent=2))
        except (json.JSONDecodeError, OSError):
            pass

    # ── metadata sidecar: dimensions, and no leftover orientation claim ─────
    sidecar = Path(str(src) + ".json")
    if sidecar.exists():
        try:
            data = json.loads(sidecar.read_text())
            results = data.get("results") or []
            meta = results[0].get("metadata") if results else data.get("metadata")
            if meta:
                dims = (meta.setdefault("media", {})).setdefault("dimensions", {})
                dims["width"], dims["height"] = new_w, new_h
                # Report what the file actually says, rather than asserting it
                # is upright. jpegtran leaves the orientation tag in place, so
                # claiming "Horizontal (normal)" after a lossless rotate left
                # the sidecar disagreeing with the photograph it describes —
                # and anything trusting the sidecar over the file would then
                # size or transpose it wrongly. Only the Pillow path really
                # clears the tag.
                dims["orientation"] = ORIENTATION_NAMES.get(
                    _exif_orientation(src), "Horizontal (normal)"
                )
                sidecar.write_text(json.dumps(data, indent=2))
        except (json.JSONDecodeError, OSError):
            pass

    # ── derived copies: stale the moment the pixels move ────────────────────
    touched["thumb"] = clear_derived_caches(rel_path, thumbs_root, cache_roots)

    poster = Path(str(src) + ".poster.jpg")
    if poster.is_file():
        try:
            _rotate_with_pillow(poster, degrees)
        except Exception:
            pass

    # A per-file cache token. The URL for a photograph never changes, so a
    # browser that has already fetched it keeps showing the old pixels no
    # matter how thoroughly the file was rewritten. The modification time is
    # the cheapest thing that is guaranteed to differ afterwards.
    try:
        version = int(src.stat().st_mtime)
    except OSError:
        version = 0

    return {"width": new_w, "height": new_h, "version": version, **touched}
