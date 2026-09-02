"""Lifting the shadows, the midtones and the highlights of a photograph.

Not the restoration model — this is arithmetic, and it does what it is told.
The model repaints; this only moves brightness about, so a scan that came out
flat can be opened up without anything being invented.

Three controls, each -100 to 100, each pulling one part of the range:

    shadows     the dark end     weight (1-x)^2
    midtones    the middle       weight 4x(1-x), peaking at half brightness
    highlights  the bright end   weight x^2

The weights overlap deliberately. Bands with hard edges leave visible seams
where one stops and the next starts — a sky gaining a step halfway up. These
blend, so lifting the shadows also lifts the low midtones a little, which is
what somebody means when they say "open up the shadows".

MAX_LIFT caps a single control at just over a third of the range. Beyond that
the picture stops looking like a photograph, and an archive is not the place
for that.
"""
from __future__ import annotations

MAX_LIFT = 0.35


def build_lut(shadows: int = 0, midtones: int = 0, highlights: int = 0) -> list[int]:
    """The 256-entry table these three controls describe.

    A table rather than per-pixel arithmetic: the same 256 answers serve every
    pixel, and the browser builds an identical one — which is what makes the
    preview honest about what will be saved.
    """
    s, m, h = shadows / 100, midtones / 100, highlights / 100
    lut = []
    for i in range(256):
        x = i / 255
        lift = (
            s * (1 - x) ** 2
            + m * 4 * x * (1 - x)
            + h * x ** 2
        )
        y = x + MAX_LIFT * lift
        lut.append(max(0, min(255, round(y * 255))))
    return lut


def is_noop(shadows: int, midtones: int, highlights: int) -> bool:
    return shadows == 0 and midtones == 0 and highlights == 0


def apply_tone(
    photos_root, rel_path: str, *,
    shadows: int = 0, midtones: int = 0, highlights: int = 0,
    thumbs_root=None, cache_roots=None,
) -> dict:
    """Write the adjusted photograph, keeping what it looked like before.

    Kept for the same reason a restoration is: this re-encodes, so applying it
    twice compounds the loss, and somebody who overdoes the shadows should be
    able to get back to the scan rather than to their own last attempt.
    """
    from pathlib import Path

    from PIL import Image, ImageOps

    from app.services.restore import preserve_original
    from app.services.rotate import JPEG_SUFFIXES, _clear_orientation, clear_derived_caches

    if is_noop(shadows, midtones, highlights):
        raise ValueError("nothing to change")

    src = (Path(photos_root) / rel_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(rel_path)

    kept = preserve_original(Path(photos_root), rel_path)
    lut = build_lut(shadows, midtones, highlights)

    with Image.open(src) as im:
        exif = im.info.get("exif")
        icc = im.info.get("icc_profile")
        upright = ImageOps.exif_transpose(im).convert("RGB")
        # The same table for all three channels: this is a brightness control,
        # and applying different curves per channel would tint the picture.
        adjusted = upright.point(lut * 3)

        params = {}
        if src.suffix.lower() in JPEG_SUFFIXES:
            params.update(quality=97, optimize=True)
        if exif:
            params["exif"] = _clear_orientation(exif)
        if icc:
            params["icc_profile"] = icc
        adjusted.save(src, **params)
        width, height = adjusted.size

    clear_derived_caches(rel_path, thumbs_root, cache_roots)

    try:
        version = int(src.stat().st_mtime)
    except OSError:
        version = 0

    return {
        "width": width, "height": height, "version": version,
        "original": kept,
        "shadows": shadows, "midtones": midtones, "highlights": highlights,
    }
