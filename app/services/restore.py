"""Restoring a photograph, with a look before you leap.

Wraps Stephen's restoration pipeline (scratch and dust removal, no face
enhancement). Two steps on purpose: the first writes a PREVIEW beside the
archive and touches nothing, the second replaces the original only once
somebody has looked at the result and said yes.

That split is the whole point. A restoration is a machine's opinion about what
a photograph used to look like, and it is right most of the time — "most" being
exactly why it should not be applied unseen.

The pipeline emits PNG and changes the dimensions slightly (615x943 comes back
608x944), so applying is not a file copy: the result is re-encoded to the
original's format, the EXIF is carried across, and every face box is scaled to
the new size.

The pre-restoration photograph is kept under originals/, mirroring the
archive's own shape. Cropping deliberately keeps nothing; restoration is a
different kind of change — a model has repainted the picture, and the scan is
the record of the print.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

# Overridable, because the script lives outside the repo and will not be on a
# deployed host at all.
RESTORE_SCRIPT = Path(
    os.environ.get("RESTORE_SCRIPT", "/home/stephen/Desktop/restore/restore.sh")
)

# Previews live under the photos root so the existing media route can serve
# them, in their own tree so they are never mistaken for archive files.
PREVIEW_DIR = "__restore"

TIMEOUT_SECONDS = 600

# Where the pre-restoration photograph goes, mirroring the archive's own shape
# minus the "archive/" prefix: archive/1951/04/x.jpg -> originals/1951/04/x.jpg.
ORIGINALS_DIR = "originals"


def preview_path(photos_root: Path, rel_path: str) -> Path:
    return photos_root / PREVIEW_DIR / f"{rel_path}.png"


def make_preview(photos_root: Path, rel_path: str) -> dict:
    """Run the pipeline against a copy and leave the result beside the archive.

    The original is not touched, read or written — the script is handed a copy,
    so a crash halfway through cannot leave a half-restored photograph where
    the real one was.
    """
    if not RESTORE_SCRIPT.is_file():
        raise FileNotFoundError(f"restore script not found at {RESTORE_SCRIPT}")

    src = (photos_root / rel_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(rel_path)

    out = preview_path(photos_root, rel_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    work = out.with_suffix(".input" + src.suffix)
    work.write_bytes(src.read_bytes())
    produced = work.with_name(f"{work.stem}.restored.png")

    try:
        subprocess.run(
            [str(RESTORE_SCRIPT), str(work)],
            check=True, capture_output=True, timeout=TIMEOUT_SECONDS,
        )
        if not produced.is_file():
            raise RuntimeError("the pipeline produced no output")
        produced.replace(out)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            (e.stderr or b"").decode()[-400:] or "restoration failed"
        ) from e
    finally:
        work.unlink(missing_ok=True)
        produced.unlink(missing_ok=True)

    from PIL import Image
    with Image.open(out) as im:
        width, height = im.size

    return {
        "preview": f"{PREVIEW_DIR}/{rel_path}.png",
        "width": width, "height": height,
        "bytes": out.stat().st_size,
    }


def discard_preview(photos_root: Path, rel_path: str) -> bool:
    out = preview_path(photos_root, rel_path)
    existed = out.is_file()
    out.unlink(missing_ok=True)
    return existed


def apply_preview(photos_root: Path, rel_path: str, *, thumbs_root=None, cache_roots=None) -> dict:
    """Replace the photograph with its restored preview.

    Re-encoded rather than copied: the pipeline writes PNG, and the archive
    identifies a photograph by its path — changing the extension would orphan
    the Media node, the sidecars and every face crop that points at it.
    """
    from PIL import Image, ImageOps

    from app.services.rotate import JPEG_SUFFIXES, clear_derived_caches

    src = (photos_root / rel_path).resolve()
    preview = preview_path(photos_root, rel_path)
    if not preview.is_file():
        raise FileNotFoundError("no preview to apply")

    kept = preserve_original(photos_root, rel_path)

    with Image.open(src) as original:
        exif = original.info.get("exif")
        icc = original.info.get("icc_profile")
        old_w, old_h = ImageOps.exif_transpose(original).size

    with Image.open(preview) as restored:
        result = restored.convert("RGB")
        new_w, new_h = result.size
        params = {}
        if src.suffix.lower() in JPEG_SUFFIXES:
            # 97: this is the copy the archive keeps, and it has already been
            # through one lossy generation getting here.
            params.update(quality=97, optimize=True)
        if exif:
            # The pipeline keeps none of it, and the date, the place and the
            # camera are most of what makes a photograph findable.
            from app.services.rotate import _clear_orientation
            params["exif"] = _clear_orientation(exif)
        if icc:
            params["icc_profile"] = icc
        result.save(src, **params)

    scale_x, scale_y = new_w / old_w, new_h / old_h
    faces = _rescale_faces(src, scale_x, scale_y)
    _update_sidecar(src, new_w, new_h)
    clear_derived_caches(rel_path, thumbs_root, cache_roots)
    preview.unlink(missing_ok=True)

    try:
        version = int(src.stat().st_mtime)
    except OSError:
        version = 0

    return {
        "width": new_w, "height": new_h, "version": version,
        "faces_moved": faces, "scaled": scale_x != 1 or scale_y != 1,
        "original": kept,
    }


def preserve_original(photos_root: Path, rel_path: str) -> str:
    """Keep the photograph as it was before the machine had an opinion.

    Cropping does not do this — trimming a border is a decision about framing,
    and the archive cannot carry a second copy of everything anybody tidies.
    Restoration is different in kind: a model has repainted grain, edges and
    faces, and however good the result, the scan is the record of the print.
    Only the handful actually restored are kept, so the cost is nil.

    Never overwrites: restore a photograph twice and the copy kept is the one
    that came off the scanner, not the first restoration.
    """
    src = photos_root / rel_path
    dst = photos_root / ORIGINALS_DIR / rel_path.removeprefix("archive/")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(src, dst)
    return str(dst.relative_to(photos_root))


def _rescale_faces(src: Path, scale_x: float, scale_y: float) -> int:
    """The restoration nudges the dimensions, so every box moves with them."""
    import json

    faces_file = Path(str(src) + ".faces.json")
    if not faces_file.exists() or (scale_x == 1 and scale_y == 1):
        return 0
    try:
        data = json.loads(faces_file.read_text())
        moved = 0
        for face in data.get("faces", []):
            box = face.get("bbox")
            if not box or len(box) < 4:
                continue
            face["bbox"] = [
                box[0] * scale_x, box[1] * scale_y,
                box[2] * scale_x, box[3] * scale_y,
            ]
            moved += 1
        faces_file.write_text(json.dumps(data, indent=2))
        return moved
    except (json.JSONDecodeError, OSError):
        return 0


def _update_sidecar(src: Path, width: int, height: int) -> None:
    import json

    sidecar = Path(str(src) + ".json")
    if not sidecar.exists():
        return
    try:
        data = json.loads(sidecar.read_text())
        results = data.get("results") or []
        meta = results[0].get("metadata") if results else data.get("metadata")
        if meta:
            dims = (meta.setdefault("media", {})).setdefault("dimensions", {})
            dims["width"], dims["height"] = width, height
            sidecar.write_text(json.dumps(data, indent=2))
    except (json.JSONDecodeError, OSError):
        pass
