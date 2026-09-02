"""Regression: rotating a mirrored photograph turned it the wrong way.

jpegtran rewrites the stored pixels and leaves the EXIF orientation tag alone,
so a viewer applies that tag afterwards. The old code always turned the pixels
by the requested amount, which is right only because rotations commute — and a
mirrored orientation is not a rotation. For orientations 2, 4, 5 and 7 the
photograph came out 180 degrees from where it was asked to go.

Writing `displayed = O(stored)`, the pixels need `O-inverse . R . O`, which in
the dihedral group collapses to a plain rotation: with the request when O is a
pure rotation, against it when O contains a mirror.

The sidecar lied about this too: it recorded "Horizontal (normal)" after every
rotate, while the file's tag was untouched.
"""
import statistics

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageOps

from app.services.rotate import ORIENTATION_NAMES, rotate_media, stored_rotation

MIRRORED = (2, 4, 5, 7)
PURE = (1, 3, 6, 8)


def write_jpeg(path, orientation):
    """A 64x32 image whose corners differ, so a wrong turn cannot pass.

    Dimensions are multiples of 16 so jpegtran's -perfect will accept them;
    it refuses rather than discarding edge pixels, which is the right call for
    an archive but makes odd sizes fall back to re-encoding.
    """
    im = Image.new("RGB", (64, 32), (20, 20, 20))
    draw = ImageDraw.Draw(im)
    draw.rectangle([0, 0, 15, 7], fill=(255, 0, 0))
    draw.rectangle([56, 24, 63, 31], fill=(0, 255, 0))
    exif = im.getexif()
    exif[274] = orientation
    im.save(path, quality=95, exif=exif)
    return path


def mean_difference(a, b):
    """JPEG is lossy, so exact equality never holds. A correct turn scores
    fractionally; a wrong one puts the markers in other corners and scores 30+.
    """
    if a.size != b.size:
        return 255.0
    return statistics.mean(ImageChops.difference(a, b).getdata(0))


class TestStoredRotation:
    def test_pure_rotations_turn_with_the_request(self):
        assert all(stored_rotation(90, o) == 90 for o in PURE)

    def test_mirrored_orientations_turn_against_it(self):
        assert all(stored_rotation(90, o) == 270 for o in MIRRORED)

    def test_half_turn_is_its_own_inverse(self):
        assert all(stored_rotation(180, o) == 180 for o in MIRRORED + PURE)


class TestRotatingRealFiles:
    @pytest.mark.parametrize("orientation", PURE + MIRRORED)
    def test_the_viewer_sees_a_quarter_turn(self, tmp_path, orientation):
        path = write_jpeg(tmp_path / "p.jpg", orientation)
        with Image.open(path) as im:
            expected = ImageOps.exif_transpose(im).rotate(-90, expand=True)

        result = rotate_media(tmp_path, path.name, 90)

        with Image.open(path) as im:
            actual = ImageOps.exif_transpose(im)
        assert mean_difference(actual, expected) < 5
        assert (result["width"], result["height"]) == expected.size

    def test_every_derived_cache_is_dropped(self, tmp_path):
        """A thumbnail regenerates on demand, so deleting it is the whole fix
        — but there are three caches and only one used to be cleared. The grid
        went on showing the old thumbnail after a rotate.
        """
        photos = tmp_path / "photos"
        (photos / "archive" / "2026").mkdir(parents=True)
        write_jpeg(photos / "archive" / "2026" / "p.jpg", 1)

        key = "2026/p.jpg.webp"
        roots = [photos / "__thumbs", tmp_path / "thumbs", tmp_path / "medium"]
        for root in roots:
            (root / key).parent.mkdir(parents=True, exist_ok=True)
            (root / key).write_bytes(b"stale")

        rotate_media(
            photos, "archive/2026/p.jpg", 90,
            thumbs_root=roots[0], cache_roots=tuple(roots[1:]),
        )

        assert not any((root / key).exists() for root in roots)

    def test_version_is_the_new_modification_time(self, tmp_path):
        path = write_jpeg(tmp_path / "p.jpg", 1)
        before = path.stat().st_mtime
        assert rotate_media(tmp_path, path.name, 90)["version"] >= int(before)

    def test_sidecar_reports_the_tag_the_file_actually_carries(self, tmp_path):
        import json
        path = write_jpeg(tmp_path / "p.jpg", 6)
        sidecar = tmp_path / "p.jpg.json"
        sidecar.write_text(json.dumps({"metadata": {"media": {"dimensions": {}}}}))

        rotate_media(tmp_path, path.name, 90)

        dims = json.loads(sidecar.read_text())["metadata"]["media"]["dimensions"]
        # jpegtran leaves the tag alone, so it is still 6 — claiming upright
        # would leave the sidecar contradicting the photograph.
        assert dims["orientation"] == ORIENTATION_NAMES[6]
        # Orientation 6 already displays the 64x32 file as 32x64, so a
        # quarter turn puts it back to 64x32.
        assert (dims["width"], dims["height"]) == (64, 32)
