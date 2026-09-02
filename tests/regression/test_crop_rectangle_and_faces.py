"""Cropping a photograph, which is destructive and keeps no copy.

Rotation is reversible — four quarter-turns and the file is bit-for-bit where
it started. A crop is not, and nothing is preserved: at 1.3TB the archive
cannot carry a second copy of everything anybody trims.

The rectangle is drawn on the photograph as DISPLAYED. A portrait phone photo
is stored landscape with an EXIF orientation tag, so cropping "the top" without
mapping the rectangle back through that tag takes a strip off the side instead.
"""
import json

import pytest
from PIL import Image, ImageChops, ImageDraw, ImageOps

from app.services.crop import MIN_SIDE, crop_bbox, crop_media, stored_rect

ORIENTATIONS = (1, 2, 3, 4, 5, 6, 7, 8)
# MCU-aligned, so jpegtran is exact rather than rounding outward.
RECT = (96, 48, 320, 240)


def write_jpeg(path, orientation=1):
    """Four distinct quadrants: a wrong mapping lands on the wrong colours."""
    im = Image.new("RGB", (640, 480), (20, 20, 20))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 319, 239], fill=(255, 0, 0))
    d.rectangle([320, 0, 639, 239], fill=(0, 255, 0))
    d.rectangle([0, 240, 319, 479], fill=(0, 0, 255))
    d.rectangle([320, 240, 639, 479], fill=(255, 255, 0))
    exif = im.getexif()
    exif[274] = orientation
    im.save(path, quality=95, exif=exif)
    return path


def difference(a, b):
    if a.size != b.size:
        return 255.0
    import statistics
    return statistics.mean(ImageChops.difference(a, b).getdata(0))


class TestTheRectangleIsWhatWasDrawn:
    @pytest.mark.parametrize("orientation", ORIENTATIONS)
    def test_crops_what_the_reader_sees(self, tmp_path, orientation):
        path = write_jpeg(tmp_path / "p.jpg", orientation)
        x, y, w, h = RECT
        with Image.open(path) as im:
            expected = ImageOps.exif_transpose(im).crop((x, y, x + w, y + h))

        result = crop_media(tmp_path, path.name, RECT)

        with Image.open(path) as im:
            actual = ImageOps.exif_transpose(im)
        assert difference(actual, expected) < 5
        assert (result["width"], result["height"]) == (w, h)
        assert result["lossless"]

    def test_upright_files_map_straight_through(self):
        assert stored_rect((10, 20, 30, 40), 640, 480, 1) == (10, 20, 30, 40)

    def test_a_portrait_phone_photo_maps_onto_its_landscape_file(self):
        # Orientation 6: displayed 3024x4032, stored 4032x3024. The displayed
        # top-left corner lives at the bottom-left of the file.
        assert stored_rect((0, 0, 100, 200), 3024, 4032, 6) == (0, 2924, 200, 100)


class TestItIsDestructive:
    """No copy is kept. Worth a test, because the opposite was true for an
    afternoon and a reader of this file could reasonably assume it still is."""

    def test_no_original_is_squirrelled_away(self, tmp_path):
        photos = tmp_path / "photos"
        (photos / "archive").mkdir(parents=True)
        write_jpeg(photos / "archive" / "p.jpg")

        result = crop_media(photos, "archive/p.jpg", RECT)

        assert "original" not in result
        assert not (photos / "originals").exists()
        with Image.open(photos / "archive" / "p.jpg") as im:
            assert im.size == (320, 240)


class TestFaces:
    def test_boxes_move_with_the_picture(self):
        assert crop_bbox([120, 80, 220, 180], 100, 50, 400, 300) == [20, 30, 120, 130]

    def test_a_face_cropped_away_is_dropped(self):
        assert crop_bbox([10, 10, 40, 40], 100, 100, 200, 200) is None

    def test_a_sliver_of_a_face_is_dropped_rather_than_clamped(self):
        """A tenth of somebody's ear is not a face, and boxing it makes the
        panel look broken."""
        assert crop_bbox([90, 90, 190, 190], 180, 180, 200, 200) is None

    def test_the_sidecar_is_rewritten(self, tmp_path):
        path = write_jpeg(tmp_path / "p.jpg")
        faces = tmp_path / "p.jpg.faces.json"
        faces.write_text(json.dumps({"faces": [
            {"face_index": 0, "bbox": [200, 100, 300, 200]},   # inside
            {"face_index": 1, "bbox": [0, 0, 40, 40]},         # cropped away
        ]}))

        result = crop_media(tmp_path, path.name, RECT)

        assert (result["faces_kept"], result["faces_dropped"]) == (1, 1)
        kept = json.loads(faces.read_text())["faces"]
        assert [f["face_index"] for f in kept] == [0]
        assert kept[0]["bbox"] == [104, 52, 204, 152]


class TestRefusals:
    def test_a_crop_smaller_than_a_thumbnail_is_a_mistake(self, tmp_path):
        path = write_jpeg(tmp_path / "p.jpg")
        with pytest.raises(ValueError):
            crop_media(tmp_path, path.name, (0, 0, MIN_SIDE - 1, MIN_SIDE - 1))

    def test_cropping_to_the_whole_photograph_is_refused(self, tmp_path):
        path = write_jpeg(tmp_path / "p.jpg")
        with pytest.raises(ValueError):
            crop_media(tmp_path, path.name, (0, 0, 640, 480))

    def test_a_missing_file_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            crop_media(tmp_path, "nope.jpg", RECT)
