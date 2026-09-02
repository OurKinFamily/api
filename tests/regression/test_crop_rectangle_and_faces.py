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

from app.services.crop import (
    MIN_SIDE, align_inward, crop_bbox, crop_media, stored_rect,
)

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


class TestUnalignedRectangles:
    """jpegtran snaps the origin back to the MCU grid, so an arbitrary
    rectangle starts a few pixels earlier than asked. Compensating for that in
    DISPLAY space is wrong for an oriented file — the snapping happens on the
    stored axes, which are not the displayed ones — and the boxes came out
    slightly off every face.
    """

    @pytest.mark.parametrize("orientation", ORIENTATIONS)
    def test_a_face_box_still_lands_on_the_face(self, tmp_path, orientation):
        from PIL import ImageOps

        def red_bbox(img):
            px = img.convert("RGB").load()
            w, h = img.size
            xs, ys = [], []
            for j in range(h):
                for i in range(w):
                    r, g, b = px[i, j]
                    if r > 180 and g < 80 and b < 80:
                        xs.append(i)
                        ys.append(j)
            return [min(xs), min(ys), max(xs) + 1, max(ys) + 1] if xs else None

        im = Image.new("RGB", (640, 480), (20, 20, 20))
        ImageDraw.Draw(im).rectangle([280, 200, 360, 280], fill=(255, 0, 0))
        exif = im.getexif()
        exif[274] = orientation
        path = tmp_path / "p.jpg"
        im.save(path, quality=95, exif=exif)

        faces = tmp_path / "p.jpg.faces.json"
        with Image.open(path) as src:
            faces.write_text(json.dumps({"faces": [
                {"face_index": 0, "bbox": red_bbox(ImageOps.exif_transpose(src))},
            ]}))

        # Deliberately off the 16px grid.
        crop_media(tmp_path, path.name, (101, 53, 400, 340))

        moved = json.loads(faces.read_text())["faces"][0]["bbox"]
        with Image.open(path) as out:
            actual = red_bbox(ImageOps.exif_transpose(out))
        assert max(abs(a - b) for a, b in zip(actual, moved)) <= 2


class TestSmallTrims:
    """The case that made crop look broken.

    jpegtran snaps an unaligned origin BACKWARD and keeps the requested extent,
    so it hands back the very pixels you asked it to remove. Trimming a 9px
    border returned the border: the crop reported success, the photograph was
    unchanged, and from the outside it looked like a button that did nothing.
    The origin is snapped forward now, cutting up to 15px more than asked —
    the right direction when the point is to remove a border.
    """

    def test_a_nine_pixel_border_actually_goes(self, tmp_path):
        from PIL import ImageOps

        im = Image.new("RGB", (896, 910), (0, 0, 0))
        ImageDraw.Draw(im).rectangle([9, 8, 886, 901], fill=(230, 230, 230))
        path = tmp_path / "b.jpg"
        im.save(path, quality=95)

        result = crop_media(tmp_path, path.name, (9, 8, 878, 894))

        with Image.open(path) as out:
            shown = ImageOps.exif_transpose(out)
            px = shown.convert("RGB").load()
            corners = [
                px[0, 0], px[shown.width - 1, 0],
                px[0, shown.height - 1], px[shown.width - 1, shown.height - 1],
            ]
        assert not [c for c in corners if sum(c) < 200], "border survived the crop"
        assert result["lossless"]

    def test_the_origin_moves_forward_not_back(self):
        # Asking to start at 9 must never start at 0, which is what returned
        # the trimmed pixels.
        x, y, w, h = align_inward((9, 8, 887, 902))
        assert (x, y) == (16, 16)
        assert x + w == 9 + 887 and y + h == 8 + 902   # far edge respected


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
