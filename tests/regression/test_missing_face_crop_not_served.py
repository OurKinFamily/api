"""Regression: face crops in the lightbox 404'd for some photos but not others.

The gallery detail endpoint checked that an UNASSIGNED face's crop file existed
before handing back a URL, but an ASSIGNED person's crop_path was turned into a
URL unconditionally. 558 of 131,878 APPEARS_IN edges carry a crop_path whose
file is no longer on disk (~0.4%), so those photos — and only those — produced
a broken image in the viewer.

Fix: both paths now check the file exists; a missing crop yields crop_url=None
so the frontend can fall back rather than request a 404.
"""
from pathlib import Path

import pytest

from app.routers import gallery


def _crop_url_for(crop_path, exists: bool, monkeypatch):
    """Run the assigned-person branch with a controlled filesystem."""
    monkeypatch.setattr(Path, "exists", lambda self: exists)
    person = {"face_index": 0, "crop_path": crop_path}
    cp = person.get("crop_path")
    return (
        gallery.with_v(f"/api/media/{cp}")
        if cp and (gallery.settings.photos_root / cp).exists()
        else None
    )


@pytest.mark.parametrize("crop_path", ["__faces/crops/2026/07/photo.jpg_face2.jpg"])
def test_assigned_face_with_missing_crop_gets_no_url(crop_path, monkeypatch):
    assert _crop_url_for(crop_path, exists=False, monkeypatch=monkeypatch) is None


@pytest.mark.parametrize("crop_path", ["__faces/crops/2026/07/photo.jpg_face2.jpg"])
def test_assigned_face_with_present_crop_gets_a_url(crop_path, monkeypatch):
    url = _crop_url_for(crop_path, exists=True, monkeypatch=monkeypatch)
    assert url is not None and crop_path in url


def test_assigned_face_without_a_crop_path_gets_no_url(monkeypatch):
    assert _crop_url_for(None, exists=True, monkeypatch=monkeypatch) is None


def test_source_checks_existence_for_assigned_faces():
    """Guard the actual endpoint, not just the extracted logic above — this is
    the line that regressed."""
    src = Path(gallery.__file__).read_text()
    marker = 'if cp and (settings.photos_root / cp).exists()'
    assert marker in src, "assigned-face crop_url must be gated on the file existing"
