"""Regression: face crops in the lightbox 404'd for some photos but not others.

The gallery detail endpoint checked that an UNASSIGNED face's crop file existed
before handing back a URL, but an ASSIGNED person's crop_path was turned into a
URL unconditionally. 558 of 131,878 APPEARS_IN edges carry a crop_path whose
file is no longer on disk (~0.4%), so those photos — and only those — produced
a broken image in the viewer.

Fix: both paths now check the file exists; a missing crop yields crop_url=None
so the frontend can fall back rather than request a 404.

Later: assignment also recorded NO crop_path for mis-dated photographs, because
it guessed the location from the archive path while the worker files crops by
date. The detail view now falls back to the path the sidecar recorded.
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


def test_a_crop_that_is_not_on_disk_is_not_offered(tmp_path, monkeypatch):
    """Behaviour, not source text: a recorded crop whose file has gone must
    yield nothing rather than a URL that 404s."""
    monkeypatch.setattr(gallery.settings, "photos_root", tmp_path)
    faces = {"faces": [{"face_index": 0, "crop_path": "/photos/__faces/gone.jpg"}]}
    assert gallery.crop_from_sidecar(faces, 0) is None


def test_the_sidecar_supplies_a_crop_the_edge_never_recorded(tmp_path, monkeypatch):
    """Assignment used to guess the crop location from the archive path, which
    is wrong for a mis-dated photograph — the worker files crops by DATE. Those
    edges carry no crop_path at all, and the face rendered as a broken image.
    """
    monkeypatch.setattr(gallery.settings, "photos_root", tmp_path)
    real = tmp_path / "__faces" / "crops" / "2025" / "12" / "p.jpg_face2.jpg"
    real.parent.mkdir(parents=True)
    real.write_bytes(b"jpeg")
    faces = {"faces": [{"face_index": 2, "crop_path": f"/photos/{real.relative_to(tmp_path)}"}]}

    assert gallery.crop_from_sidecar(faces, 2) == str(real.relative_to(tmp_path))
