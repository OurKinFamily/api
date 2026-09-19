"""The deployed containers mount /photos read-only, so the editing controls —
rotate, crop, tone, restore — cannot work there. /admin/me says so, and the
frontend hides them rather than offering a button that fails.

This exists because the first version read `settings` without importing it in
that module. Every test passed: nothing covered /admin/me at all, and the
error only appeared when a real request hit it.
"""
import os

from app.routers.admin import media_is_writable


def test_a_writable_archive_reports_writable(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "photos_root", tmp_path)
    assert media_is_writable() is True


def test_a_read_only_archive_reports_read_only(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "photos_root", tmp_path)
    os.chmod(tmp_path, 0o500)
    try:
        assert media_is_writable() is False
    finally:
        # Or pytest cannot clean the directory up afterwards.
        os.chmod(tmp_path, 0o700)


def test_a_missing_archive_is_not_writable(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "photos_root", tmp_path / "nothing-here")
    assert media_is_writable() is False
