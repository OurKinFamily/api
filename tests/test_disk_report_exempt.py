"""Drift exemption for trees that are on disk and meant to stay out of the graph.

`archive/bios/` holds illustrations linked from biography prose as
`![caption](archive/bios/<name>/file.jpg)` (see PersonBiography.jsx). They are
page furniture, deliberately never imported. Before this, the Health page
counted all 27 as drift and offered to ingest them — which would have put
Edward's diner in the gallery dated 1970-01-01.

They still belong in the inventory (they are real files taking real space),
so the exemption applies to the drift comparison only.
"""
import os
from pathlib import Path

import pytest

from app.services import disk_report
from app.services.disk_report import (
    DRIFT_EXEMPT_DIRS,
    DRIFT_EXEMPT_PREFIXES,
    build_disk_report,
)


def make_tree(root: Path, rel_paths):
    """Create empty media files at each relative path under `root`."""
    for rel in rel_paths:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg")
    return root


@pytest.fixture
def photos(tmp_path, monkeypatch):
    """A miniature /photos with one archive year, one bios dir, one heritage."""
    root = tmp_path / "photos"
    make_tree(root, [
        "archive/2004/09/real-one.jpg",
        "archive/2004/09/real-two.jpg",
        "archive/bios/edward/diner-aerial.jpg",
        "archive/bios/edward/diner-exterior-1.jpg",
        "archive/bios/bagdasar/portrait.jpg",
        "heritage/someone/scan.jpg",
        "heritage/someone/the_great_war/_bw_originals/1918-plate.jpg",
        "heritage/someone/_bw_originals_old/not-exempt.jpg",
    ])
    # build_disk_report() takes no arguments -- it reads settings.photos_root
    # and opens its own driver, so both are redirected here.
    monkeypatch.setattr(disk_report.settings, "photos_root", root)
    # No Neo4j in tests: an empty graph makes every file "on disk, not in
    # the graph", which is exactly the condition under test.
    monkeypatch.setattr(disk_report, "_connect_neo4j", lambda: None)
    return root


class TestDriftExemption:

    def test_bios_is_exempt_and_counted_separately(self, photos):
        rep = build_disk_report()
        drift = rep["drift"]

        unindexed = drift["on_disk_not_in_graph"]["samples"]
        assert not any("archive/bios/" in p for p in unindexed), \
            "bios images are still being reported as drift"

        assert drift["exempt"]["by_prefix"]["archive/bios/"] == 3

    def test_real_media_is_still_drift(self, photos):
        """The exemption must not swallow anything outside the exempt tree."""
        drift = build_disk_report()["drift"]
        unindexed = set(drift["on_disk_not_in_graph"]["samples"])
        assert "archive/2004/09/real-one.jpg" in unindexed
        assert "archive/2004/09/real-two.jpg" in unindexed
        assert "heritage/someone/scan.jpg" in unindexed
        assert drift["on_disk_not_in_graph"]["count"] == 4

    def test_exempt_files_stay_in_the_inventory(self, photos):
        """They are real files on disk and should still be counted as such."""
        rep = build_disk_report()
        # 5 under archive/ — 2 real + 3 bios
        assert rep["directories"]["archive"]["media_total"] == 5

    def test_readiness_does_not_probe_exempt_files(self, photos):
        """Bios images were the whole of the green 'Ready' pile."""
        readiness = build_disk_report()["drift"] \
            ["on_disk_not_in_graph"]["readiness"]
        assert sum(readiness.values()) == 4, \
            "readiness should only classify the non-exempt files"

    def test_prefix_list_matches_the_face_pipeline(self, photos):
        """Both trees exclude bios; if one moves the other has to follow."""
        assert "archive/bios/" in DRIFT_EXEMPT_PREFIXES

    def test_a_path_merely_containing_bios_is_not_exempt(self, photos):
        """Prefix match, not substring — `archive/2004/biostats.jpg` is media."""
        (photos / "archive/2004/09/biosphere-trip.jpg").write_bytes(b"\xff\xd8\xff")
        drift = build_disk_report()["drift"]
        assert "archive/2004/09/biosphere-trip.jpg" in \
            set(drift["on_disk_not_in_graph"]["samples"])
        assert drift["exempt"]["count"] == 4

    def test_bw_originals_is_exempt_anywhere_under_heritage(self, photos):
        """Source scans sit one per collection, so it is a segment not a prefix."""
        drift = build_disk_report()["drift"]
        unindexed = set(drift["on_disk_not_in_graph"]["samples"])
        assert "heritage/someone/the_great_war/_bw_originals/1918-plate.jpg" \
            not in unindexed
        assert drift["exempt"]["by_prefix"]["_bw_originals"] == 1

    def test_a_similarly_named_dir_is_not_exempt(self, photos):
        """Exact segment match — `_bw_originals_old` is still drift."""
        drift = build_disk_report()["drift"]
        assert "heritage/someone/_bw_originals_old/not-exempt.jpg" in \
            set(drift["on_disk_not_in_graph"]["samples"])

    def test_heritage_is_not_blanket_exempt(self, photos):
        """Heritage is a system of record; only its source-scan dirs are exempt."""
        drift = build_disk_report()["drift"]
        assert "heritage/someone/scan.jpg" in \
            set(drift["on_disk_not_in_graph"]["samples"])
        assert "_bw_originals" in DRIFT_EXEMPT_DIRS


class TestPageNarration:
    """Read-aloud audio beside a scanned page is a sidecar, not media.

    regression(2026-11-24): 145 .wav files — TTS readings of scrapbook page
    transcripts — were counted as media the graph was missing. They filled
    the Health page's "Videos" pile and could never be resolved, because
    there is nothing to import. Their .txt siblings were already treated as
    sidecars; the .wav was not.
    """

    @pytest.fixture
    def photos(self, tmp_path, monkeypatch):
        root = tmp_path / "photos"
        make_tree(root, [
            # convention A: <image>.<ext>.wav
            "heritage/p/journal/page-02.jpg",
            "heritage/p/journal/page-02.jpg.wav",
            # convention B: <stem>.wav beside <stem>.jpg
            "heritage/p/photos/001-front-000.jpg",
            "heritage/p/photos/001-front-000.wav",
            # a real voice memo — no image beside it, so it IS media
            "staging/voice-memo-2024.wav",
        ])
        monkeypatch.setattr(disk_report.settings, "photos_root", root)
        monkeypatch.setattr(disk_report, "_connect_neo4j", lambda: None)
        return root

    def test_narration_is_not_counted_as_media(self, photos):
        rep = build_disk_report()
        # 2 page images, neither .wav
        assert rep["directories"]["heritage"]["media_total"] == 2

    def test_both_naming_conventions_are_caught(self, photos):
        drift = build_disk_report()["drift"]["on_disk_not_in_graph"]
        assert not any(p.endswith(".wav") and p.startswith("heritage/")
                       for p in drift["samples"])

    def test_a_voice_memo_with_no_image_is_still_media(self, photos):
        """Staging holds real recordings; only paired audio is a sidecar."""
        rep = build_disk_report()
        assert rep["directories"]["staging"]["media_total"] == 1
        assert "staging/voice-memo-2024.wav" in \
            set(rep["drift"]["on_disk_not_in_graph"]["samples"])

    def test_helper_is_exact_about_the_partner(self):
        sibs = {"page-02.jpg", "page-02.jpg.wav", "other.wav"}
        assert disk_report._is_page_narration("page-02.jpg.wav", sibs)
        assert not disk_report._is_page_narration("other.wav", sibs)
        assert not disk_report._is_page_narration("page-02.jpg", sibs)
