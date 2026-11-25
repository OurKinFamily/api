"""On-disk archive scanner.

Walks the three media trees (`archive/`, `heritage/`, `staging/`),
counts media files + sidecar presence per directory, and compares the
union of disk paths against the Media nodes in Neo4j to surface drift
(on-disk-not-in-graph and graph-not-on-disk).

Written so it can run from a background thread — the work is plain
filesystem walking + dict bookkeeping, no async needed.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

from neo4j import GraphDatabase

from app.config import settings


# ── Constants ────────────────────────────────────────────────────────────────

MEDIA_EXTS = {
    # images
    '.jpg', '.jpeg', '.png', '.gif', '.heic', '.heif', '.tiff', '.tif',
    '.webp', '.bmp',
    # videos
    '.mp4', '.mov', '.avi', '.mkv', '.mts', '.m2ts', '.m4v', '.webm',
    '.3gp', '.wmv', '.flv', '.mpg', '.mpeg',
    # audio (rare, but staging has voice memos)
    '.m4a', '.aac', '.wav', '.mp3', '.ogg',
}
# RAW formats mpp deliberately ignores in favour of the JPG sibling iPhones
# emit alongside them. Tracked separately so we can surface them as a
# "shadow" stat later without polluting the unindexed drift number — see
# OurKinFamily/app#53.
RAW_EXTS = {
    '.cr2', '.nef', '.arw', '.dng', '.orf', '.rw2',
}

# Trees that live under a scanned root but are not archive media, and are
# deliberately never imported into the graph. They are still real files, so
# they stay in the inventory counts; they are only held out of the DRIFT
# comparison, where they would otherwise sit forever as false positives with
# a suggested fix that would make things worse.
#
#   archive/bios/  illustrations linked from biography prose as
#                  `![caption](archive/bios/<name>/file.jpg)` -- see
#                  app/src/pages/PersonBiography.jsx. Page furniture, not
#                  photographs; ingesting them would put them in the gallery
#                  and in date browsing as if they were family snapshots.
#
# The face pipeline already excludes this same tree for the same reason --
# see `workshop/services/face-recognition/build_embedding_index.py`, which
# skips any sidecar with "bios" in its parts. Keep the two in step.
DRIFT_EXEMPT_PREFIXES = (
    'archive/bios/',
)

# Same idea, but matched as a path SEGMENT anywhere under a root rather than
# as a fixed prefix, because these sit one per collection rather than in one
# place.
#
#   _bw_originals   the black-and-white scans a colourised image was made
#                   from, kept beside the finished version. The colourised
#                   file is what the app shows; the original is kept so the
#                   work can be redone. e.g. the 1918 Signal Corps plates
#                   under the_great_war_wwi/.
DRIFT_EXEMPT_DIRS = (
    '_bw_originals',
)


def _drift_exempt(rel_path: str) -> bool:
    """True for files that are on disk and deliberately not in the graph.

    Prefix match for whole trees, exact segment match for per-collection
    subfolders -- segment and not substring, so `archive/2004/biosphere.jpg`
    and a folder called `_bw_originals_old` are both still drift.
    """
    if rel_path.startswith(DRIFT_EXEMPT_PREFIXES):
        return True
    return any(part in DRIFT_EXEMPT_DIRS for part in rel_path.split('/')[:-1])

# Each media file `foo.jpg` may have sidecar siblings of these shapes.
# Tuple is (suffix-on-disk, internal-key).
# (suffix-on-disk, internal-key, what it applies to, is it expected)
#
# `applies` matters: counting a missing poster against 156k photographs makes
# poster coverage read 14% when every video has one. A percentage needs the
# right denominator or it is just a scary-looking number -- which is why this
# block was collected but never shown on the Health page.
#
# `expected` separates "this should exist and does not" from "this is optional
# and usually absent". Subtitles and transcriptions are produced by hand for a
# few items; reporting them as 0% coverage says nothing.
SIDECAR_SPECS = [
    ('.json',         'mpp',     'all',   True),   # primary mpp metadata
    ('.objects.json', 'objects', 'all',   True),   # YOLO output
    ('.clip.json',    'clip',    'all',   True),   # CLIP embedding
    ('.scenes.json',  'scenes',  'image', True),   # DINOv2 -- stills only in practice
    ('.faces.json',   'faces',   'all',   True),   # InsightFace output
    ('.poster.jpg',   'poster',  'video', True),   # video poster frame
    ('.srt',          'srt',     'video', False),  # ffmpeg-extracted subtitles
    ('.txt',          'txt',     'all',   False),  # whisper plain transcription
]
SIDECAR_SUFFIXES = [(suf, key) for suf, key, _, _ in SIDECAR_SPECS]


def _sidecar_applies(applies: str, kind: str) -> bool:
    """Audio has no sidecar pipeline of its own yet, so it is never eligible."""
    if kind == 'audio':
        return False
    return applies == 'all' or applies == kind

# What we treat as "the three trees". Each maps to a friendly section name.
ROOTS = [
    ('archive',  'archive'),
    ('heritage', 'heritage'),
    ('staging',  'staging'),
]

DRIFT_SAMPLE_LIMIT = 25       # how many sample paths to surface per drift bucket
CACHE_PATH = Path(os.environ.get(
    'DISK_REPORT_JSON',
    str(settings.photos_root / '__data' / 'status' / 'disk-report.json'),
))


# ── Scan ─────────────────────────────────────────────────────────────────────

_SIDECAR_TAILS = tuple(suf.lower() for suf, _ in SIDECAR_SUFFIXES)

# Audio that is narration OF a page rather than a recording in its own right.
# A scanned page gets `<page>.txt` (the transcript) and, where it has been
# generated, a read-aloud `.wav` of that transcript. The .txt is already
# treated as a sidecar; the .wav was not, so every one of them counted as a
# media file the graph was missing -- 145 of them, all in the "Videos" pile.
#
# Matched by its partner, not by extension: staging holds genuine voice
# memos with no image beside them, and those are real media.
_AUDIO_EXTS = {'.wav', '.mp3', '.m4a', '.aac', '.ogg'}
_PAGE_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.heic', '.webp')


def _is_page_narration(name: str, siblings: set[str]) -> bool:
    """True when `name` is audio read from a page image sitting beside it.

    Both naming conventions in the archive are accepted:
        1998-...-page-02.jpg.wav  ->  1998-...-page-02.jpg
        001-front-000.wav         ->  001-front-000.jpg
    """
    stem, ext = os.path.splitext(name)
    if ext.lower() not in _AUDIO_EXTS:
        return False
    if stem in siblings:                       # <image>.<ext>.wav
        return True
    base = os.path.splitext(stem)[0]           # strip a doubled extension
    return any(stem + e in siblings or base + e in siblings
               for e in _PAGE_IMAGE_EXTS)


def _walk_media(root: Path) -> Iterable[Path]:
    """Yield every media file under `root` (case-insensitive suffix match).
    Skips files whose name ends with a known sidecar suffix even when the
    final extension matches a media one — e.g. `foo.mp4.poster.jpg` looks
    like a JPEG by suffix but is really a video poster, not media. Also
    skips read-aloud audio sitting beside the page it narrates (see
    `_is_page_narration`)."""
    for dirpath, _, filenames in os.walk(root):
        siblings = set(filenames)
        for name in filenames:
            ext = Path(name).suffix.lower()
            if ext not in MEDIA_EXTS:
                continue
            lname = name.lower()
            if any(lname.endswith(tail) for tail in _SIDECAR_TAILS):
                continue
            if _is_page_narration(name, siblings):
                continue
            yield Path(dirpath) / name


def _sidecar_present(media: Path, suffix: str) -> tuple[bool, int]:
    """Returns (exists, size_bytes). Size is 0 when missing."""
    side = Path(str(media) + suffix)
    try:
        return True, side.stat().st_size
    except FileNotFoundError:
        return False, 0


def _empty_bucket() -> dict:
    """A fresh per-root counter bag."""
    return {
        'media_total':     0,
        'media_bytes':     0,
        'sidecar_bytes':   0,
        'by_kind':         {'image': 0, 'video': 0, 'audio': 0},
        'by_extension':    {},
        'by_year':         {},
        'sidecars':        {key: {'present': 0, 'missing': 0, 'bytes': 0,
                                  'eligible': 0, 'expected': exp, 'applies': app}
                            for _, key, app, exp in SIDECAR_SPECS},
    }


def _classify_kind(ext: str) -> str:
    if ext in {'.jpg', '.jpeg', '.png', '.gif', '.heic', '.heif', '.tiff',
               '.tif', '.webp', '.bmp', '.cr2', '.nef', '.arw', '.dng',
               '.orf', '.rw2'}:
        return 'image'
    if ext in {'.m4a', '.aac', '.wav', '.mp3', '.ogg'}:
        return 'audio'
    return 'video'


def _year_from_path(rel: Path) -> str | None:
    """The first 4-digit path segment is treated as the year bucket.
    `archive/2024/05/foo.jpg` → `2024`. Returns None if no year found."""
    for part in rel.parts[:3]:
        if len(part) == 4 and part.isdigit():
            return part
    return None


def _neo4j_paths(driver) -> set[str]:
    """All Media.path values from the graph as a set (relative to photos_root,
    same shape the disk walker produces)."""
    paths: set[str] = set()
    with driver.session() as sess:
        for row in sess.run("MATCH (m:Media) WHERE m.path IS NOT NULL RETURN m.path AS p"):
            p = row['p']
            if isinstance(p, str):
                paths.add(p)
    return paths


def _connect_neo4j():
    """Use the same URI/credentials as the rest of the app. Returns the driver
    or None on failure (so a broken graph doesn't break the disk-side scan)."""
    try:
        return GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
    except Exception:
        return None


def build_disk_report() -> dict:
    """Run the scan. Returns a dict matching the API's serialisation contract
    (see /admin/disk-report). Side-effect-free until `cache_disk_report` writes
    the result to disk."""
    started = time.perf_counter()
    root = settings.photos_root

    # Pull graph paths up front; cheap relative to the disk walk.
    driver = _connect_neo4j()
    graph_paths = _neo4j_paths(driver) if driver else set()
    if driver is not None:
        try: driver.close()
        except Exception: pass

    directories: dict[str, dict] = {}
    on_disk_all: set[str] = set()

    for dir_key, sub in ROOTS:
        bucket = _empty_bucket()
        directories[dir_key] = bucket
        scan_root = root / sub
        if not scan_root.exists():
            continue

        for media in _walk_media(scan_root):
            rel = media.relative_to(root)
            rel_str = str(rel)
            on_disk_all.add(rel_str)

            ext = media.suffix.lower()
            bucket['media_total'] += 1
            bucket['by_extension'][ext] = bucket['by_extension'].get(ext, 0) + 1
            bucket['by_kind'][_classify_kind(ext)] += 1

            try:
                bucket['media_bytes'] += media.stat().st_size
            except OSError:
                pass

            yr = _year_from_path(rel)
            if yr:
                bucket['by_year'][yr] = bucket['by_year'].get(yr, 0) + 1

            media_kind = _classify_kind(ext)
            for suffix, key, applies, _expected in SIDECAR_SPECS:
                present, size = _sidecar_present(media, suffix)
                cell = bucket['sidecars'][key]
                cell['bytes'] += size
                if key == 'mpp' and not present and rel_str in graph_paths:
                    # The mpp row answers "is this file known", not "did one
                    # particular tool touch it". Cayce's 2,834 journal pages
                    # come in through import_documents.py, which builds
                    # Document nodes straight from the scan and never writes
                    # an mpp sidecar. Counting the sidecar alone reported
                    # heritage as half-unprocessed when every one of those
                    # pages is in the app and in a collection.
                    present = True
                    cell['via_graph'] = cell.get('via_graph', 0) + 1
                # Count presence always -- the bytes and the file are real --
                # but only let an eligible file move the percentage.
                if present:
                    cell['present'] += 1
                if _sidecar_applies(applies, media_kind):
                    cell['eligible'] += 1
                    if not present:
                        cell['missing'] += 1
            bucket['sidecar_bytes'] += sum(
                bucket['sidecars'][k]['bytes'] for _, k in SIDECAR_SUFFIXES
            ) - bucket['sidecar_bytes']  # incremental

    # Sort year + extension buckets for stable output.
    for bucket in directories.values():
        bucket['by_year'] = dict(sorted(bucket['by_year'].items()))
        bucket['by_extension'] = dict(sorted(
            bucket['by_extension'].items(), key=lambda x: -x[1]
        ))

    # Graph ↔ disk drift.
    # Restrict the comparison to graph paths that *should* live inside the
    # three scanned trees. Anything in `m.path` that points elsewhere
    # (face crops under `__faces/`, audio sidecars, etc.) is out of scope
    # for this scan — comparing it would yield false "missing from disk"
    # entries.
    in_scope_prefixes = tuple(f"{sub}/" for _, sub in ROOTS)
    graph_in_scope = {p for p in graph_paths if p.startswith(in_scope_prefixes)}
    unindexed_all = on_disk_all - graph_in_scope
    # Hold the exempt trees out of drift, but count them so the number is
    # visibly set aside rather than silently vanished.
    drift_exempt = sorted(p for p in unindexed_all if _drift_exempt(p))
    on_disk_not_in_graph = sorted(p for p in unindexed_all if not _drift_exempt(p))
    in_graph_not_on_disk = sorted(graph_in_scope - on_disk_all)

    # Per-root graph counts so the UI can split "production health"
    # (archive + heritage — strict comparison) from "staging" (a
    # workspace where unindexed files are by-design backlog).
    for _, sub in ROOTS:
        if sub in directories:
            prefix = f"{sub}/"
            directories[sub]['graph_total'] = sum(1 for p in graph_paths if p.startswith(prefix))
    def _split_by_root(paths: list[str]) -> dict[str, int]:
        """Bucket drift paths by their top-level dir (archive / heritage /
        staging). Lets the UI answer 'where are the gaps?' without a
        separate query."""
        counts = {root: 0 for _, root in ROOTS}
        for p in paths:
            head = p.split('/', 1)[0]
            if head in counts:
                counts[head] += 1
        return counts

    def _readiness_bucket(rel_path: str) -> str:
        """Classify how `mm archive` would handle this unindexed file
        without actually doing anything. Strictly read-only — opens the
        file to read EXIF, then closes. Cheap (~5 ms per image).

        Returns one of:
            ready          — has GPS + capture date, will ingest cleanly
            needs_no_gps   — has date, missing GPS (needs --allow-no-gps)
            needs_no_date  — has GPS, missing date (needs --allow-no-date)
            video          — not an image; mm has separate handling
            hopeless       — image with neither GPS nor date
            unreadable     — couldn't open / parse EXIF
        """
        full = root / rel_path
        ext = full.suffix.lower()
        # Anything that isn't a still image gets its own bucket — PIL
        # doesn't read EXIF from videos / audio and mm has a separate
        # ingest path for them anyway.
        if ext not in {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.heic', '.heif', '.webp', '.bmp', '.dng', '.cr2', '.nef', '.arw', '.orf', '.rw2'}:
            return 'video'
        try:
            from PIL import Image  # local import keeps disk-report importable on machines without PIL
            with Image.open(full) as img:
                exif = img.getexif()
                # Tag IDs from the EXIF spec:
                #   34853 = GPSInfo, 36867 = DateTimeOriginal, 306 = DateTime
                has_gps  = 34853 in exif and bool(exif.get(34853))
                has_date = bool(exif.get(36867) or exif.get(306))
        except Exception:
            return 'unreadable'

        if has_gps and has_date: return 'ready'
        if has_date:             return 'needs_no_gps'
        if has_gps:              return 'needs_no_date'
        return 'hopeless'

    def _classify_readiness(paths: list[str]) -> dict[str, int]:
        """Bucket-count every path by readiness. Sequential — could be
        parallelised but ~10s on 2.2K is fine for now."""
        buckets = {'ready': 0, 'needs_no_gps': 0, 'needs_no_date': 0,
                   'video': 0, 'hopeless': 0, 'unreadable': 0}
        for p in paths:
            buckets[_readiness_bucket(p)] += 1
        return buckets

    # Readiness only applies to "production" roots (archive + heritage).
    # Staging is a workspace and its files are deliberate backlog — running
    # the EXIF probe on them just inflates the Ready bucket with files no
    # one wants ingested yet.
    PROD_PREFIXES = ('archive/', 'heritage/')
    prod_unindexed = [p for p in on_disk_not_in_graph if p.startswith(PROD_PREFIXES)]
    drift = {
        'on_disk_not_in_graph': {
            'count':     len(on_disk_not_in_graph),
            'samples':   on_disk_not_in_graph[:DRIFT_SAMPLE_LIMIT],
            'by_root':   _split_by_root(on_disk_not_in_graph),
            'readiness': _classify_readiness(prod_unindexed),
        },
        'in_graph_not_on_disk': {
            'count':    len(in_graph_not_on_disk),
            'samples':  in_graph_not_on_disk[:DRIFT_SAMPLE_LIMIT],
            'by_root':  _split_by_root(in_graph_not_on_disk),
        },
        'exempt': {
            'count':     len(drift_exempt),
            'by_prefix': {
                **{prefix: sum(1 for p in drift_exempt if p.startswith(prefix))
                   for prefix in DRIFT_EXEMPT_PREFIXES},
                **{name: sum(1 for p in drift_exempt
                             if name in p.split('/')[:-1])
                   for name in DRIFT_EXEMPT_DIRS},
            },
            'reason': 'On disk and deliberately not in the graph. Not drift.',
        },
        'on_disk_total':  len(on_disk_all),
        'in_graph_total': len(graph_paths),
    }

    elapsed = round(time.perf_counter() - started, 2)
    return {
        'generated':         datetime.now().isoformat(timespec='seconds'),
        'duration_seconds':  elapsed,
        'photos_root':       str(root),
        'directories':       directories,
        'drift':             drift,
    }


def cache_disk_report(report: dict) -> Path:
    """Persist the most recent scan to disk so /admin/disk-report can serve it
    without re-scanning every request."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(report, indent=2))
    return CACHE_PATH


def read_cached_report() -> dict | None:
    if not CACHE_PATH.exists():
        return None
    try:
        return json.loads(CACHE_PATH.read_text())
    except Exception:
        return None
