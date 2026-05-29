"""Backfill: for every `media.deleted` event found in the rotated api
logs, move the file (and its sidecar siblings) into /photos/trash/,
preserving the original folder structure.

This is the ONLY honest source of "things the user actually deleted via
the app" — Neo4j retains no audit trail after DETACH DELETE. Any deletion
older than the logs' retention is lost and can't be backfilled.

Usage:
    venv/bin/python3 scripts/backfill_trash.py            # dry-run (default)
    venv/bin/python3 scripts/backfill_trash.py --dry-run  # explicit, same thing
    venv/bin/python3 scripts/backfill_trash.py --apply    # actually move
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings

LOG_GLOB = str(Path(__file__).resolve().parent.parent / 'logs' / 'api*.jsonl')

SIDECARS = [
    '.json', '.objects.json', '.clip.json', '.scenes.json',
    '.faces.json', '.srt', '.txt', '.poster.jpg', '_plain.txt',
]


def collect_deleted_paths() -> list[str]:
    """Scan every api JSONL log file for `media.deleted` events. Returns
    a deduplicated list of relative paths in the order they were first
    deleted (oldest first)."""
    seen: dict[str, None] = {}  # insertion-ordered set
    for log in sorted(glob.glob(LOG_GLOB)):
        try:
            with open(log) as f:
                for line in f:
                    if 'media.deleted' not in line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    ex = rec.get('record', {}).get('extra', {})
                    if ex.get('event') != 'media.deleted':
                        continue
                    p = ex.get('path')
                    if isinstance(p, str):
                        seen.setdefault(p, None)
        except OSError:
            continue
    return list(seen.keys())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true',
                    help='Actually move files. Without this, dry-run only.')
    ap.add_argument('--dry-run', action='store_true',
                    help='No-op alias — dry-run is the default. Kept for muscle memory.')
    args = ap.parse_args()

    print(f"reading delete events from {LOG_GLOB}...")
    paths = collect_deleted_paths()
    print(f"  {len(paths)} distinct paths recorded as deleted via the app")
    if not paths:
        print()
        print("Nothing to do — no delete events found in retained logs.")
        return

    photos_root = settings.photos_root
    trash_root = photos_root / 'trash'
    moved_files = 0
    moved_sidecars = 0
    missing_files = 0

    print()
    for rel in paths:
        src = photos_root / rel
        action = 'WOULD MOVE' if not args.apply else 'MOVE'
        if not src.exists():
            print(f"  [skip — already gone] {rel}")
            missing_files += 1
            continue

        dst = trash_root / rel
        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                src.rename(dst)
            except OSError as e:
                print(f"  [ERROR] {rel}: {e}")
                continue
        moved_files += 1

        sidecar_moves = []
        for suf in SIDECARS:
            side_src = Path(str(src) + suf)
            if side_src.exists():
                side_dst = Path(str(dst) + suf)
                if args.apply:
                    try:
                        side_src.rename(side_dst)
                    except OSError:
                        continue
                sidecar_moves.append(suf)
                moved_sidecars += 1

        sc_note = f" (+ sidecars: {', '.join(sidecar_moves)})" if sidecar_moves else ""
        print(f"  [{action}] {rel}{sc_note}")

    print()
    verb = "moved" if args.apply else "would move"
    print(f"done. {verb} {moved_files} files + {moved_sidecars} sidecars  ·  {missing_files} already gone")
    if not args.apply:
        print()
        print("  this was a DRY RUN. Re-run with --apply to actually move files.")


if __name__ == '__main__':
    main()
