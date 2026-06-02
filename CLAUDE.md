# ourkin — api

> Part of the **ourkin stack** at `/home/stephen/Documents/ourkin/`.
> Siblings: `app`, `db`, `workers`, `workshop`, `deploy`. Read the root
> [`../CLAUDE.md`](../CLAUDE.md) for stack-wide conventions and
> [`../CLAUDE.index.md`](../CLAUDE.index.md) for the full repo map.

FastAPI backend for the ourkin family archive.

## GitHub

https://github.com/OurKinFamily/api

## Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # set NEO4J_PASSWORD
uvicorn app.main:app --reload
```

Docs: http://localhost:8000/docs

## Structure

```
app/
  main.py         # app entry point, lifespan, middleware
  config.py       # settings via pydantic-settings, reads .env
  db/
    neo4j.py      # driver init/close, session context manager
  middleware/
    auth.py       # auth placeholder (no-op for now)
  models/
    person.py     # Pydantic response models
  routers/
    people.py     # GET /people, GET /people/{id}, GET /people/{id}/relatives
```

## Current Endpoints

- `GET /people` — list all people
- `GET /people/{id}` — get one person
- `GET /people/{id}/relatives` — parents, children, spouses

## Neo4j Schema

### Media Labels (multi-label pattern)

Every media node carries `:Media` plus one or more sublabels:

```
:Media:Photo          — modern digital/phone image
:Media:Photo:Heritage — scanned/manually catalogued image
:Media:Video          — modern digital/phone video
:Media:Video:Heritage — film reel, VHS, home tape
:Media:Document       — scrapbook pages, yearbooks, school papers
:Media:Audio          — voice recordings, tapes
```

Additional `source` property for fine grain: `"phone"`, `"digital_camera"`, `"film_scan"`, `"vhs"`, `"reel"`, `"document_scan"`

**Key rules:**
- Always `MATCH (m:Media)` for cross-type queries (gallery, places, travel)
- Use sublabel for type-specific: `MATCH (m:Photo)` for images only, `MATCH (m:Video)` for videos only
- Face detection applies to `:Photo` AND `:Video` (frames extracted) — use `:Media` on APPEARS_IN targets
- `is_video` property is legacy — use `:Video` label instead
- `:Heritage` tagged via sidecar signals: `processor=HeritageProcessor` OR `heritage.context.type=family`

### Heritage Detection (from sidecars)

```python
# A media file is heritage if:
processor == "HeritageProcessor"          # old pipeline
OR heritage.context.type == "family"      # mpp-tagged
# NOT: context.type == "other" (mpp default, not real heritage)
```

### Other Node Types

```
:Person           — individuals
:Collection       — heritage document collections (baby books, yearbooks etc.)
:Group            — social groups, organizations
:Suggestion       — AI-generated relationship suggestions
```

### Key Relationships

```
(Person)-[:APPEARS_IN]->(Media)        # depicted/mentioned in a specific media item
(Person)-[:PARENT_OF]->(Person)
(Person)-[:MARRIED_TO]->(Person)
(Person)-[:KNOWS]->(Person)
(Person)-[:MEMBER_OF]->(Group)
(Collection)-[:BELONGS_TO]->(Person)   # collection ownership
(Collection)-[:CONTAINS]->(Document)   # collection → its pages
```

**Ownership vs appearance** — two distinct concepts:
- `Collection-[:BELONGS_TO]->Person` = who owns the whole collection (e.g. Dorothy owns her address book)
- `Person-[:APPEARS_IN]->Document` = who is on a *specific page* (e.g. Patricia's address written in it)

### Documents — Source of Truth

`:Media:Document` nodes are the source of truth for collection pages. The
`/collections/{id}/items` endpoint reads from Document nodes (NOT from disk).
Pages live on disk under `heritage/<person>/<collection>/`, but adding new
pages requires re-running `import_documents.py` to pick them up.

Import scripts (in `/home/stephen/photo-intelligence/`):
- `import_documents.py` — creates Document nodes from collection sidecars
- `tag_stephen_documents.py` — backfills people in untagged sidecars
- `tag_heritage_nodes.py` — adds `:Heritage` to existing Media by sidecar signal
- `patch_null_timestamps.py` — backfills timestamp/is_video from sidecars

### Heritage Video Import (Old Reels / home movies)

`import_heritage_videos.py` (in `/home/stephen/photo-intelligence/`) does the full flow:
1. Archives `.mp4` + `.json` + `.poster.jpg` → `archive/YYYY/MM/` by `contentDate`
2. Creates/merges a `:Collection {type:'home_movies'}` owned by Stephen E. Young
3. Creates `:Media:Video:Heritage {source:'reel'}` nodes (title, notes, place, poster, gps, physical)
4. Edges: `(collection)-[:CONTAINS]->(video)` + `(person)-[:APPEARS_IN]->(video)`

Dry-run by default; `--execute` writes; `--only <substr>` limits to one file.
People names in sidecars must be canonical DB names — `normalize_video_people.py`
rewrites short names (e.g. "Dorothy" → "Dorothy Chooljian") in the sidecars first.

Scrapbook shows a collection if the person **owns it OR appears in any contained media**
(`get_collections` UNIONs both). So home-movie reels surface in every tagged person's
scrapbook, not just the owner's.

### Face Detection on Videos

`services/face-recognition/extract_faces.py <path>` handles single files or dirs.
Videos: `video_face_sidecar.py` samples 8 frames (10–90%), runs InsightFace, dedups
across frames, writes the same `.faces.json` + `__faces/crops/` format as images.
GPU (cu128 / RTX 5060 Ti) currently errors → falls back to CPU automatically (~9s/video).
Detected faces are NOT auto-matched to the `APPEARS_IN` people — that's manual assignment.

## Staging DB Access (dev)

The data-rich DB is **ourkin staging** (`ourkin-graph-staging`):
- Bolt: `bolt://localhost:7688`  ·  Browser: `http://localhost:7475`
- Creds: `neo4j` / `${NEO4J_PASSWORD}`
- Local `api/.env` already points `NEO4J_URI` here.
Prod (`ourkin-graph`, volume `deploy_neo4j_data`) has no host port — query via `docker exec`.

## Logging — required when adding endpoints

**See `LOGGING.md` (api/) for full conventions.** Quick rule:

- Every new mutation endpoint emits one `logger.bind(event="...", by=..., request_id=..., <subject_id>=...).info("...")` after the Cypher write succeeds.
- Bulk writes: `bulk_id = uuid.uuid4().hex[:12]` on the parent + per-item lines.
- Event names: `subject.verb` past tense (`person.created`, `face.assigned`).
- Reads: skip unless surprising / failing.
- After adding a new event name → add a row to the "What to log" table in `LOGGING.md` so Grafana's Edits Feed regex stays exhaustive.

Dashboards already wired: `localhost:3002` → folder *Ourkin* (API Live, Edits Feed, Face Activity, Errors). New events show up automatically — no Grafana edit needed unless adding a new event family prefix.

## Coming Next

- Batch remaining Old Reels (110) + 1987 videos (3) via `import_heritage_videos.py`
- Manual face assignment / clustering for detected video faces
- Real auth middleware
