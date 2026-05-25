# ourkin — api

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

Import scripts (in photo-intelligence repo):
- `import_documents.py` — creates Document nodes from collection sidecars
- `tag_stephen_documents.py` — backfills people in untagged sidecars
- `tag_heritage_nodes.py` — adds `:Heritage` to existing Media by sidecar signal

## Coming Next

- `:Media:Video:Heritage` nodes for Old Reels + 1987 home videos
- Real auth middleware
