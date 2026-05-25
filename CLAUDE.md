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
(Person)-[:APPEARS_IN]->(Media)
(Person)-[:PARENT_OF]->(Person)
(Person)-[:MARRIED_TO]->(Person)
(Person)-[:KNOWS]->(Person)
(Person)-[:MEMBER_OF]->(Group)
(Collection)-[:BELONGS_TO]->(Person)
```

## Coming Next

- `:Media:Video:Heritage` nodes for Old Reels + 1987 home videos
- `:Media:Document` nodes for Collection items
- Real auth middleware
