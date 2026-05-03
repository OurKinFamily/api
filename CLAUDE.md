# mykin — api

FastAPI backend for the mykin family archive.

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

## Coming Next

- POST/PUT/DELETE for people
- Event, Place, Media endpoints
- Real auth middleware
