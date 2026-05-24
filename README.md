# ourkin / api

FastAPI backend for the Ourkin family archive. Connects to Neo4j for the people/relationship graph.

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

The staging Neo4j is exposed on `localhost:7688` — use that for local dev so you're working against real data without spinning up a separate database.

Copy the example env file:

```bash
cp .env.local.example .env
# fill in NEO4J_PASSWORD
```

Or set manually:

```bash
NEO4J_URI=bolt://localhost:7688
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password
```

Start the server:

```bash
uvicorn app.main:app --reload
```

API available at http://localhost:8000. Docs at http://localhost:8000/docs.

## Testing

```bash
pytest
```

Tests use mocked Neo4j sessions — no running database required.

## CI/CD & Branch Flow

```
feature/* → PR → staging → PR → main
```

- PRs target `staging` by default
- Merging to `staging` → tests → builds `ghcr.io/ourkinfamily/api:staging` → deploys to staging.ourkin.family
- Merging to `main` → tests → builds `ghcr.io/ourkinfamily/api:latest` → deploys to www.ourkin.family
- PRs to `main` are blocked unless the source branch is `staging`

Both branches are protected — no direct pushes. `staging` allows admin bypass for emergencies.
