# ourkin / api

FastAPI backend for the Ourkin family archive. Connects to Neo4j for the people/relationship graph.

## Running locally

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
```

Requires a running Neo4j instance. Start one from `../db/graph/`:

```bash
cd ../db/graph
docker compose up -d
bash load.sh   # load schema + seed data (first time only)
```

Set env vars (copy from `.env.example` or create `.env`):

```bash
NEO4J_URI=bolt://localhost:7687
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

## CI/CD

Push to `staging` → runs tests → builds `ghcr.io/ourkinfamily/api:staging` → deploys to staging.ourkin.family  
Push to `main` → runs tests → builds `ghcr.io/ourkinfamily/api:latest` → deploys to www.ourkin.family
