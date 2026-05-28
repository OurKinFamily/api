# API Logging Conventions

Loguru + JSONL + Loki/Promtail/Grafana. This file is the contract: read it
before adding new endpoints, follow it without re-deriving from scratch.

## The pipe

```
api code → loguru → stdout (terminal/Docker)
                  → logs/api.jsonl (rotated daily, 14d retention)
                                          ↓
                                       Promtail (tails JSONL)
                                          ↓
                                        Loki (store)
                                          ↓
                                       Grafana (query)
```

Same Grafana for dev/staging/prod; filter by the `env` label.

## Using the logger

```python
from app.log import logger

# Reads: a single line is usually enough.
logger.bind(event="person.read", person_id=pid, by=user).debug("person fetched")

# Mutations: ALWAYS log on success (after the Cypher write).
logger.bind(event="person.created", person_id=p["id"], by=user).info("person created")

# Errors caught + handled: log at WARN, include the exception.
logger.bind(event="person.create.failed", by=user).warning(f"validation failed: {exc}")
```

`request.state.user_email` and `request.state.request_id` are set by
`RequestLogMiddleware` for every HTTP request. Bind them:

```python
async def my_route(request: Request, ...):
    log = logger.bind(
        by=request.state.user_email,
        request_id=request.state.request_id,
    )
    ...
    log.bind(event="thing.done", thing_id=tid).info("thing done")
```

## Event naming

`subject.verb` — past tense for completed actions.

Good:
- `person.created`, `person.updated`, `person.deleted`
- `media.redated`, `media.deleted`
- `face.assigned`, `face.unassigned`, `face.bulk_assigned`
- `album.media.added`, `album.media.removed`
- `favorite.added`, `favorite.removed`
- `cluster.assigned`, `cluster.skipped`
- `suggestion.accepted`, `suggestion.rejected`
- `job.started`, `job.completed`, `job.failed`

Bad:
- `created_person` (noun-first, hard to query)
- `creating person` (present tense — reserve for in-flight events only)
- `do_thing` (generic verb)
- `success` (no subject)

## Required fields per event

Every mutation event MUST include:

| Field | Why |
|---|---|
| `event` | The verb. Lets Grafana group by event type. |
| `by` | User email (from `request.state.user_email`). Audit trail. |
| primary id | `person_id`, `path`, `cluster_id`, `album_id`, etc. — whatever identifies the subject. |

Every event SHOULD include (when applicable):

| Field | Why |
|---|---|
| `request_id` | Correlate multi-line events in the same request. |
| `bulk_id` | Group lines from a bulk operation (face.assigned x 500 share one bulk_id). |
| `count` | For bulk ops: include on the summary line. |
| `old` / `new` | For updates: enough to reconstruct the change. |
| `source` | Where the action originated (e.g. `cluster_assign` vs `search_assign`). |

## What to log (the must list)

Per router, every mutation endpoint emits one terminal event:

| Router | Endpoints → Events |
|---|---|
| `people` | POST → `person.created`, PATCH → `person.updated`, DELETE → `person.deleted`, POST /relationships → `relationship.created`, DELETE /relationships → `relationship.deleted`, POST /merge → `person.merged` |
| `media` | (currently read-mostly; if any mutations exist, add them) |
| `gallery` | DELETE → `media.deleted`, PATCH → `media.redated` ✓ |
| `faces` | POST /search/assign → `face.assigned` per face ✓, DELETE /assignment → `face.unassigned`, POST /clusters/{id}/assign → `cluster.assigned` + per-face `face.assigned`, POST /clusters/{id}/skip → `cluster.skipped` |
| `albums` | POST → `album.created`, PATCH → `album.updated`, DELETE → `album.deleted`, POST /media → `album.media.added` per path, DELETE /media → `album.media.removed` per path |
| `me` | PUT /favorites → `favorite.added`, DELETE /favorites → `favorite.removed` |
| `heritage` | every collection / page CRUD endpoint |
| `groups` | every group CRUD + member add/remove |
| `suggestions` | POST accept/reject/dismiss → `suggestion.{accept,reject,dismiss}ed` |
| `places` | (read-only currently) |
| `admin` | PUT /avatar → `person.avatar_updated`, etc. |
| `jobs` | START/COMPLETE/FAIL events |

## What NOT to log

- **Health pings / hot polling** — `GET /jobs/{id}/status` called every 2s
  by the UI. Floods Loki. Either skip (don't log read endpoints) or
  filter in Promtail.
- **Static assets / 404s on missing files** — noise floor.
- **Large payloads in fields** — full image bytes, full embedding vectors,
  full Cypher result sets. Log counts and ids, not contents.
- **Secrets** — tokens, passwords, raw CF JWTs. Easy to leak via logs.
- **Per-row reads** — fetching a list of 100 people = ONE event with
  `count=100`, not 100 events. (The exception: bulk *writes* — those
  should log per item so each mutation is traceable.)

## Levels

- `DEBUG` — reads, internal noise, off in prod
- `INFO` — every successful state change (mutations)
- `WARNING` — recoverable failures, validation rejections, "user asked
  for something we can't serve" cases
- `ERROR` — unhandled exceptions (auto-captured by the middleware), DB
  outages, anything that needs operator attention

## Bulk operations

Bulk writes get two patterns:

```python
import uuid
bulk_id = uuid.uuid4().hex[:12]
log.bind(event="face.bulk_assign.started", count=len(faces), bulk_id=bulk_id).info("bulk started")

for f in faces:
    # ... actual cypher write ...
    log.bind(
      event="face.assigned",
      person_id=person_id,
      photo_path=f.path,
      face_index=f.face_index,
      bulk_id=bulk_id,        # ← correlates back to the parent line
    ).info("face assigned")
```

Then in Grafana:

```logql
{job="api"} | json | bulk_id="abc1234567ab"     # every line of that batch
{job="api"} | json | event="face.assigned" | person_id="p1"   # all faces ever assigned to p1
```

## Auditing a person's edit history

```logql
# Every change anyone made to this person, last 30 days
{job="api", env="prod"} | json | person_id="abc-uuid"
```

## Auditing your own activity

```logql
{job="api"} | json | by="stephenyoung7267@gmail.com" | level="info"
```

## Adding a new endpoint

1. Bind `by` + `request_id` from `request.state` at the top of the handler.
2. After every successful Cypher mutation, emit one `.info()` with the
   `event` + subject id(s).
3. After every caught exception that doesn't re-raise, emit one
   `.warning()` with the exception + context.
4. Don't log on the read path unless something's surprising.
5. If it's bulk, attach a `bulk_id`.
6. Cross-reference: every new event name added → entry in the table
   above.

## Future

- Move `LOGGING.md`'s "what to log" table into a structured registry
  (JSON?) that Grafana dashboards can read, so new event types
  auto-surface in the audit feed.
- Browser logger → `/api/logs` endpoint → loguru → Loki (when we get to
  shipping client-side errors).
- Background-job loggers (face extraction, mpp, dedup) follow the same
  conventions — separate JSONL file per job, same Promtail config.
