import json
import os
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from app.config import with_v
from app.db.neo4j import get_session
from app.log import logger
from app.services import mosaic as mosaic_svc


WEEKDAY_INT = {"Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6, "Sun": 7}

HUE_CASE = """
CASE
    WHEN substring({col}, 1, 1) = '0' AND substring({col}, 3, 1) = '0' AND substring({col}, 5, 1) = '0' THEN 'black'
    WHEN substring({col}, 1, 1) = 'f' AND substring({col}, 3, 1) = 'f' AND substring({col}, 5, 1) = 'f' THEN 'white'
    WHEN substring({col}, 1, 1) = substring({col}, 3, 1) AND substring({col}, 3, 1) = substring({col}, 5, 1) THEN 'gray'
    WHEN substring({col}, 1, 1) > substring({col}, 3, 1) AND substring({col}, 1, 1) > substring({col}, 5, 1) THEN 'red'
    WHEN substring({col}, 3, 1) > substring({col}, 1, 1) AND substring({col}, 3, 1) > substring({col}, 5, 1) THEN 'green'
    WHEN substring({col}, 5, 1) > substring({col}, 1, 1) AND substring({col}, 5, 1) > substring({col}, 3, 1) THEN 'blue'
    WHEN substring({col}, 1, 1) = substring({col}, 3, 1) AND substring({col}, 1, 1) > substring({col}, 5, 1) THEN 'yellow'
    WHEN substring({col}, 1, 1) = substring({col}, 5, 1) AND substring({col}, 1, 1) > substring({col}, 3, 1) THEN 'magenta'
    WHEN substring({col}, 3, 1) = substring({col}, 5, 1) AND substring({col}, 3, 1) > substring({col}, 1, 1) THEN 'cyan'
    ELSE 'mixed'
END
"""


def _hex_nibble(var: str, pos: int) -> str:
    """Cypher CASE that maps one hex char at position `pos` of `var` to 0..15."""
    return (
        f"(CASE substring({var},{pos},1) "
        "WHEN '0' THEN 0 WHEN '1' THEN 1 WHEN '2' THEN 2 WHEN '3' THEN 3 "
        "WHEN '4' THEN 4 WHEN '5' THEN 5 WHEN '6' THEN 6 WHEN '7' THEN 7 "
        "WHEN '8' THEN 8 WHEN '9' THEN 9 WHEN 'a' THEN 10 WHEN 'b' THEN 11 "
        "WHEN 'c' THEN 12 WHEN 'd' THEN 13 WHEN 'e' THEN 14 WHEN 'f' THEN 15 "
        "ELSE 0 END)"
    )


# Shared HSV→hue-bucket classifier used by /archive-overview and /hue/.../samples.
# Same thresholds in both must stay in sync.
HUE_BUCKET_CASE = """
CASE
    WHEN V < 0.18                                       THEN 'black'
    WHEN V > 0.88 AND S < 0.30                          THEN 'pale'
    WHEN S < 0.12 AND V < 0.25                          THEN 'black'
    WHEN S < 0.10 AND V > 0.82                          THEN 'white'
    WHEN S < 0.10                                       THEN 'gray'
    WHEN H < 0                                          THEN 'gray'
    WHEN H >= 15 AND H < 45 AND V < 0.45                THEN 'brown'
    WHEN H >= 15 AND H < 45 AND V > 0.80 AND S < 0.35   THEN 'tan'
    WHEN H >= 15 AND H < 45 AND V > 0.80 AND S >= 0.35  THEN 'peach'
    WHEN (H >= 345 OR H < 15) AND V < 0.35              THEN 'maroon'
    WHEN H >= 345 OR H < 8                              THEN 'red'
    WHEN H < 15                                         THEN 'scarlet'
    WHEN H < 30                                         THEN 'orange'
    WHEN H < 45                                         THEN 'goldenrod'
    WHEN H < 65                                         THEN 'yellow'
    WHEN H < 90 AND V < 0.45                            THEN 'olive'
    WHEN H < 90                                         THEN 'lime'
    WHEN H < 150 AND V < 0.30                           THEN 'forest'
    WHEN H < 150                                        THEN 'green'
    WHEN H < 185                                        THEN 'teal'
    WHEN H < 200                                        THEN 'cyan'
    WHEN H < 235 AND V < 0.28                           THEN 'navy'
    WHEN H < 235 AND V > 0.75 AND S < 0.55              THEN 'sky'
    WHEN H < 235                                        THEN 'blue'
    WHEN H < 270                                        THEN 'indigo'
    WHEN H < 295                                        THEN 'violet'
    WHEN H < 325                                        THEN 'magenta'
    ELSE                                                     'pink'
END
"""


def _hue_bucket_aggregate_query(prop: str) -> str:
    """Cypher that aggregates Media by hue bucket of `prop`. Returns rows of
    (hue, count, c_min, c_33, c_66, c_max) — last 4 are 10/40/70/90th percentile
    representative hex colors within the bucket.
    """
    return f"""
        MATCH (m:Media) WHERE m.{prop} IS NOT NULL AND size(m.{prop}) = 7
        WITH m.{prop} AS c
        WITH c,
             {_hex_nibble('c', 1)} * 16 + {_hex_nibble('c', 2)} AS R,
             {_hex_nibble('c', 3)} * 16 + {_hex_nibble('c', 4)} AS G,
             {_hex_nibble('c', 5)} * 16 + {_hex_nibble('c', 6)} AS B
        WITH c, R, G, B,
             CASE WHEN R >= G AND R >= B THEN R WHEN G >= B THEN G ELSE B END AS mx,
             CASE WHEN R <= G AND R <= B THEN R WHEN G <= B THEN G ELSE B END AS mn
        WITH c, R, G, B, mx, mn, (mx - mn) AS d
        WITH c, R, G, B, mx, mn, d,
             CASE WHEN d = 0 THEN -1.0
                  WHEN mx = R THEN (((G - B) * 60.0 / d) + 360.0) % 360.0
                  WHEN mx = G THEN (((B - R) * 60.0 / d) + 120.0)
                  ELSE             (((R - G) * 60.0 / d) + 240.0) END AS H,
             CASE WHEN mx = 0 THEN 0.0 ELSE d * 1.0 / mx END AS S,
             mx / 255.0 AS V
        WITH c, V, {HUE_BUCKET_CASE} AS hue
        WITH hue, c, V
        ORDER BY hue, V
        WITH hue, collect(c) AS samples
        WITH hue, size(samples) AS n,
             samples[toInteger(size(samples)*0.10)]  AS c_min,
             samples[toInteger(size(samples)*0.40)]  AS c_33,
             samples[toInteger(size(samples)*0.70)]  AS c_66,
             samples[toInteger(size(samples)*0.90)]  AS c_max
        RETURN hue, n AS count, c_min, c_33, c_66, c_max
        ORDER BY count DESC
    """


def _hue_bucket_filter_query(prop: str) -> str:
    """Cypher that returns sample Media nodes whose `prop` matches a given
    `$hue` bucket. Requires `$hue` and `$limit` params at runtime.
    """
    return f"""
        MATCH (m:Media) WHERE m.{prop} IS NOT NULL AND size(m.{prop}) = 7
        WITH m, m.{prop} AS c
        WITH m, c,
             {_hex_nibble('c', 1)} * 16 + {_hex_nibble('c', 2)} AS R,
             {_hex_nibble('c', 3)} * 16 + {_hex_nibble('c', 4)} AS G,
             {_hex_nibble('c', 5)} * 16 + {_hex_nibble('c', 6)} AS B
        WITH m, c, R, G, B,
             CASE WHEN R >= G AND R >= B THEN R WHEN G >= B THEN G ELSE B END AS mx,
             CASE WHEN R <= G AND R <= B THEN R WHEN G <= B THEN G ELSE B END AS mn
        WITH m, c, R, G, B, mx, mn, (mx - mn) AS d
        WITH m, c, R, G, B, mx, mn, d,
             CASE WHEN d = 0 THEN -1.0
                  WHEN mx = R THEN (((G - B) * 60.0 / d) + 360.0) % 360.0
                  WHEN mx = G THEN (((B - R) * 60.0 / d) + 120.0)
                  ELSE             (((R - G) * 60.0 / d) + 240.0) END AS H,
             CASE WHEN mx = 0 THEN 0.0 ELSE d * 1.0 / mx END AS S,
             mx / 255.0 AS V
        WITH m, c, {HUE_BUCKET_CASE} AS hue
        WHERE hue = $hue
        WITH m, c, rand() AS rnd
        ORDER BY rnd
        LIMIT $limit
        RETURN m.path AS path, c AS color, m.timestamp AS ts
    """


_PROP_BY_SOURCE = {
    "dominant": "dominant_color",
    "mean":     "mean_color",
    "salient":  "salient_color",
}


def _build_filters(color: Optional[str], weekday: Optional[str], city: Optional[str]):
    """Turn ?color/?weekday/?city into a Cypher WHERE-fragment + params dict.

    Returns (clause, params). Clause is empty string when no filters set.
    The clause assumes the query has already MATCH-ed a Media node `m`.
    """
    clauses = []
    params: dict = {}

    if color:
        clauses.append(
            "m.dominant_color IS NOT NULL AND size(m.dominant_color) = 7 AND "
            + HUE_CASE.format(col="m.dominant_color") + " = $facet_color"
        )
        params['facet_color'] = color

    if weekday:
        wd_int = WEEKDAY_INT.get(weekday)
        if wd_int is not None:
            clauses.append(
                "m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01' "
                "AND datetime(toString(m.timestamp)).weekday = $facet_weekday"
            )
            params['facet_weekday'] = wd_int

    if city:
        clauses.append("m.location_city = $facet_city")
        params['facet_city'] = city

    if not clauses:
        return "", params
    return " AND ".join(f"({c})" for c in clauses), params


_MEDIA_MATCH_RE = None  # lazy compile so module-level import order doesn't matter


def _prepend_where(query: str, extra_where: str) -> str:
    """Inject `extra_where` as a WHERE clause immediately after the first
    `MATCH (m:Media[:...])` in `query`. Existing WHERE clauses on the same
    pattern get an AND with the existing clause wrapped in parens so we
    don't break OR-precedence.
    """
    if not extra_where:
        return query

    import re
    global _MEDIA_MATCH_RE
    if _MEDIA_MATCH_RE is None:
        _MEDIA_MATCH_RE = re.compile(r'MATCH \(m:Media(?::[A-Za-z]+)*\)\s*')

    match = _MEDIA_MATCH_RE.search(query)
    if not match:
        return query
    insert_at = match.end()
    tail = query[insert_at:]

    # If the existing query immediately has WHERE ..., we wrap that clause.
    where_match = re.match(r'WHERE\s+(.+?)(?=\s+(?:RETURN|WITH|MATCH|OPTIONAL\s+MATCH|UNWIND|ORDER|LIMIT|$))', tail, re.DOTALL)
    if where_match:
        existing = where_match.group(1).strip()
        before = query[:insert_at]
        after = tail[where_match.end():]
        return f"{before}WHERE ({extra_where}) AND ({existing})\n{after}"

    return query[:insert_at] + f"WHERE {extra_where}\n" + tail

router = APIRouter(prefix="/admin", tags=["admin"])

# Cloudflare Access injects this header on every authenticated request when the
# app sits behind a Zero-Trust Access policy. In local dev (no Cloudflare in
# front), the middleware falls back to env DEV_USER_EMAIL.
CF_EMAIL_HEADER = "cf-access-authenticated-user-email"


def _current_email(request: Request) -> str | None:
    return (
        request.headers.get(CF_EMAIL_HEADER)
        or os.environ.get("DEV_USER_EMAIL", "stephenyoung7267@gmail.com")
    )

REPORT_PATH = Path(os.environ.get("REPORT_JSON", "/photos/__data/status/report.json"))


@router.get("/disk-report")
async def get_disk_report(refresh: bool = False):
    """On-disk scan + graph-drift report. Cached to /photos/__data/status/disk-report.json;
    pass `?refresh=true` to rebuild (blocks until the scan finishes — ~30-60s
    for the full archive). For a background refresh, POST /disk-report/refresh
    instead."""
    from app.services.disk_report import build_disk_report, cache_disk_report, read_cached_report
    if refresh:
        report = await run_in_threadpool(build_disk_report)
        cache_disk_report(report)
        return report
    cached = read_cached_report()
    if cached is None:
        raise HTTPException(404, "disk-report cache not built yet — call with ?refresh=true once")
    return cached


@router.post("/disk-report/refresh", status_code=202)
async def refresh_disk_report():
    """Kick off a fresh scan in the background. The next GET will see the
    new report once the scan finishes."""
    from app.services.disk_report import build_disk_report, cache_disk_report
    def _bg():
        try:
            cache_disk_report(build_disk_report())
        except Exception:
            pass
    import threading
    threading.Thread(target=_bg, daemon=True).start()
    return {"status": "scanning"}


@router.get("/report")
async def get_report():
    if not REPORT_PATH.exists():
        raise HTTPException(404, "report.json not found — run the Archive Report job first")
    import json
    return json.loads(REPORT_PATH.read_text())


@router.get("/archive-overview")
async def archive_overview(
    color:   Optional[str] = Query(default=None),
    weekday: Optional[str] = Query(default=None),
    city:    Optional[str] = Query(default=None),
):
    """Graph-native archive overview. Every number is a live Cypher query
    against Neo4j — no disk scan, no sidecar inspection. Sibling to
    /admin/report, which still reports the on-disk sidecar side.

    Sections:
    - media:       totals by label + disk size + decade distribution
    - people:      Person counts + GEDCOM enrichment progress
    - edges:       graph density (APPEARS_IN, FAVORITED, MARRIED_TO, PARENT_OF, LIFE_STAGE)
    - timestamps:  source + confidence + precision breakdowns
    - places:      GPS / city coverage + top cities
    - cameras:     top make+model
    - heritage:    heritage-specific fields (content_date, transcription, physical_status)
    - engagement:  user-driven artifacts (favorites, life-stage locks, albums, etc.)
    """
    import asyncio
    from datetime import datetime
    filter_where, filter_params = _build_filters(color, weekday, city)
    active_filters = {
        k: v for k, v in {"color": color, "weekday": weekday, "city": city}.items() if v
    }

    # Parallel-query infra. Each query opens its own short-lived Neo4j
    # session so asyncio.gather can fan them out. Auto-filter injects the
    # facet WHERE only into queries that touch (m:Media).
    def _maybe_filter(query: str) -> str:
        if filter_where and ('(m:Media' in query or '(m:Media:' in query):
            return _prepend_where(query, filter_where)
        return query

    # Cap simultaneous Neo4j sessions. With 45+ queries fanned out at
    # once we'd otherwise blow past the server's transient-conflict
    # threshold and the client gets a 500. 10 keeps the wall-clock fast
    # without putting the DB in a bad mood.
    sem = asyncio.Semaphore(10)

    async def _run_with_retry(query: str, fetch: str):
        from neo4j.exceptions import TransientError
        last_exc = None
        for attempt in range(3):
            try:
                async with sem, get_session() as s:
                    r = await s.run(_maybe_filter(query), **filter_params)
                    if fetch == "single":
                        row = await r.single()
                        return (row.get(row.keys()[0]) if row else 0) or 0
                    return await r.data()
            except TransientError as e:
                last_exc = e
                await asyncio.sleep(0.05 * (2 ** attempt))  # 50ms, 100ms, 200ms
        raise last_exc

    async def one(query: str):
        return await _run_with_retry(query, "single")

    async def rows(query: str):
        return await _run_with_retry(query, "data")

    # ── Fan out every query in parallel ──────────────────────────────────
    # Order here matches the unpacking below. Keep them aligned.
    (
        media_total, media_by_label, media_bytes, media_with_ts, media_by_decade,
        people_total, people_with_gedcom, people_with_gender,
        people_with_birth, people_with_avatar, people_deceased, people_with_gallery,
        edges_appears_in, edges_favorited, edges_married_to,
        edges_parent_of, edges_knows, edges_life_stage,
        ts_by_source, ts_by_confidence, ts_by_precision,
        with_gps, with_city, distinct_cities, top_cities, top_cameras,
        heritage_total, heritage_with_content_date, heritage_with_description,
        heritage_with_transcription, heritage_by_status,
        colors_dom_hue, colors_mean_hue, colors_salient_hue, colors_dom_brightness,
        trivia_busiest_day, trivia_busiest_weekday, trivia_busiest_hour,
        trivia_distinct_cameras, trivia_avg_megapixels, trivia_faces_per_photo,
        trivia_longest_streak,
        eng_manual_redates, eng_albums, eng_collections, eng_groups, eng_suggestions_open,
    ) = await asyncio.gather(
        one("MATCH (m:Media) RETURN count(m)"),
        rows("MATCH (m:Media) RETURN labels(m) AS labels, count(*) AS count ORDER BY count DESC"),
        one("MATCH (m:Media) WHERE m.file_size IS NOT NULL RETURN sum(m.file_size)"),
        one("MATCH (m:Media) WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01' RETURN count(m)"),
        rows("""
            MATCH (m:Media)
            WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH (toInteger(substring(toString(m.timestamp), 0, 4)) / 10) * 10 AS decade, m
            RETURN decade, count(m) AS count
            ORDER BY decade
        """),

        one("MATCH (p:Person) RETURN count(p)"),
        one("MATCH (p:Person) WHERE p.gedcom_id IS NOT NULL RETURN count(p)"),
        one("MATCH (p:Person) WHERE p.gender IS NOT NULL RETURN count(p)"),
        one("MATCH (p:Person) WHERE p.birth_date IS NOT NULL RETURN count(p)"),
        one("MATCH (p:Person) WHERE p.avatar IS NOT NULL RETURN count(p)"),
        one("MATCH (p:Person) WHERE p.is_living = false RETURN count(p)"),
        one("MATCH (p:Person)-[:APPEARS_IN]->() RETURN count(DISTINCT p)"),

        one("MATCH ()-[r:APPEARS_IN]->() RETURN count(r)"),
        one("MATCH ()-[r:FAVORITED]->() RETURN count(r)"),
        one("MATCH ()-[r:MARRIED_TO]->() RETURN count(r)"),
        one("MATCH ()-[r:PARENT_OF]->() RETURN count(r)"),
        one("MATCH ()-[r:KNOWS]->() RETURN count(r)"),
        one("MATCH ()-[r:LIFE_STAGE]->() RETURN count(r)"),

        rows("""
            MATCH (m:Media) WHERE m.timestamp_source IS NOT NULL
            RETURN m.timestamp_source AS source, count(*) AS count
            ORDER BY count DESC
        """),
        rows("""
            MATCH (m:Media) WHERE m.timestamp_confidence IS NOT NULL
            RETURN m.timestamp_confidence AS confidence, count(*) AS count
            ORDER BY count DESC
        """),
        rows("""
            MATCH (m:Media) WHERE m.timestamp_precision IS NOT NULL
            RETURN m.timestamp_precision AS precision, count(*) AS count
            ORDER BY count DESC
        """),

        one("MATCH (m:Media) WHERE m.latitude IS NOT NULL RETURN count(m)"),
        one("MATCH (m:Media) WHERE m.location_city IS NOT NULL RETURN count(m)"),
        one("MATCH (m:Media) WHERE m.location_city IS NOT NULL RETURN count(DISTINCT m.location_city)"),
        rows("""
            MATCH (m:Media) WHERE m.location_city IS NOT NULL
            RETURN m.location_city AS city, m.location_state AS state, count(*) AS count
            ORDER BY count DESC LIMIT 15
        """),
        rows("""
            MATCH (m:Media) WHERE m.camera_make IS NOT NULL OR m.camera_model IS NOT NULL
            RETURN m.camera_make AS make, m.camera_model AS model, count(*) AS count
            ORDER BY count DESC LIMIT 10
        """),

        one("MATCH (m:Media:Heritage) RETURN count(m)"),
        one("MATCH (m:Media:Heritage) WHERE m.content_date IS NOT NULL RETURN count(m)"),
        one("MATCH (m:Media:Heritage) WHERE m.description IS NOT NULL RETURN count(m)"),
        one("MATCH (m:Media:Heritage) WHERE m.transcription IS NOT NULL AND m.transcription <> '' RETURN count(m)"),
        rows("""
            MATCH (m:Media:Heritage) WHERE m.physical_status IS NOT NULL
            RETURN m.physical_status AS status, count(*) AS count
            ORDER BY count DESC
        """),

        rows(_hue_bucket_aggregate_query("dominant_color")),
        rows(_hue_bucket_aggregate_query("mean_color")),
        rows(_hue_bucket_aggregate_query("salient_color")),
        rows("""
            MATCH (m:Media) WHERE m.dominant_color IS NOT NULL AND size(m.dominant_color) = 7
            WITH (substring(m.dominant_color, 1, 1) + substring(m.dominant_color, 3, 1) + substring(m.dominant_color, 5, 1)) AS rgb
            WITH CASE
                WHEN rgb <= '555' THEN 'dark'
                WHEN rgb <= 'aaa' THEN 'mid'
                ELSE 'bright'
            END AS band
            RETURN band, count(*) AS count
            ORDER BY count DESC
        """),

        rows("""
            MATCH (m:Media) WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH substring(toString(m.timestamp), 0, 10) AS day
            RETURN day, count(*) AS count
            ORDER BY count DESC LIMIT 5
        """),
        rows("""
            MATCH (m:Media) WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH datetime(toString(m.timestamp)).weekday AS dow
            WITH CASE dow
                WHEN 1 THEN 'Mon' WHEN 2 THEN 'Tue' WHEN 3 THEN 'Wed'
                WHEN 4 THEN 'Thu' WHEN 5 THEN 'Fri' WHEN 6 THEN 'Sat'
                WHEN 7 THEN 'Sun'
            END AS weekday
            RETURN weekday, count(*) AS count
            ORDER BY count DESC
        """),
        rows("""
            MATCH (m:Media) WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH datetime(toString(m.timestamp)).hour AS hour
            RETURN hour, count(*) AS count
            ORDER BY hour ASC
        """),
        one("""
            MATCH (m:Media) WHERE m.camera_model IS NOT NULL
            RETURN count(DISTINCT m.camera_model)
        """),
        one("""
            MATCH (m:Media:Photo) WHERE m.megapixels IS NOT NULL
            RETURN round(avg(m.megapixels) * 10) / 10.0
        """),
        rows("""
            MATCH (m:Media:Photo)
            OPTIONAL MATCH (p:Person)-[:APPEARS_IN]->(m)
            WITH m, count(p) AS faces
            WITH CASE
                WHEN faces = 0 THEN '0'
                WHEN faces = 1 THEN '1'
                WHEN faces = 2 THEN '2'
                WHEN faces <= 5 THEN '3-5'
                ELSE '6+'
            END AS bucket
            RETURN bucket, count(*) AS count
            ORDER BY bucket
        """),
        one("""
            MATCH (m:Media) WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH DISTINCT substring(toString(m.timestamp), 0, 10) AS day
            WITH day ORDER BY day
            WITH collect(day) AS days
            WITH days, range(0, size(days)-2) AS idxs
            UNWIND idxs AS i
            WITH days, days[i] AS d1, days[i+1] AS d2,
                 duration.inDays(date(days[i]), date(days[i+1])).days AS gap
            WITH days, [i IN range(0, size(days)-2) WHERE
                duration.inDays(date(days[i]), date(days[i+1])).days = 1] AS streaks
            RETURN size(streaks) AS longest
            LIMIT 1
        """),

        one("MATCH (m:Media) WHERE m.timestamp_source = 'manual' RETURN count(m)"),
        one("MATCH (a:Album) RETURN count(a)"),
        one("MATCH (c:Collection) RETURN count(c)"),
        one("MATCH (g:Group) RETURN count(g)"),
        one("MATCH (s:Suggestion) WHERE coalesce(s.status, 'pending') = 'pending' RETURN count(s)"),
    )

    edges = {
        'appears_in': edges_appears_in,
        'favorited':  edges_favorited,
        'married_to': edges_married_to,
        'parent_of':  edges_parent_of,
        'knows':      edges_knows,
        'life_stage': edges_life_stage,
    }

    engagement = {
        'favorites':         edges['favorited'],
        'life_stage_locks':  edges['life_stage'],
        'manual_redates':    eng_manual_redates,
        'albums':            eng_albums,
        'collections':       eng_collections,
        'groups':            eng_groups,
        'suggestions_open':  eng_suggestions_open,
    }

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "neo4j",
        "filters": active_filters,
        "media": {
            "total": media_total,
            "by_type": [{"labels": r["labels"], "count": r["count"]} for r in media_by_label],
            "total_bytes": media_bytes,
            "with_timestamp": media_with_ts,
            "by_decade": [{"decade": r["decade"], "count": r["count"]} for r in media_by_decade],
        },
        "people": {
            "total":            people_total,
            "with_gedcom_id":   people_with_gedcom,
            "with_gender":      people_with_gender,
            "with_birth_date":  people_with_birth,
            "with_avatar":      people_with_avatar,
            "with_gallery":     people_with_gallery,
            "deceased":         people_deceased,
        },
        "edges": edges,
        "timestamps": {
            "by_source":     [{"source": r["source"],         "count": r["count"]} for r in ts_by_source],
            "by_confidence": [{"confidence": r["confidence"], "count": r["count"]} for r in ts_by_confidence],
            "by_precision":  [{"precision": r["precision"],   "count": r["count"]} for r in ts_by_precision],
        },
        "places": {
            "with_gps":         with_gps,
            "with_city":        with_city,
            "distinct_cities":  distinct_cities,
            "top_cities": [
                {"city": r["city"], "state": r["state"], "count": r["count"]}
                for r in top_cities
            ],
        },
        "cameras": [
            {"make": r["make"], "model": r["model"], "count": r["count"]}
            for r in top_cameras
        ],
        "heritage": {
            "total":              heritage_total,
            "with_content_date":  heritage_with_content_date,
            "with_description":   heritage_with_description,
            "with_transcription": heritage_with_transcription,
            "by_physical_status": [
                {"status": r["status"], "count": r["count"]} for r in heritage_by_status
            ],
        },
        "engagement": engagement,
        "colors": {
            "dominant_by_hue":        [{
                "hue":      r["hue"],
                "count":    r["count"],
                "samples":  [r["c_min"], r["c_33"], r["c_66"], r["c_max"]],
            } for r in colors_dom_hue],
            "mean_by_hue":            [{
                "hue":      r["hue"],
                "count":    r["count"],
                "samples":  [r["c_min"], r["c_33"], r["c_66"], r["c_max"]],
            } for r in colors_mean_hue],
            "salient_by_hue":         [{
                "hue":      r["hue"],
                "count":    r["count"],
                "samples":  [r["c_min"], r["c_33"], r["c_66"], r["c_max"]],
            } for r in colors_salient_hue],
            "dominant_by_brightness": [{"band": r["band"], "count": r["count"]} for r in colors_dom_brightness],
        },
        "trivia": {
            "busiest_days":          [{"day": r["day"], "count": r["count"]} for r in trivia_busiest_day],
            "by_weekday":            [{"weekday": r["weekday"], "count": r["count"]} for r in trivia_busiest_weekday],
            "by_hour_of_day":        [{"hour": r["hour"], "count": r["count"]} for r in trivia_busiest_hour],
            "distinct_cameras":      trivia_distinct_cameras,
            "avg_megapixels":        trivia_avg_megapixels,
            "people_per_photo":      [{"bucket": r["bucket"], "count": r["count"]} for r in trivia_faces_per_photo],
            "longest_daily_streak":  trivia_longest_streak,
        },
    }


@router.get("/archive-overview/hue/{hue}/samples")
async def hue_samples(hue: str, limit: int = 24, source: str = "dominant"):
    """Return up to `limit` sample Media rows whose color (per `source`) falls
    into the given hue bucket. `source` is one of: dominant, mean.
    """
    prop = _PROP_BY_SOURCE.get(source)
    if prop is None:
        raise HTTPException(400, f"unknown source '{source}' — try one of {sorted(_PROP_BY_SOURCE)}")
    query = _hue_bucket_filter_query(prop)

    async def fetch():
        async with get_session() as s:
            res = await s.run(query, hue=hue, limit=int(limit))
            data = [r async for r in res]
            return data

    rows_out = await fetch()
    samples = []
    for r in rows_out:
        p = (r["path"] or "").removeprefix("archive/")
        samples.append({
            "path":      r["path"],
            "thumb_url": with_v(f"/api/media/thumb/{p}") if p else None,
            "color":     r["color"],
            "timestamp": str(r["ts"]) if r["ts"] is not None else None,
        })
    return {"hue": hue, "source": source, "count": len(samples), "samples": samples}


def _hue_filter_prefix(prop: str) -> str:
    """Reusable Cypher prefix that binds `m` to Media nodes whose `prop`
    falls in the `$hue` bucket. Downstream queries can add their own
    aggregation tail (objects, people, locations, ...) without re-deriving
    the HSV classifier.
    """
    return f"""
        MATCH (m:Media) WHERE m.{prop} IS NOT NULL AND size(m.{prop}) = 7
        WITH m, m.{prop} AS c
        WITH m, c,
             {_hex_nibble('c', 1)} * 16 + {_hex_nibble('c', 2)} AS R,
             {_hex_nibble('c', 3)} * 16 + {_hex_nibble('c', 4)} AS G,
             {_hex_nibble('c', 5)} * 16 + {_hex_nibble('c', 6)} AS B
        WITH m,
             CASE WHEN R >= G AND R >= B THEN R WHEN G >= B THEN G ELSE B END AS mx,
             CASE WHEN R <= G AND R <= B THEN R WHEN G <= B THEN G ELSE B END AS mn,
             R, G, B
        WITH m, R, G, B, mx, mn, (mx - mn) AS d
        WITH m,
             CASE WHEN d = 0 THEN -1.0
                  WHEN mx = R THEN (((G - B) * 60.0 / d) + 360.0) % 360.0
                  WHEN mx = G THEN (((B - R) * 60.0 / d) + 120.0)
                  ELSE             (((R - G) * 60.0 / d) + 240.0) END AS H,
             CASE WHEN mx = 0 THEN 0.0 ELSE d * 1.0 / mx END AS S,
             mx / 255.0 AS V
        WITH m, {HUE_BUCKET_CASE} AS hue
        WHERE hue = $hue
    """


async def _insights_bundle(filter_prefix: str, params: dict, limit: int = 5) -> dict:
    """Run the canonical "deeper data" Cypher tails over any Media-filter
    prefix. Always returns the same shape so the InsightsModal on the
    frontend can be aspect-agnostic. The caller passes a `filter_prefix`
    that ends with `m` bound to Media nodes in the bucket of interest.
    """
    import asyncio

    objects_q = filter_prefix + """
        WITH m WHERE m.objects IS NOT NULL
        WITH count(m) AS total_in_bucket, collect(m) AS bucket_media
        UNWIND bucket_media AS m
        UNWIND m.objects AS obj
        WITH obj, total_in_bucket, count(DISTINCT m) AS object_count
        WHERE obj IS NOT NULL AND obj <> '' AND obj <> 'person'
        RETURN obj AS object, object_count AS count, total_in_bucket AS total
        ORDER BY object_count DESC
        LIMIT $insights_limit
    """

    person_q = filter_prefix + """
        MATCH (p:Person)-[:APPEARS_IN]->(m)
        WITH p, count(DISTINCT m) AS appearances
        ORDER BY appearances DESC
        LIMIT 1
        RETURN p.id AS id, p.name AS name, p.avatar AS avatar, appearances AS count
    """

    city_q = filter_prefix + """
        WITH m WHERE m.location_city IS NOT NULL AND m.location_city <> ''
        RETURN m.location_city AS city, m.location_state AS state, count(m) AS count
        ORDER BY count DESC
        LIMIT 1
    """

    weekday_q = filter_prefix + """
        WITH m WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
        WITH datetime(toString(m.timestamp)).weekday AS dow
        WITH dow, count(*) AS count
        ORDER BY count DESC
        LIMIT 1
        RETURN dow, count
    """

    camera_q = filter_prefix + """
        WITH m WHERE m.camera_make IS NOT NULL OR m.camera_model IS NOT NULL
        WITH m.camera_make AS make, m.camera_model AS model, count(m) AS count
        ORDER BY count DESC
        LIMIT 1
        RETURN make, model, count
    """

    async def run(query):
        async with get_session() as s:
            res = await s.run(query, insights_limit=limit, **params)
            return await res.data()

    objects_rows, person_rows, city_rows, weekday_rows, camera_rows = await asyncio.gather(
        run(objects_q), run(person_q), run(city_q), run(weekday_q), run(camera_q),
    )

    total = objects_rows[0]["total"] if objects_rows else 0
    top_person = top_city = top_weekday = top_camera = None
    if person_rows:
        r = person_rows[0]
        top_person = {"id": r["id"], "name": r["name"], "avatar": r["avatar"], "count": r["count"]}
    if city_rows:
        r = city_rows[0]
        top_city = {"city": r["city"], "state": r["state"], "count": r["count"]}
    if weekday_rows:
        WEEKDAY = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
        r = weekday_rows[0]
        top_weekday = {"weekday": WEEKDAY.get(r["dow"], "?"), "count": r["count"]}
    if camera_rows:
        r = camera_rows[0]
        top_camera = {"make": r["make"], "model": r["model"], "count": r["count"]}

    return {
        "total":       total,
        "top_person":  top_person,
        "top_city":    top_city,
        "top_weekday": top_weekday,
        "top_camera":  top_camera,
        "objects": [
            {
                "object": r["object"],
                "count":  r["count"],
                "share":  (r["count"] / total) if total else 0,
            }
            for r in objects_rows
        ],
    }


async def _samples_for(filter_prefix: str, params: dict, limit: int = 48) -> list[dict]:
    """Return `limit` random sample Media (path/color/timestamp) from any
    Media-filter prefix. Same response shape regardless of aspect.
    """
    query = filter_prefix + """
        WITH m, rand() AS rnd
        ORDER BY rnd
        LIMIT $samples_limit
        RETURN m.path AS path, m.dominant_color AS color, m.timestamp AS ts
    """
    async with get_session() as s:
        res = await s.run(query, samples_limit=int(limit), **params)
        rows_out = await res.data()
    samples = []
    for r in rows_out:
        p = (r["path"] or "").removeprefix("archive/")
        samples.append({
            "path":      r["path"],
            "thumb_url": with_v(f"/api/media/thumb/{p}") if p else None,
            "color":     r["color"],
            "timestamp": str(r["ts"]) if r["ts"] is not None else None,
        })
    return samples


@router.get("/archive-overview/hue/{hue}/objects")
async def hue_objects(hue: str, source: str = "dominant", limit: int = 5):
    """Top objects + top person + top city + top weekday + top camera for
    Media whose `source` color lands in the given hue bucket. (Name kept
    as `/objects` for backwards compat — response shape grew.)
    """
    prop = _PROP_BY_SOURCE.get(source)
    if prop is None:
        raise HTTPException(400, f"unknown source '{source}'")

    bundle = await _insights_bundle(
        _hue_filter_prefix(prop),
        params={"hue": hue},
        limit=limit,
    )
    return {"hue": hue, "source": source, **bundle}


# ── Weekday aspect ───────────────────────────────────────────────────────

_WEEKDAY_FILTER = """
    MATCH (m:Media)
    WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
      AND datetime(toString(m.timestamp)).weekday = $dow
"""


@router.get("/analytics/weekday/buckets")
async def weekday_buckets():
    """Return 7 weekday buckets (Mon..Sun) with photo counts."""
    query = """
        MATCH (m:Media)
        WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
        WITH datetime(toString(m.timestamp)).weekday AS dow
        RETURN dow, count(*) AS count
        ORDER BY dow
    """
    WEEKDAY = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    async with get_session() as s:
        res = await s.run(query)
        rows_out = await res.data()
    return {
        "aspect": "weekday",
        "buckets": [
            {"key": r["dow"], "label": WEEKDAY.get(r["dow"], "?"), "count": r["count"]}
            for r in rows_out
        ],
    }


@router.get("/analytics/weekday/{dow}/insights")
async def weekday_insights(dow: int, limit: int = 5):
    if dow < 1 or dow > 7:
        raise HTTPException(400, "dow must be 1..7 (Mon..Sun)")
    bundle = await _insights_bundle(_WEEKDAY_FILTER, params={"dow": dow}, limit=limit)
    return {"aspect": "weekday", "key": dow, **bundle}


@router.get("/analytics/weekday/{dow}/samples")
async def weekday_samples(dow: int, limit: int = 48):
    if dow < 1 or dow > 7:
        raise HTTPException(400, "dow must be 1..7 (Mon..Sun)")
    samples = await _samples_for(_WEEKDAY_FILTER, params={"dow": dow}, limit=limit)
    return {"aspect": "weekday", "key": dow, "count": len(samples), "samples": samples}


# ── Hour-of-day aspect ───────────────────────────────────────────────────

_HOUR_FILTER = """
    MATCH (m:Media)
    WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
      AND datetime(toString(m.timestamp)).hour = $hour
"""


@router.get("/analytics/hour/buckets")
async def hour_buckets():
    """Return 24 hour-of-day buckets (0..23) with photo counts."""
    query = """
        MATCH (m:Media)
        WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
        WITH datetime(toString(m.timestamp)).hour AS h
        RETURN h, count(*) AS count
        ORDER BY h
    """
    async with get_session() as s:
        res = await s.run(query)
        rows_out = await res.data()
    return {
        "aspect": "hour",
        "buckets": [
            {"key": r["h"], "label": f"{r['h']:02d}:00", "count": r["count"]}
            for r in rows_out
        ],
    }


@router.get("/analytics/hour/{hour}/insights")
async def hour_insights(hour: int, limit: int = 5):
    if hour < 0 or hour > 23:
        raise HTTPException(400, "hour must be 0..23")
    bundle = await _insights_bundle(_HOUR_FILTER, params={"hour": hour}, limit=limit)
    return {"aspect": "hour", "key": hour, **bundle}


# ── Single day aspect ────────────────────────────────────────────────────

_DAY_FILTER = """
    MATCH (m:Media)
    WHERE m.timestamp IS NOT NULL AND substring(toString(m.timestamp), 0, 10) = $day
"""


@router.get("/analytics/day/{day}/insights")
async def day_insights(day: str, limit: int = 5):
    bundle = await _insights_bundle(_DAY_FILTER, params={"day": day}, limit=limit)
    return {"aspect": "day", "key": day, **bundle}


@router.get("/analytics/day/{day}/samples")
async def day_samples(day: str, limit: int = 48):
    samples = await _samples_for(_DAY_FILTER, params={"day": day}, limit=limit)
    return {"aspect": "day", "key": day, "count": len(samples), "samples": samples}


# ── People-per-photo bucket aspect ───────────────────────────────────────

_PPP_RANGES = {
    "0":   (0, 0),
    "1":   (1, 1),
    "2":   (2, 2),
    "3-5": (3, 5),
    "6+":  (6, 9999),
}


@router.get("/analytics/people_per_photo/{bucket}/insights")
async def ppp_insights(bucket: str, limit: int = 5):
    if bucket not in _PPP_RANGES:
        raise HTTPException(400, f"unknown bucket '{bucket}'")
    lo, hi = _PPP_RANGES[bucket]
    filter_q = """
        MATCH (m:Media)
        WITH m, size([(p:Person)-[:APPEARS_IN]->(m) | p]) AS pc
        WHERE pc >= $lo AND pc <= $hi
    """
    bundle = await _insights_bundle(filter_q, params={"lo": lo, "hi": hi}, limit=limit)

    # Replace top_person with top_group — the most-common set of people
    # who appear together in this bucket. For "0 people" there's no group;
    # for "1 person" the group is just one name (same as top_person).
    top_group = None
    if hi >= 1:
        group_q = filter_q + """
            MATCH (p:Person)-[:APPEARS_IN]->(m)
            WITH m, collect(DISTINCT p.name) AS names
            RETURN names
        """
        async with get_session() as s:
            res = await s.run(group_q, lo=lo, hi=hi)
            rows = await res.data()
        from collections import Counter
        counts: Counter = Counter()
        for r in rows:
            key = tuple(sorted(r["names"]))
            if key:
                counts[key] += 1
        if counts:
            (names, n) = counts.most_common(1)[0]
            top_group = {"names": list(names), "count": n}

    bundle["top_group"] = top_group
    # Hide the single-person card; the group card replaces it for this aspect.
    bundle["top_person"] = None
    return {"aspect": "people_per_photo", "key": bucket, **bundle}


@router.get("/analytics/people_per_photo/{bucket}/samples")
async def ppp_samples(bucket: str, limit: int = 48):
    if bucket not in _PPP_RANGES:
        raise HTTPException(400, f"unknown bucket '{bucket}'")
    lo, hi = _PPP_RANGES[bucket]
    filter_q = """
        MATCH (m:Media)
        WITH m, size([(p:Person)-[:APPEARS_IN]->(m) | p]) AS pc
        WHERE pc >= $lo AND pc <= $hi
    """
    samples = await _samples_for(filter_q, params={"lo": lo, "hi": hi}, limit=limit)
    return {"aspect": "people_per_photo", "key": bucket, "count": len(samples), "samples": samples}


# ── Month aspect ─────────────────────────────────────────────────────────

_MONTH_FILTER = """
    MATCH (m:Media)
    WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
      AND datetime(toString(m.timestamp)).month = $month
"""


@router.get("/analytics/month/buckets")
async def month_buckets():
    """Return 12 month buckets (Jan..Dec) with photo counts."""
    query = """
        MATCH (m:Media)
        WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
        WITH datetime(toString(m.timestamp)).month AS month
        RETURN month, count(*) AS count
        ORDER BY month
    """
    MONTH = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
             7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    async with get_session() as s:
        res = await s.run(query)
        rows_out = await res.data()
    return {
        "aspect": "month",
        "buckets": [
            {"key": r["month"], "label": MONTH.get(r["month"], "?"), "count": r["count"]}
            for r in rows_out
        ],
    }


@router.get("/analytics/month/{month}/insights")
async def month_insights(month: int, limit: int = 5):
    if month < 1 or month > 12:
        raise HTTPException(400, "month must be 1..12")
    bundle = await _insights_bundle(_MONTH_FILTER, params={"month": month}, limit=limit)
    return {"aspect": "month", "key": month, **bundle}


@router.get("/analytics/month/{month}/samples")
async def month_samples(month: int, limit: int = 48):
    if month < 1 or month > 12:
        raise HTTPException(400, "month must be 1..12")
    samples = await _samples_for(_MONTH_FILTER, params={"month": month}, limit=limit)
    return {"aspect": "month", "key": month, "count": len(samples), "samples": samples}


# ── Camera aspect ────────────────────────────────────────────────────────

_CAMERA_FILTER = """
    MATCH (m:Media)
    WHERE m.camera_model = $model
"""


@router.get("/analytics/camera/buckets")
async def camera_buckets(limit: int = 15):
    """Return top-N camera models (by model name) with photo counts."""
    query = """
        MATCH (m:Media) WHERE m.camera_model IS NOT NULL AND m.camera_model <> ''
        WITH m.camera_model AS model, m.camera_make AS make, count(*) AS count
        ORDER BY count DESC
        LIMIT $limit
        RETURN model, make, count
    """
    async with get_session() as s:
        res = await s.run(query, limit=int(limit))
        rows_out = await res.data()
    return {
        "aspect": "camera",
        "buckets": [
            {
                "key":   r["model"],
                "label": r["model"],
                "sub":   r["make"],
                "count": r["count"],
            }
            for r in rows_out
        ],
    }


@router.get("/analytics/camera/{model}/insights")
async def camera_insights(model: str, limit: int = 5):
    bundle = await _insights_bundle(_CAMERA_FILTER, params={"model": model}, limit=limit)
    return {"aspect": "camera", "key": model, **bundle}


@router.get("/analytics/camera/{model}/samples")
async def camera_samples(model: str, limit: int = 48):
    samples = await _samples_for(_CAMERA_FILTER, params={"model": model}, limit=limit)
    return {"aspect": "camera", "key": model, "count": len(samples), "samples": samples}


# ── Location aspect ──────────────────────────────────────────────────────

_LOCATION_FILTER = """
    MATCH (m:Media)
    WHERE m.location_city = $city
"""


@router.get("/analytics/location/buckets")
async def location_buckets(limit: int = 3):
    query = """
        MATCH (m:Media)
        WHERE m.location_city IS NOT NULL AND m.location_city <> ''
        WITH m.location_city AS city, m.location_state AS state, count(*) AS count
        ORDER BY count DESC
        LIMIT $limit
        RETURN city, state, count
    """
    async with get_session() as s:
        res = await s.run(query, limit=int(limit))
        rows_out = await res.data()
    return {
        "aspect": "location",
        "buckets": [
            {
                "key":   r["city"],
                "label": r["city"],
                "sub":   r["state"],
                "count": r["count"],
            }
            for r in rows_out
        ],
    }


@router.get("/analytics/location/{city}/insights")
async def location_insights(city: str, limit: int = 5):
    bundle = await _insights_bundle(_LOCATION_FILTER, params={"city": city}, limit=limit)
    return {"aspect": "location", "key": city, **bundle}


@router.get("/analytics/location/{city}/samples")
async def location_samples(city: str, limit: int = 48):
    samples = await _samples_for(_LOCATION_FILTER, params={"city": city}, limit=limit)
    return {"aspect": "location", "key": city, "count": len(samples), "samples": samples}


# ── Person aspect ────────────────────────────────────────────────────────

_PERSON_FILTER = """
    MATCH (subj:Person {id: $person_id})-[:APPEARS_IN]->(m:Media)
"""


@router.get("/analytics/people/buckets")
async def people_buckets(limit: int = 10):
    """Return top people by photo count, with avatar info for the leaderboard."""
    query = """
        MATCH (p:Person)-[:APPEARS_IN]->(m:Media)
        WITH p, count(DISTINCT m) AS appearances
        ORDER BY appearances DESC
        LIMIT $limit
        RETURN p.id AS id, p.name AS name, p.avatar AS avatar, appearances AS count
    """
    async with get_session() as s:
        res = await s.run(query, limit=int(limit))
        rows_out = await res.data()
    return {
        "aspect": "people",
        "buckets": [
            {
                "key":    r["id"],
                "label":  r["name"],
                "avatar": r["avatar"],
                "count":  r["count"],
            }
            for r in rows_out
        ],
    }


@router.get("/analytics/people/{person_id}/insights")
async def people_insights(person_id: str, limit: int = 5):
    bundle = await _insights_bundle(_PERSON_FILTER, params={"person_id": person_id}, limit=limit)
    # The "top person" card is meaningless when we're filtering to one
    # specific person. Hide it; favour camera/location/weekday instead.
    bundle["top_person"] = None
    return {"aspect": "people", "key": person_id, **bundle}


@router.get("/analytics/people/{person_id}/samples")
async def people_samples(person_id: str, limit: int = 48):
    samples = await _samples_for(_PERSON_FILTER, params={"person_id": person_id}, limit=limit)
    return {"aspect": "people", "key": person_id, "count": len(samples), "samples": samples}


# ── Face cluster summary ─────────────────────────────────────────────────

@router.get("/analytics/scenes/summary")
async def scenes_summary():
    """DINOv2 scene-embedding stats, read from the generated archive report.

    Deliberately not a live scan: there is one .scenes.json per image (~25 KB of
    float each), so walking them per request would take minutes. The Archive
    Report job already visits every sidecar, so it does the counting and this
    just serves the result.

    `mixed_models` is the one worth acting on -- embeddings from different DINOv2
    variants live in different vector spaces and different dimensions, so a mixed
    archive cannot be compared end to end.
    """
    if not REPORT_PATH.exists():
        return {"available": False, "reason": "report.json not found — run the Archive Report job"}
    import json
    try:
        report = json.loads(REPORT_PATH.read_text())
    except Exception as e:
        raise HTTPException(500, f"failed to read report.json: {e}")
    scenes = report.get("scenes")
    if not scenes:
        return {"available": False, "reason": "report predates scene stats — re-run the Archive Report job"}
    return {
        "available": True,
        "generated": report.get("generated"),
        "total":     scenes.get("total", 0),
        "images":    scenes.get("images", 0),
        "videos":    scenes.get("videos", 0),
        "eligible":  scenes.get("eligible", 0),
        "pct":       scenes.get("pct"),
        "missing":   report.get("issues", {}).get("missing_scenes"),
        "mixed_models": scenes.get("mixed_models", False),
        "models": [{"name": k, "count": v} for k, v in (scenes.get("models") or {}).items()],
        "dims":   [{"dim": k, "count": v} for k, v in (scenes.get("dims") or {}).items()],
    }


@router.get("/analytics/face_clusters/summary")
async def face_clusters_summary():
    """Stats from the face cluster sidecars: total clusters, total faces,
    assigned vs unassigned counts, and the top 5 unassigned cluster sizes.
    """
    import json
    from pathlib import Path
    clusters_file = mosaic_svc.settings.photos_root / "__faces" / "clusters" / "clusters.json"
    if not clusters_file.exists():
        return {"available": False}

    try:
        data = json.loads(clusters_file.read_text())
    except Exception as e:
        raise HTTPException(500, f"failed to read clusters.json: {e}")

    # clusters.json shape: { "<cluster_id>": [ {photo_path, face_index, ...}, ... ] }
    if not isinstance(data, dict):
        raise HTTPException(500, "unexpected clusters.json shape (not a dict)")

    assigned_pairs = set()
    async with get_session() as s:
        res = await s.run("MATCH (:Person)-[r:APPEARS_IN]->(m:Media) WHERE r.face_index IS NOT NULL RETURN m.path AS path, r.face_index AS fi")
        async for r in res:
            assigned_pairs.add((r["path"], r["fi"]))

    cluster_sizes = []
    total_faces = 0
    unassigned_faces = 0
    for cluster_id, members in data.items():
        if not isinstance(members, list):
            continue
        size = len(members)
        if size == 0:
            continue
        total_faces += size
        unassigned_in_c = sum(
            1 for m in members
            if isinstance(m, dict)
            # Cluster sidecars store absolute paths (/photos/archive/...)
            # while Neo4j stores relative ones (archive/...). Strip the
            # leading photos_root prefix before comparing.
            and (
                (m.get("photo_path") or "").removeprefix(f"{mosaic_svc.settings.photos_root}/"),
                m.get("face_index"),
            ) not in assigned_pairs
        )
        unassigned_faces += unassigned_in_c
        cluster_sizes.append({
            "cluster_id": cluster_id,
            "size":       size,
            "unassigned": unassigned_in_c,
        })

    cluster_sizes.sort(key=lambda x: -x["unassigned"])
    top_unassigned = [c for c in cluster_sizes if c["unassigned"] > 0][:5]

    return {
        "available":         True,
        "total_clusters":    len(cluster_sizes),
        "total_faces":       total_faces,
        "assigned_faces":    total_faces - unassigned_faces,
        "unassigned_faces":  unassigned_faces,
        "top_unassigned":    top_unassigned,
    }


# ── State aspect ─────────────────────────────────────────────────────────

_STATE_FILTER = """
    MATCH (m:Media)
    WHERE m.location_state = $state
"""


@router.get("/analytics/state/buckets")
async def state_buckets(limit: int = 15):
    query = """
        MATCH (m:Media)
        WHERE m.location_state IS NOT NULL AND m.location_state <> ''
        WITH m.location_state AS state, count(*) AS count
        ORDER BY count DESC
        LIMIT $limit
        RETURN state, count
    """
    async with get_session() as s:
        res = await s.run(query, limit=int(limit))
        rows_out = await res.data()
    return {
        "aspect": "state",
        "buckets": [
            {"key": r["state"], "label": r["state"], "count": r["count"]}
            for r in rows_out
        ],
    }


@router.get("/analytics/state/{state}/insights")
async def state_insights(state: str, limit: int = 5):
    bundle = await _insights_bundle(_STATE_FILTER, params={"state": state}, limit=limit)
    return {"aspect": "state", "key": state, **bundle}


@router.get("/analytics/state/{state}/samples")
async def state_samples(state: str, limit: int = 48):
    samples = await _samples_for(_STATE_FILTER, params={"state": state}, limit=limit)
    return {"aspect": "state", "key": state, "count": len(samples), "samples": samples}


# ── Decade aspect ────────────────────────────────────────────────────────

_DECADE_FILTER = """
    MATCH (m:Media)
    WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
      AND (toInteger(substring(toString(m.timestamp), 0, 4)) / 10) * 10 = $decade
"""


@router.get("/analytics/decade/buckets")
async def decade_buckets(limit: int = 3):
    query = """
        MATCH (m:Media)
        WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
        WITH (toInteger(substring(toString(m.timestamp), 0, 4)) / 10) * 10 AS decade
        WITH decade, count(*) AS count
        ORDER BY count DESC
        LIMIT $limit
        RETURN decade, count
    """
    async with get_session() as s:
        res = await s.run(query, limit=int(limit))
        rows_out = await res.data()
    return {
        "aspect": "decade",
        "buckets": [
            {
                "key":   r["decade"],
                "label": f"{r['decade']}s",
                "count": r["count"],
            }
            for r in rows_out
        ],
    }


@router.get("/analytics/decade/{decade}/insights")
async def decade_insights(decade: int, limit: int = 5):
    bundle = await _insights_bundle(_DECADE_FILTER, params={"decade": decade}, limit=limit)
    return {"aspect": "decade", "key": decade, **bundle}


@router.get("/analytics/decade/{decade}/samples")
async def decade_samples(decade: int, limit: int = 48):
    samples = await _samples_for(_DECADE_FILTER, params={"decade": decade}, limit=limit)
    return {"aspect": "decade", "key": decade, "count": len(samples), "samples": samples}


@router.get("/analytics/hour/{hour}/samples")
async def hour_samples(hour: int, limit: int = 48):
    if hour < 0 or hour > 23:
        raise HTTPException(400, "hour must be 0..23")
    samples = await _samples_for(_HOUR_FILTER, params={"hour": hour}, limit=limit)
    return {"aspect": "hour", "key": hour, "count": len(samples), "samples": samples}


@router.post("/mosaic/render")
async def mosaic_render(
    request:     Request,
    source:      Optional[UploadFile] = File(None),
    source_path: Optional[str]        = Form(None),
    grid_w:      int = Form(60),
    grid_h:      int = Form(80),
    tile_size:   int = Form(48),
    color:       str = Form("mean"),
    max_reuse:   int = Form(8),
    shape:              str = Form("square"),
    person_ids:         str = Form(""),
    exclude_person_ids: str = Form(""),
    color_distance:     str = Form("lab"),
    edge_aware:     bool = Form(False),
    source_smooth:  int  = Form(1),
    crop:           str  = Form("center"),
):
    if color not in mosaic_svc.PROP_BY_SOURCE:
        raise HTTPException(400, f"unknown color source '{color}'")
    if shape not in ("square", "hex", "dot"):
        raise HTTPException(400, f"unknown shape '{shape}'")
    if color_distance not in ("rgb", "lab"):
        raise HTTPException(400, f"unknown color_distance '{color_distance}'")
    if source_smooth not in (1, 3, 5):
        raise HTTPException(400, f"source_smooth must be 1, 3, or 5")
    if crop not in ("center", "saliency"):
        raise HTTPException(400, f"unknown crop '{crop}'")
    grid_w    = max(8,  min(grid_w, 300))
    grid_h    = max(8,  min(grid_h, 300))
    tile_size = max(16, min(tile_size, 128))
    max_reuse = max(1,  min(max_reuse, 500))

    log = logger.bind(
        by=getattr(request.state, "user_email", None),
        request_id=getattr(request.state, "request_id", None),
    )
    params = {
        "source_path":    source_path,
        "source_upload":  source.filename if source else None,
        "grid_w":         grid_w,
        "grid_h":         grid_h,
        "tile_size":      tile_size,
        "color":          color,
        "max_reuse":      max_reuse,
        "shape":          shape,
        "color_distance": color_distance,
        "edge_aware":     edge_aware,
        "source_smooth":  source_smooth,
        "crop":           crop,
        "person_ids":         [s.strip() for s in person_ids.split(",") if s.strip()] if person_ids else [],
        "exclude_person_ids": [s.strip() for s in exclude_person_ids.split(",") if s.strip()] if exclude_person_ids else [],
    }
    log.bind(event="mosaic.render.started", **params).info("mosaic render started")

    import time as _time
    t_start = _time.perf_counter()
    started_at = _time.time()
    try:
        if source is not None:
            source_bytes = await source.read()
        elif source_path:
            full = (mosaic_svc.settings.photos_root / source_path).resolve()
            try:
                full.relative_to(mosaic_svc.settings.photos_root.resolve())
            except ValueError:
                raise HTTPException(400, "source_path escapes photos_root")
            if not full.exists():
                raise HTTPException(404, f"source_path not found: {source_path}")
            source_bytes = full.read_bytes()
        else:
            raise HTTPException(400, "provide either `source` (upload) or `source_path` (archive path)")

        ids_list    = [s.strip() for s in person_ids.split(",") if s.strip()] if person_ids else None
        exclude_ids = [s.strip() for s in exclude_person_ids.split(",") if s.strip()] if exclude_person_ids else None
        pool = await mosaic_svc.fetch_pool(color, person_ids=ids_list, exclude_person_ids=exclude_ids)
        if not pool:
            raise HTTPException(400, "no candidate tiles for the chosen filters")

        jpeg, meta = await run_in_threadpool(
            mosaic_svc.build_mosaic,
            source_bytes, pool, grid_w, grid_h, tile_size, max_reuse, shape, color_distance, edge_aware, source_smooth, crop,
        )
    except HTTPException as e:
        elapsed_ms = int((_time.perf_counter() - t_start) * 1000)
        log.bind(
            event="mosaic.render.failed",
            elapsed_ms=elapsed_ms,
            status=e.status_code,
            detail=e.detail,
            **params,
        ).warning("mosaic render failed")
        raise
    except Exception as e:
        elapsed_ms = int((_time.perf_counter() - t_start) * 1000)
        log.bind(
            event="mosaic.render.failed",
            elapsed_ms=elapsed_ms,
            error=repr(e),
            **params,
        ).error("mosaic render crashed")
        raise

    elapsed_ms = int((_time.perf_counter() - t_start) * 1000)
    log.bind(
        event="mosaic.render.completed",
        started_at=started_at,
        ended_at=_time.time(),
        elapsed_ms=elapsed_ms,
        pool_size=meta.get("pool_size"),
        unique_tiles=meta.get("unique_tiles"),
        solid_fallbacks=meta.get("solid_fallbacks"),
        out_w=meta.get("out_w"),
        out_h=meta.get("out_h"),
        file_size_bytes=meta.get("file_size_bytes"),
        **params,
    ).info("mosaic render completed")

    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"x-mosaic-meta": json.dumps(meta)},
    )


@router.get("/archive-overview/person/{person_id}")
async def archive_overview_for_person(person_id: str):
    """Same shape as /archive-overview, scoped to one Person — every metric
    is restricted to photos where (Person {id})-[:APPEARS_IN]->(m:Media).
    Plus a `subject` block (the person's profile) and a `partners` list
    (the top 15 people they co-appear with).

    Sections that have no meaningful per-person analogue (global edge
    counts, all-graph engagement) are dropped from the response.
    """
    from datetime import datetime
    async with get_session() as session:
        subj_row = await (await session.run(
            """
            MATCH (p:Person {id: $pid})
            RETURN p.id AS id, p.name AS name, p.known_as AS known_as,
                   p.avatar AS avatar, p.birth_date AS birth_date,
                   p.gender AS gender, p.is_living AS is_living
            """,
            pid=person_id,
        )).single()
        if subj_row is None:
            raise HTTPException(404, f"Person {person_id} not found")
        subject = {
            "id":         subj_row["id"],
            "name":       subj_row["name"],
            "known_as":   subj_row["known_as"],
            "avatar":     subj_row["avatar"],
            "birth_date": subj_row["birth_date"],
            "gender":     subj_row["gender"],
            "is_living":  subj_row["is_living"],
        }

        async def one(query, **params):
            r = await session.run(query, pid=person_id, **params)
            row = await r.single()
            return (row.get(row.keys()[0]) if row else 0) or 0

        async def rows(query, **params):
            r = await session.run(query, pid=person_id, **params)
            return await r.data()

        # Subject-scoped subqueries — `(subj)-[:APPEARS_IN]->(m)` is the
        # shared scoping pattern.
        appears_in = "MATCH (subj:Person {id: $pid})-[:APPEARS_IN]->(m:Media)"

        media_total      = await one(f"{appears_in} RETURN count(DISTINCT m)")
        media_bytes      = await one(f"{appears_in} WHERE m.file_size IS NOT NULL RETURN sum(m.file_size)")
        media_with_ts    = await one(f"{appears_in} WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01' RETURN count(DISTINCT m)")
        media_by_label   = await rows(f"""
            {appears_in}
            RETURN labels(m) AS labels, count(DISTINCT m) AS count
            ORDER BY count DESC
        """)
        media_by_decade  = await rows(f"""
            {appears_in}
            WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH (toInteger(substring(toString(m.timestamp), 0, 4)) / 10) * 10 AS decade
            RETURN decade, count(*) AS count
            ORDER BY decade
        """)

        ts_by_source     = await rows(f"{appears_in} WHERE m.timestamp_source IS NOT NULL RETURN m.timestamp_source AS source, count(*) AS count ORDER BY count DESC")
        ts_by_confidence = await rows(f"{appears_in} WHERE m.timestamp_confidence IS NOT NULL RETURN m.timestamp_confidence AS confidence, count(*) AS count ORDER BY count DESC")
        ts_by_precision  = await rows(f"{appears_in} WHERE m.timestamp_precision IS NOT NULL RETURN m.timestamp_precision AS precision, count(*) AS count ORDER BY count DESC")

        with_gps         = await one(f"{appears_in} WHERE m.latitude IS NOT NULL RETURN count(DISTINCT m)")
        with_city        = await one(f"{appears_in} WHERE m.location_city IS NOT NULL RETURN count(DISTINCT m)")
        distinct_cities  = await one(f"{appears_in} WHERE m.location_city IS NOT NULL RETURN count(DISTINCT m.location_city)")
        top_cities       = await rows(f"""
            {appears_in} WHERE m.location_city IS NOT NULL
            RETURN m.location_city AS city, m.location_state AS state, count(DISTINCT m) AS count
            ORDER BY count DESC LIMIT 15
        """)

        top_cameras      = await rows(f"""
            {appears_in} WHERE m.camera_make IS NOT NULL OR m.camera_model IS NOT NULL
            RETURN m.camera_make AS make, m.camera_model AS model, count(DISTINCT m) AS count
            ORDER BY count DESC LIMIT 10
        """)

        heritage_total              = await one(f"MATCH (subj:Person {{id: $pid}})-[:APPEARS_IN]->(m:Media:Heritage) RETURN count(DISTINCT m)")
        heritage_with_content_date  = await one(f"MATCH (subj:Person {{id: $pid}})-[:APPEARS_IN]->(m:Media:Heritage) WHERE m.content_date IS NOT NULL RETURN count(DISTINCT m)")
        heritage_with_description   = await one(f"MATCH (subj:Person {{id: $pid}})-[:APPEARS_IN]->(m:Media:Heritage) WHERE m.description IS NOT NULL RETURN count(DISTINCT m)")
        heritage_with_transcription = await one(f"MATCH (subj:Person {{id: $pid}})-[:APPEARS_IN]->(m:Media:Heritage) WHERE m.transcription IS NOT NULL AND m.transcription <> '' RETURN count(DISTINCT m)")
        heritage_by_status          = await rows(f"""
            MATCH (subj:Person {{id: $pid}})-[:APPEARS_IN]->(m:Media:Heritage)
            WHERE m.physical_status IS NOT NULL
            RETURN m.physical_status AS status, count(DISTINCT m) AS count
            ORDER BY count DESC
        """)

        colors_dom_hue = await rows(f"""
            {appears_in} WHERE m.dominant_color IS NOT NULL AND size(m.dominant_color) = 7
            WITH substring(m.dominant_color, 1, 1) AS r,
                 substring(m.dominant_color, 3, 1) AS g,
                 substring(m.dominant_color, 5, 1) AS b
            WITH CASE
                WHEN r = '0' AND g = '0' AND b = '0' THEN 'black'
                WHEN r = 'f' AND g = 'f' AND b = 'f' THEN 'white'
                WHEN r = g AND g = b                  THEN 'gray'
                WHEN r > g AND r > b                  THEN 'red'
                WHEN g > r AND g > b                  THEN 'green'
                WHEN b > r AND b > g                  THEN 'blue'
                WHEN r = g AND r > b                  THEN 'yellow'
                WHEN r = b AND r > g                  THEN 'magenta'
                WHEN g = b AND g > r                  THEN 'cyan'
                ELSE 'mixed'
            END AS hue
            RETURN hue, count(*) AS count
            ORDER BY count DESC
        """)
        colors_dom_brightness = await rows(f"""
            {appears_in} WHERE m.dominant_color IS NOT NULL AND size(m.dominant_color) = 7
            WITH (substring(m.dominant_color, 1, 1) + substring(m.dominant_color, 3, 1) + substring(m.dominant_color, 5, 1)) AS rgb
            WITH CASE
                WHEN rgb <= '555' THEN 'dark'
                WHEN rgb <= 'aaa' THEN 'mid'
                ELSE 'bright'
            END AS band
            RETURN band, count(*) AS count
            ORDER BY count DESC
        """)

        trivia_busiest_days = await rows(f"""
            {appears_in} WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH substring(toString(m.timestamp), 0, 10) AS day
            RETURN day, count(*) AS count
            ORDER BY count DESC LIMIT 5
        """)
        trivia_by_weekday = await rows(f"""
            {appears_in} WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH datetime(toString(m.timestamp)).weekday AS dow
            WITH CASE dow
                WHEN 1 THEN 'Mon' WHEN 2 THEN 'Tue' WHEN 3 THEN 'Wed'
                WHEN 4 THEN 'Thu' WHEN 5 THEN 'Fri' WHEN 6 THEN 'Sat'
                WHEN 7 THEN 'Sun'
            END AS weekday
            RETURN weekday, count(*) AS count
            ORDER BY count DESC
        """)
        trivia_by_hour = await rows(f"""
            {appears_in} WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH datetime(toString(m.timestamp)).hour AS hour
            RETURN hour, count(*) AS count
            ORDER BY hour ASC
        """)
        trivia_busiest_month = await rows(f"""
            {appears_in} WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH toInteger(substring(toString(m.timestamp), 5, 2)) AS mo
            RETURN mo, count(*) AS count
            ORDER BY mo ASC
        """)
        trivia_distinct_cameras = await one(f"{appears_in} WHERE m.camera_model IS NOT NULL RETURN count(DISTINCT m.camera_model)")
        trivia_avg_megapixels = await one(f"MATCH (subj:Person {{id: $pid}})-[:APPEARS_IN]->(m:Media:Photo) WHERE m.megapixels IS NOT NULL RETURN round(avg(m.megapixels) * 10) / 10.0")

        # People-per-photo distribution for THIS person's photos only.
        trivia_people_per_photo = await rows(f"""
            MATCH (subj:Person {{id: $pid}})-[:APPEARS_IN]->(m:Media:Photo)
            OPTIONAL MATCH (q:Person)-[:APPEARS_IN]->(m)
            WITH m, count(q) AS faces
            WITH CASE
                WHEN faces = 1 THEN '1'
                WHEN faces = 2 THEN '2'
                WHEN faces <= 5 THEN '3-5'
                ELSE '6+'
            END AS bucket
            RETURN bucket, count(*) AS count
            ORDER BY bucket
        """)

        # Co-appearance partners — who shows up most often with the subject.
        partners = await rows("""
            MATCH (subj:Person {id: $pid})-[:APPEARS_IN]->(m:Media)<-[:APPEARS_IN]-(other:Person)
            WHERE other.id <> $pid
            RETURN other.id AS id,
                   coalesce(other.known_as, other.name, '?') AS name,
                   other.avatar AS avatar,
                   count(DISTINCT m) AS shared_photos
            ORDER BY shared_photos DESC
            LIMIT 15
        """)

        # Manual redates / life-stage locks scoped to this person's photos.
        manual_redates = await one(f"{appears_in} WHERE m.timestamp_source = 'manual' RETURN count(DISTINCT m)")
        life_stage_locks = await one("MATCH (subj:Person {id: $pid})-[r:LIFE_STAGE]->() RETURN count(r)")
        favorites = await one(f"{appears_in} OPTIONAL MATCH (:Person)-[fav:FAVORITED]->(m) WITH m, count(fav) > 0 AS faved WHERE faved RETURN count(DISTINCT m)")

    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source": "neo4j",
        "subject": subject,
        "media": {
            "total": media_total,
            "by_type": [{"labels": r["labels"], "count": r["count"]} for r in media_by_label],
            "total_bytes": media_bytes,
            "with_timestamp": media_with_ts,
            "by_decade": [{"decade": r["decade"], "count": r["count"]} for r in media_by_decade],
        },
        "timestamps": {
            "by_source":     [{"source": r["source"],         "count": r["count"]} for r in ts_by_source],
            "by_confidence": [{"confidence": r["confidence"], "count": r["count"]} for r in ts_by_confidence],
            "by_precision":  [{"precision": r["precision"],   "count": r["count"]} for r in ts_by_precision],
        },
        "places": {
            "with_gps":         with_gps,
            "with_city":        with_city,
            "distinct_cities":  distinct_cities,
            "top_cities": [
                {"city": r["city"], "state": r["state"], "count": r["count"]}
                for r in top_cities
            ],
        },
        "cameras": [
            {"make": r["make"], "model": r["model"], "count": r["count"]}
            for r in top_cameras
        ],
        "heritage": {
            "total":              heritage_total,
            "with_content_date":  heritage_with_content_date,
            "with_description":   heritage_with_description,
            "with_transcription": heritage_with_transcription,
            "by_physical_status": [{"status": r["status"], "count": r["count"]} for r in heritage_by_status],
        },
        "colors": {
            "dominant_by_hue":        [{"hue": r["hue"], "count": r["count"]} for r in colors_dom_hue],
            "dominant_by_brightness": [{"band": r["band"], "count": r["count"]} for r in colors_dom_brightness],
        },
        "trivia": {
            "busiest_days":         [{"day": r["day"], "count": r["count"]} for r in trivia_busiest_days],
            "by_weekday":           [{"weekday": r["weekday"], "count": r["count"]} for r in trivia_by_weekday],
            "by_hour_of_day":       [{"hour": r["hour"], "count": r["count"]} for r in trivia_by_hour],
            "by_month":             [{"month": r["mo"], "count": r["count"]} for r in trivia_busiest_month],
            "distinct_cameras":     trivia_distinct_cameras,
            "avg_megapixels":       trivia_avg_megapixels,
            "people_per_photo":     [{"bucket": r["bucket"], "count": r["count"]} for r in trivia_people_per_photo],
        },
        "engagement": {
            "favorites":         favorites,
            "manual_redates":    manual_redates,
            "life_stage_locks":  life_stage_locks,
        },
        "partners": [
            {"id": r["id"], "name": r["name"], "avatar": r["avatar"], "shared_photos": r["shared_photos"]}
            for r in partners
        ],
    }


@router.get("/whoami")
async def whoami(request: Request):
    """Debug: dump every Cloudflare / forwarded header reaching the backend.
    Use this to confirm Cloudflare Access is injecting identity headers."""
    return {
        "resolved_email":  _current_email(request),
        "cf_headers":      {k: v for k, v in request.headers.items() if k.lower().startswith("cf-")},
        "forwarded":       {k: v for k, v in request.headers.items() if "forwarded" in k.lower()},
        "client_host":     request.client.host if request.client else None,
        "all_lower_keys":  sorted(request.headers.keys()),
    }


@router.get("/me")
async def me(request: Request):
    """Current user identity. Resolved from Cloudflare Access header (or dev
    fallback), then matched to a Person by email. Returns null fields if no
    matching Person record exists yet."""
    email = _current_email(request)
    if not email:
        return {"email": None, "person": None}
    async with get_session() as session:
        result = await session.run(
            "MATCH (p:Person {email: $email}) "
            "RETURN p.id AS id, p.name AS name, p.known_as AS known_as, p.avatar AS avatar, p.email AS email",
            email=email,
        )
        row = await result.single()
    from app.deps import is_admin_email, can_see_gallery
    return {
        "email":           email,
        "is_admin":        is_admin_email(email),
        "can_see_gallery": can_see_gallery(email),
        "person":          dict(row) if row else None,
    }
