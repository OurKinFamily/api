"""
LLM-powered search.

Pipeline:
  1. planner   — Ollama (qwen3:8b) parses free-text → structured plan JSON
  2. resolvers — resolve free-text person + place names to IDs / coords
  3. executor  — translate plan into Cypher, return Media list
  4. router    — POST /search returns { media, debug }

Phase 1 v0: structured signals only (person, year, place, date_match,
date_match-from-person-dob, media_type, sort, limit). CLIP fallback
deferred to Phase 2.

The whole pipeline is single-shot and side-effect-free — every call is
idempotent. Plans get cached by query hash so identical queries skip
the LLM round-trip.
"""

import json
import re
import time
import hashlib
from typing import Any, Optional

import httpx

from app.db.neo4j import get_session
from app.log import logger as log


# ── Config ───────────────────────────────────────────────────────────────────

OLLAMA_HOST  = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:8b"
# Cold-start on qwen3:8b can take 30+s. Keep generous; users will rarely
# wait that long since the model stays loaded after the first call.
OLLAMA_TIMEOUT = 120.0

# In-memory plan cache. Keyed by query hash. Cleared on process restart.
_plan_cache: dict[str, dict] = {}

# Dynamic context fetched on startup + every 1h.
_archive_meta: dict = {"min_year": 1916, "max_year": 2026, "fetched_at": 0.0}
_ARCHIVE_META_TTL = 3600.0


# ── Place shortcut table (mirrors /home/stephen/.config/mmp/config.json) ─────
# Inlined here so the API doesn't depend on a user's home config. Update both
# when adding a new shortcut.
PLACE_SHORTCUTS: dict[str, tuple[float, float]] = {
    "cottage":          (42.7070, -71.1631),
    "didi-house":       (42.787539, -71.075354),
    "didis":            (42.787539, -71.075354),
    "lawrence":         (42.715292, -71.142224),
    "haverhill":        (42.7762, -71.0773),
    "haverhill-house":  (42.771948, -71.065564),
    "fish-tale-diner":  (42.816813, -70.870638),
    "sandown":          (42.958108, -71.167963),
    "sandown-house":    (42.9284, -71.1872),
    "plaistow":         (42.843390, -71.115378),
    "salisbury":        (42.84655, -70.86163),
    "salisbury-beach":  (42.84655, -70.86163),
    "hampton-beach":    (42.909441, -70.810173),
    "canobie":          (42.793876, -71.248950),
    "loon":             (44.037048, -71.621969),
    "newbury":          (42.77742, -70.85022),
    "newburyport":      (42.81237, -70.88785),
    "miller-field":     (42.917874, -71.170236),
    "syracuse":         (43.04356, -76.15065),
    "miami":            (25.77976, -80.19877),
    "san-diego":        (32.7157, -117.1611),
    "exeter":           (42.977231, -70.948613),
    "durham":           (43.138861, -70.931741),
    "north-hampton":    (42.966193, -70.836039),
    "kingston-lake":    (42.919274, -71.063061),
    "timberlane":       (42.843390, -71.115379),
    "lancaster-pa":     (40.0470, -76.3040),
    "lancaster-general": (40.0470, -76.3040),
    "point-sebago":     (43.8467, -70.5456),
    "jellystone":       (44.1053, -71.1822),
    "pine-acres":       (43.0193, -71.1642),
    "golden-hill":      (42.773494, -71.066392),
    "austin":           (30.284904, -97.715400),
    "unca":             (35.615105, -82.567113),
}

# Default GPS-match radius in degrees (~22km at MA latitudes). Generous —
# place names often refer to a region, not a single coordinate.
PLACE_RADIUS_DEG = 0.2


# ── Dynamic context ──────────────────────────────────────────────────────────

async def _archive_year_range() -> tuple[int, int]:
    """Min/max year from Neo4j Media nodes, cached 1h."""
    if (time.time() - _archive_meta["fetched_at"]) < _ARCHIVE_META_TTL:
        return _archive_meta["min_year"], _archive_meta["max_year"]
    try:
        async with get_session() as session:
            res = await session.run(
                """
                MATCH (m:Media) WHERE m.timestamp IS NOT NULL
                RETURN min(toInteger(substring(toString(m.timestamp), 0, 4))) AS lo,
                       max(toInteger(substring(toString(m.timestamp), 0, 4))) AS hi
                """
            )
            row = await res.single()
        if row and row["lo"] and row["hi"]:
            _archive_meta["min_year"] = int(row["lo"])
            _archive_meta["max_year"] = int(row["hi"])
    except Exception as e:
        log.warning(f"archive_year_range fetch failed: {e}")
    _archive_meta["fetched_at"] = time.time()
    return _archive_meta["min_year"], _archive_meta["max_year"]


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """You are the query planner for a family photo/video archive. Convert
natural-language queries into a structured JSON plan for the server to execute.

OUTPUT RULES
- Return ONLY a single JSON object. No prose, no markdown, no code fences, no <think> tags.
- All fields optional. Omit any field you can't fill confidently.
- Server resolves free-text names + places to IDs — do not invent IDs.

SCHEMA
{
  "intent":        "media_search" | "count" | "fact" | "unknown",
                   // media_search: return photos/videos matching filters (default)
                   // count: user asked "how many ..." — return a NUMBER + samples
                   // fact: user asked a question about a person/thing — return text answer
                   // unknown: query made no sense — explain politely
  "person_names":  [string],         // free-text; server resolves
  "require_all":   boolean,           // true=AND, false=OR. default true for 2+ names
  "year_from":     int|null,          // inclusive
  "year_to":       int|null,          // inclusive; same as year_from for single year
  "age_from":      int|null,          // person's age. Server applies to person_names[0] DOB
  "age_to":        int|null,          // works WITH person_names — leave year_from/to null
  "date_match":    {"month": int, "day": int} | {"from_person_dob": string} | null,
  "place_name":    string|null,
  "media_type":    "photo"|"video"|"document"|"all",  // default "all"
  "sort":          "asc"|"desc",      // default "desc"
  "limit":         int,                // default 48; use 1 for "first/last X"
  "clip_text":     string|null,        // LAST RESORT: only when no structured signal fits
  "fact_kind":     "age"|"birthdate"|"birthplace"|"photo_count"|null
                   // for intent=fact: which attribute to look up about person_names[0]
}

INTENT DETECTION
- "how many ..." / "count ..." / "what's the total ..." → intent="count"
- "how old is X" / "when was X born" / "where was X born" → intent="fact" + fact_kind
- "show me / find / videos of / photos of / first photo of" → intent="media_search" (default)
- otherwise → intent="media_search"

PRONOUNS (he/she/her/him/them/they/his/hers/their)
- NEVER put pronouns in person_names. Resolve to the actual NAME from the prior
  conversation turn. Copy `person_names` from the prior plan exactly.
- Example: prior plan has person_names=["Cayce"]; user says "photos of her in 2000"
  → emit person_names=["Cayce"] (NOT ["her"]).

AGE DESCRIPTORS — emit age_from/age_to instead of year_from/year_to when query
describes the SUBJECT'S age. Server computes the year window from that person's DOB.
  - "as a baby"        → age_from=0, age_to=2
  - "as a toddler"     → age_from=1, age_to=4
  - "as a kid"/"young" → age_from=5, age_to=12
  - "as a teen"/"teenager" → age_from=13, age_to=19
  - "in their 20s"     → age_from=20, age_to=29
  - "in their 30s"     → age_from=30, age_to=39
  - "older"/"recent"   → age_from=40 (when no prior context)

ROUTING PRIORITY
1. ALWAYS prefer structured (person, year, place, date) over CLIP.
2. clip_text is a fallback. If person+year+place+date cover intent, leave clip_text null.
3. Use clip_text only for purely visual queries (e.g. "red bucket", "snowy backyard").

HIGH-PRIORITY PATTERNS
- "<X>'s birthday" / "<X>'s anniversary" → person_names=[X], date_match={"from_person_dob": X}
- "first photo of <X>" → person_names=[X], sort="asc", limit=1
- "last photo of <X>" → person_names=[X], sort="desc", limit=1
- "<X> and <Y>" → person_names=[X, Y], require_all=true
- bare year ("1994") → year_from=1994, year_to=1994
- "in the 70s"/"in the 1970s" → year_from=1970, year_to=1979
- "at the <place>" / "from <place>" → place_name="<place>"
- "videos of <X>" → media_type="video", person_names=[X]
- "every Halloween" → date_match={"month": 10, "day": 31}
- "every Christmas" → date_match={"month": 12, "day": 25}
- "thanksgiving" → no date_match (date varies year to year — leave to user clarify)

FAMILY ROLES (pass through verbatim, server resolves):
"mom"/"mother"/"mum", "dad"/"father", "grandma", "grandpa", "uncle X", "aunt X"

EXAMPLES
Q: Margaret's birthday
A: {"person_names":["Margaret"],"date_match":{"from_person_dob":"Margaret"}}

Q: Stephen and Patty at the beach in 1994
A: {"person_names":["Stephen","Patty"],"require_all":true,"year_from":1994,"year_to":1994,"place_name":"beach"}

Q: first photo of Henry and Dorothy together
A: {"person_names":["Henry","Dorothy"],"require_all":true,"sort":"asc","limit":1}

Q: videos at the cottage
A: {"media_type":"video","place_name":"cottage"}

Q: every Halloween photo
A: {"date_match":{"month":10,"day":31}}

Q: 1995
A: {"year_from":1995,"year_to":1995}

Q: Mom in the 80s
A: {"person_names":["Mom"],"year_from":1980,"year_to":1989}

Q: Stephen as a kid
A: {"person_names":["Stephen"],"age_from":5,"age_to":12}

Q: Patty as a teenager
A: {"person_names":["Patty"],"age_from":13,"age_to":19}

Q: photos with a red bucket
A: {"clip_text":"red bucket"}

Q: how many photos of Henry are there?
A: {"intent":"count","person_names":["Henry"]}

Q: how many videos of Patty at the cottage?
A: {"intent":"count","person_names":["Patty"],"place_name":"cottage","media_type":"video"}

Q: how old is Margaret?
A: {"intent":"fact","person_names":["Margaret"],"fact_kind":"age"}

Q: when was Eddie born?
A: {"intent":"fact","person_names":["Eddie"],"fact_kind":"birthdate"}

Q: where was Patty born?
A: {"intent":"fact","person_names":["Patty"],"fact_kind":"birthplace"}
"""


def build_system_prompt(min_year: int, max_year: int, today: str) -> str:
    return SYSTEM_PROMPT_BASE + f"""

DYNAMIC CONTEXT
ARCHIVE_DATE_RANGE: {min_year} to {max_year} (pre-1948 is very sparse — heritage scans only)
KNOWN_PLACE_SHORTCUTS: {", ".join(sorted(PLACE_SHORTCUTS.keys()))}
TODAY: {today}
"""


# ── Planner (Ollama) ─────────────────────────────────────────────────────────

async def _ollama_generate(prompt: str, system: str) -> tuple[str, dict]:
    """Call Ollama /api/generate. Returns (response_text, raw_response_dict).

    `think: False` disables Qwen 3's chain-of-thought block — for structured
    query parsing we don't need (or want) the model to "think aloud".
    Cuts response time from 10–30s to ~1s once the model is warm.

    `keep_alive: 30m` tells Ollama to keep the model in VRAM longer than the
    default 5min idle eviction — avoids the cold-load penalty between bursts.
    """
    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        r = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model":      OLLAMA_MODEL,
                "system":     system,
                "prompt":     prompt,
                "stream":     False,
                "format":     "json",
                "think":      False,
                "keep_alive": "30m",
                "options":    {"temperature": 0.0},
            },
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", ""), data


def _query_hash(query: str, history: list[dict] | None = None) -> str:
    payload = f"{OLLAMA_MODEL}|{query.strip().lower()}"
    if history:
        # Hash the history shape too so refined queries don't collide with bare ones
        payload += "||" + json.dumps([{"q": h.get("q"), "plan": h.get("plan")} for h in history], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _format_history(history: list[dict]) -> str:
    """Render prior turns for the LLM prompt. Trims to last 3."""
    if not history:
        return ""
    recent = history[-3:]
    lines = [
        "",
        "PRIOR CONVERSATION",
        "Use the most recent plan as a starting point. If the new query is a REFINEMENT",
        "(e.g. 'older', 'younger', 'a bit later', 'and at the cottage', 'without Mom'),",
        "emit a plan that BUILDS on the most recent prior plan: copy unchanged fields,",
        "modify only what the new query implies.",
        "",
    ]
    for i, h in enumerate(recent, 1):
        lines.append(f"Turn {i} user: {h.get('q', '')}")
        lines.append(f"Turn {i} plan: {json.dumps(h.get('plan') or {})}")
    lines.append("")
    lines.append("REFINEMENT RULES (apply when query is a follow-up):")
    lines.append('- "older" / "a little older" / "later" → shift age range UP by 5 (e.g. age_from 5→10, age_to 12→17). If prior plan used year_from/year_to, shift those up by 5 instead.')
    lines.append('- "younger" / "a little younger" / "earlier" → shift age range DOWN by 5. Same for year.')
    lines.append('- "much older/younger" → shift by 10 instead of 5.')
    lines.append('- "and at the cottage" / "but at X" → add or replace place_name; keep other fields.')
    lines.append('- "without <name>" → remove from person_names.')
    lines.append('- "but only photos" / "videos only" → change media_type; keep other fields.')
    lines.append("")
    lines.append("RESET: if the new query introduces a completely different subject (e.g. prior was 'Stephen as a kid' and new query is 'photos of Dorothy in 1965'), ignore prior turns and plan fresh.")
    return "\n".join(lines)


PRONOUNS = {"he", "she", "her", "him", "his", "hers", "they", "them", "their", "theirs", "it", "its"}


def _coref_substitute(plan: dict, history: list[dict] | None) -> dict:
    """If the plan contains pronouns in person_names, swap them out for the
    most recent prior turn's person_names. Qwen sometimes ignores prompt
    instructions about coreference; this is a deterministic backstop."""
    if not history or not plan:
        return plan
    names = plan.get("person_names") or []
    if not names:
        return plan
    has_pronoun = any((n or "").strip().lower() in PRONOUNS for n in names)
    if not has_pronoun:
        return plan
    # Walk history backwards for the first turn with non-pronoun person_names
    for turn in reversed(history):
        prior_names = (turn.get("plan") or {}).get("person_names") or []
        clean = [n for n in prior_names if (n or "").strip().lower() not in PRONOUNS]
        if clean:
            plan = {**plan, "person_names": clean, "_coref_substituted": {"from": names, "to": clean}}
            return plan
    return plan


async def plan_query(query: str, today: str, history: list[dict] | None = None) -> dict:
    """Returns {plan, raw_response, system_prompt, cached, ms, error}.
    Never raises — errors come back in the `error` field so the search
    endpoint can render a useful debug payload instead of returning 500."""
    qh = _query_hash(query, history)
    if qh in _plan_cache:
        return {**_plan_cache[qh], "cached": True}

    min_y, max_y = await _archive_year_range()
    system = build_system_prompt(min_y, max_y, today)
    if history:
        system = system + "\n" + _format_history(history)
    t0 = time.time()
    raw = ""
    error = None
    try:
        raw, _ = await _ollama_generate(query, system)
    except httpx.ReadTimeout:
        error = f"Ollama timed out after {OLLAMA_TIMEOUT}s — model may be cold-loading. Retry in a few seconds."
    except httpx.ConnectError:
        error = "Could not reach Ollama at " + OLLAMA_HOST + " — is `ollama serve` running?"
    except Exception as e:
        error = f"Ollama call failed: {type(e).__name__}: {e}"
    ms = int((time.time() - t0) * 1000)

    plan = {}
    if raw:
        try:
            plan = json.loads(raw)
        except Exception:
            # Sometimes the model wraps JSON in markdown or appends a <think> tag.
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                try:
                    plan = json.loads(m.group(0))
                except Exception:
                    pass

    # Server-side coreference: pronouns → prior person names
    plan = _coref_substitute(plan, history)

    result = {
        "plan":           plan,
        "raw_response":   raw,
        "system_prompt":  system,
        "user_query":     query,
        "model":          OLLAMA_MODEL,
        "ms":             ms,
        "cached":         False,
        "error":          error,
    }
    # Don't cache errors so the next retry actually retries.
    if not error:
        _plan_cache[qh] = result
    return result


# ── Resolvers ────────────────────────────────────────────────────────────────

async def resolve_person(name: str, session) -> Optional[dict]:
    """Fuzzy-match a free-text name to a Person node. Returns {id, name, dob}
    or None. Ranks by name match quality then face-count."""
    if not name or not name.strip():
        return None
    needle = name.strip().lower()

    res = await session.run(
        """
        MATCH (p:Person)
        WHERE toLower(p.name) CONTAINS $q
           OR toLower(coalesce(p.known_as, '')) CONTAINS $q
           OR toLower(coalesce(p.maiden_name, '')) CONTAINS $q
        OPTIONAL MATCH (p)-[r:APPEARS_IN]->(m:Media)
        WITH p, count(DISTINCT m) AS face_count
        RETURN p.id AS id, p.name AS name, p.known_as AS known_as,
               toString(coalesce(p.birth_date, p.dob)) AS dob, face_count
        ORDER BY
          CASE WHEN toLower(p.name) = $q OR toLower(coalesce(p.known_as,'')) = $q THEN 0 ELSE 1 END,
          face_count DESC
        LIMIT 5
        """,
        q=needle,
    )
    rows = await res.data()
    if not rows:
        return None
    top = rows[0]
    return {
        "id":           top["id"],
        "name":         top["name"],
        "known_as":     top.get("known_as"),
        "dob":          top.get("dob"),
        "face_count":   top.get("face_count", 0),
        "alternatives": [
            {"id": r["id"], "name": r["name"], "face_count": r["face_count"]}
            for r in rows[1:]
        ],
    }


def resolve_place(name: str) -> Optional[dict]:
    """Match a free-text place name against PLACE_SHORTCUTS. Returns
    {slug, lat, lng, radius_deg} or None."""
    if not name:
        return None
    needle = name.strip().lower().replace(" ", "-")
    # Exact slug match
    if needle in PLACE_SHORTCUTS:
        lat, lng = PLACE_SHORTCUTS[needle]
        return {"slug": needle, "lat": lat, "lng": lng, "radius_deg": PLACE_RADIUS_DEG, "match": "exact"}
    # Substring match
    for slug, (lat, lng) in PLACE_SHORTCUTS.items():
        if needle in slug or slug in needle:
            return {"slug": slug, "lat": lat, "lng": lng, "radius_deg": PLACE_RADIUS_DEG, "match": "substring"}
    return None


async def resolve_plan(plan: dict, session) -> dict:
    """Resolve free-text names / places in a plan. Returns enriched plan +
    resolution metadata for debugging."""
    resolved = {
        "person_ids":    [],
        "unresolved":    [],
        "place":         None,
        "dob_lookups":   {},
        "alternatives":  {},
    }

    for n in plan.get("person_names") or []:
        p = await resolve_person(n, session)
        if p:
            resolved["person_ids"].append(p["id"])
            if p.get("dob"):
                resolved["dob_lookups"][n] = p["dob"]
            if p.get("alternatives"):
                resolved["alternatives"][n] = p["alternatives"]
        else:
            resolved["unresolved"].append(n)

    if plan.get("place_name"):
        resolved["place"] = resolve_place(plan["place_name"])

    # If date_match references a person DOB, resolve it.
    dm = plan.get("date_match")
    if dm and isinstance(dm, dict) and "from_person_dob" in dm:
        name = dm["from_person_dob"]
        dob = resolved["dob_lookups"].get(name)
        person_found = False
        if not dob:
            # Resolve fresh if not already in person_names
            p = await resolve_person(name, session)
            if p:
                person_found = True
                if p.get("dob"):
                    dob = p["dob"]
                    resolved["dob_lookups"][name] = dob
        else:
            person_found = True
        if dob:
            # dob format: "1961-04-12" or "1961-04-12T00:00:00"
            try:
                m_int = int(dob[5:7])
                d_int = int(dob[8:10])
                resolved["date_match_resolved"] = {"month": m_int, "day": d_int, "from": name, "dob": dob}
            except Exception:
                resolved["date_match_unresolved"] = {"from": name, "reason": f"could not parse dob: {dob!r}"}
        else:
            resolved["date_match_unresolved"] = {
                "from":   name,
                "reason": f"no DOB set in Neo4j for '{name}'" if person_found else f"could not find person '{name}'",
            }
    elif dm and isinstance(dm, dict) and "month" in dm and "day" in dm:
        resolved["date_match_resolved"] = {"month": int(dm["month"]), "day": int(dm["day"])}

    # Resolve age_from/age_to into year_from/year_to using the first resolved
    # person's DOB. Year fields take precedence — if the LLM emitted both,
    # we leave year_from/year_to alone.
    age_from = plan.get("age_from")
    age_to   = plan.get("age_to")
    if (age_from is not None or age_to is not None) and not plan.get("year_from") and not plan.get("year_to"):
        # Look up first person's DOB. Reuse dob_lookups if already resolved.
        person_dob_year = None
        person_dob_name = None
        for n in plan.get("person_names") or []:
            dob = resolved["dob_lookups"].get(n)
            if not dob:
                p = await resolve_person(n, session)
                if p and p.get("dob"):
                    dob = p["dob"]
                    resolved["dob_lookups"][n] = dob
            if dob and len(dob) >= 4 and dob[:4].isdigit():
                person_dob_year = int(dob[:4])
                person_dob_name = n
                break
        if person_dob_year:
            if age_from is not None:
                plan["year_from"] = person_dob_year + int(age_from)
            if age_to is not None:
                plan["year_to"] = person_dob_year + int(age_to)
            resolved["age_resolved"] = {
                "from_person": person_dob_name,
                "dob_year":    person_dob_year,
                "age_from":    age_from,
                "age_to":      age_to,
                "year_from":   plan.get("year_from"),
                "year_to":     plan.get("year_to"),
            }
        else:
            resolved["age_unresolved"] = {
                "reason":         "no DOB found for any named person",
                "person_names":   plan.get("person_names"),
            }

    return resolved


# ── Executor ─────────────────────────────────────────────────────────────────

def _build_filters(plan: dict, resolved: dict, active: set[str]) -> tuple[list[str], dict]:
    """Build Cypher WHERE clauses + bind params from plan + resolved, but ONLY
    for filters in the `active` set. Used by the auto-broaden loop to try
    successively looser queries."""
    where = ["m.timestamp IS NOT NULL"]
    params: dict[str, Any] = {}

    if "person" in active:
        pids = resolved.get("person_ids") or []
        if pids:
            require_all = plan.get("require_all", True if len(pids) >= 2 else False)
            if require_all and len(pids) >= 2:
                where.append(
                    "ALL(pid IN $person_ids WHERE EXISTS { (:Person {id: pid})-[:APPEARS_IN]->(m) })"
                )
            else:
                where.append(
                    "ANY(pid IN $person_ids WHERE EXISTS { (:Person {id: pid})-[:APPEARS_IN]->(m) })"
                )
            params["person_ids"] = pids

    if "year" in active:
        if plan.get("year_from"):
            where.append("toInteger(substring(toString(m.timestamp), 0, 4)) >= $year_from")
            params["year_from"] = int(plan["year_from"])
        if plan.get("year_to"):
            where.append("toInteger(substring(toString(m.timestamp), 0, 4)) <= $year_to")
            params["year_to"] = int(plan["year_to"])

    if "year_wide" in active:
        # Widened year range: ±2 years around the original window
        yf = plan.get("year_from")
        yt = plan.get("year_to")
        if yf:
            where.append("toInteger(substring(toString(m.timestamp), 0, 4)) >= $year_from")
            params["year_from"] = int(yf) - 2
        if yt:
            where.append("toInteger(substring(toString(m.timestamp), 0, 4)) <= $year_to")
            params["year_to"] = int(yt) + 2

    if "date_match" in active:
        dmr = resolved.get("date_match_resolved")
        if dmr:
            where.append("toInteger(substring(toString(m.timestamp), 5, 2)) = $dm_month")
            where.append("toInteger(substring(toString(m.timestamp), 8, 2)) = $dm_day")
            params["dm_month"] = dmr["month"]
            params["dm_day"]   = dmr["day"]

    if "place" in active or "place_wide" in active:
        place = resolved.get("place")
        if place:
            radius = place["radius_deg"] * (3.0 if "place_wide" in active else 1.0)
            where.append(
                "m.latitude IS NOT NULL AND m.longitude IS NOT NULL"
                " AND abs(m.latitude - $place_lat)  < $place_radius"
                " AND abs(m.longitude - $place_lng) < $place_radius"
            )
            params["place_lat"]    = place["lat"]
            params["place_lng"]    = place["lng"]
            params["place_radius"] = radius

    if "media_type" in active:
        media_type = plan.get("media_type", "all")
        if media_type == "video":
            where.append("m.is_video = true")
        elif media_type == "photo":
            where.append("(m.is_video IS NULL OR m.is_video = false)")

    return where, params


async def _run_cypher(plan: dict, where: list[str], params: dict, session) -> tuple[list[dict], str]:
    """Build the final Cypher with sort + limit, run it, materialize media."""
    sort = (plan.get("sort") or "desc").lower()
    if sort not in ("asc", "desc"):
        sort = "desc"
    limit = int(plan.get("limit") or 48)
    limit = max(1, min(limit, 500))
    params = {**params, "limit": limit}

    where_clause = " AND ".join(where) if where else "true"
    cypher = f"""
    MATCH (m:Media)
    WHERE {where_clause}
    RETURN m {{
      .path, .timestamp, .timestamp_confidence, .dominant_color,
      .is_video, .width, .height, .place_name, .city,
      .poster_path, .filename
    }} AS p
    ORDER BY m.timestamp {sort}, m.path {sort}
    LIMIT $limit
    """
    res = await session.run(cypher, **params)
    rows = await res.data()

    from app.config import with_v
    from pathlib import Path

    media = []
    for r in rows:
        p = r["p"]
        path = p.get("path", "")
        poster = p.get("poster_path")
        thumbnail_url = with_v(
            f"/api/media/{poster}" if poster
            else f"/api/media/thumb/{path.removeprefix('archive/')}"
        )
        media.append({
            "path":           path,
            "url":            with_v(f"/api/media/{path}"),
            "thumbnail_url":  thumbnail_url,
            "filename":       p.get("filename") or Path(path).name,
            "timestamp":      p.get("timestamp"),
            "confidence":     p.get("timestamp_confidence"),
            "dominant_color": p.get("dominant_color"),
            "is_video":       p.get("is_video", False),
            "width":          p.get("width"),
            "height":         p.get("height"),
            "place_name":     p.get("place_name"),
            "city":           p.get("city"),
        })
    return media, cypher.strip()


# Drop order: from most-restrictive / most-likely-to-be-wrong → least.
# `person` is never dropped — it's the user's anchor.
# YEAR is never dropped either — if user said "2000", returning 2026 photos is
# worse than returning nothing. Year is only WIDENED.
def _broaden_steps(active: set[str], plan: dict, resolved: dict) -> list[tuple[str, set[str]]]:
    """Return [(reason, new_active_set), …] in priority order.

    Only emits steps for filters that were ACTUALLY applied in the original
    plan — avoids the misleading "dropped place filter" message when no place
    was ever in the query.
    """
    steps = []
    has_place = bool(resolved.get("place"))
    has_date_match = bool(resolved.get("date_match_resolved"))
    has_year = bool(plan.get("year_from") or plan.get("year_to"))
    has_media_type = bool(plan.get("media_type") and plan.get("media_type") != "all")

    if "place" in active and has_place:
        next_set = (active - {"place"}) | {"place_wide"}
        steps.append(("widened place radius (≈60km instead of ≈20km)", next_set))
    if has_place and ("place" in active or "place_wide" in active):
        steps.append(("dropped place filter", active - {"place", "place_wide"}))
    if "date_match" in active and has_date_match:
        steps.append(("dropped exact-date filter (kept year)", active - {"date_match"}))
    if "year" in active and has_year:
        next_set = (active - {"year"}) | {"year_wide"}
        steps.append(("widened year range ±2", next_set))
    if "media_type" in active and has_media_type:
        steps.append(("dropped media type filter", active - {"media_type"}))
    # NOTE: year is deliberately never dropped — "Cayce in 2000" should not
    # silently return 2026 photos. Empty result is honest.
    return steps


def build_answer_card(
    query: str,
    plan: dict,
    resolved: dict,
    dropped: list[str],
    media_count: int,
    *,
    intent: str = "media_search",
    total_count: int | None = None,
    fact_result: dict | None = None,
) -> dict:
    """Compose a human-readable answer card from the resolution + execution
    outputs. Server-side templates only (no second LLM round-trip). Aim:
    quick, deterministic explanation of what the search did/didn't do.

    Returned shape: { message, tone, details: [str] }.
    """
    person_names = [
        # Use resolved name from dob_lookups if available, else free-text
        (resolved.get("alternatives") or {}).get(n) and n or n
        for n in (plan.get("person_names") or [])
    ]
    # Get nice display names for resolved persons by re-deriving from resolved
    # — alternatives store losing matches, so first-line resolved is implicit.
    # For human messaging, free-text names are fine.
    resolved_names = list(plan.get("person_names") or [])

    media_type = plan.get("media_type") or "all"
    media_noun = {"video": "video", "photo": "photo", "document": "document", "all": "result"}[media_type]
    if media_count != 1:
        media_noun = media_noun + "s"

    details: list[str] = []
    notes:   list[str] = []
    tone = "info"

    # Person resolution notes
    if resolved.get("unresolved"):
        notes.append(f"couldn't find {', '.join(resolved['unresolved'])} in your archive")
        tone = "warning"
    if resolved.get("alternatives"):
        for name, alts in resolved["alternatives"].items():
            alt_names = ", ".join(a["name"] for a in alts[:2])
            details.append(f"Picked top match for '{name}' (also matched: {alt_names})")

    # Date / age unresolved warnings
    dm_un = resolved.get("date_match_unresolved")
    if dm_un:
        notes.append(f"couldn't find a birthdate for {dm_un['from']}")
        tone = "warning"
    age_un = resolved.get("age_unresolved")
    if age_un:
        notes.append("couldn't apply age filter (no birthdate)")
        tone = "warning"

    # Auto-broadened
    if dropped:
        notes.append("had to broaden the search (" + "; ".join(dropped) + ")")
        tone = "warning"

    # Compose primary message
    subject = ""
    if resolved_names:
        subject = " of " + (
            " and ".join(resolved_names) if len(resolved_names) <= 2
            else ", ".join(resolved_names[:-1]) + ", and " + resolved_names[-1]
        )
    where = ""
    if resolved.get("place"):
        where = f" at the {resolved['place']['slug']}"
    when = ""
    if plan.get("year_from") and plan.get("year_to") and plan["year_from"] == plan["year_to"]:
        when = f" in {plan['year_from']}"
    elif plan.get("year_from") and plan.get("year_to"):
        when = f" between {plan['year_from']} and {plan['year_to']}"
    elif plan.get("year_from"):
        when = f" from {plan['year_from']}"
    elif plan.get("year_to"):
        when = f" up through {plan['year_to']}"
    dm_r = resolved.get("date_match_resolved")
    if dm_r:
        when = (when + ", " if when else " ") + f"on {dm_r['from']}'s birthday ({dm_r['month']:02d}-{dm_r['day']:02d})"

    # FACT intent: short answer, no media context needed
    if intent == "fact" and fact_result:
        if fact_result.get("error"):
            return {"message": fact_result["error"], "tone": "warning", "details": details}
        return {"message": fact_result.get("text", ""), "tone": "info", "details": details}

    # COUNT intent: lead with the real total
    if intent == "count" and total_count is not None:
        # singular/plural-correct noun off real total, not sample size
        n = total_count
        media_noun_real = {"video": "video", "photo": "photo", "document": "document", "all": "result"}[media_type]
        if n != 1:
            media_noun_real += "s"
        if n == 0:
            message = f"I couldn't find any {media_noun_real}{subject}{where}{when}."
            tone = "warning"
        else:
            message = f"There {'are' if n != 1 else 'is'} {n:,} {media_noun_real}{subject}{where}{when}."
            if media_count > 0:
                message += f" Showing the {'most recent ' if (plan.get('sort') or 'desc') == 'desc' else 'earliest '}{media_count}."
        if notes:
            message = "I " + " and ".join(notes) + ". " + message
        return {"message": message.strip(), "tone": tone, "details": details}

    # MEDIA_SEARCH (default)
    if media_count == 0:
        # When nothing matched, the broadening notes describe attempts that DIDN'T
        # help — don't lead with them. Just say: nothing found, here's what was tried.
        tone = "warning"
        target = f"{media_noun}{subject}{where}{when}".strip() or "matches"
        message = f"I couldn't find any {target}."
        if dropped:
            message += " I tried widening the search too — still nothing."
        if resolved.get("date_match_unresolved"):
            message += f" (Couldn't apply the birthday filter: {resolved['date_match_unresolved']['reason']}.)"
        if resolved.get("age_unresolved"):
            message += " (Couldn't apply the age filter — no birthdate on record.)"
        if resolved.get("unresolved"):
            message += f" (Couldn't find: {', '.join(resolved['unresolved'])}.)"
    elif notes:
        joined_notes = "I " + " and ".join(notes) + "."
        message = f"{joined_notes} Here are {media_count} {media_noun}{subject}{where}{when}."
    else:
        message = f"Here are {media_count} {media_noun}{subject}{where}{when}."

    return {"message": message.strip(), "tone": tone, "details": details}


def capitalize_first(s: str) -> str:
    return (s[:1].upper() + s[1:]) if s else s


async def count_plan(plan: dict, resolved: dict, session) -> tuple[int, str, dict]:
    """For intent=count: run a COUNT query, return (count, cypher, params)."""
    active = {"person", "year", "date_match", "place", "media_type"}
    where, params = _build_filters(plan, resolved, active)
    where_clause = " AND ".join(where) if where else "true"
    cypher = f"MATCH (m:Media) WHERE {where_clause} RETURN count(m) AS n"
    res = await session.run(cypher, **params)
    row = await res.single()
    return int(row["n"] if row else 0), cypher.strip(), params


async def fact_plan(plan: dict, resolved: dict, session) -> dict:
    """For intent=fact: look up an attribute about the first named person.
    Returns {kind, value, text} or {error}.
    """
    pids = resolved.get("person_ids") or []
    if not pids:
        return {"error": "couldn't find that person"}
    pid = pids[0]
    kind = plan.get("fact_kind") or "age"
    res = await session.run(
        """
        MATCH (p:Person {id: $id})
        OPTIONAL MATCH (p)-[r:APPEARS_IN]->(m:Media)
        WITH p, count(DISTINCT m) AS photo_count
        RETURN p.name AS name, p.known_as AS known_as,
               toString(p.birth_date) AS birth_date,
               p.birth_place AS birth_place,
               p.is_living AS is_living,
               photo_count
        """,
        id=pid,
    )
    row = await res.single()
    if not row:
        return {"error": "person not found in DB"}
    name = row["name"] or "unknown"
    bd   = row.get("birth_date")
    bp   = row.get("birth_place")
    pc   = row.get("photo_count") or 0
    if kind == "age":
        if not bd:
            return {"kind": "age", "value": None, "text": f"I don't have a birthdate for {name}."}
        from datetime import date
        try:
            by, bm, bday = int(bd[0:4]), int(bd[5:7]), int(bd[8:10])
            today = date.today()
            age = today.year - by - ((today.month, today.day) < (bm, bday))
            return {"kind": "age", "value": age, "text": f"{name} is {age} (born {bd})."}
        except Exception:
            return {"kind": "age", "value": None, "text": f"Couldn't parse birthdate for {name}: {bd!r}"}
    if kind == "birthdate":
        if bd:
            return {"kind": "birthdate", "value": bd, "text": f"{name} was born on {bd}."}
        return {"kind": "birthdate", "value": None, "text": f"I don't have a birthdate for {name}."}
    if kind == "birthplace":
        if bp:
            return {"kind": "birthplace", "value": bp, "text": f"{name} was born in {bp}."}
        return {"kind": "birthplace", "value": None, "text": f"I don't have a birthplace for {name}."}
    if kind == "photo_count":
        return {"kind": "photo_count", "value": pc, "text": f"{name} appears in {pc:,} photos."}
    return {"error": f"unknown fact_kind: {kind}"}


async def execute_plan(plan: dict, resolved: dict, session) -> tuple[list[dict], str, dict, list[str]]:
    """Run the plan as written. If zero results, progressively broaden until
    we get hits or exhaust strategies. Returns (media, final_cypher, params, dropped).

    `dropped` is a human-readable list of what was loosened to get the result.
    Empty list = original query worked. Non-empty = auto-broadened.
    """
    active = {"person", "year", "date_match", "place", "media_type"}

    # First pass: original plan
    where, params = _build_filters(plan, resolved, active)
    media, cypher = await _run_cypher(plan, where, params, session)
    if media:
        return media, cypher, params, []

    # Zero results — start broadening
    dropped: list[str] = []
    for reason, next_active in _broaden_steps(active, plan, resolved):
        active = next_active
        dropped.append(reason)
        where, params = _build_filters(plan, resolved, active)
        media, cypher = await _run_cypher(plan, where, params, session)
        if media:
            return media, cypher, params, dropped

    # Truly nothing — return last attempt's (empty) result
    return media, cypher, params, dropped
