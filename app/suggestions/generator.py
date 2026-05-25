"""
Suggestion generator — runs all rules and upserts Suggestion nodes into Neo4j.
MERGE on (type, person_id, target_id) means rejected/accepted suggestions are never overwritten.
"""
import json
import uuid
from datetime import datetime, timezone

from app.db.neo4j import get_session


async def _upsert(session, *, type, person_id, target_id, reason, confidence, metadata=None):
    await session.run(
        """
        MERGE (s:Suggestion {type: $type, person_id: $person_id, target_id: $target_id})
        ON CREATE SET
            s.id         = $id,
            s.status     = 'pending',
            s.reason     = $reason,
            s.confidence = $confidence,
            s.metadata   = $metadata,
            s.created_at = $created_at
        """,
        type=type,
        person_id=person_id or '',
        target_id=target_id or '',
        id=str(uuid.uuid4()),
        reason=reason,
        confidence=confidence,
        metadata=json.dumps(metadata or {}),
        created_at=datetime.now(timezone.utc).isoformat(),
    )


# ── group_membership ──────────────────────────────────────────────────────────

async def _group_membership_photo(session):
    """≥ half of a photo's people are in group X → suggest the rest."""
    result = await session.run("""
        MATCH (photo:Photo)<-[:APPEARS_IN]-(person:Person)
        WITH photo, collect(DISTINCT person) AS photo_people, count(DISTINCT person) AS photo_count
        WHERE photo_count >= 3
        MATCH (photo)<-[:APPEARS_IN]-(member:Person)-[:MEMBER_OF]->(g:Group)
        WITH photo, photo_people, photo_count, g, collect(DISTINCT member) AS in_group
        WHERE size(in_group) * 2 >= photo_count
        UNWIND photo_people AS candidate
        WITH candidate, g, in_group, photo_count
        WHERE NOT (candidate)-[:MEMBER_OF]->(g)
          AND NOT candidate IN in_group
        RETURN DISTINCT
            candidate.id AS person_id,
            g.id         AS group_id,
            g.name       AS group_name,
            size(in_group)          AS matched,
            photo_count,
            toFloat(size(in_group)) / photo_count AS confidence
    """)
    for r in await result.data():
        await _upsert(
            session,
            type='group_membership',
            person_id=r['person_id'],
            target_id=r['group_id'],
            reason=f"{r['matched']} of their photo companions are members of {r['group_name']}",
            confidence=min(r['confidence'], 0.9),
        )


# ── relationship ──────────────────────────────────────────────────────────────

async def _relationship_name_photos(session):
    """Same last name + appear in same photos → suggest family relationship."""
    result = await session.run("""
        MATCH (a:Person)-[:APPEARS_IN]->(photo:Photo)<-[:APPEARS_IN]-(b:Person)
        WHERE a.id < b.id
          AND a.name IS NOT NULL AND b.name IS NOT NULL
          AND size(split(a.name, ' ')) > 1
          AND size(split(b.name, ' ')) > 1
          AND split(a.name, ' ')[-1] = split(b.name, ' ')[-1]
          AND NOT (a)-[:PARENT_OF*]-(b)
          AND NOT (a)-[:MARRIED_TO]-(b)
        WITH a, b, split(a.name, ' ')[-1] AS last_name, count(DISTINCT photo) AS shared_photos
        RETURN a.id AS person_id, b.id AS target_id, last_name, shared_photos
        ORDER BY shared_photos DESC
    """)
    for r in await result.data():
        await _upsert(
            session,
            type='relationship',
            person_id=r['person_id'],
            target_id=r['target_id'],
            reason=f"Share last name '{r['last_name']}' and appear in {r['shared_photos']} photo(s) together",
            confidence=min(0.5 + r['shared_photos'] * 0.05, 0.85),
        )


async def _relationship_name_group(session):
    """Same last name + share a group → suggest family relationship."""
    result = await session.run("""
        MATCH (a:Person)-[:MEMBER_OF]->(g:Group)<-[:MEMBER_OF]-(b:Person)
        WHERE a.id < b.id
          AND a.name IS NOT NULL AND b.name IS NOT NULL
          AND size(split(a.name, ' ')) > 1
          AND size(split(b.name, ' ')) > 1
          AND split(a.name, ' ')[-1] = split(b.name, ' ')[-1]
          AND NOT (a)-[:PARENT_OF*]-(b)
          AND NOT (a)-[:MARRIED_TO]-(b)
        RETURN DISTINCT
            a.id AS person_id, b.id AS target_id,
            split(a.name, ' ')[-1] AS last_name, g.name AS group_name
    """)
    for r in await result.data():
        await _upsert(
            session,
            type='relationship',
            person_id=r['person_id'],
            target_id=r['target_id'],
            reason=f"Share last name '{r['last_name']}' and are both in {r['group_name']}",
            confidence=0.6,
        )


# ── connection ────────────────────────────────────────────────────────────────

async def _connection_shared_group(session):
    """Share a group but no KNOWS edge."""
    result = await session.run("""
        MATCH (a:Person)-[:MEMBER_OF]->(g:Group)<-[:MEMBER_OF]-(b:Person)
        WHERE a.id < b.id
          AND NOT (a)-[:KNOWS]-(b)
          AND NOT (a)-[:PARENT_OF*]-(b)
          AND NOT (a)-[:MARRIED_TO]-(b)
        RETURN DISTINCT a.id AS person_id, b.id AS target_id, g.name AS group_name
    """)
    for r in await result.data():
        await _upsert(
            session,
            type='connection',
            person_id=r['person_id'],
            target_id=r['target_id'],
            reason=f"Both members of {r['group_name']}",
            confidence=0.7,
        )


async def _connection_photos(session):
    """≥ 10 shared photos, no KNOWS edge."""
    result = await session.run("""
        MATCH (a:Person)-[:APPEARS_IN]->(photo:Photo)<-[:APPEARS_IN]-(b:Person)
        WHERE a.id < b.id
          AND NOT (a)-[:KNOWS]-(b)
          AND NOT (a)-[:PARENT_OF*]-(b)
          AND NOT (a)-[:MARRIED_TO]-(b)
        WITH a, b, count(DISTINCT photo) AS shared_photos
        WHERE shared_photos >= 10
        RETURN a.id AS person_id, b.id AS target_id, shared_photos
        ORDER BY shared_photos DESC
    """)
    for r in await result.data():
        await _upsert(
            session,
            type='connection',
            person_id=r['person_id'],
            target_id=r['target_id'],
            reason=f"Appear together in {r['shared_photos']} photos",
            confidence=min(0.5 + r['shared_photos'] * 0.02, 0.9),
        )


async def _connection_friend_of_friend(session):
    """A-B-C path exists but no A-C edge."""
    result = await session.run("""
        MATCH (a:Person)-[:KNOWS]-(b:Person)-[:KNOWS]-(c:Person)
        WHERE a.id < c.id
          AND NOT (a)-[:KNOWS]-(c)
          AND NOT (a)-[:PARENT_OF*]-(c)
          AND NOT (a)-[:MARRIED_TO]-(c)
        WITH a, c, collect(DISTINCT b.name) AS through_names, count(DISTINCT b) AS mutual
        RETURN a.id AS person_id, c.id AS target_id, through_names[0] AS through_name, mutual
        ORDER BY mutual DESC
        LIMIT 300
    """)
    for r in await result.data():
        extra = f" and {r['mutual'] - 1} others" if r['mutual'] > 1 else ""
        await _upsert(
            session,
            type='connection',
            person_id=r['person_id'],
            target_id=r['target_id'],
            reason=f"Both know {r['through_name']}{extra}",
            confidence=min(0.4 + r['mutual'] * 0.1, 0.75),
        )


# ── birth_year ────────────────────────────────────────────────────────────────

async def _birth_year(session):
    """Classmates where one has DOB → suggest the other's birth year."""
    result = await session.run("""
        MATCH (a:Person)-[ra:MEMBER_OF]->(g:Group)<-[rb:MEMBER_OF]-(b:Person)
        WHERE g.type IN ['school', 'school_class']
          AND (ra.role IS NULL OR ra.role IN ['Student', 'Classmate'])
          AND (rb.role IS NULL OR rb.role IN ['Student', 'Classmate'])
          AND a.birth_date IS NOT NULL
          AND b.birth_date IS NULL
          AND a.id <> b.id
        WITH b, a, g,
             toInteger(substring(a.birth_date, 0, 4)) AS ref_year,
             CASE g.type WHEN 'school_class' THEN 0 ELSE 2 END AS tolerance
        RETURN DISTINCT
            b.id AS person_id,
            ref_year, tolerance,
            g.name   AS group_name,
            a.name   AS ref_person
        LIMIT 100
    """)
    for r in await result.data():
        await _upsert(
            session,
            type='birth_year',
            person_id=r['person_id'],
            target_id='',
            reason=f"Classmate of {r['ref_person']} in {r['group_name']} (born ~{r['ref_year']})",
            confidence=0.6,
            metadata={'suggested_year': r['ref_year'], 'tolerance': r['tolerance']},
        )


# ── maiden_name ───────────────────────────────────────────────────────────────

async def _maiden_name(session):
    """Has spouse but no maiden_name set."""
    result = await session.run("""
        MATCH (a:Person)-[:MARRIED_TO]-(b:Person)
        WHERE (a.maiden_name IS NULL OR a.maiden_name = '')
        RETURN DISTINCT a.id AS person_id, b.id AS spouse_id, b.name AS spouse_name
    """)
    for r in await result.data():
        await _upsert(
            session,
            type='maiden_name',
            person_id=r['person_id'],
            target_id=r['spouse_id'],
            reason=f"Married to {r['spouse_name']} — maiden name not recorded",
            confidence=0.5,
        )


# ── location ──────────────────────────────────────────────────────────────────

async def _group_location(session):
    """Group has no location but ≥ 2 members share a birth_place."""
    result = await session.run("""
        MATCH (p:Person)-[:MEMBER_OF]->(g:Group)
        WHERE (g.location_name IS NULL OR g.location_name = '')
          AND p.birth_place IS NOT NULL AND p.birth_place <> ''
        WITH g, p.birth_place AS place, count(*) AS cnt
        ORDER BY cnt DESC
        WITH g, collect({place: place, cnt: cnt})[0] AS top
        WHERE top.cnt >= 2
        RETURN g.id AS group_id, top.place AS suggested_location, top.cnt AS member_count
    """)
    for r in await result.data():
        await _upsert(
            session,
            type='location',
            person_id='',
            target_id=r['group_id'],
            reason=f"{r['member_count']} members share birthplace '{r['suggested_location']}'",
            confidence=0.6,
            metadata={'suggested_location': r['suggested_location']},
        )


# ── group_year ────────────────────────────────────────────────────────────────

async def _group_year(session):
    """school_class group has no year but ≥ 2 student members have birth dates."""
    result = await session.run("""
        MATCH (p:Person)-[r:MEMBER_OF]->(g:Group)
        WHERE g.type = 'school_class'
          AND (g.year IS NULL OR g.year = '')
          AND p.birth_date IS NOT NULL
          AND (r.role IS NULL OR r.role IN ['Student', 'Classmate'])
        WITH g, avg(toInteger(substring(p.birth_date, 0, 4))) AS avg_birth, count(p) AS cnt
        WHERE cnt >= 2
        RETURN g.id AS group_id, toInteger(avg_birth) + 18 AS suggested_year, cnt
    """)
    for r in await result.data():
        await _upsert(
            session,
            type='group_year',
            person_id='',
            target_id=r['group_id'],
            reason=f"Based on {r['cnt']} members' birth dates, graduation year ~{r['suggested_year']}",
            confidence=0.65,
            metadata={'suggested_year': r['suggested_year']},
        )


# ── missing_ancestry ─────────────────────────────────────────────────────────

async def _missing_ancestry(session):
    """Person has no parents in the graph."""
    result = await session.run("""
        MATCH (p:Person)
        WHERE NOT ()-[:PARENT_OF]->(p)
        RETURN p.id AS person_id, p.name AS name
        ORDER BY p.name
    """)
    for r in await result.data():
        await _upsert(
            session,
            type='missing_ancestry',
            person_id=r['person_id'],
            target_id='',
            reason=f"No parents recorded for {r['name']}",
            confidence=1.0,
        )


# ── entry point ───────────────────────────────────────────────────────────────

async def run_all():
    async with get_session() as session:
        await session.run("MATCH (s:Suggestion {status: 'pending'}) DELETE s")
        await _group_membership_photo(session)
        await _relationship_name_photos(session)
        await _relationship_name_group(session)
        await _connection_shared_group(session)
        await _connection_photos(session)
        await _connection_friend_of_friend(session)
        await _birth_year(session)
        await _maiden_name(session)
        await _group_location(session)
        await _group_year(session)
        await _missing_ancestry(session)
