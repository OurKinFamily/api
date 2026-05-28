import uuid
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from app.db.neo4j import get_session
from app.log import logger
from app.models.person import Person, PersonCreate, RelationshipAdd


def _ctx(request: Request) -> dict:
    """Standard bindings every mutation logger needs."""
    return {
        "by": getattr(request.state, "user_email", None),
        "request_id": getattr(request.state, "request_id", None),
    }


class AvatarSet(BaseModel):
    crop_path: str


class CoverSet(BaseModel):
    photo_path: str | None = None  # null = clear cover, fall back to random
    position:   str | None = None  # 'top' | 'center' | 'bottom' (default center)

router = APIRouter(prefix="/people", tags=["people"])


@router.get("/search")
async def search_people(q: str, limit: int = 20):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person)
            WHERE toLower(p.name) CONTAINS toLower($q)
               OR toLower(coalesce(p.known_as, '')) CONTAINS toLower($q)
            RETURN p ORDER BY p.name LIMIT $limit
            """,
            q=q, limit=limit
        )
        records = await result.data()
        return [Person(**r["p"]) for r in records]


@router.get("/", response_model=list[Person])
async def list_people():
    async with get_session() as session:
        result = await session.run("MATCH (p:Person) RETURN p ORDER BY p.name")
        records = await result.data()
        return [Person(**r["p"]) for r in records]


@router.post("/", response_model=Person, status_code=201)
async def create_person(data: PersonCreate, request: Request):
    async with get_session() as session:
        person_id = str(uuid.uuid4())
        result = await session.run(
            """
            CREATE (p:Person {
              id: $id, name: $name, known_as: $known_as,
              birth_date: $birth_date, birth_date_precision: $birth_date_precision,
              is_living: true
            })
            RETURN p
            """,
            id=person_id,
            name=data.name,
            known_as=data.known_as,
            birth_date=data.birth_date,
            birth_date_precision=data.birth_date_precision,
        )
        record = await result.single()
        logger.bind(
            event="person.created",
            person_id=person_id,
            name=data.name,
            **_ctx(request),
        ).info("person created")
        return Person(**record["p"])


class PersonUpdate(BaseModel):
    name: str
    known_as: str | None = None
    maiden_name: str | None = None
    birth_date: str | None = None
    birth_date_precision: str | None = None
    birth_place: str | None = None
    death_date: str | None = None
    death_date_precision: str | None = None
    death_place: str | None = None
    burial_place: str | None = None
    immigration_date: str | None = None
    immigration_place: str | None = None
    naturalization_date: str | None = None
    naturalization_place: str | None = None
    ssn: str | None = None
    is_living: bool = True
    notes: str | None = None


@router.patch("/{person_id}", response_model=Person)
async def update_person(person_id: str, data: PersonUpdate, request: Request):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})
            SET p.name = $name, p.known_as = $known_as, p.maiden_name = $maiden_name,
                p.birth_date = $birth_date, p.birth_date_precision = $birth_date_precision,
                p.birth_place = $birth_place, p.death_date = $death_date,
                p.death_date_precision = $death_date_precision, p.death_place = $death_place,
                p.burial_place = $burial_place,
                p.immigration_date = $immigration_date, p.immigration_place = $immigration_place,
                p.naturalization_date = $naturalization_date, p.naturalization_place = $naturalization_place,
                p.ssn = $ssn,
                p.is_living = $is_living, p.notes = $notes
            RETURN p
            """,
            id=person_id, name=data.name, known_as=data.known_as,
            maiden_name=data.maiden_name, birth_date=data.birth_date,
            birth_date_precision=data.birth_date_precision, birth_place=data.birth_place,
            death_date=data.death_date, death_date_precision=data.death_date_precision,
            death_place=data.death_place,
            burial_place=data.burial_place,
            immigration_date=data.immigration_date, immigration_place=data.immigration_place,
            naturalization_date=data.naturalization_date, naturalization_place=data.naturalization_place,
            ssn=data.ssn,
            is_living=data.is_living, notes=data.notes,
        )
        record = await result.single()
        if not record:
            raise HTTPException(404, "Person not found")
        logger.bind(
            event="person.updated",
            person_id=person_id,
            name=data.name,
            **_ctx(request),
        ).info("person updated")
        return Person(**record["p"])


@router.get("/year-density")
async def people_year_density(min_photos: int = Query(default=30, ge=1)):
    """Per-person yearly photo counts. Returns one row per person whose
    total >= min_photos: `[{id, name, known_as, avatar, total, points: [{year, count}]}]`.
    Powers the Family page ridgeline view."""
    async with get_session() as session:
        res = await session.run(
            """
            MATCH (p:Person)-[:APPEARS_IN]->(m:Media)
            WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH p, toInteger(substring(toString(m.timestamp), 0, 4)) AS year
            WITH p, year, count(*) AS c
            WITH p, collect({year: year, count: c}) AS points,
                 sum(c) AS total,
                 min(year) AS first_year
            WHERE total >= $min_photos
            RETURN p.id        AS id,
                   p.name      AS name,
                   p.known_as  AS known_as,
                   p.avatar    AS avatar,
                   points,
                   total,
                   first_year
            ORDER BY first_year ASC, total DESC
            """,
            min_photos=min_photos,
        )
        return await res.data()


@router.get("/timeline")
async def people_timeline(min_photos: int = Query(default=30, ge=1)):
    """Per-person gantt rows across ALL photos: first / last timestamp +
    total photo count. Same row shape as /connection-timeline so the same
    UI component can render it. Powers the Family page.

    Filters out the `1700-01-01` undated sentinel so min() doesn't latch
    onto it — keeps real heritage photos (1900s+) intact."""
    async with get_session() as session:
        res = await session.run(
            """
            MATCH (p:Person)-[:APPEARS_IN]->(m:Media)
            WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH p,
                 min(m.timestamp) AS first_ts,
                 max(m.timestamp) AS last_ts,
                 count(DISTINCT m) AS photo_count
            WHERE photo_count >= $min_photos
            RETURN p.id        AS id,
                   p.name      AS name,
                   p.known_as  AS known_as,
                   p.avatar    AS avatar,
                   toString(first_ts) AS first_ts,
                   toString(last_ts)  AS last_ts,
                   photo_count
            ORDER BY first_ts ASC
            """,
            min_photos=min_photos,
        )
        return await res.data()


@router.get("/{person_id}", response_model=Person)
async def get_person(person_id: str):
    async with get_session() as session:
        result = await session.run(
            "MATCH (p:Person {id: $id}) RETURN p",
            id=person_id
        )
        record = await result.single()
        if not record:
            raise HTTPException(status_code=404, detail="Person not found")
        return Person(**record["p"])


@router.get("/{person_id}/connection-timeline")
async def connection_timeline(person_id: str, min_photos: int = Query(default=30, ge=1)):
    """For each other person who appears in ≥ min_photos with the subject,
    return the date range of their co-appearances and the photo count.
    Used by the PersonPage Timeline tab to render a gantt-style view of
    overlapping social eras."""
    async with get_session() as session:
        res = await session.run(
            """
            MATCH (subject:Person {id: $id})-[:APPEARS_IN]->(m:Media)<-[:APPEARS_IN]-(other:Person)
            WHERE other.id <> $id AND m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            WITH other,
                 min(m.timestamp) AS first_ts,
                 max(m.timestamp) AS last_ts,
                 count(DISTINCT m) AS photo_count
            WHERE photo_count >= $min_photos
            RETURN other.id        AS id,
                   other.name      AS name,
                   other.known_as  AS known_as,
                   other.avatar    AS avatar,
                   toString(first_ts) AS first_ts,
                   toString(last_ts)  AS last_ts,
                   photo_count
            ORDER BY first_ts ASC
            """,
            id=person_id, min_photos=min_photos,
        )
        return await res.data()


@router.post("/{person_id}/relationships", status_code=204)
async def add_relationship(person_id: str, body: RelationshipAdd, request: Request):
    async with get_session() as session:
        # Verify both people exist
        check = await session.run(
            "MATCH (a:Person {id: $a}), (b:Person {id: $b}) RETURN count(*) AS n",
            a=person_id, b=body.target_id
        )
        row = await check.single()
        if not row or row["n"] == 0:
            raise HTTPException(status_code=404, detail="One or both people not found")

        if body.rel_type == "spouse":
            await session.run(
                """
                MATCH (a:Person {id: $a}), (b:Person {id: $b})
                MERGE (a)-[:MARRIED_TO]->(b)
                MERGE (b)-[:MARRIED_TO]->(a)
                """,
                a=person_id, b=body.target_id
            )

        elif body.rel_type == "child":
            await session.run(
                "MATCH (a:Person {id: $a}), (b:Person {id: $b}) MERGE (a)-[:PARENT_OF]->(b)",
                a=person_id, b=body.target_id
            )

        elif body.rel_type == "parent":
            await session.run(
                "MATCH (a:Person {id: $a}), (b:Person {id: $b}) MERGE (b)-[:PARENT_OF]->(a)",
                a=person_id, b=body.target_id
            )

        elif body.rel_type == "sibling":
            # Connect each of the focal person's parents to the target
            await session.run(
                """
                MATCH (parent:Person)-[:PARENT_OF]->(focal:Person {id: $focal_id})
                MATCH (target:Person {id: $target_id})
                MERGE (parent)-[:PARENT_OF]->(target)
                """,
                focal_id=person_id, target_id=body.target_id
            )

        else:
            raise HTTPException(status_code=400, detail=f"Unknown relationship type: {body.rel_type}")

        logger.bind(
            event="relationship.created",
            person_id=person_id,
            target_id=body.target_id,
            rel_type=body.rel_type,
            **_ctx(request),
        ).info("relationship created")


class RelationshipRemove(BaseModel):
    target_id: str
    rel_type: str  # spouse, parent, child


@router.delete("/{person_id}/relationships", status_code=204)
async def remove_relationship(person_id: str, body: RelationshipRemove, request: Request):
    async with get_session() as session:
        if body.rel_type == "spouse":
            await session.run(
                "MATCH (a:Person {id: $a})-[r:MARRIED_TO]-(b:Person {id: $b}) DELETE r",
                a=person_id, b=body.target_id,
            )
        elif body.rel_type == "parent":
            await session.run(
                "MATCH (b:Person {id: $b})-[r:PARENT_OF]->(a:Person {id: $a}) DELETE r",
                a=person_id, b=body.target_id,
            )
        elif body.rel_type == "child":
            await session.run(
                "MATCH (a:Person {id: $a})-[r:PARENT_OF]->(b:Person {id: $b}) DELETE r",
                a=person_id, b=body.target_id,
            )
        elif body.rel_type == "sibling":
            await session.run(
                """
                MATCH (parent:Person)-[:PARENT_OF]->(focal:Person {id: $focal_id})
                MATCH (parent)-[r:PARENT_OF]->(target:Person {id: $target_id})
                DELETE r
                """,
                focal_id=person_id, target_id=body.target_id,
            )
        else:
            raise HTTPException(400, f"Unknown rel_type: {body.rel_type}")

        logger.bind(
            event="relationship.deleted",
            person_id=person_id,
            target_id=body.target_id,
            rel_type=body.rel_type,
            **_ctx(request),
        ).info("relationship deleted")


@router.put("/{person_id}/avatar", status_code=204)
async def set_avatar(person_id: str, body: AvatarSet, request: Request):
    async with get_session() as session:
        result = await session.run(
            "MATCH (p:Person {id: $id}) SET p.avatar = $avatar RETURN p",
            id=person_id, avatar=body.crop_path
        )
        record = await result.single()
        if not record:
            raise HTTPException(status_code=404, detail="Person not found")
        logger.bind(
            event="person.avatar_updated",
            person_id=person_id,
            crop_path=body.crop_path,
            **_ctx(request),
        ).info("avatar updated")


@router.put("/{person_id}/cover", status_code=204)
async def set_cover(person_id: str, body: CoverSet, request: Request):
    """Set (or clear, with null) the cover_image used by PersonPage's hero
    banner. `position` controls vertical crop alignment: top / center / bottom."""
    async with get_session() as session:
        result = await session.run(
            "MATCH (p:Person {id: $id}) SET p.cover_image = $cover, p.cover_position = $pos RETURN p",
            id=person_id, cover=body.photo_path, pos=body.position,
        )
        record = await result.single()
        if not record:
            raise HTTPException(status_code=404, detail="Person not found")
        logger.bind(
            event="person.cover_updated",
            person_id=person_id,
            cover_image=body.photo_path,
            position=body.position,
            **_ctx(request),
        ).info("cover updated")


class FaceAssign(BaseModel):
    photo_path: str
    face_index: int
    crop_path:  str

class PhotoTag(BaseModel):
    photo_path: str

@router.post("/{person_id}/photos", status_code=204)
async def tag_photo(person_id: str, body: PhotoTag, request: Request):
    """Tag a person in a photo with no face bbox (manual addition when face
    extraction missed them). Creates APPEARS_IN without face_index/crop_path."""
    async with get_session() as session:
        await session.run(
            """
            MATCH (person:Person {id: $person_id})
            MATCH (photo:Media {path: $photo_path})
            MERGE (person)-[:APPEARS_IN]->(photo)
            """,
            person_id=person_id, photo_path=body.photo_path,
        )
        logger.bind(
            event="person.photo_tagged",
            person_id=person_id,
            photo_path=body.photo_path,
            source="manual",
            **_ctx(request),
        ).info("person tagged in photo")


@router.post("/{person_id}/faces", status_code=204)
async def assign_face(person_id: str, body: FaceAssign, request: Request):
    async with get_session() as session:
        await session.run(
            """
            MATCH (person:Person {id: $person_id})
            MATCH (photo:Media {path: $photo_path})
            MERGE (person)-[r:APPEARS_IN]->(photo)
            SET r.face_index = $face_index,
                r.crop_path  = $crop_path,
                r.photo_path = $photo_path
            """,
            person_id=person_id,
            photo_path=body.photo_path,
            face_index=body.face_index,
            crop_path=body.crop_path,
        )
        logger.bind(
            event="face.assigned",
            person_id=person_id,
            photo_path=body.photo_path,
            face_index=body.face_index,
            crop_path=body.crop_path,
            source="single_assign",
            **_ctx(request),
        ).info("face assigned")


@router.get("/{person_id}/faces")
async def get_faces(person_id: str):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})-[r:APPEARS_IN]->(photo:Media)
            WHERE r.crop_path IS NOT NULL AND r.crop_path <> ''
            RETURN r.crop_path AS crop_path,
                   r.face_index AS face_index,
                   photo.path AS photo_path
            ORDER BY photo.path
            """,
            id=person_id
        )
        records = await result.data()
        return [
            {
                "crop_path":  r["crop_path"],
                "face_index": r["face_index"],
                "photo_path": r["photo_path"],
            }
            for r in records
        ]


def _format_age(age_years: float) -> str:
    if age_years < 0:
        return ""
    if age_years < 1:
        days = age_years * 365.25
        if days < 7:
            return "newborn"
        if days < 60:
            return f"{int(days)} days"
        months = max(1, int(age_years * 12))
        return f"{months} months"
    return f"age {int(age_years)}"


# Bucket definitions: (low_inclusive, high_inclusive, label)
_LIFE_BUCKETS = [
    (0,  1,  "baby"),
    (2,  4,  "toddler"),
    (5,  9,  "kid"),
    (10, 14, "preteen"),
    (15, 19, "teen"),
    (20, 29, "twenties"),
    (30, 39, "thirties"),
    (40, 49, "forties"),
    (50, 59, "fifties"),
    (60, 69, "sixties"),
    (70, 999, "seventies+"),
]


@router.get("/{person_id}/life-stages")
async def life_stages(person_id: str):
    """For each age bucket (baby → seventies+), pick one representative
    photo of this person. Selection: favorited > chronologically first.
    Used by the Person Overview to render the life-stages strip."""
    from datetime import datetime
    async with get_session() as session:
        res = await session.run(
            """
            MATCH (p:Person {id: $id})
            OPTIONAL MATCH (p)-[mine:APPEARS_IN]->(m:Media)
              WHERE m.timestamp IS NOT NULL AND toString(m.timestamp) > '1800-01-01'
            OPTIONAL MATCH (other:Person)-[:APPEARS_IN]->(m)
            OPTIONAL MATCH (:Person)-[fav:FAVORITED]->(m)
            WITH p, m, mine,
                 count(DISTINCT other) AS appears_count,
                 count(fav) > 0        AS favorited
            RETURN p.birth_date AS birth,
                   p.birth_date_precision AS prec,
                   collect(DISTINCT {
                     path: m.path,
                     ts: toString(m.timestamp),
                     is_video: m.is_video,
                     poster_path: m.poster_path,
                     crop_path: mine.crop_path,
                     favorited: favorited,
                     solo: appears_count = 1
                   }) AS photos
            """,
            id=person_id,
        )
        rec = await res.single()
    if not rec or not rec["birth"]:
        return {"buckets": []}

    # Birth date — accept YYYY / YYYY-MM / YYYY-MM-DD
    raw = str(rec["birth"]).split("T")[0]
    parts = raw.split("-")
    try:
        y = int(parts[0])
        m = int(parts[1]) if len(parts) > 1 else 1
        d = int(parts[2]) if len(parts) > 2 else 1
        birth_dt = datetime(y, m, d)
    except (ValueError, IndexError):
        return {"buckets": []}

    photos_in = [p for p in rec["photos"] if p.get("path") and p.get("ts")]
    photos = []
    for p in photos_in:
        try:
            ts = datetime.fromisoformat(p["ts"].split(".")[0].replace("Z", "").rstrip("+"))
        except ValueError:
            continue
        years = (ts - birth_dt).days / 365.25
        if years < 0:
            continue
        photos.append({**p, "ts_parsed": ts, "age_years": years})

    # Stills only — a video's poster frame can't be trusted to actually
    # show the person we're profiling, so drop videos from the candidate
    # pool entirely. Bucket disappears if no stills exist for that age.
    stills_all = [p for p in photos if not p.get("is_video")]
    buckets_out = []
    for lo, hi, label in _LIFE_BUCKETS:
        in_bucket = [p for p in stills_all if lo <= p["age_years"] <= hi]
        if not in_bucket:
            continue
        # Preference cascade: favorited+solo > solo > favorited > anything.
        # "Solo" = target is the only tagged person → most likely a portrait.
        solo = [p for p in in_bucket if p.get("solo")]
        favs = [p for p in in_bucket if p["favorited"]]
        fav_solo = [p for p in solo if p["favorited"]]
        pool = fav_solo or solo or favs or in_bucket
        chosen = min(pool, key=lambda x: x["ts_parsed"])
        # Thumb URL: videos use poster jpg; images use thumb webp route.
        thumb_url = (
            f"/api/media/{chosen['poster_path']}"
            if chosen.get("is_video") and chosen.get("poster_path")
            else f"/api/media/thumb/{chosen['path']}"
        )
        crop_path = chosen.get("crop_path")
        buckets_out.append({
            "bucket": label,
            "age_text": _format_age(chosen["age_years"]),
            "path": chosen["path"],
            "url": f"/api/media/{chosen['path']}",
            "thumb_url": thumb_url,
            "crop_url": f"/api/media/{crop_path}" if crop_path else None,
            "is_video": bool(chosen.get("is_video")),
            "count": len(in_bucket),
        })
    return {"buckets": buckets_out}


@router.get("/{person_id}/photos")
async def get_photos(person_id: str, limit: int = 100, offset: int = 0):
    async with get_session() as session:
        count_result = await session.run(
            "MATCH (p:Person {id: $id})-[:APPEARS_IN]->(photo:Media) RETURN count(DISTINCT photo) AS n",
            id=person_id
        )
        count_rec = await count_result.single()
        total = count_rec["n"] if count_rec else 0

        result = await session.run(
            """
            MATCH (p:Person {id: $id})-[:APPEARS_IN]->(photo:Media)
            RETURN DISTINCT photo.path AS photo_path
            ORDER BY photo.path
            SKIP $skip LIMIT $lim
            """,
            id=person_id, skip=offset, lim=limit
        )
        records = await result.data()
        return {"total": total, "paths": [r["photo_path"] for r in records]}


@router.get("/{person_id}/groups")
async def get_groups(person_id: str):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})-[rel:MEMBER_OF]->(g:Group)
            RETURN g, rel.role AS role
            ORDER BY g.name
            """,
            id=person_id,
        )
        records = await result.data()
    return [{**dict(r["g"]), "role": r["role"]} for r in records]


class ConnectionAdd(BaseModel):
    target_id: str
    context: str | None = None
    since: str | None = None
    through_group_id: str | None = None


@router.get("/{person_id}/connections")
async def get_connections(person_id: str):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})-[r:KNOWS]-(other:Person)
            OPTIONAL MATCH (g:Group) WHERE g.id IN coalesce(r.through_group_ids, [])
            RETURN DISTINCT other, r.context AS context, r.since AS since,
                   coalesce(r.through_group_ids, []) AS through_group_ids,
                   collect(DISTINCT {id: g.id, name: g.name}) AS through_groups
            ORDER BY other.name
            """,
            id=person_id,
        )
        records = await result.data()
    return [
        {
            "id":            r["other"]["id"],
            "name":          r["other"]["name"],
            "known_as":      r["other"].get("known_as"),
            "avatar":        r["other"].get("avatar"),
            "context":       r["context"],
            "since":         r["since"],
            "through_groups": [g for g in r["through_groups"] if g.get("id")],
        }
        for r in records
    ]


@router.post("/{person_id}/connections", status_code=204)
async def add_connection(person_id: str, body: ConnectionAdd, request: Request):
    async with get_session() as session:
        check = await session.run(
            "MATCH (a:Person {id: $a}), (b:Person {id: $b}) RETURN count(*) AS n",
            a=person_id, b=body.target_id,
        )
        row = await check.single()
        if not row or row["n"] == 0:
            raise HTTPException(404, "One or both people not found")
        await session.run(
            """
            MATCH (a:Person {id: $a}), (b:Person {id: $b})
            MERGE (a)-[r:KNOWS]-(b)
            SET r.context = coalesce($context, r.context),
                r.since   = coalesce($since,   r.since),
                r.through_group_ids = CASE
                  WHEN $gid IS NULL THEN coalesce(r.through_group_ids, [])
                  WHEN $gid IN coalesce(r.through_group_ids, []) THEN coalesce(r.through_group_ids, [])
                  ELSE coalesce(r.through_group_ids, []) + [$gid]
                END
            """,
            a=person_id, b=body.target_id,
            context=body.context, since=body.since,
            gid=body.through_group_id,
        )
        logger.bind(
            event="connection.created",
            person_id=person_id,
            target_id=body.target_id,
            context=body.context,
            through_group_id=body.through_group_id,
            **_ctx(request),
        ).info("connection created")


@router.delete("/{person_id}/connections/{other_id}", status_code=204)
async def remove_connection(person_id: str, other_id: str, request: Request):
    async with get_session() as session:
        await session.run(
            """
            MATCH (a:Person {id: $a})-[r:KNOWS]-(b:Person {id: $b})
            DELETE r
            """,
            a=person_id, b=other_id,
        )
        logger.bind(
            event="connection.deleted",
            person_id=person_id,
            target_id=other_id,
            **_ctx(request),
        ).info("connection deleted")


@router.get("/{person_id}/travel/points")
async def get_person_travel_points(person_id: str):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (person:Person {id: $id})-[:APPEARS_IN]->(photo:Media)
            WHERE photo.latitude IS NOT NULL AND photo.longitude IS NOT NULL
            RETURN photo.latitude AS lat, photo.longitude AS lng, photo.path AS path
            """,
            id=person_id,
        )
        records = await result.data()
    points = [[r["lat"], r["lng"], r["path"]] for r in records]
    return {"points": points}


@router.get("/{person_id}/relatives")
async def get_relatives(person_id: str):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})
            OPTIONAL MATCH (p)-[:PARENT_OF]->(child:Person)
            WITH p, collect(DISTINCT {id: child.id, name: child.name, known_as: child.known_as, avatar: child.avatar}) AS children

            // Parents with their parents (grandparents) nested inside.
            OPTIONAL MATCH (parent:Person)-[:PARENT_OF]->(p)
            OPTIONAL MATCH (gp:Person)-[:PARENT_OF]->(parent)
            WITH p, children, parent,
                 collect(DISTINCT {id: gp.id, name: gp.name, known_as: gp.known_as, avatar: gp.avatar}) AS gps_per_parent
            WITH p, children,
                 collect(DISTINCT {
                   id: parent.id, name: parent.name, known_as: parent.known_as, avatar: parent.avatar,
                   parents: [g IN gps_per_parent WHERE g.id IS NOT NULL]
                 }) AS parents

            OPTIONAL MATCH (p)-[:MARRIED_TO]->(spouse:Person)
            WITH p, children, parents, collect(DISTINCT {id: spouse.id, name: spouse.name, known_as: spouse.known_as, avatar: spouse.avatar}) AS spouses

            OPTIONAL MATCH (par2:Person)-[:PARENT_OF]->(p)
            OPTIONAL MATCH (par2)-[:PARENT_OF]->(sibling:Person) WHERE sibling.id <> p.id
            WITH children, parents, spouses, collect(DISTINCT {id: sibling.id, name: sibling.name, known_as: sibling.known_as, avatar: sibling.avatar}) AS siblings
            RETURN children, parents, spouses, siblings
            """,
            id=person_id
        )
        record = await result.single()
        if not record:
            raise HTTPException(status_code=404, detail="Person not found")
        return {
            "parents":  [r for r in record["parents"]  if r["id"]],
            "children": [r for r in record["children"] if r["id"]],
            "spouses":  [r for r in record["spouses"]  if r["id"]],
            "siblings": [r for r in record["siblings"] if r["id"]],
        }
