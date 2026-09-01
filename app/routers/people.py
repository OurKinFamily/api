import math
import re
import uuid
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from app.config import with_v
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
async def list_people(request: Request, viewer_id: str | None = None):
    """People list. When we can tell who's viewing (their Cloudflare email maps
    to a Person, or an admin passes ?viewer_id to preview), the list is ordered
    by closeness to them — their own family first, then anyone sharing their
    surname (e.g. all the Dezazzos for a Dezazzo), then everyone else. Falls back
    to photo-count ordering when there's no known viewer."""
    from app.routers.me import _current_email
    from app.deps import is_admin_email
    email = _current_email(request)
    async with get_session() as session:
        vid = None
        if viewer_id and is_admin_email(email):
            vid = viewer_id
        elif email:
            r = await session.run("MATCH (p:Person {email: $e}) RETURN p.id AS id", e=email)
            row = await r.single()
            vid = row.get("id") if row else None

        if vid:
            result = await session.run(
                """
                MATCH (viewer:Person {id: $vid})
                OPTIONAL MATCH (viewer)-[:PARENT_OF|MARRIED_TO*1..4]-(kin:Person)
                WITH viewer, collect(DISTINCT kin.id) AS kin_ids,
                     last(split(viewer.name, ' ')) AS surname
                MATCH (p:Person)
                OPTIONAL MATCH (p)-[:APPEARS_IN]->(m:Media)
                WITH p, viewer, kin_ids, surname, count(m) AS photo_count
                RETURN p,
                  CASE
                    WHEN p.id = viewer.id THEN 0
                    WHEN p.id IN kin_ids THEN 1
                    WHEN surname <> '' AND p.name CONTAINS surname THEN 2
                    ELSE 3
                  END AS tier, photo_count
                ORDER BY tier, photo_count DESC, p.name
                """,
                vid=vid,
            )
        else:
            result = await session.run(
                """
                MATCH (p:Person)
                OPTIONAL MATCH (p)-[:APPEARS_IN]->(m:Media)
                WITH p, count(m) AS photo_count
                RETURN p ORDER BY photo_count DESC, p.name
                """
            )
        records = await result.data()
        return [Person(**r["p"]) for r in records]


@router.get("/with-biography")
async def people_with_biography(request: Request):
    """Slim list of people who have a non-empty biography — for the Biographies
    index page. Declared before /{person_id} so it isn't captured as an id."""
    from app.deps import is_admin_request
    # Family viewers don't see bios the owner marked private.
    priv_filter = "" if is_admin_request(request) else "AND NOT coalesce(p.bio_private, false)"
    async with get_session() as session:
        result = await session.run(
            f"""
            MATCH (p:Person)
            WHERE p.biography IS NOT NULL AND trim(p.biography) <> '' {priv_filter}
            RETURN p.id AS id, p.name AS name, p.known_as AS known_as,
                   p.avatar AS avatar, p.cover_image AS cover_image,
                   p.birth_date AS birth_date, p.death_date AS death_date,
                   coalesce(p.bio_private, false) AS bio_private
            ORDER BY p.name
            """
        )
        return await result.data()


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
async def get_person(person_id: str, request: Request):
    async with get_session() as session:
        result = await session.run(
            "MATCH (p:Person {id: $id}) RETURN p",
            id=person_id
        )
        record = await result.single()
        if not record:
            raise HTTPException(status_code=404, detail="Person not found")
        from app.deps import is_admin_request
        props = dict(record["p"])
        # Hide the biography from family viewers when the owner marked it private.
        # The person otherwise stays fully visible (tree, photos, relationships).
        if props.get("bio_private") and not is_admin_request(request):
            props["biography"] = None
        return Person(**props)


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


class BiographyBody(BaseModel):
    biography: str | None = None


@router.put("/{person_id}/biography", status_code=204)
async def set_biography(person_id: str, body: BiographyBody, request: Request):
    """Set (or clear, with null) the person's freeform markdown biography.
    Kept as its own endpoint so the general person-edit form can't wipe it."""
    async with get_session() as session:
        result = await session.run(
            "MATCH (p:Person {id: $id}) SET p.biography = $bio RETURN p.id AS id",
            id=person_id, bio=body.biography,
        )
        if not await result.single():
            raise HTTPException(status_code=404, detail="Person not found")
        logger.bind(
            event="person.biography_updated",
            person_id=person_id,
            **_ctx(request),
        ).info("biography updated")


class BioPrivateBody(BaseModel):
    bio_private: bool


@router.patch("/{person_id}/bio-private", status_code=204)
async def set_bio_private(person_id: str, body: BioPrivateBody, request: Request):
    """Mark a person's biography private (hidden from family viewers) or public.
    Admin-only — enforced by the write-lock middleware."""
    async with get_session() as session:
        result = await session.run(
            "MATCH (p:Person {id: $id}) SET p.bio_private = $v RETURN p.id AS id",
            id=person_id, v=body.bio_private,
        )
        if not await result.single():
            raise HTTPException(status_code=404, detail="Person not found")
        logger.bind(
            event="person.bio_private_set",
            person_id=person_id,
            bio_private=body.bio_private,
            **_ctx(request),
        ).info("bio privacy set")


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

        try:
            from app.services.brain import rebuild_person_brain
            from app.routers.faces import _invalidate_brain
            await rebuild_person_brain(person_id, session)
            _invalidate_brain()
        except Exception as e:
            logger.warning(f"brain rebuild failed for {person_id}: {e}")


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


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        s = "th"
    else:
        s = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{s}"


def _format_relationship(n: int, m: int, subject_gender: str | None) -> str:
    """Map LCA distances to an English relationship label, viewer→subject.

    n = viewer's hops up to LCA;  m = subject's hops up to LCA.
    Gendered terms when subject_gender known; fall back to neutral phrasing.
    """
    male = subject_gender == "male"
    female = subject_gender == "female"
    M = lambda f, m, n: f if female else (m if male else n)  # noqa: E731

    if n == 0 and m == 0:
        return "yourself"

    # Subject is viewer's descendant (n=0)
    if n == 0:
        if m == 1: return f"your {M('daughter','son','child')}"
        if m == 2: return f"your grand{M('daughter','son','child')}"
        if m == 3: return f"your great-grand{M('daughter','son','child')}"
        return f"your {m-2}× great-grand{M('daughter','son','child')}"

    # Subject is viewer's ancestor (m=0)
    if m == 0:
        if n == 1: return f"your {M('mother','father','parent')}"
        if n == 2: return f"your grand{M('mother','father','parent')}"
        if n == 3: return f"your great-grand{M('mother','father','parent')}"
        return f"your {n-2}× great-grand{M('mother','father','parent')}"

    # Siblings
    if n == 1 and m == 1:
        return f"your {M('sister','brother','sibling')}"

    # Niblings (subject is descendant of viewer's sibling)
    if n == 1:
        prefix = "great-" * (m - 2)
        if m == 2: return f"your {M('niece','nephew','niece or nephew')}"
        return f"your {prefix}grand-{M('niece','nephew','niece or nephew')}"

    # Aunts / uncles (subject is sibling of viewer's ancestor)
    if m == 1:
        prefix = "great-" * (n - 2)
        if n == 2: return f"your {M('aunt','uncle','aunt or uncle')}"
        return f"your {prefix}grand{M('aunt','uncle','aunt or uncle')}"

    # Cousins
    cousin_n = min(n, m) - 1
    removed = abs(n - m)
    base = f"your {_ordinal(cousin_n)} cousin"
    if removed == 0: return base
    if removed == 1: return f"{base} once removed"
    if removed == 2: return f"{base} twice removed"
    return f"{base} {removed} times removed"


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


# How many frames the strip aims for. The old ladder was a fixed set of named
# brackets (baby / toddler / kid / ...), which meant an 8-year-old could only
# ever fill three of them — Margaret had 15,180 photos rendered as 3 tiles.
# Stride is derived instead, so the strip stays about this long at any age.
_TARGET_TILES = 12

# Bucket key looks like "y5" (a single year) or "y4-7" (a stride of four).
_AGE_BUCKET_RE = re.compile(r"^y(\d+)(?:-(\d+))?$")


def _life_buckets_for(lo_age: int, hi_age: int):
    """(low, high, label) whole-year buckets covering lo_age..hi_age, strided
    so the result lands near _TARGET_TILES.

    Bounds are whole years and `high` is inclusive of that year, so a bucket
    (2, 2) means "age two" and matches a photo at age 2.4 — callers compare with
    `lo <= age < hi + 1`, since ages are fractional.

    Under ten that works out to one bucket per year; a twenty-year-old gets
    two-year strides, a forty-year-old four. Bounds come from the ages actually
    photographed rather than from the person's current age — someone whose
    pictures stop at five should get a dense strip of those five years, not ten
    empty brackets covering a life the archive has nothing from.
    """
    lo_age = max(0, int(lo_age))
    hi_age = max(lo_age, int(hi_age))
    span = hi_age - lo_age + 1
    step = max(1, math.ceil(span / _TARGET_TILES))
    out = []
    for lo in range(lo_age, hi_age + 1, step):
        hi = lo + step - 1
        out.append((lo, hi, f"y{lo}" if step == 1 else f"y{lo}-{hi}"))
    return out


def _resolve_locks(locked_by_label, stills, buckets):
    """Re-key locks onto the current buckets via the AGE OF THE PHOTO each one
    points at, rather than the label it was saved under.

    Bucket labels are not stable: the stride widens as a person ages, and these
    replaced the old named brackets outright. Matching on the stored label would
    silently orphan every existing lock (Margaret's 'toddler' and 'kid' among
    them) the moment the boundaries moved.
    """
    age_by_path = {p["path"]: p["age_years"] for p in stills}
    out = {}
    for path in locked_by_label.values():
        age = age_by_path.get(path)
        if age is None:
            continue  # locked photo no longer in the pool
        for lo, hi, label in buckets:
            if lo <= age < hi + 1:
                out.setdefault(label, path)
                break
    return out


def _bucket_bounds(label):
    """Parse a bucket key back into (lo, hi). None if it isn't an age bucket."""
    m = _AGE_BUCKET_RE.match(label)
    if not m:
        return None
    lo = int(m.group(1))
    return lo, int(m.group(2)) if m.group(2) else lo

# Decade bucket label for the no-birth-date fallback strip, e.g. 1994 -> "1990s".
_DECADE_RE = re.compile(r"^(\d{4})s$")


def _decade_label(ts) -> str:
    return f"{(ts.year // 10) * 10}s"


@router.get("/{person_id}/relationship")
async def relationship_to_viewer(person_id: str, request: Request, viewer_id: str | None = None):
    """Viewer→subject English relationship line for the Person header.

    Walks PARENT_OF up from both viewer and subject, finds the lowest common
    ancestor, formats the (n, m) hop pair into English (parent / cousin /
    great-grand-uncle / etc), and tags maternal/paternal side using the
    viewer's parent gender. Spouse + in-law shortcuts handled before the
    blood-line walk.

    viewer_id: optional override for admin preview-as-person mode.
    """
    from app.routers.me import _current_email
    from app.deps import is_admin_email

    async with get_session() as session:
        # Resolve viewer — use viewer_id override if caller is admin
        if viewer_id and is_admin_email(_current_email(request)):
            resolved_viewer_id = viewer_id
        else:
            email = _current_email(request)
            if not email:
                return {"label": None, "side": None}
            rv = await session.run("MATCH (v:Person {email: $e}) RETURN v.id AS id", e=email)
            vrow = await rv.single()
            if not vrow:
                return {"label": None, "side": None}
            resolved_viewer_id = vrow["id"]

        viewer_id = resolved_viewer_id
        if viewer_id == person_id:
            return {"label": "yourself", "side": None}

        # Direct spouse shortcut
        rs = await session.run(
            """
            MATCH (v:Person {id: $vid})-[:MARRIED_TO]-(s:Person {id: $sid})
            RETURN count(*) > 0 AS yes
            """,
            vid=viewer_id, sid=person_id,
        )
        srow = await rs.single()
        if srow and srow["yes"]:
            return {"label": "your spouse", "side": None}

        # Pull viewer's ancestors (incl self at dist 0). For each non-self
        # ancestor, also capture which immediate parent of the viewer leads
        # to that ancestor (so we can label maternal/paternal side).
        rva = await session.run(
            """
            MATCH path = (v:Person {id: $vid})<-[:PARENT_OF*1..8]-(anc:Person)
            RETURN anc.id           AS id,
                   length(path)     AS dist,
                   nodes(path)[1].id AS first_parent_id
            """,
            vid=viewer_id,
        )
        v_anc: dict[str, tuple[int, str | None]] = {viewer_id: (0, None)}
        for row in await rva.data():
            d = row["dist"]
            cur = v_anc.get(row["id"])
            if cur is None or d < cur[0]:
                v_anc[row["id"]] = (d, row["first_parent_id"])

        # Pull subject's ancestors
        rsa = await session.run(
            """
            MATCH path = (s:Person {id: $sid})<-[:PARENT_OF*0..8]-(anc:Person)
            RETURN anc.id AS id, min(length(path)) AS dist
            """,
            sid=person_id,
        )
        s_anc = {row["id"]: row["dist"] for row in await rsa.data()}

        # Subject's own props (for gendered terms)
        rsp = await session.run("MATCH (s:Person {id: $sid}) RETURN s.gender AS g", sid=person_id)
        subj_row = await rsp.single()
        subj_gender = subj_row["g"] if subj_row else None

        common = set(v_anc) & set(s_anc)
        if common:
            best = min(common, key=lambda x: v_anc[x][0] + s_anc[x])
            n, parent_via = v_anc[best]
            m = s_anc[best]
            label = _format_relationship(n, m, subj_gender)
            side = None
            # Side only meaningful for lateral relationships (cousins, uncles, etc.)
            # Not for direct lines: n=0 (your child) or m=0 (your parent/grandparent).
            # Siblings (n==1, m==1) share both parents so side is arbitrary too.
            if parent_via and n >= 1 and m >= 1 and not (n == 1 and m == 1):
                rp = await session.run(
                    "MATCH (p:Person {id: $pid}) RETURN p.gender AS g",
                    pid=parent_via,
                )
                prow = await rp.single()
                if prow and prow["g"] == "female":
                    side = "your mother's side"
                elif prow and prow["g"] == "male":
                    side = "your father's side"
            return {"label": label, "side": side}

        # In-law: subject married to one of viewer's blood relatives
        rl = await session.run(
            """
            MATCH (s:Person {id: $sid})-[:MARRIED_TO]-(rel:Person)
            RETURN rel.id AS id, rel.gender AS gender
            """,
            sid=person_id,
        )
        for row in await rl.data():
            rel_id = row["id"]
            if rel_id in v_anc:
                vn, parent_via = v_anc[rel_id]
                rel_label = _format_relationship(vn, 0, row["gender"])
                spouse_word = ("wife" if subj_gender == "female"
                               else "husband" if subj_gender == "male"
                               else "spouse")
                label = f"{rel_label}'s {spouse_word}"
                side = None
                if parent_via:
                    rp = await session.run(
                        "MATCH (p:Person {id: $pid}) RETURN p.gender AS g", pid=parent_via,
                    )
                    prow = await rp.single()
                    if prow and prow["g"] == "female": side = "your mother's side"
                    elif prow and prow["g"] == "male": side = "your father's side"
                return {"label": label, "side": side}

        return {"label": None, "side": None}


def _parse_birth(birth_raw):
    from datetime import datetime
    raw = str(birth_raw).split("T")[0]
    parts = raw.split("-")
    try:
        y = int(parts[0])
        mo = int(parts[1]) if len(parts) > 1 else 1
        d = int(parts[2]) if len(parts) > 2 else 1
        return datetime(y, mo, d)
    except (ValueError, IndexError):
        return None


def _life_stage_score(p):
    """Higher = better. Solo+favorited tops the list."""
    return (
        (4 if p["favorited"] and p.get("solo") else 0)
        + (2 if p.get("solo") else 0)
        + (1 if p["favorited"] else 0)
    )


def _life_stage_tile(label, chosen, age_years, count, locked, age_text=None):
    thumb_url = with_v(
        f"/api/media/{chosen['poster_path']}"
        if chosen.get("is_video") and chosen.get("poster_path")
        else f"/api/media/thumb/{chosen['path']}"
    )
    crop_path = chosen.get("crop_path")
    return {
        "bucket": label,
        # Caption ribbon. Age buckets format the numeric age; the no-birth-date
        # fallback passes an explicit decade label (e.g. "1990s").
        "age_text": age_text if age_text is not None else _format_age(age_years),
        "path": chosen["path"],
        "url": with_v(f"/api/media/{chosen['path']}"),
        "thumb_url": thumb_url,
        "crop_url": with_v(f"/api/media/{crop_path}") if crop_path else None,
        "is_video": bool(chosen.get("is_video")),
        "count": count,
        "locked": locked,
    }


async def _fetch_life_stage_pool(session, person_id):
    """Returns (birth_dt, stills_with_age, locked_by_bucket) — shared by
    the list endpoint, the candidates endpoint, and any future tooling.

    If the person has a `death_date`, photos taken meaningfully after
    death are dropped: those are posthumous shots (e.g. a framed portrait
    in the background of a later photo), not "this is what they looked
    like at age 92"."""
    from datetime import datetime
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
        WITH p, collect(DISTINCT {
              path: m.path, ts: toString(m.timestamp),
              is_video: m.is_video, poster_path: m.poster_path,
              crop_path: mine.crop_path,
              favorited: favorited, solo: appears_count = 1
            }) AS photos
        OPTIONAL MATCH (p)-[ls:LIFE_STAGE]->(locked:Media)
        RETURN p.birth_date AS birth, p.death_date AS death, photos,
               collect(DISTINCT {bucket: ls.bucket, path: locked.path}) AS locks
        """,
        id=person_id,
    )
    rec = await res.single()
    if not rec:
        return None, [], {}

    # birth_dt may be None — the caller falls back to a decade-based strip.
    birth_dt = _parse_birth(rec["birth"]) if rec.get("birth") else None

    # Death cutoff (only meaningful with a birth date). Allow a small grace
    # window (1 year) — funeral / memorial photos within ~12 months are still
    # likely the same era. After that, exclude.
    death_dt = _parse_birth(rec["death"]) if rec.get("death") else None
    max_age = (
        ((death_dt - birth_dt).days / 365.25) + 1.0
        if (death_dt and birth_dt) else None
    )

    raw_photos = [p for p in rec["photos"] if p.get("path") and p.get("ts")]
    photos = []
    for p in raw_photos:
        try:
            ts = datetime.fromisoformat(
                p["ts"].split(".")[0].replace("Z", "").rstrip("+")
            )
        except ValueError:
            continue
        if birth_dt is not None:
            years = (ts - birth_dt).days / 365.25
            if years < 0:
                continue
            if max_age is not None and years > max_age:
                continue
            photos.append({**p, "ts_parsed": ts, "age_years": years})
        else:
            photos.append({**p, "ts_parsed": ts, "age_years": None})
    stills = [p for p in photos if not p.get("is_video")]
    locked = {row["bucket"]: row["path"] for row in rec["locks"] if row.get("bucket")}
    return birth_dt, stills, locked


def _sort_bucket(in_bucket):
    """In-place sort: highest score first; oldest first within score."""
    in_bucket.sort(key=lambda x: (-_life_stage_score(x), x["ts_parsed"]))
    return in_bucket


def _pick_chosen(in_bucket, locked_path):
    """Return (chosen_photo, locked_bool). Honors a locked override when the
    locked photo is still in the pool; otherwise auto-picks the best (the
    bucket must already be sorted best-first)."""
    if locked_path:
        chosen = next((p for p in in_bucket if p["path"] == locked_path), None)
        if chosen is not None:
            return chosen, True
    return in_bucket[0], False  # no lock, or locked photo gone from pool


def _decade_buckets(stills):
    """Group dated stills into decade buckets, oldest decade first. Used for
    the no-birth-date fallback strip — chronology stands in for age."""
    groups: dict[str, list] = {}
    for p in stills:
        groups.setdefault(_decade_label(p["ts_parsed"]), []).append(p)
    return [(lbl, groups[lbl]) for lbl in sorted(groups, key=lambda l: int(l[:-1]))]


@router.get("/{person_id}/life-stages")
async def life_stages(person_id: str):
    """For each age bucket (baby → seventies+), pick one representative
    photo of this person. Honors any locked override (LIFE_STAGE edge);
    otherwise auto-picks via favorited+solo > solo > favorited > newest."""
    async with get_session() as session:
        birth_dt, stills, locked_by_bucket = await _fetch_life_stage_pool(session, person_id)
    if not stills:
        return {"buckets": []}

    out = []
    if birth_dt is not None:
        # Age buckets, strided to the range of ages actually photographed.
        ages = [p["age_years"] for p in stills]
        buckets = _life_buckets_for(min(ages), max(ages))
        locks = _resolve_locks(locked_by_bucket, stills, buckets)
        for lo, hi, label in buckets:
            in_bucket = [p for p in stills if lo <= p["age_years"] < hi + 1]
            if not in_bucket:
                continue
            _sort_bucket(in_bucket)
            chosen, locked = _pick_chosen(in_bucket, locks.get(label))
            out.append(_life_stage_tile(label, chosen, chosen["age_years"], len(in_bucket), locked))
    else:
        # No birth date → fall back to a chronological "through the years"
        # strip keyed by photo decade.
        for label, in_bucket in _decade_buckets(stills):
            _sort_bucket(in_bucket)
            chosen, locked = _pick_chosen(in_bucket, locked_by_bucket.get(label))
            out.append(_life_stage_tile(label, chosen, None, len(in_bucket), locked, age_text=label))
    return {"buckets": out}


@router.get("/{person_id}/life-stages/{bucket}/candidates")
async def life_stage_candidates(person_id: str, bucket: str, limit: int = 24):
    """All candidate photos for one age bucket, sorted best-first.
    Drives the 'swap to a different one' UI."""
    async with get_session() as session:
        birth_dt, stills, _ = await _fetch_life_stage_pool(session, person_id)
    if not stills:
        return {"candidates": []}

    if birth_dt is not None:
        bounds = _bucket_bounds(bucket)
        if bounds is None:
            raise HTTPException(404, f"Unknown bucket {bucket!r}")
        lo, hi = bounds
        in_bucket = [p for p in stills if lo <= p["age_years"] < hi + 1]
        _sort_bucket(in_bucket)
        out = [
            _life_stage_tile(bucket, p, p["age_years"], len(in_bucket), False)
            for p in in_bucket[:limit]
        ]
        return {"candidates": out}

    # No-birth-date fallback: the bucket is a decade label like "1990s".
    m = _DECADE_RE.match(bucket)
    if not m:
        raise HTTPException(404, f"Unknown bucket {bucket!r}")
    decade = int(m.group(1))
    in_bucket = [p for p in stills if (p["ts_parsed"].year // 10) * 10 == decade]
    _sort_bucket(in_bucket)
    out = [
        _life_stage_tile(bucket, p, None, len(in_bucket), False, age_text=bucket)
        for p in in_bucket[:limit]
    ]
    return {"candidates": out}


async def _stale_lock_paths(session, person_id, bucket):
    """Paths of this person's stills whose age falls inside `bucket`.

    A lock saved under an older label ("toddler", or a narrower stride from
    before a birthday) covers the same stretch of life as the bucket the user is
    acting on now. Deleting only by label would leave that edge in place, and
    since reads resolve locks by photo age it would immediately reappear as a
    lock the user just tried to remove.
    """
    bounds = _bucket_bounds(bucket)
    if bounds is None:
        return []
    lo, hi = bounds
    _birth, stills, _locks = await _fetch_life_stage_pool(session, person_id)
    return [
        p["path"] for p in stills
        if p["age_years"] is not None and lo <= p["age_years"] < hi + 1
    ]


class LifeStageLockBody(BaseModel):
    bucket: str
    path: str


@router.put("/{person_id}/life-stages")
async def lock_life_stage(person_id: str, body: LifeStageLockBody, request: Request):
    """Pin one photo as the persisted representative for a bucket.
    Idempotent: replaces any existing lock for that bucket."""
    async with get_session() as session:
        # Verify the media exists + the person actually appears in it.
        check = await session.run(
            """
            MATCH (p:Person {id: $pid})-[:APPEARS_IN]->(m:Media {path: $path})
            RETURN count(*) AS n
            """,
            pid=person_id, path=body.path,
        )
        row = await check.single()
        if not row or row["n"] == 0:
            raise HTTPException(404, "Person doesn't appear in that media")
        stale = await _stale_lock_paths(session, person_id, body.bucket)
        await session.run(
            """
            MATCH (p:Person {id: $pid})
            OPTIONAL MATCH (p)-[old:LIFE_STAGE]->(om:Media)
              WHERE old.bucket = $bucket OR om.path IN $stale
            DELETE old
            WITH DISTINCT p
            MATCH (m:Media {path: $path})
            MERGE (p)-[:LIFE_STAGE {bucket: $bucket}]->(m)
            """,
            pid=person_id, bucket=body.bucket, path=body.path, stale=stale,
        )
    logger.bind(
        event="life_stage.locked",
        person_id=person_id, bucket=body.bucket, path=body.path,
        **_ctx(request),
    ).info("life-stage locked")
    return {"ok": True}


@router.delete("/{person_id}/life-stages/{bucket}", status_code=204)
async def unlock_life_stage(person_id: str, bucket: str, request: Request):
    """Drop the lock on a bucket so it goes back to auto-pick."""
    async with get_session() as session:
        stale = await _stale_lock_paths(session, person_id, bucket)
        await session.run(
            """
            MATCH (p:Person {id: $pid})-[ls:LIFE_STAGE]->(m:Media)
            WHERE ls.bucket = $bucket OR m.path IN $stale
            DELETE ls
            """,
            pid=person_id, bucket=bucket, stale=stale,
        )
    logger.bind(
        event="life_stage.unlocked",
        person_id=person_id, bucket=bucket,
        **_ctx(request),
    ).info("life-stage unlocked")


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
