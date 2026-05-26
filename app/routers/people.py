import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.db.neo4j import get_session
from app.models.person import Person, PersonCreate, RelationshipAdd


class AvatarSet(BaseModel):
    crop_path: str

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
async def create_person(data: PersonCreate):
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
    is_living: bool = True
    notes: str | None = None


@router.patch("/{person_id}", response_model=Person)
async def update_person(person_id: str, data: PersonUpdate):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})
            SET p.name = $name, p.known_as = $known_as, p.maiden_name = $maiden_name,
                p.birth_date = $birth_date, p.birth_date_precision = $birth_date_precision,
                p.birth_place = $birth_place, p.death_date = $death_date,
                p.death_date_precision = $death_date_precision, p.death_place = $death_place,
                p.is_living = $is_living, p.notes = $notes
            RETURN p
            """,
            id=person_id, name=data.name, known_as=data.known_as,
            maiden_name=data.maiden_name, birth_date=data.birth_date,
            birth_date_precision=data.birth_date_precision, birth_place=data.birth_place,
            death_date=data.death_date, death_date_precision=data.death_date_precision,
            death_place=data.death_place, is_living=data.is_living, notes=data.notes,
        )
        record = await result.single()
        if not record:
            raise HTTPException(404, "Person not found")
        return Person(**record["p"])


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


@router.post("/{person_id}/relationships", status_code=204)
async def add_relationship(person_id: str, body: RelationshipAdd):
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


class RelationshipRemove(BaseModel):
    target_id: str
    rel_type: str  # spouse, parent, child


@router.delete("/{person_id}/relationships", status_code=204)
async def remove_relationship(person_id: str, body: RelationshipRemove):
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


@router.put("/{person_id}/avatar", status_code=204)
async def set_avatar(person_id: str, body: AvatarSet):
    async with get_session() as session:
        result = await session.run(
            "MATCH (p:Person {id: $id}) SET p.avatar = $avatar RETURN p",
            id=person_id, avatar=body.crop_path
        )
        record = await result.single()
        if not record:
            raise HTTPException(status_code=404, detail="Person not found")


class FaceAssign(BaseModel):
    photo_path: str
    face_index: int
    crop_path:  str

class PhotoTag(BaseModel):
    photo_path: str

@router.post("/{person_id}/photos", status_code=204)
async def tag_photo(person_id: str, body: PhotoTag):
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

@router.post("/{person_id}/faces", status_code=204)
async def assign_face(person_id: str, body: FaceAssign):
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
async def add_connection(person_id: str, body: ConnectionAdd):
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


@router.delete("/{person_id}/connections/{other_id}", status_code=204)
async def remove_connection(person_id: str, other_id: str):
    async with get_session() as session:
        await session.run(
            """
            MATCH (a:Person {id: $a})-[r:KNOWS]-(b:Person {id: $b})
            DELETE r
            """,
            a=person_id, b=other_id,
        )


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
            OPTIONAL MATCH (parent:Person)-[:PARENT_OF]->(p)
            WITH p, children, collect(DISTINCT {id: parent.id, name: parent.name, known_as: parent.known_as, avatar: parent.avatar}) AS parents
            OPTIONAL MATCH (p)-[:MARRIED_TO]->(spouse:Person)
            WITH p, children, parents, collect(DISTINCT {id: spouse.id, name: spouse.name, known_as: spouse.known_as, avatar: spouse.avatar}) AS spouses
            OPTIONAL MATCH (parent:Person)-[:PARENT_OF]->(p)
            OPTIONAL MATCH (parent)-[:PARENT_OF]->(sibling:Person) WHERE sibling.id <> p.id
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
