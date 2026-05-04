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


@router.get("/{person_id}/faces")
async def get_faces(person_id: str):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})-[r:APPEARS_IN]->(photo:Photo)
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
async def get_photos(person_id: str):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})-[:APPEARS_IN]->(photo:Photo)
            RETURN DISTINCT photo.path AS photo_path
            ORDER BY photo.path
            """,
            id=person_id
        )
        records = await result.data()
        return [r["photo_path"] for r in records]


@router.get("/{person_id}/relatives")
async def get_relatives(person_id: str):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})
            OPTIONAL MATCH (p)-[:PARENT_OF]->(child:Person)
            WITH p, collect(DISTINCT {id: child.id, name: child.name, known_as: child.known_as}) AS children
            OPTIONAL MATCH (parent:Person)-[:PARENT_OF]->(p)
            WITH p, children, collect(DISTINCT {id: parent.id, name: parent.name, known_as: parent.known_as}) AS parents
            OPTIONAL MATCH (p)-[:MARRIED_TO]->(spouse:Person)
            WITH p, children, parents, collect(DISTINCT {id: spouse.id, name: spouse.name, known_as: spouse.known_as}) AS spouses
            OPTIONAL MATCH (parent:Person)-[:PARENT_OF]->(p)
            OPTIONAL MATCH (parent)-[:PARENT_OF]->(sibling:Person) WHERE sibling.id <> p.id
            WITH children, parents, spouses, collect(DISTINCT {id: sibling.id, name: sibling.name, known_as: sibling.known_as}) AS siblings
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
