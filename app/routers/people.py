from fastapi import APIRouter, HTTPException
from app.db.neo4j import get_session
from app.models.person import Person

router = APIRouter(prefix="/people", tags=["people"])


@router.get("/", response_model=list[Person])
async def list_people():
    async with get_session() as session:
        result = await session.run("MATCH (p:Person) RETURN p ORDER BY p.name")
        records = await result.data()
        return [Person(**r["p"]) for r in records]


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


@router.get("/{person_id}/relatives")
async def get_relatives(person_id: str):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})
            OPTIONAL MATCH (p)-[:PARENT_OF]->(child:Person)
            OPTIONAL MATCH (parent:Person)-[:PARENT_OF]->(p)
            OPTIONAL MATCH (p)-[:MARRIED_TO]-(spouse:Person)
            RETURN
                collect(DISTINCT {id: child.id, name: child.name, known_as: child.known_as})  AS children,
                collect(DISTINCT {id: parent.id, name: parent.name, known_as: parent.known_as}) AS parents,
                collect(DISTINCT {id: spouse.id, name: spouse.name, known_as: spouse.known_as}) AS spouses
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
        }
