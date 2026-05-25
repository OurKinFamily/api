import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.db.neo4j import get_session

router = APIRouter(prefix="/groups", tags=["groups"])

GROUP_TYPES = [
    "sports_team", "fitness", "hobby_club",
    "school_class", "school", "extracurricular",
    "workplace", "professional_org",
    "neighborhood", "religious", "civic",
    "family_friend", "extended_network",
    "camp",
]


class GroupCreate(BaseModel):
    name: str
    type: str
    year: int | None = None
    season: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    location_name: str | None = None
    notes: str | None = None


class MemberAdd(BaseModel):
    person_id: str
    role: str | None = None
    since: str | None = None
    until: str | None = None


@router.get("/types")
async def list_types():
    return GROUP_TYPES


@router.get("/")
async def list_groups(type: str | None = None, q: str | None = None):
    async with get_session() as session:
        filters = ["WHERE 1=1"]
        params: dict = {}
        if type:
            filters.append("AND g.type = $type")
            params["type"] = type
        if q:
            filters.append("AND toLower(g.name) CONTAINS toLower($q)")
            params["q"] = q
        where = " ".join(filters)
        result = await session.run(
            f"""
            MATCH (g:Group) {where}
            OPTIONAL MATCH (g)<-[:MEMBER_OF]-(p:Person)
            RETURN g, count(p) AS member_count
            ORDER BY g.name
            """,
            **params,
        )
        records = await result.data()
    return [
        {**dict(r["g"]), "member_count": r["member_count"]}
        for r in records
    ]


@router.post("/", status_code=201)
async def create_group(data: GroupCreate):
    if data.type not in GROUP_TYPES:
        raise HTTPException(400, f"Invalid type. Must be one of: {GROUP_TYPES}")
    async with get_session() as session:
        group_id = str(uuid.uuid4())
        result = await session.run(
            """
            CREATE (g:Group {
              id: $id, name: $name, type: $type,
              year: $year, season: $season,
              latitude: $latitude, longitude: $longitude,
              location_name: $location_name, notes: $notes
            })
            RETURN g
            """,
            id=group_id,
            name=data.name,
            type=data.type,
            year=data.year,
            season=data.season,
            latitude=data.latitude,
            longitude=data.longitude,
            location_name=data.location_name,
            notes=data.notes,
        )
        record = await result.single()
    return dict(record["g"])


@router.get("/{group_id}")
async def get_group(group_id: str):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (g:Group {id: $id})
            OPTIONAL MATCH (g)<-[rel:MEMBER_OF]-(p:Person)
            RETURN g,
                   collect({
                     id: p.id, name: p.name, known_as: p.known_as,
                     avatar: p.avatar, role: rel.role,
                     since: rel.since, until: rel.until
                   }) AS members
            """,
            id=group_id,
        )
        record = await result.single()
        if not record:
            raise HTTPException(404, "Group not found")
    members = [m for m in record["members"] if m.get("id")]
    return {**dict(record["g"]), "members": members}


@router.patch("/{group_id}")
async def update_group(group_id: str, data: GroupCreate):
    if data.type not in GROUP_TYPES:
        raise HTTPException(400, f"Invalid type")
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (g:Group {id: $id})
            SET g.name = $name, g.type = $type, g.year = $year,
                g.season = $season, g.latitude = $latitude,
                g.longitude = $longitude, g.location_name = $location_name,
                g.notes = $notes
            RETURN g
            """,
            id=group_id,
            name=data.name,
            type=data.type,
            year=data.year,
            season=data.season,
            latitude=data.latitude,
            longitude=data.longitude,
            location_name=data.location_name,
            notes=data.notes,
        )
        record = await result.single()
        if not record:
            raise HTTPException(404, "Group not found")
    return dict(record["g"])


@router.delete("/{group_id}", status_code=204)
async def delete_group(group_id: str):
    async with get_session() as session:
        await session.run(
            "MATCH (g:Group {id: $id}) DETACH DELETE g",
            id=group_id,
        )


@router.post("/{group_id}/members", status_code=204)
async def add_member(group_id: str, body: MemberAdd):
    async with get_session() as session:
        check = await session.run(
            "MATCH (g:Group {id: $gid}), (p:Person {id: $pid}) RETURN count(*) AS n",
            gid=group_id, pid=body.person_id,
        )
        row = await check.single()
        if not row or row["n"] == 0:
            raise HTTPException(404, "Group or person not found")
        await session.run(
            """
            MATCH (g:Group {id: $gid}), (p:Person {id: $pid})
            MERGE (p)-[r:MEMBER_OF]->(g)
            SET r.role = $role, r.since = $since, r.until = $until
            """,
            gid=group_id, pid=body.person_id,
            role=body.role, since=body.since, until=body.until,
        )


@router.delete("/{group_id}/members/{person_id}", status_code=204)
async def remove_member(group_id: str, person_id: str):
    async with get_session() as session:
        await session.run(
            """
            MATCH (p:Person {id: $pid})-[r:MEMBER_OF]->(g:Group {id: $gid})
            DELETE r
            """,
            gid=group_id, pid=person_id,
        )


@router.get("/{group_id}/connection/{other_id}")
async def find_connection(group_id: str, other_id: str):
    """Shortest path between two people through any relationship."""
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (a:Person {id: $a}), (b:Person {id: $b})
            MATCH path = shortestPath((a)-[*..10]-(b))
            RETURN [node in nodes(path) | {
              labels: labels(node),
              id: node.id,
              name: coalesce(node.name, node.name),
              type: node.type
            }] AS nodes,
            [rel in relationships(path) | type(rel)] AS rels
            LIMIT 1
            """,
            a=group_id, b=other_id,
        )
        record = await result.single()
        if not record:
            return {"connected": False}
    return {"connected": True, "nodes": record["nodes"], "rels": record["rels"]}
