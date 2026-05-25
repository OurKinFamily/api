from fastapi import APIRouter
from app.db.neo4j import get_session

router = APIRouter(prefix="/places", tags=["places"])


@router.get("/points")
async def get_points():
    """Return all photos with GPS as compact [lat, lng, path] tuples."""
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Media)
            WHERE p.latitude IS NOT NULL AND p.longitude IS NOT NULL
            RETURN p.latitude AS lat, p.longitude AS lng, p.path AS path
            """
        )
        records = await result.data()

    points = [[r["lat"], r["lng"], r["path"]] for r in records]
    return {"points": points}
