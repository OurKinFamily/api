"""Current-user-scoped endpoints.

Identity is read from the Cloudflare Access `cf-access-authenticated-user-email`
header (with a DEV_USER_EMAIL fallback for local dev). All endpoints here
operate on the logged-in user's data — favorites, preferences, etc.
"""
import os
from fastapi import APIRouter, HTTPException, Request, Query
from app.db.neo4j import get_session

router = APIRouter(prefix="/me", tags=["me"])

CF_EMAIL_HEADER = "cf-access-authenticated-user-email"


def _current_email(request: Request) -> str | None:
    return (
        request.headers.get(CF_EMAIL_HEADER)
        or os.environ.get("DEV_USER_EMAIL", "stephenyoung7267@gmail.com")
    )


async def _require_user_id(request: Request) -> str:
    email = _current_email(request)
    if not email:
        raise HTTPException(401, "Not authenticated")
    async with get_session() as session:
        r = await session.run(
            "MATCH (p:Person {email: $email}) RETURN p.id AS id",
            email=email,
        )
        row = await r.single()
    if not row:
        raise HTTPException(404, f"No Person matches {email}")
    return row["id"]


# ── Favorites ────────────────────────────────────────────────────────────────


@router.get("/favorites/paths")
async def favorites_paths(request: Request) -> list[str]:
    """Lightweight: just the set of favorited Media paths. Used to hydrate UI
    on initial load so the heart icons render in their correct state."""
    user_id = await _require_user_id(request)
    async with get_session() as session:
        r = await session.run(
            """
            MATCH (p:Person {id: $uid})-[:FAVORITED]->(m:Media)
            RETURN m.path AS path
            """,
            uid=user_id,
        )
        rows = await r.data()
    return [row["path"] for row in rows]


@router.get("/favorites")
async def favorites(request: Request, limit: int = 100, offset: int = 0):
    """Full list of favorited media, most-recently-favorited first."""
    user_id = await _require_user_id(request)
    async with get_session() as session:
        tot_res = await session.run(
            "MATCH (p:Person {id: $uid})-[:FAVORITED]->(:Media) RETURN count(*) AS n",
            uid=user_id,
        )
        total = (await tot_res.single())["n"]

        res = await session.run(
            """
            MATCH (p:Person {id: $uid})-[r:FAVORITED]->(m:Media)
            RETURN m.path AS path, m.timestamp AS timestamp,
                   m.width AS width, m.height AS height,
                   m.dominant_color AS dominant_color,
                   m.kind AS kind,
                   r.at AS favorited_at
            ORDER BY r.at DESC
            SKIP $offset LIMIT $limit
            """,
            uid=user_id, offset=offset, limit=limit,
        )
        rows = await res.data()
    items = [
        {
            "path":           row["path"],
            "timestamp":      row["timestamp"],
            "width":          row["width"],
            "height":         row["height"],
            "dominant_color": row["dominant_color"],
            "is_video":       (row["kind"] == "video"),
            "favorited_at":   row["favorited_at"],
            "thumbnail_url":  f"/api/media/thumb/{(row['path'] or '').removeprefix('archive/')}",
        }
        for row in rows
    ]
    return {"items": items, "total": total, "offset": offset}


@router.put("/favorites", status_code=204)
async def favorite_add(request: Request, path: str = Query(...)):
    """Favorite a media path. Idempotent — repeated calls just refresh `at`."""
    user_id = await _require_user_id(request)
    async with get_session() as session:
        await session.run(
            """
            MATCH (p:Person {id: $uid})
            MATCH (m:Media   {path: $path})
            MERGE (p)-[r:FAVORITED]->(m)
            SET r.at = datetime()
            """,
            uid=user_id, path=path,
        )


@router.delete("/favorites", status_code=204)
async def favorite_remove(request: Request, path: str = Query(...)):
    user_id = await _require_user_id(request)
    async with get_session() as session:
        await session.run(
            """
            MATCH (p:Person {id: $uid})-[r:FAVORITED]->(m:Media {path: $path})
            DELETE r
            """,
            uid=user_id, path=path,
        )
