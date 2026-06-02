"""Albums — user-curated collections of media.

Visibility model:
- `is_private = False` (default) → visible to everyone, listed for all users
- `is_private = True`             → visible only to the creator

Albums are created by the logged-in user (resolved from Cloudflare Access
header). Other family members can still ADD media to a shared album they
didn't create — same shared-archive feel as Groups / Connections.
"""
import os
import uuid
from fastapi import APIRouter, HTTPException, Request, Query
from pydantic import BaseModel
from app.config import with_v
from app.db.neo4j import get_session
from app.log import logger


def _ctx(request: Request) -> dict:
    return {
        "by": getattr(request.state, "user_email", None),
        "request_id": getattr(request.state, "request_id", None),
    }

router = APIRouter(prefix="/albums", tags=["albums"])

CF_EMAIL_HEADER = "cf-access-authenticated-user-email"


def _current_email(request: Request) -> str | None:
    return (
        request.headers.get(CF_EMAIL_HEADER)
        or os.environ.get("DEV_USER_EMAIL", "stephenyoung7267@gmail.com")
    )


async def _current_user_id(request: Request) -> str | None:
    email = _current_email(request)
    if not email: return None
    async with get_session() as session:
        r = await session.run("MATCH (p:Person {email: $email}) RETURN p.id AS id", email=email)
        row = await r.single()
    return row["id"] if row else None


# ── Schemas ──────────────────────────────────────────────────────────────────

class AlbumCreate(BaseModel):
    name:        str
    description: str | None = None
    is_private:  bool       = False


class AlbumUpdate(BaseModel):
    name:        str | None  = None
    description: str | None  = None
    cover_path:  str | None  = None
    is_private:  bool | None = None


class AlbumAddMedia(BaseModel):
    paths: list[str]


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/")
async def list_albums(request: Request, viewer_id: str | None = None):
    """All albums visible to the current user: public + own private.
    `cover_path` falls back to the most-recently-added media when the album
    has no explicit cover set.

    viewer_id: admin-only override for preview-as-person mode.
    """
    from app.deps import is_admin_email
    if viewer_id and is_admin_email(_current_email(request)):
        uid = viewer_id
    else:
        uid = await _current_user_id(request)
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (a:Album)
            WHERE coalesce(a.is_private, false) = false
               OR EXISTS { MATCH (:Person {id: $uid})-[:CREATED_ALBUM]->(a) }
            OPTIONAL MATCH (a)-[c:CONTAINS]->(:Media)
            WITH a, count(c) AS item_count
            OPTIONAL MATCH (a)-[c2:CONTAINS]->(latest:Media)
            WITH a, item_count, latest
            ORDER BY c2.added_at DESC
            WITH a, item_count, collect(latest)[0] AS latest_media
            OPTIONAL MATCH (creator:Person)-[:CREATED_ALBUM]->(a)
            RETURN a.id AS id, a.name AS name, a.description AS description,
                   coalesce(a.cover_path, latest_media.path) AS cover_path,
                   a.cover_path IS NOT NULL AS cover_is_explicit,
                   a.is_private AS is_private,
                   a.created_at AS created_at, item_count,
                   creator.id AS created_by_id, creator.name AS created_by_name
            ORDER BY a.created_at DESC
            """,
            uid=uid,
        )
        rows = await result.data()
    return rows


@router.post("/", status_code=201)
async def create_album(body: AlbumCreate, request: Request):
    uid = await _current_user_id(request)
    if not uid:
        raise HTTPException(401, "Cannot identify the current user")
    aid = str(uuid.uuid4())
    async with get_session() as session:
        await session.run(
            """
            MATCH (creator:Person {id: $uid})
            CREATE (a:Album {
              id:           $aid,
              name:         $name,
              description:  $description,
              is_private:   $is_private,
              created_at:   datetime()
            })
            MERGE (creator)-[:CREATED_ALBUM]->(a)
            """,
            uid=uid, aid=aid,
            name=body.name, description=body.description, is_private=body.is_private,
        )
    logger.bind(
        event="album.created",
        album_id=aid,
        name=body.name,
        is_private=body.is_private,
        **_ctx(request),
    ).info("album created")
    return {"id": aid}


@router.get("/{album_id}")
async def get_album(album_id: str, request: Request, limit: int = 500, offset: int = 0, viewer_id: str | None = None):
    from app.deps import is_admin_email
    if viewer_id and is_admin_email(_current_email(request)):
        uid = viewer_id
    else:
        uid = await _current_user_id(request)
    async with get_session() as session:
        # Album metadata + visibility check (+ implicit cover from latest media)
        meta_res = await session.run(
            """
            MATCH (a:Album {id: $aid})
            WHERE coalesce(a.is_private, false) = false
               OR EXISTS { MATCH (:Person {id: $uid})-[:CREATED_ALBUM]->(a) }
            OPTIONAL MATCH (a)-[c:CONTAINS]->(latest:Media)
            WITH a, latest
            ORDER BY c.added_at DESC
            WITH a, collect(latest)[0] AS latest_media
            OPTIONAL MATCH (creator:Person)-[:CREATED_ALBUM]->(a)
            RETURN a, latest_media.path AS latest_path,
                   creator.id AS created_by_id, creator.name AS created_by_name
            """,
            aid=album_id, uid=uid,
        )
        meta_row = await meta_res.single()
        if not meta_row:
            raise HTTPException(404, "Album not found")
        a = dict(meta_row["a"])
        if not a.get("cover_path") and meta_row["latest_path"]:
            a["cover_path"]        = meta_row["latest_path"]
            a["cover_is_explicit"] = False
        else:
            a["cover_is_explicit"] = a.get("cover_path") is not None

        tot = await session.run(
            "MATCH (:Album {id: $aid})-[:CONTAINS]->(m:Media) RETURN count(m) AS n",
            aid=album_id,
        )
        total = (await tot.single())["n"]

        items_res = await session.run(
            """
            MATCH (:Album {id: $aid})-[c:CONTAINS]->(m:Media)
            RETURN m.path AS path, m.timestamp AS timestamp,
                   m.width AS width, m.height AS height,
                   m.dominant_color AS dominant_color, m.kind AS kind,
                   c.added_at AS added_at
            ORDER BY c.added_at DESC
            SKIP $offset LIMIT $limit
            """,
            aid=album_id, offset=offset, limit=limit,
        )
        rows = await items_res.data()

    items = [
        {
            "path":           r["path"],
            "timestamp":      r["timestamp"],
            "width":          r["width"],
            "height":         r["height"],
            "dominant_color": r["dominant_color"],
            "is_video":       (r["kind"] == "video"),
            "added_at":       r["added_at"],
            "thumbnail_url":  with_v(f"/api/media/thumb/{(r['path'] or '').removeprefix('archive/')}"),
        }
        for r in rows
    ]
    return {
        **a,
        "created_by_id":   meta_row["created_by_id"],
        "created_by_name": meta_row["created_by_name"],
        "items":           items,
        "total":           total,
        "offset":          offset,
    }


@router.patch("/{album_id}", status_code=204)
async def update_album(album_id: str, body: AlbumUpdate, request: Request):
    uid = await _current_user_id(request)
    sets   = []
    params = {"aid": album_id, "uid": uid}
    if body.name        is not None: sets.append("a.name = $name");               params["name"]        = body.name
    if body.description is not None: sets.append("a.description = $description"); params["description"] = body.description
    if body.cover_path  is not None: sets.append("a.cover_path = $cover_path");   params["cover_path"]  = body.cover_path
    if body.is_private  is not None: sets.append("a.is_private = $is_private");   params["is_private"]  = body.is_private
    if not sets:
        return
    async with get_session() as session:
        res = await session.run(
            f"""
            MATCH (:Person {{id: $uid}})-[:CREATED_ALBUM]->(a:Album {{id: $aid}})
            SET {', '.join(sets)}
            RETURN a.id
            """,
            **params,
        )
        if not await res.single():
            raise HTTPException(404, "Album not found or not owned by current user")
    logger.bind(
        event="album.updated",
        album_id=album_id,
        fields=list(params.keys() - {"aid", "uid"}),
        **_ctx(request),
    ).info("album updated")


@router.delete("/{album_id}", status_code=204)
async def delete_album(album_id: str, request: Request):
    uid = await _current_user_id(request)
    async with get_session() as session:
        res = await session.run(
            """
            MATCH (:Person {id: $uid})-[:CREATED_ALBUM]->(a:Album {id: $aid})
            DETACH DELETE a RETURN 1 AS ok
            """,
            aid=album_id, uid=uid,
        )
        if not await res.single():
            raise HTTPException(404, "Album not found or not owned by current user")
    logger.bind(
        event="album.deleted",
        album_id=album_id,
        **_ctx(request),
    ).info("album deleted")


@router.post("/{album_id}/media", status_code=204)
async def add_media(album_id: str, body: AlbumAddMedia, request: Request):
    """Bulk-add media paths to an album. Shared (non-private) albums accept
    adds from any logged-in user. Private albums accept adds only from
    their creator."""
    uid = await _current_user_id(request)
    async with get_session() as session:
        # Visibility check
        vis = await session.run(
            """
            MATCH (a:Album {id: $aid})
            RETURN coalesce(a.is_private, false) AS is_private,
                   EXISTS { MATCH (:Person {id: $uid})-[:CREATED_ALBUM]->(a) } AS is_owner
            """,
            aid=album_id, uid=uid,
        )
        row = await vis.single()
        if not row:
            raise HTTPException(404, "Album not found")
        if row["is_private"] and not row["is_owner"]:
            raise HTTPException(403, "Cannot modify a private album you didn't create")
        await session.run(
            """
            MATCH (a:Album {id: $aid})
            UNWIND $paths AS p
            MATCH (m:Media {path: p})
            MERGE (a)-[r:CONTAINS]->(m)
            ON CREATE SET r.added_at = datetime()
            """,
            aid=album_id, paths=body.paths,
        )
    bulk_id = uuid.uuid4().hex[:12]
    logger.bind(
        event="album.media.bulk_added",
        album_id=album_id,
        count=len(body.paths),
        bulk_id=bulk_id,
        **_ctx(request),
    ).info("album media bulk-added")
    for p in body.paths:
        logger.bind(
            event="album.media.added",
            album_id=album_id,
            path=p,
            bulk_id=bulk_id,
            **_ctx(request),
        ).info("album media added")


@router.delete("/{album_id}/media", status_code=204)
async def remove_media(album_id: str, request: Request, path: str = Query(...)):
    uid = await _current_user_id(request)
    async with get_session() as session:
        vis = await session.run(
            """
            MATCH (a:Album {id: $aid})
            RETURN coalesce(a.is_private, false) AS is_private,
                   EXISTS { MATCH (:Person {id: $uid})-[:CREATED_ALBUM]->(a) } AS is_owner
            """,
            aid=album_id, uid=uid,
        )
        row = await vis.single()
        if not row:
            raise HTTPException(404, "Album not found")
        if row["is_private"] and not row["is_owner"]:
            raise HTTPException(403, "Cannot modify a private album you didn't create")
        await session.run(
            """
            MATCH (:Album {id: $aid})-[r:CONTAINS]->(:Media {path: $path})
            DELETE r
            """,
            aid=album_id, path=path,
        )
    logger.bind(
        event="album.media.removed",
        album_id=album_id,
        path=path,
        **_ctx(request),
    ).info("album media removed")
