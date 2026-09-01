import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.db.neo4j import get_session
from app.config import settings, with_v
from app.log import logger
from app.deps import is_admin_request


def _ctx(request: Request) -> dict:
    return {
        "by": getattr(request.state, "user_email", None),
        "request_id": getattr(request.state, "request_id", None),
    }


router = APIRouter(tags=["heritage"])

PHOTOS_ROOT = settings.photos_root


# ── Models ─────────────────────────────────────────────────────────────────────

class CollectionCreate(BaseModel):
    name: str
    type: str
    is_series: bool = False
    base_path: str
    cover_path: Optional[str] = None
    description: Optional[str] = None
    parent_id: Optional[str] = None
    story: Optional[str] = None


class CollectionUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    is_series: Optional[bool] = None
    cover_path: Optional[str] = None
    description: Optional[str] = None
    # `description` is the one-line card subtitle. `story` is the long-form
    # markdown shown at the top of the collection page -- a narrative about the
    # whole collection, the way `biography` works on a Person.
    story: Optional[str] = None
    private: Optional[bool] = None       # owner-only when true
    parent_id: Optional[str] = None      # nest under another collection; "" un-nests


# ── Helpers ────────────────────────────────────────────────────────────────────

def _item_from_node(d: dict) -> dict:
    path = d["path"]

    is_video = bool(d.get("is_video"))

    audio_url = None
    if d.get("audio_file"):
        audio_rel = f"{Path(path).parent}/{d['audio_file']}"
        if (PHOTOS_ROOT / audio_rel).exists():
            audio_url = with_v(f"/api/media/{audio_rel}")

    # videos use their poster jpg as the thumbnail; images use the webp thumb route
    if is_video and d.get("poster_path"):
        thumb_url = with_v(f"/api/media/{d['poster_path']}")
    else:
        thumb_url = with_v(f"/api/media/thumb/{path}")

    subtitle_url = with_v(f"/api/media/vtt/{path}") if d.get("has_subtitles") else None

    return {
        "path": path,
        "url": with_v(f"/api/media/{path}"),
        "thumb_url": thumb_url,
        "is_video": is_video,
        "subtitle_url": subtitle_url,
        "page_number": d.get("page_number"),
        "transcription": d.get("transcription"),
        "content_date": d.get("content_date"),
        "content_date_precision": d.get("content_date_precision"),
        "content_date_explanation": d.get("content_date_explanation"),
        "context_type": d.get("context_type"),
        "context_subject": d.get("title"),
        "context_notes": d.get("notes"),
        "description": d.get("description"),
        "place_name": d.get("place_name"),
        "physical_status": d.get("physical_status"),
        "physical_condition": d.get("physical_condition"),
        "audio_url": audio_url,
        "audio_description": d.get("audio_description"),
    }


def _collection_record(rec: dict) -> dict:
    c = dict(rec["c"])
    c["created_at"] = str(c.get("created_at", ""))
    c["private"] = bool(c.get("private", False))
    if "item_count" in rec:
        c["item_count"] = rec["item_count"]
    # Sub-collections. A parent may hold children, loose items, or both, so
    # both counts are reported and the UI decides what to show.
    if "child_count" in rec:
        c["child_count"] = rec["child_count"]
    if "descendant_item_count" in rec:
        c["descendant_item_count"] = rec["descendant_item_count"]
    if "earliest" in rec:
        c["earliest"] = rec["earliest"]
    return c


# Privacy is inherited: a collection is hidden from family viewers if it OR any
# ancestor is private. Without this a child of a private parent would still be
# reachable by its own URL.
_VISIBLE = (
    "NOT EXISTS { MATCH (c)-[:PART_OF*0..]->(a:Collection) "
    "WHERE coalesce(a.private, false) }"
)


# ── Person collections ─────────────────────────────────────────────────────────

@router.get("/people/{person_id}/collections")
async def get_collections(person_id: str, request: Request):
    # Family viewers don't see collections the owner marked private, nor any
    # collection nested under a private one.
    priv_filter = "" if is_admin_request(request) else f"AND {_VISIBLE}"
    async with get_session() as session:
        # A person's scrapbook shows the TOP-LEVEL collections they OWN. A
        # collection nested under another is reached by opening its parent, not
        # listed beside it. Items they merely appear in are surfaced
        # individually via /people/{id}/items, not as grouped collections.
        result = await session.run(
            f"""
            MATCH (c:Collection)-[:BELONGS_TO]->(p:Person {{id: $id}})
            WHERE NOT (c)-[:PART_OF]->(:Collection)
            {priv_filter}
            OPTIONAL MATCH (c)-[:CONTAINS]->(m:Media)
            WITH c, count(m) AS item_count
            OPTIONAL MATCH (c)<-[:PART_OF]-(kid:Collection)
            WITH c, item_count, count(DISTINCT kid) AS child_count
            OPTIONAL MATCH (c)<-[:PART_OF*0..]-(:Collection)-[:CONTAINS]->(dm:Media)
            RETURN c, item_count, child_count,
                   count(DISTINCT dm) AS descendant_item_count,
                   min(dm.content_date) AS earliest
            // Chronological, by the earliest thing inside. A scrapbook reads as a
            // life in order, so alphabetical ("1st Grade" before "Holy Angels")
            // is actively wrong. Derived rather than a stored sort key so new
            // collections slot in by themselves. Mixed precision sorts fine:
            // "2023" < "2023-11-02" lexicographically. Undated collections have
            // a null earliest and Cypher sorts nulls last, which is what we want.
            ORDER BY earliest, c.name
            """,
            id=person_id,
        )
        records = await result.data()
        return [_collection_record(r) for r in records]


@router.get("/people/{person_id}/items")
async def get_person_items(person_id: str):
    """Individual heritage items a person appears in, from collections they do
    NOT own (those are shown grouped). Each is its own scrapbook entry."""
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})-[:APPEARS_IN]->(m:Media)<-[:CONTAINS]-(c:Collection)
            WHERE NOT (c)-[:BELONGS_TO]->(p)
            RETURN DISTINCT m, head(collect(c.name)) AS collection_name
            ORDER BY m.content_date, m.filename
            """,
            id=person_id,
        )
        rows = await result.data()
    items = []
    for r in rows:
        it = _item_from_node(dict(r["m"]))
        it["collection_name"] = r["collection_name"]
        items.append(it)
    return {"items": items, "total": len(items)}


@router.post("/people/{person_id}/collections", status_code=201)
async def create_collection(person_id: str, body: CollectionCreate, request: Request):
    cid = str(uuid.uuid4())
    async with get_session() as session:
        exists = await session.run("MATCH (p:Person {id: $id}) RETURN p", id=person_id)
        if not await exists.single():
            raise HTTPException(status_code=404, detail="Person not found")
        await session.run(
            """
            CREATE (c:Collection {
                id: $id, name: $name, type: $type, is_series: $is_series,
                base_path: $base_path, cover_path: $cover_path,
                description: $description, story: $story,
                item_count: 0, created_at: datetime()
            })
            WITH c
            MATCH (p:Person {id: $person_id})
            CREATE (c)-[:BELONGS_TO]->(p)
            WITH c
            // Children keep their own BELONGS_TO so ownership queries stay flat;
            // PART_OF only controls where they appear.
            OPTIONAL MATCH (parent:Collection {id: $parent_id})
            FOREACH (_ IN CASE WHEN parent IS NULL THEN [] ELSE [1] END |
                MERGE (c)-[:PART_OF]->(parent))
            """,
            id=cid, person_id=person_id,
            name=body.name, type=body.type, is_series=body.is_series,
            base_path=body.base_path, cover_path=body.cover_path,
            description=body.description, story=body.story,
            parent_id=body.parent_id,
        )
    logger.bind(
        event="collection.created",
        collection_id=cid,
        person_id=person_id,
        name=body.name,
        type=body.type,
        **_ctx(request),
    ).info("collection created")
    return {"id": cid}


# ── Single collection ──────────────────────────────────────────────────────────

@router.get("/collections/{collection_id}")
async def get_collection(collection_id: str, request: Request):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (c:Collection {id: $id})-[:BELONGS_TO]->(p:Person)
            RETURN c, p.id AS person_id, p.name AS person_name, p.known_as AS person_known_as
            """,
            id=collection_id,
        )
        rec = await result.single()
        if not rec:
            raise HTTPException(status_code=404, detail="Collection not found")
        data = _collection_record({"c": rec["c"]})
        data["person_id"] = rec["person_id"]
        data["person_name"] = rec["person_known_as"] or rec["person_name"]

        # Private collections are invisible to family viewers (404, not 403, so
        # their existence isn't even revealed). Inherited: a child of a private
        # parent is hidden too, or it would leak via its own URL.
        if not is_admin_request(request):
            vis = await session.run(
                f"MATCH (c:Collection {{id: $id}}) RETURN {_VISIBLE} AS visible",
                id=collection_id,
            )
            vrec = await vis.single()
            if not vrec or not vrec["visible"]:
                raise HTTPException(status_code=404, detail="Collection not found")

        # Ancestors, nearest first — the UI reverses this for a breadcrumb and
        # uses ancestors[0] as the target for its back button.
        anc_result = await session.run(
            """
            MATCH (c:Collection {id: $id})-[:PART_OF*1..]->(a:Collection)
            RETURN a.id AS id, a.name AS name
            """,
            id=collection_id,
        )
        data["ancestors"] = [
            {"id": r["id"], "name": r["name"]} for r in await anc_result.data()
        ]

        # Direct children, each with its own counts so the cards can be
        # labelled without a second round trip.
        kid_filter = "" if is_admin_request(request) else f"WHERE {_VISIBLE}"
        kids_result = await session.run(
            f"""
            MATCH (parent:Collection {{id: $id}})<-[:PART_OF]-(c:Collection)
            {kid_filter}
            OPTIONAL MATCH (c)-[:CONTAINS]->(m:Media)
            WITH c, count(m) AS item_count
            OPTIONAL MATCH (c)<-[:PART_OF]-(kid:Collection)
            WITH c, item_count, count(DISTINCT kid) AS child_count
            OPTIONAL MATCH (c)<-[:PART_OF*0..]-(:Collection)-[:CONTAINS]->(dm:Media)
            RETURN c, item_count, child_count,
                   count(DISTINCT dm) AS descendant_item_count,
                   min(dm.content_date) AS earliest
            ORDER BY earliest, c.name
            """,
            id=collection_id,
        )
        data["children"] = [
            _collection_record(r) for r in await kids_result.data()
        ]
        return data


@router.patch("/collections/{collection_id}")
async def update_collection(collection_id: str, body: CollectionUpdate, request: Request):
    async with get_session() as session:
        result = await session.run(
            "MATCH (c:Collection {id: $id}) RETURN c", id=collection_id
        )
        if not await result.single():
            raise HTTPException(status_code=404, detail="Collection not found")
        sets = []
        params = {"id": collection_id}
        for field in ("name", "type", "is_series", "cover_path", "description",
                      "story", "private"):
            val = getattr(body, field)
            if val is not None:
                sets.append(f"c.{field} = ${field}")
                params[field] = val
        if sets:
            await session.run(
                f"MATCH (c:Collection {{id: $id}}) SET {', '.join(sets)}", **params
            )

        # Re-parent. Empty string un-nests, making the collection top-level.
        if body.parent_id is not None:
            if body.parent_id == "":
                await session.run(
                    "MATCH (c:Collection {id: $id})-[r:PART_OF]->(:Collection) DELETE r",
                    id=collection_id,
                )
            else:
                if body.parent_id == collection_id:
                    raise HTTPException(
                        status_code=400, detail="A collection cannot contain itself"
                    )
                # A cycle would make every ancestor/descendant traversal in this
                # module run forever, so refuse a parent that sits below us.
                cyc = await session.run(
                    """
                    MATCH (p:Collection {id: $parent_id})
                    RETURN EXISTS { MATCH (p)-[:PART_OF*1..]->(c:Collection {id: $id}) }
                           AS would_cycle
                    """,
                    parent_id=body.parent_id, id=collection_id,
                )
                crec = await cyc.single()
                if not crec:
                    raise HTTPException(status_code=404, detail="Parent collection not found")
                if crec["would_cycle"]:
                    raise HTTPException(
                        status_code=400,
                        detail="That would nest a collection inside its own descendant",
                    )
                await session.run(
                    """
                    MATCH (c:Collection {id: $id})
                    OPTIONAL MATCH (c)-[old:PART_OF]->(:Collection)
                    DELETE old
                    WITH c
                    MATCH (p:Collection {id: $parent_id})
                    MERGE (c)-[:PART_OF]->(p)
                    """,
                    id=collection_id, parent_id=body.parent_id,
                )
    logger.bind(
        event="collection.updated",
        collection_id=collection_id,
        fields=list(params.keys() - {"id"}),
        **_ctx(request),
    ).info("collection updated")
    return {"ok": True}


@router.get("/collections/{collection_id}/items")
async def get_collection_items(collection_id: str, request: Request):
    async with get_session() as session:
        result = await session.run(
            f"MATCH (c:Collection {{id: $id}}) "
            f"RETURN c.is_series AS is_series, NOT ({_VISIBLE}) AS hidden",
            id=collection_id,
        )
        rec = await result.single()
        if not rec:
            raise HTTPException(status_code=404, detail="Collection not found")
        if rec["hidden"] and not is_admin_request(request):
            raise HTTPException(status_code=404, detail="Collection not found")
        is_series = rec["is_series"]

        docs_result = await session.run(
            "MATCH (c:Collection {id: $id})-[:CONTAINS]->(d:Media) RETURN d",
            id=collection_id,
        )
        docs = [dict(r["d"]) for r in await docs_result.data()]

    items = [_item_from_node(d) for d in docs]

    if is_series:
        items.sort(key=lambda x: (x["page_number"] is None, x["page_number"] or 0))
    else:
        items.sort(key=lambda x: x["path"])

    return {"items": items, "total": len(items), "is_series": is_series}
