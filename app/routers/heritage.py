import uuid
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db.neo4j import get_session
from app.config import settings

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


class CollectionUpdate(BaseModel):
    name: Optional[str] = None
    type: Optional[str] = None
    is_series: Optional[bool] = None
    cover_path: Optional[str] = None
    description: Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _item_from_node(d: dict) -> dict:
    path = d["path"]

    is_video = bool(d.get("is_video"))

    audio_url = None
    if d.get("audio_file"):
        audio_rel = f"{Path(path).parent}/{d['audio_file']}"
        if (PHOTOS_ROOT / audio_rel).exists():
            audio_url = f"/api/media/{audio_rel}"

    # videos use their poster jpg as the thumbnail; images use the webp thumb route
    if is_video and d.get("poster_path"):
        thumb_url = f"/api/media/{d['poster_path']}"
    else:
        thumb_url = f"/api/media/thumb/{path}"

    subtitle_url = f"/api/media/vtt/{path}" if d.get("has_subtitles") else None

    return {
        "path": path,
        "url": f"/api/media/{path}",
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
    return c


# ── Person collections ─────────────────────────────────────────────────────────

@router.get("/people/{person_id}/collections")
async def get_collections(person_id: str):
    async with get_session() as session:
        result = await session.run(
            """
            MATCH (p:Person {id: $id})
            CALL {
              WITH p
              MATCH (c:Collection)-[:BELONGS_TO]->(p)
              RETURN c
              UNION
              WITH p
              MATCH (p)-[:APPEARS_IN]->(:Media)<-[:CONTAINS]-(c:Collection)
              RETURN c
            }
            RETURN DISTINCT c ORDER BY c.name
            """,
            id=person_id,
        )
        records = await result.data()
        return [_collection_record(r) for r in records]


@router.post("/people/{person_id}/collections", status_code=201)
async def create_collection(person_id: str, body: CollectionCreate):
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
                description: $description, item_count: 0, created_at: datetime()
            })
            WITH c
            MATCH (p:Person {id: $person_id})
            CREATE (c)-[:BELONGS_TO]->(p)
            """,
            id=cid, person_id=person_id,
            name=body.name, type=body.type, is_series=body.is_series,
            base_path=body.base_path, cover_path=body.cover_path,
            description=body.description,
        )
    return {"id": cid}


# ── Single collection ──────────────────────────────────────────────────────────

@router.get("/collections/{collection_id}")
async def get_collection(collection_id: str):
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
        return data


@router.patch("/collections/{collection_id}")
async def update_collection(collection_id: str, body: CollectionUpdate):
    async with get_session() as session:
        result = await session.run(
            "MATCH (c:Collection {id: $id}) RETURN c", id=collection_id
        )
        if not await result.single():
            raise HTTPException(status_code=404, detail="Collection not found")
        sets = []
        params = {"id": collection_id}
        for field in ("name", "type", "is_series", "cover_path", "description"):
            val = getattr(body, field)
            if val is not None:
                sets.append(f"c.{field} = ${field}")
                params[field] = val
        if sets:
            await session.run(
                f"MATCH (c:Collection {{id: $id}}) SET {', '.join(sets)}", **params
            )
    return {"ok": True}


@router.get("/collections/{collection_id}/items")
async def get_collection_items(collection_id: str):
    async with get_session() as session:
        result = await session.run(
            "MATCH (c:Collection {id: $id}) RETURN c.is_series AS is_series",
            id=collection_id,
        )
        rec = await result.single()
        if not rec:
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
