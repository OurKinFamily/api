"""Sub-collections: a Collection may sit inside another via
(child)-[:PART_OF]->(parent).

Three things have to hold, and each is easy to get wrong:

  1. A nested collection must NOT also appear in the person's top-level list,
     or opening "Kindergarten" and seeing its five children still listed
     alongside it defeats the point.
  2. Privacy is inherited. A child of a private parent must 404 for family
     viewers even when they hit its own URL directly.
  3. Nesting must refuse cycles. A -> B -> A would make every ancestor and
     descendant traversal in the router run forever.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app

# A logged-in family viewer. Without this header the dev fallback resolves to
# the owner's email and every privacy branch is skipped as admin.
FAMILY = {"cf-access-authenticated-user-email": "billdezazzo@gmail.com"}


def _capture_session(seen_cypher: list, singles=None, data=None):
    """Session double that records every Cypher string it is handed.

    `singles` / `data` are queues consumed in call order, so a test can script
    a multi-query endpoint without caring which query is which.
    """
    singles = list(singles or [])
    data = list(data or [])

    @asynccontextmanager
    async def _session():
        session = AsyncMock()

        def _run(*args, **kwargs):
            seen_cypher.append(args[0] if args else "")
            result = AsyncMock()
            # Pop lazily, at the moment single()/data() is awaited. Popping both
            # queues eagerly per run() would let a query that only calls single()
            # swallow the data destined for the next one.
            async def _single():
                return singles.pop(0) if singles else None

            async def _data():
                return data.pop(0) if data else []

            result.single = _single
            result.data = _data
            return result

        session.run.side_effect = _run
        yield session

    return _session


async def _get(path: str, headers: dict | None = None):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path, headers=headers or {})


async def _patch(path: str, body: dict):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.patch(path, json=body)


@pytest.mark.asyncio
async def test_top_level_list_excludes_nested_collections(monkeypatch):
    seen: list = []
    monkeypatch.setattr("app.routers.heritage.get_session", _capture_session(seen))

    r = await _get("/people/p1/collections")

    assert r.status_code == 200
    assert seen, "expected a Cypher query"
    # The child is hidden from the top level by its outgoing PART_OF edge.
    assert "WHERE NOT (c)-[:PART_OF]->(:Collection)" in seen[0]


@pytest.mark.asyncio
async def test_top_level_list_reports_child_counts(monkeypatch):
    seen: list = []
    monkeypatch.setattr("app.routers.heritage.get_session", _capture_session(seen))

    await _get("/people/p1/collections")

    # A parent holding only sub-collections has no direct items, so the card
    # needs both counts or it renders "0 items".
    assert "child_count" in seen[0]
    assert "descendant_item_count" in seen[0]


@pytest.mark.asyncio
async def test_collection_detail_returns_children_and_ancestors(monkeypatch):
    seen: list = []
    child = {
        "id": "c2", "name": "Stories", "type": "school_papers",
        "is_series": True, "private": False, "created_at": "",
    }
    monkeypatch.setattr(
        "app.routers.heritage.get_session",
        _capture_session(
            seen,
            singles=[
                {"c": {"id": "c1", "name": "Kindergarten", "type": "school_papers",
                       "is_series": False, "private": False, "created_at": ""},
                 "person_id": "p1", "person_name": "Margaret",
                 "person_known_as": None},
                {"visible": True},
            ],
            data=[
                [{"id": "c0", "name": "School"}],                       # ancestors
                [{"c": child, "item_count": 12, "child_count": 0,
                  "descendant_item_count": 12}],                        # children
            ],
        ),
    )

    r = await _get("/collections/c1")

    assert r.status_code == 200
    body = r.json()
    assert [a["name"] for a in body["ancestors"]] == ["School"]
    assert [c["name"] for c in body["children"]] == ["Stories"]
    assert body["children"][0]["item_count"] == 12


@pytest.mark.asyncio
async def test_child_of_private_parent_is_hidden_from_family(monkeypatch):
    """The child is not itself private — its parent is. Without inherited
    privacy it would still be reachable by its own URL."""
    seen: list = []
    monkeypatch.setattr(
        "app.routers.heritage.get_session",
        _capture_session(
            seen,
            singles=[
                {"c": {"id": "c2", "name": "Stories", "type": "school_papers",
                       "is_series": False, "private": False, "created_at": ""},
                 "person_id": "p1", "person_name": "Margaret",
                 "person_known_as": None},
                {"visible": False},   # an ancestor is private
            ],
        ),
    )

    r = await _get("/collections/c2", headers=FAMILY)

    # 404 rather than 403 — a family viewer shouldn't learn it exists.
    assert r.status_code == 404
    assert any("PART_OF*0.." in c for c in seen)


@pytest.mark.asyncio
async def test_cannot_nest_a_collection_inside_itself(monkeypatch):
    seen: list = []
    monkeypatch.setattr(
        "app.routers.heritage.get_session",
        _capture_session(seen, singles=[{"c": {"id": "c1"}}]),
    )

    r = await _patch("/collections/c1", {"parent_id": "c1"})

    assert r.status_code == 400
    assert "itself" in r.json()["detail"]


@pytest.mark.asyncio
async def test_cannot_nest_a_collection_inside_its_own_descendant(monkeypatch):
    """A -> B already exists; making A a child of B would close the loop and
    hang every *-traversal in this router."""
    seen: list = []
    monkeypatch.setattr(
        "app.routers.heritage.get_session",
        _capture_session(
            seen,
            singles=[{"c": {"id": "parent"}}, {"would_cycle": True}],
        ),
    )

    r = await _patch("/collections/parent", {"parent_id": "child"})

    assert r.status_code == 400
    assert "descendant" in r.json()["detail"]


@pytest.mark.asyncio
async def test_empty_parent_id_unnests(monkeypatch):
    seen: list = []
    monkeypatch.setattr(
        "app.routers.heritage.get_session",
        _capture_session(seen, singles=[{"c": {"id": "c1"}}]),
    )

    r = await _patch("/collections/c1", {"parent_id": ""})

    assert r.status_code == 200
    # Deletes the edge, and must not create a new one.
    assert any("DELETE r" in c for c in seen)
    assert not any("MERGE (c)-[:PART_OF]->(p)" in c for c in seen)


@pytest.mark.asyncio
async def test_items_endpoint_honours_inherited_privacy(monkeypatch):
    seen: list = []
    monkeypatch.setattr(
        "app.routers.heritage.get_session",
        _capture_session(seen, singles=[{"is_series": False, "hidden": True}]),
    )

    r = await _get("/collections/c2/items", headers=FAMILY)

    assert r.status_code == 404
    assert any("PART_OF*0.." in c for c in seen)


@pytest.mark.asyncio
async def test_collections_are_ordered_chronologically(monkeypatch):
    """A scrapbook reads as a life in order. Alphabetical puts "1st Grade"
    before "Holy Angels", which is three school years out of sequence."""
    seen: list = []
    monkeypatch.setattr("app.routers.heritage.get_session", _capture_session(seen))

    await _get("/people/p1/collections")

    assert "min(dm.content_date) AS earliest" in seen[0]
    assert "ORDER BY earliest, c.name" in seen[0]
    assert "ORDER BY c.name\n" not in seen[0]
