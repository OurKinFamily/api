import pytest
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport
from app.main import app

PERSON = {
    "id": "person-stephen",
    "name": "Stephen Young",
    "known_as": "Stephen",
    "maiden_name": None,
    "former_names": None,
    "gender": None,
    "birth_date": "1986-04-15",
    "birth_date_precision": "exact",
    "birth_place": None,
    "death_date": None,
    "death_date_precision": None,
    "death_place": None,
    "is_living": True,
    "notes": None,
}

_MISSING = object()


def mock_session(records=None, single=_MISSING):
    @asynccontextmanager
    async def _session():
        session = AsyncMock()
        result = AsyncMock()
        result.data.return_value = records or []
        result.single.return_value = {"p": PERSON} if single is _MISSING else single
        session.run.return_value = result
        yield session
    return _session


@pytest.mark.asyncio
async def test_list_people():
    with patch("app.routers.people.get_session", mock_session(records=[{"p": PERSON}])):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/people/")
    assert res.status_code == 200
    data = res.json()
    assert len(data) == 1
    assert data[0]["id"] == "person-stephen"


@pytest.mark.asyncio
async def test_get_person_found():
    with patch("app.routers.people.get_session", mock_session(single={"p": PERSON})):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/people/person-stephen")
    assert res.status_code == 200
    assert res.json()["name"] == "Stephen Young"


@pytest.mark.asyncio
async def test_get_person_not_found():
    with patch("app.routers.people.get_session", mock_session(single=None)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/people/nobody")
    assert res.status_code == 404


PERSON_PRIVATE_BIO = {**PERSON, "biography": "Her whole story.", "bio_private": True}
FAMILY = {"cf-access-authenticated-user-email": "billdezazzo@gmail.com"}


@pytest.mark.asyncio
async def test_private_bio_visible_to_admin():
    # No CF header → dev fallback resolves to the admin email → bio comes through.
    with patch("app.routers.people.get_session", mock_session(single={"p": PERSON_PRIVATE_BIO})):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/people/person-stephen")
    assert res.status_code == 200
    assert res.json()["biography"] == "Her whole story."


@pytest.mark.asyncio
async def test_private_bio_hidden_from_family():
    # A non-admin family viewer gets the person but the private bio is stripped.
    with patch("app.routers.people.get_session", mock_session(single={"p": PERSON_PRIVATE_BIO})):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            res = await client.get("/people/person-stephen", headers=FAMILY)
    assert res.status_code == 200
    body = res.json()
    assert body["name"] == "Stephen Young"   # person still visible
    assert body["biography"] is None         # but the bio is hidden


@pytest.mark.asyncio
async def test_family_cannot_toggle_bio_private():
    # The write-lock middleware 403s any non-admin mutation.
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        res = await client.patch("/people/person-stephen/bio-private",
                                 json={"bio_private": True}, headers=FAMILY)
    assert res.status_code == 403
