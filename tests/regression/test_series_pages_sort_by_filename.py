"""A paginated collection with no page numbers came back shuffled.

/collections/{id}/items sorts a series by `page_number`. The 125 pages of the
Timberlane Regional 2004 yearbook have no `page_number` property at all -- the
sidecars carry a full `heritage` block but never the `pageNumber` key, unlike
every other yearbook in the archive -- so every item produced the identical
sort key `(True, 0)`. Python's sort is stable, so the list came back in Neo4j's
CONTAINS traversal order: arbitrary, and arbitrary differently from one query
plan to the next. Page 42 first, then 26, then 119.

EXIF cannot rescue this. All 125 pages were scanned in one 2018 session and
share 78 distinct EXIF:ModifyDate values between them, so a timestamp sort
would still interleave pages. The filename is the only clean signal:
`timberlane-2004-yearbook-page-042.jpg`, 001 through 125, no gaps.
"""
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.routers.heritage import _natural_key

YEARBOOK = "heritage/stephen_e_young_3cf/timberlane_yearbook_2004"


def _page(n: int, page_number=None) -> dict:
    """One scanned page as the graph actually holds it."""
    d = {
        "path": f"{YEARBOOK}/timberlane-2004-yearbook-page-{n:03d}.jpg",
        "title": "Timberlane Regional High School Yearbook, Class of 2004",
        "content_date": "2004",
    }
    if page_number is not None:
        d["page_number"] = page_number
    return d


def _session(nodes, *, is_series=True):
    """get_collection_items runs: the visibility single(), then the docs data()."""
    @asynccontextmanager
    async def _s():
        session = AsyncMock()
        first = True

        def _run(*args, **kwargs):
            nonlocal first
            result = AsyncMock()
            if first:
                first = False
                result.single.return_value = {"is_series": is_series, "hidden": False}
                result.data.return_value = []
            else:
                result.single.return_value = None
                result.data.return_value = [{"d": n} for n in nodes]
            return result

        session.run = AsyncMock(side_effect=_run)
        yield session

    return _s


async def _items(monkeypatch, nodes, **kw):
    import app.routers.heritage as heritage_mod
    monkeypatch.setattr(heritage_mod, "get_session", _session(nodes, **kw))
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        res = await c.get("/collections/c1/items")
    assert res.status_code == 200
    return res.json()["items"]


def _numbers(items):
    """The page number read back off each filename, in returned order."""
    return [int(i["path"][-7:-4]) for i in items]


class TestASeriesWithNoPageNumbers:
    """The Timberlane case. Nothing to sort by but the filename."""

    async def test_pages_come_back_in_filename_order(self, monkeypatch):
        shuffled = [_page(n) for n in (42, 26, 119, 125, 8, 1, 97)]
        items = await _items(monkeypatch, shuffled)
        assert _numbers(items) == [1, 8, 26, 42, 97, 119, 125]

    async def test_the_first_page_is_first(self, monkeypatch):
        """The collection's cover_path points at page-001; it has to lead."""
        items = await _items(monkeypatch, [_page(n) for n in (64, 1, 110)])
        assert _numbers(items)[0] == 1

    async def test_the_whole_yearbook_is_ordered(self, monkeypatch):
        """125 pages, reversed on the way in, as the real collection is sized."""
        items = await _items(monkeypatch, [_page(n) for n in range(125, 0, -1)])
        assert _numbers(items) == list(range(1, 126))

    async def test_the_order_does_not_depend_on_what_neo4j_returned(self, monkeypatch):
        """The defect was exactly this: traversal order leaking through a
        constant sort key. Two different input orders must agree."""
        a = await _items(monkeypatch, [_page(n) for n in (3, 1, 2)])
        b = await _items(monkeypatch, [_page(n) for n in (2, 3, 1)])
        assert _numbers(a) == _numbers(b) == [1, 2, 3]


class TestASeriesThatDoesHavePageNumbers:
    """Must not regress. Most of the archive's 52 series are in this state --
    3,900 of 4,034 items carry a page_number."""

    async def test_page_number_wins_over_the_filename(self, monkeypatch):
        """Scan filenames and printed page numbers need not agree, and when
        they disagree the sidecar's number is the deliberate answer."""
        nodes = [_page(1, page_number=3), _page(2, page_number=1), _page(3, page_number=2)]
        items = await _items(monkeypatch, nodes)
        assert [i["page_number"] for i in items] == [1, 2, 3]
        assert _numbers(items) == [2, 3, 1]

    async def test_numbered_pages_precede_unnumbered_inserts(self, monkeypatch):
        """Seven collections have one or two unnumbered items. Guessing where an
        insert belongs is worse than putting it at the end, so nulls stay last."""
        nodes = [_page(9), _page(1, page_number=1), _page(2, page_number=2)]
        items = await _items(monkeypatch, nodes)
        assert _numbers(items) == [1, 2, 9]
        assert items[-1]["page_number"] is None

    async def test_unnumbered_inserts_are_ordered_among_themselves(self, monkeypatch):
        nodes = [_page(9), _page(4), _page(1, page_number=1)]
        items = await _items(monkeypatch, nodes)
        assert _numbers(items) == [1, 4, 9]


class TestACollectionThatIsNotASeries:
    """A box of loose papers has no page order, so it sorts by path -- but it
    sorts by path the same way, so an unpadded batch is not scrambled either."""

    async def test_unpadded_filenames_sort_as_numbers(self, monkeypatch):
        nodes = [{"path": f"{YEARBOOK}/scan-{n}.jpg"} for n in (10, 9, 1, 2)]
        items = await _items(monkeypatch, nodes, is_series=False)
        assert [i["path"].rsplit("-", 1)[1] for i in items] == [
            "1.jpg", "2.jpg", "9.jpg", "10.jpg"
        ]

    async def test_page_number_is_ignored_when_not_a_series(self, monkeypatch):
        """is_series is the switch. A stray page_number on a loose item must not
        reorder the box."""
        nodes = [_page(1, page_number=99), _page(2, page_number=1)]
        items = await _items(monkeypatch, nodes, is_series=False)
        assert _numbers(items) == [1, 2]


class TestNaturalKey:
    """The helper. Sorted() over these is the whole contract."""

    def test_it_reads_numbers_as_numbers(self):
        assert _natural_key("p9") < _natural_key("p10")

    def test_plain_string_order_would_get_that_wrong(self):
        """Guards the reason the helper exists at all."""
        assert "p10" < "p9"

    def test_zero_padding_still_works(self):
        assert _natural_key("p009") < _natural_key("p010")

    def test_it_is_case_insensitive(self):
        assert _natural_key("Page2") < _natural_key("page10")

    def test_a_name_with_no_digits_is_comparable_to_one_with(self):
        """Mixing int and str at the same tuple index raises TypeError. The
        split's fixed parity is what prevents it, and this is the case that
        would expose a slip."""
        assert sorted([_natural_key("cover.jpg"), _natural_key("page-1.jpg")])

    def test_several_numbers_in_one_name_all_count(self):
        assert _natural_key("y2004-p2") < _natural_key("y2004-p10")
        assert _natural_key("y2004-p10") < _natural_key("y2005-p2")

    def test_the_real_filenames_sort(self):
        names = [f"timberlane-2004-yearbook-page-{n:03d}.jpg" for n in range(1, 126)]
        assert sorted(names, key=_natural_key) == names
