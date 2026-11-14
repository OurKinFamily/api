"""Regression: the life-stage strip filled a slot with the wrong decade.

Guillaume Delloue, born September 1986, had an "age 39" tile whose candidates
were almost entirely Facebook photographs of himself at about 25. They sit in
archive/0000 — the date-conflict bucket — carrying timestamps of 2026-01-03
and 2025-06-20, which are the times the files were copied in, not taken. For
a man born in 1986 those compute to age 39, so an import run quietly filled
whatever bucket he happens to be in now.

The archive already knew: every one of them is graded
`timestamp_confidence: 'low'`. The strip simply never read the grade. It makes
exactly one claim — this is what somebody looked like at this age — and a date
is the whole basis for it, so an untrusted date has no business here.

Dropping them costs little: 3,644 of 182,000 dated items archive-wide, and
for Guillaume 83 of 1,867. His age-39 slot still holds twenty real
photographs.

The filter lives in the Cypher, and the other tests in this area stub the
database and hand the endpoint a record directly — so they cannot see it.
This inspects the query itself, which is the only layer where the rule exists.
"""
import re
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

from app.routers.people import _fetch_life_stage_pool


class _Recorder:
    """Captures the Cypher the pool runs."""

    def __init__(self):
        self.queries = []

    @asynccontextmanager
    async def session(self):
        s = AsyncMock()

        def _run(*args, **kwargs):
            self.queries.append(args[0] if args else "")
            result = AsyncMock()
            result.single.return_value = None
            return result

        s.run.side_effect = _run
        yield s


@pytest.mark.asyncio
async def test_the_pool_refuses_dates_the_archive_does_not_trust():
    rec = _Recorder()
    async with rec.session() as session:
        await _fetch_life_stage_pool(session, "pid")

    assert rec.queries, "the pool should have run a query"
    q = " ".join(rec.queries[0].split())

    assert "timestamp_confidence" in q, (
        "the strip must exclude low-confidence dates, or a file's copy date "
        "puts a 25-year-old in the age-39 slot"
    )
    assert re.search(r"coalesce\(m\.timestamp_confidence,\s*'high'\)\s*<>\s*'low'", q), (
        "expected low-confidence media to be filtered out, with a missing "
        "grade treated as trustworthy"
    )


@pytest.mark.asyncio
async def test_it_still_requires_a_real_date():
    # The pre-existing guards have to survive alongside the new one: no
    # timestamp at all, and the 1700 sentinel used for "unknown".
    rec = _Recorder()
    async with rec.session() as session:
        await _fetch_life_stage_pool(session, "pid")
    q = " ".join(rec.queries[0].split())
    assert "m.timestamp IS NOT NULL" in q
    assert "1800-01-01" in q
