"""Regression: one hand-dated photograph 500'd a whole life-stage strip.

Every timestamp in the archive is wall-clock time with a meaningless `Z`:
`2012-11-28T11:15:48.000Z`. Ages here are measured against a naive birth date,
so the two sides have always matched and nobody had to think about it.

Then a photograph filed under November 2012 turned out to be New Year's Eve
2009 in Montreal, and the corrected value went in as
`2009-12-31T12:00:00-05:00`. A real offset — the only one among 182,000 nodes.
It parsed to an aware datetime, subtracting the birth date raised
`TypeError: can't subtract offset-naive and offset-aware datetimes`, and the
endpoint 500'd. Every decade of the man's life disappeared behind one
corrected date, and the only visible symptom was a toast that said the people
would not load.

The stored value has since been normalised, but the parser is the real fault:
a single odd timestamp among sixteen hundred should cost that one photograph,
not the page. So `_media_dt` accepts any ISO shape and returns naive.

Offsets are dropped rather than converted, because these are local wall-clock
times. Converting the Montreal evening to UTC would move it to 01:00 on
1 January and file the photograph under the wrong year — which is the very
mistake the correction was made to fix.
"""
from datetime import datetime, timedelta, timezone

import pytest

from app.routers.people import _media_dt, _parse_birth


def test_an_offset_timestamp_parses_without_a_timezone():
    assert _media_dt("2009-12-31T12:00:00-05:00") == datetime(2009, 12, 31, 12, 0, 0)


def test_the_archive_shape_still_parses():
    assert _media_dt("2012-11-28T11:15:48.000Z") == datetime(2012, 11, 28, 11, 15, 48)


def test_a_bare_timestamp_still_parses():
    assert _media_dt("2012-11-28T11:15:48") == datetime(2012, 11, 28, 11, 15, 48)


def test_a_date_alone_still_parses():
    assert _media_dt("2012-11-28") == datetime(2012, 11, 28)


@pytest.mark.parametrize("raw", [
    "2009-12-31T12:00:00-05:00",
    "2009-12-31T12:00:00+09:00",
    "2012-11-28T11:15:48.000Z",
    "2012-11-28T11:15:48",
])
def test_every_shape_can_be_subtracted_from_a_birth_date(raw):
    # The actual crash: an age is a subtraction, and it must never raise.
    age = (_media_dt(raw) - _parse_birth("1986-09-15")).days / 365.25
    assert 20 < age < 30


def test_an_offset_is_dropped_not_converted():
    # 20:00 in Montreal was photographed in the evening of the 31st, not at
    # 01:00 on the 1st. Converting would file it under the wrong year.
    assert _media_dt("2009-12-31T20:00:00-05:00").year == 2009
    assert _media_dt("2009-12-31T20:00:00-05:00").day == 31


def test_nonsense_raises_so_the_caller_can_skip_it():
    # The caller catches ValueError and drops the photograph. A sidecar in the
    # archive genuinely holds "6583-69-67T12:00:00.000Z".
    with pytest.raises(ValueError):
        _media_dt("6583-69-67T12:00:00.000Z")
    with pytest.raises(ValueError):
        _media_dt("")


def test_an_already_naive_datetime_is_untouched():
    assert _media_dt("1700-01-01T00:00:00.000Z") == datetime(1700, 1, 1)


def test_the_result_is_always_naive():
    for raw in ("2009-12-31T12:00:00-05:00", "2012-11-28T11:15:48.000Z",
                "2012-11-28T11:15:48", "2012-11-28"):
        assert _media_dt(raw).tzinfo is None


def test_a_pool_row_with_an_offset_is_kept_not_skipped():
    # The photograph should survive the fix, not merely stop crashing.
    aware = datetime(2009, 12, 31, 12, tzinfo=timezone(timedelta(hours=-5)))
    assert _media_dt(aware.isoformat()).tzinfo is None
