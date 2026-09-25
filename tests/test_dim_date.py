"""Date dimension generation.

``dim_date`` is the conformed join that lets a 40-million-row trip fact and a
one-row-per-state-per-month price fact answer the same time predicate. It is
generated rather than sourced, so every property below is checkable exactly --
there is no sampling error and no excuse for an off-by-one.
"""
from __future__ import annotations

import calendar
import datetime as dt

import duckdb
import pytest

from config import settings
from src.model import dimensions


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> duckdb.DuckDBPyConnection:
    """A freshly generated dim_date in an isolated in-memory database."""
    connection = duckdb.connect(":memory:")
    dimensions.build_dim_date(connection)
    yield connection
    connection.close()


def test_row_count_is_exactly_the_calendar_plus_unknown(built) -> None:
    expected_days = (settings.DATE_DIM_END - settings.DATE_DIM_START).days + 1
    rows, calendar_days = built.execute(
        "SELECT count(*), count(*) FILTER (WHERE date_key > 0) FROM dim_date"
    ).fetchone()
    assert calendar_days == expected_days
    assert rows == expected_days + 1, "exactly one Unknown member"


def test_unknown_member_exists_and_is_null_safe(built) -> None:
    row = built.execute(
        "SELECT * FROM dim_date WHERE date_key = ?", [settings.UNKNOWN_KEY]
    ).df()
    assert len(row) == 1
    assert row["full_date"].isna().all()
    assert row["year_month"].iloc[0] == "Unknown"


def test_date_key_is_yyyymmdd_and_unique(built) -> None:
    duplicates = built.execute(
        "SELECT date_key FROM dim_date GROUP BY 1 HAVING count(*) > 1"
    ).fetchall()
    assert not duplicates

    mismatched = built.execute(
        """
        SELECT count(*) FROM dim_date
        WHERE date_key > 0
          AND date_key <> CAST(strftime(full_date, '%Y%m%d') AS INTEGER)
        """
    ).fetchone()[0]
    assert mismatched == 0


def test_date_key_sorts_chronologically(built) -> None:
    """The reason an integer YYYYMMDD key is used at all."""
    out_of_order = built.execute(
        """
        SELECT count(*) FROM (
            SELECT full_date,
                   lag(full_date) OVER (ORDER BY date_key) AS prev
            FROM dim_date WHERE date_key > 0
        ) WHERE prev IS NOT NULL AND full_date <= prev
        """
    ).fetchone()[0]
    assert out_of_order == 0


def test_day_of_week_is_iso_numbered(built) -> None:
    """1 = Monday .. 7 = Sunday, stated explicitly because DuckDB's own
    dayofweek() is 0 = Sunday and the two are trivially confused."""
    rows = built.execute(
        """
        SELECT full_date, day_of_week, day_name, is_weekend
        FROM dim_date
        WHERE full_date IN (DATE '2024-01-01', DATE '2024-01-06',
                            DATE '2024-01-07', DATE '2024-02-29')
        ORDER BY full_date
        """
    ).fetchall()
    lookup = {row[0]: row for row in rows}

    monday = lookup[dt.date(2024, 1, 1)]
    assert monday[1] == 1 and monday[2] == "Monday" and monday[3] is False

    saturday = lookup[dt.date(2024, 1, 6)]
    assert saturday[1] == 6 and saturday[2] == "Saturday" and saturday[3] is True

    sunday = lookup[dt.date(2024, 1, 7)]
    assert sunday[1] == 7 and sunday[2] == "Sunday" and sunday[3] is True


def test_weekend_flag_matches_day_of_week_everywhere(built) -> None:
    inconsistent = built.execute(
        """
        SELECT count(*) FROM dim_date
        WHERE date_key > 0 AND is_weekend <> (day_of_week >= 6)
        """
    ).fetchone()[0]
    assert inconsistent == 0


def test_leap_day_is_present_and_2023_has_none(built) -> None:
    assert built.execute(
        "SELECT count(*) FROM dim_date WHERE full_date = DATE '2024-02-29'"
    ).fetchone()[0] == 1
    assert built.execute(
        "SELECT count(*) FROM dim_date WHERE year = 2024"
    ).fetchone()[0] == 366
    assert built.execute(
        "SELECT count(*) FROM dim_date WHERE year = 2023"
    ).fetchone()[0] == 365


def test_month_end_flag_is_exact_for_every_month(built) -> None:
    """One month end per month, and it falls on the true last day."""
    rows = built.execute(
        """
        SELECT year, month_number, count(*) FILTER (WHERE is_month_end) AS ends,
               max(CASE WHEN is_month_end THEN day_of_month END) AS end_day
        FROM dim_date
        WHERE date_key > 0
        GROUP BY 1, 2
        """
    ).fetchall()
    assert rows
    for year, month, ends, end_day in rows:
        assert ends == 1, f"{year}-{month:02d} has {ends} month-end rows"
        assert end_day == calendar.monthrange(int(year), int(month))[1]


def test_year_month_is_sortable_and_well_formed(built) -> None:
    bad = built.execute(
        """
        SELECT count(*) FROM dim_date
        WHERE date_key > 0
          AND year_month <> printf('%04d-%02d', year, month_number)
        """
    ).fetchone()[0]
    assert bad == 0


def test_quarter_boundaries(built) -> None:
    quarters = built.execute(
        """
        SELECT month_number, min(quarter), max(quarter)
        FROM dim_date WHERE date_key > 0 GROUP BY 1 ORDER BY 1
        """
    ).fetchall()
    expected = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2,
                7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
    for month, low, high in quarters:
        assert low == high == expected[int(month)]


def test_span_covers_every_source_window(built) -> None:
    """The calendar must cover the WFP panel, the NBS panel and the trip year."""
    first, last = built.execute(
        "SELECT min(full_date), max(full_date) FROM dim_date WHERE date_key > 0"
    ).fetchone()
    assert first <= dt.date(2016, 1, 1)
    assert last >= dt.date(2026, 7, 15), "WFP publishes to mid-2026"
    assert first <= settings.NYC_WINDOW_START
    assert last >= dt.date(2024, 12, 31)


def test_generation_is_deterministic() -> None:
    """Two independent builds must be byte-identical (constraint 2.6)."""
    first = duckdb.connect(":memory:")
    second = duckdb.connect(":memory:")
    try:
        dimensions.build_dim_date(first)
        dimensions.build_dim_date(second)
        left = first.execute("SELECT * FROM dim_date ORDER BY date_key").df()
        right = second.execute("SELECT * FROM dim_date ORDER BY date_key").df()
        assert left.equals(right)
    finally:
        first.close()
        second.close()
