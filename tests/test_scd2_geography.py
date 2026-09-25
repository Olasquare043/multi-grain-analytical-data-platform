"""Exercise the Type 2 slowly changing behaviour of ``dim_geography``.

Section 4 requires this dimension to implement genuine Type 2 SCD, and requires
the behaviour to be exercised by reloading with a changed ``region_group`` and
asserting a closed prior row plus a new current row.

The test builds its own dimension in an in-memory database from the real DDL and
drives the real merge function, so it tests the shipped implementation rather
than a reimplementation of it.
"""
from __future__ import annotations

import datetime as dt

import duckdb
import pandas as pd
import pytest

from config import settings
from src.model import dim_geography
from src.utils.db import read_sql_file

EFFECTIVE = dt.date(2026, 6, 1)


def _incoming(region_group: str = "South West") -> pd.DataFrame:
    """A minimal current-truth set: one market, one state, one zone."""
    frame = pd.DataFrame([
        {
            "geo_code": "42", "geo_name": "Mile 12", "country": "Nigeria",
            "admin_level": "market", "parent_geo_name": "Lagos",
            "region_group": region_group,
        },
        {
            "geo_code": "Lagos", "geo_name": "Lagos", "country": "Nigeria",
            "admin_level": "state", "parent_geo_name": "Nigeria",
            "region_group": "South West",
        },
        {
            "geo_code": "132", "geo_name": "JFK Airport",
            "country": "United States", "admin_level": "zone",
            "parent_geo_name": "Queens", "region_group": "Airports",
        },
    ])
    for column in dim_geography.TRACKED_ATTRIBUTES:
        frame[column] = frame[column].astype("string")
    return frame


@pytest.fixture()
def dimension(memory_con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """An empty dim_geography created from the shipped DDL."""
    ddl = read_sql_file(settings.SQL_DDL_DIR / "dim_geography.sql")
    memory_con.execute(ddl.format(unknown_key=settings.UNKNOWN_KEY))
    return memory_con


def _rows(con: duckdb.DuckDBPyConnection, geo_code: str = "42") -> pd.DataFrame:
    """Versions of one natural key, with validity dates as ``datetime.date``.

    DuckDB hands DATE columns to pandas as ``Timestamp``; normalising here keeps
    every assertion below a like-for-like comparison with the configured dates.
    """
    frame = con.execute(
        "SELECT * FROM dim_geography WHERE geo_code = ? ORDER BY geo_key",
        [geo_code],
    ).df()
    for column in ("valid_from", "valid_to"):
        frame[column] = frame[column].map(lambda value: pd.Timestamp(value).date())
    return frame


# --------------------------------------------------------------------------- #
def test_initial_load_creates_one_current_version_and_unknown_member(
    dimension: duckdb.DuckDBPyConnection,
) -> None:
    outcome = dim_geography.merge(dimension, _incoming())

    assert outcome["first_load"] is True
    assert outcome["versioned"] == 0

    unknown = dimension.execute(
        "SELECT * FROM dim_geography WHERE geo_key = ?", [settings.UNKNOWN_KEY]
    ).df()
    assert len(unknown) == 1, "every dimension must carry an Unknown member"
    assert bool(unknown["is_current"].iloc[0])

    rows = _rows(dimension)
    assert len(rows) == 1
    assert bool(rows["is_current"].iloc[0])
    assert rows["valid_to"].iloc[0] == settings.SCD_HIGH_DATE
    assert rows["valid_from"].iloc[0] == settings.SCD_INITIAL_VALID_FROM, (
        "the initial load must stamp a fixed epoch, not the run date, so two "
        "cold runs on different days produce identical dimensions"
    )


def test_reload_without_change_is_idempotent(
    dimension: duckdb.DuckDBPyConnection,
) -> None:
    """The case most Type 2 implementations get wrong: no change, no version."""
    dim_geography.merge(dimension, _incoming())
    before = dimension.execute("SELECT count(*) FROM dim_geography").fetchone()[0]

    outcome = dim_geography.merge(dimension, _incoming(), effective_date=EFFECTIVE)

    after = dimension.execute("SELECT count(*) FROM dim_geography").fetchone()[0]
    assert after == before, "an unchanged reload must not add a row"
    assert outcome["inserted"] == 0
    assert outcome["versioned"] == 0
    assert outcome["unchanged"] == 3
    assert len(_rows(dimension)) == 1


def test_changed_region_group_closes_prior_and_opens_new_version(
    dimension: duckdb.DuckDBPyConnection,
) -> None:
    """The behaviour section 4 names explicitly."""
    dim_geography.merge(dimension, _incoming(region_group="South West"))

    outcome = dim_geography.merge(
        dimension, _incoming(region_group="South East"), effective_date=EFFECTIVE,
    )
    assert outcome["versioned"] == 1
    assert outcome["inserted"] == 0

    rows = _rows(dimension)
    assert len(rows) == 2, "the natural key must now hold two versions"

    closed = rows[~rows["is_current"].astype(bool)].iloc[0]
    current = rows[rows["is_current"].astype(bool)].iloc[0]

    # the prior row is closed the day before the change took effect
    assert closed["region_group"] == "South West"
    assert closed["valid_to"] == EFFECTIVE - dt.timedelta(days=1)
    assert bool(closed["is_current"]) is False

    # the new row is current, carries the new value and a NEW surrogate key
    assert current["region_group"] == "South East"
    assert current["valid_from"] == EFFECTIVE
    assert current["valid_to"] == settings.SCD_HIGH_DATE
    assert bool(current["is_current"]) is True
    assert current["geo_key"] != closed["geo_key"], (
        "a new version must get a new surrogate key, so facts loaded earlier "
        "keep pointing at the attribution they were recorded under"
    )

    # validity intervals must not overlap
    assert closed["valid_to"] < current["valid_from"]


def test_exactly_one_current_row_per_natural_key(
    dimension: duckdb.DuckDBPyConnection,
) -> None:
    """The defining Type 2 invariant, also enforced as quality rule Q078."""
    dim_geography.merge(dimension, _incoming(region_group="South West"))
    dim_geography.merge(dimension, _incoming(region_group="South East"),
                        effective_date=EFFECTIVE)
    dim_geography.merge(dimension, _incoming(region_group="North Central"),
                        effective_date=EFFECTIVE + dt.timedelta(days=30))

    duplicates = dimension.execute(
        """
        SELECT geo_code, country, admin_level, count(*) AS n
        FROM dim_geography
        WHERE is_current
        GROUP BY ALL
        HAVING count(*) > 1
        """
    ).df()
    assert duplicates.empty, (
        f"a natural key may hold many versions but exactly one current row; "
        f"found duplicates:\n{duplicates}"
    )
    assert len(_rows(dimension)) == 3, "three successive changes, three versions"


def test_new_natural_key_is_inserted_not_versioned(
    dimension: duckdb.DuckDBPyConnection,
) -> None:
    dim_geography.merge(dimension, _incoming())

    extended = pd.concat([
        _incoming(),
        pd.DataFrame([{
            "geo_code": "77", "geo_name": "Bodija", "country": "Nigeria",
            "admin_level": "market", "parent_geo_name": "Oyo",
            "region_group": "South West",
        }]),
    ], ignore_index=True)
    for column in dim_geography.TRACKED_ATTRIBUTES:
        extended[column] = extended[column].astype("string")

    outcome = dim_geography.merge(dimension, extended, effective_date=EFFECTIVE)
    assert outcome["inserted"] == 1
    assert outcome["versioned"] == 0
    assert len(_rows(dimension, "77")) == 1


def test_member_absent_from_source_is_left_current(
    dimension: duckdb.DuckDBPyConnection,
) -> None:
    """A place vanishing from one vintage is not evidence it ceased to exist.

    Closing it would orphan facts already loaded against it, so the merge leaves
    it current. This is a deliberate choice, documented in docs/scd_strategy.md.
    """
    dim_geography.merge(dimension, _incoming())

    reduced = _incoming().iloc[1:].reset_index(drop=True)  # drop the market
    dim_geography.merge(dimension, reduced, effective_date=EFFECTIVE)

    rows = _rows(dimension)
    assert len(rows) == 1
    assert bool(rows["is_current"].iloc[0]) is True
    assert rows["valid_to"].iloc[0] == settings.SCD_HIGH_DATE
