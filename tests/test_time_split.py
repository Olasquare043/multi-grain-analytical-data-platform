"""Asserts every split used by the ladder notebooks is genuinely time-ordered:
the maximum timestamp in training is strictly earlier than the minimum
timestamp in test. Runs against the real warehouse when it is built (the
same tables the notebooks read), and additionally against small synthetic
frames so the assertion itself is exercised even when the warehouse is
absent (e.g. in a CI environment that never ran the pipeline).
"""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from config import settings
from src.modelling.splits import (
    NIGERIA_TEST_END_KEY,
    NIGERIA_TEST_START_KEY,
    NIGERIA_TRAIN_END_KEY,
    NYC_TEST_MONTHS,
    NYC_TRAIN_MONTHS,
    Split,
    assert_split_is_time_ordered,
    nigeria_time_split,
    nyc_time_split,
)
from tests.conftest import require_table


# --------------------------------------------------------------------------- #
# Synthetic frames: exercise the assertion logic without needing a build.
# --------------------------------------------------------------------------- #
def test_assert_split_is_time_ordered_passes_a_genuinely_ordered_split() -> None:
    train = pd.DataFrame({"t": [1, 2, 3]})
    test = pd.DataFrame({"t": [4, 5]})
    assert_split_is_time_ordered(Split(train, test, "synthetic"), time_col="t")


def test_assert_split_is_time_ordered_catches_overlap() -> None:
    train = pd.DataFrame({"t": [1, 2, 5]})  # 5 is not earlier than test's min (4)
    test = pd.DataFrame({"t": [4, 6]})
    with pytest.raises(AssertionError):
        assert_split_is_time_ordered(Split(train, test, "synthetic"), time_col="t")


def test_nigeria_time_split_boundary_constants_do_not_overlap() -> None:
    assert NIGERIA_TRAIN_END_KEY < NIGERIA_TEST_START_KEY
    assert NIGERIA_TEST_START_KEY <= NIGERIA_TEST_END_KEY


def test_nyc_time_split_boundary_constants_do_not_overlap() -> None:
    assert max(NYC_TRAIN_MONTHS) < min(NYC_TEST_MONTHS)


def test_nigeria_time_split_on_synthetic_panel() -> None:
    frame = pd.DataFrame({
        "month_key": [20231101, 20240601, 20251101, 20251201, 20260301, 20260501],
        "price_ngn": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    split = nigeria_time_split(frame)
    assert len(split.train) == 3   # 2023-11, 2024-06, 2025-11
    assert len(split.test) == 3    # 2025-12, 2026-03, 2026-05
    assert_split_is_time_ordered(split, time_col="month_key")


def test_nyc_time_split_on_synthetic_panel() -> None:
    frame = pd.DataFrame({
        "year": [2024] * 6 + [2023],
        "month": [1, 5, 10, 11, 12, 12, 12],
        "x": range(7),
    })
    split = nyc_time_split(frame)
    assert len(split.train) == 3   # Jan, May, Oct 2024
    assert len(split.test) == 3    # Nov 2024 (1 row) + Dec 2024 (2 rows)
    # 2023-12 must be excluded from both sides, not silently swept into test.
    assert 2023 not in split.train["year"].values
    assert 2023 not in split.test["year"].values
    assert_split_is_time_ordered(split, time_col="month")


# --------------------------------------------------------------------------- #
# Against the real, already-built warehouse -- the tables the notebooks read.
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def con() -> duckdb.DuckDBPyConnection:
    if not settings.WAREHOUSE_DB.exists():
        pytest.skip(f"{settings.WAREHOUSE_DB} not found; build the pipeline first")
    connection = duckdb.connect(str(settings.WAREHOUSE_DB), read_only=True)
    try:
        yield connection
    finally:
        connection.close()


def test_nigeria_split_on_the_real_fact_table_is_time_ordered(con) -> None:
    require_table(con, "fact_fuel_price_monthly")
    frame = con.execute(
        "SELECT month_key, geo_key, price_ngn FROM fact_fuel_price_monthly"
    ).df()
    split = nigeria_time_split(frame)
    assert len(split.train) > 0
    assert len(split.test) > 0
    assert_split_is_time_ordered(split, time_col="month_key")


def test_nyc_split_on_the_real_fact_table_is_time_ordered(con) -> None:
    require_table(con, "fact_trip")
    frame = con.execute(
        "SELECT year, month, pickup_hour FROM fact_trip "
        "USING SAMPLE 200000 ROWS"
    ).df()
    split = nyc_time_split(frame)
    assert len(split.train) > 0
    assert len(split.test) > 0
    assert_split_is_time_ordered(split, time_col="month")
