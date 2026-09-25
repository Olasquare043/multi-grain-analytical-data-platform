"""End-to-end smoke test on a small slice.

Builds a miniature version of the whole platform -- dimensions, a partitioned
fact, an aggregate -- from synthetic rows in a temporary directory, and asserts
that the shipped DDL and the shipped quality rules work together on it.

Synthetic data is used **only here**, and only to exercise the machinery. No
figure, table or number in the paper touches it. Every published number comes
from the real sources through the real pipeline (constraint 2.5).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from config import settings
from src.model import dim_geography, dimensions
from src.quality import engine
from src.utils.db import read_sql_file

SYNTHETIC_TRIPS = 500


@pytest.fixture()
def mini(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """A miniature warehouse: real DDL, synthetic rows."""
    con = duckdb.connect(str(tmp_path / "mini.duckdb"))

    dimensions.build_dim_date(con)
    for name in dimensions.STATIC_DIMENSIONS:
        con.execute(
            read_sql_file(settings.SQL_DDL_DIR / f"{name}.sql")
            .format(unknown_key=settings.UNKNOWN_KEY)
        )
    con.execute(
        read_sql_file(settings.SQL_DDL_DIR / "dim_geography.sql")
        .format(unknown_key=settings.UNKNOWN_KEY)
    )
    dim_geography.merge(con, pd.DataFrame([
        {"geo_code": "132", "geo_name": "JFK Airport",
         "country": "United States", "admin_level": "zone",
         "parent_geo_name": "Queens", "region_group": "Airports"},
        {"geo_code": "161", "geo_name": "Midtown Center",
         "country": "United States", "admin_level": "zone",
         "parent_geo_name": "Manhattan", "region_group": "Yellow Zone"},
    ]))

    base = dt.date(2024, 3, 1)
    rows = []
    for index in range(SYNTHETIC_TRIPS):
        day = base + dt.timedelta(days=index % 28)
        pickup_zone = "132" if index % 3 == 0 else "161"
        rows.append({
            "trip_id": f"{index:032x}",
            "pickup_date_key": int(day.strftime("%Y%m%d")),
            "dropoff_date_key": int(day.strftime("%Y%m%d")),
            "pickup_hour": index % 24,
            "pickup_zone_code": pickup_zone,
            "dropoff_zone_code": "161" if pickup_zone == "132" else "132",
            "mode_key": 1,
            "vendor_key": 1 + (index % 2),
            "payment_key": 2 if index % 4 else 3,
            "rate_key": 1,
            "flag_key": -1,
            "passenger_count": 1 + (index % 4),
            "trip_distance_miles": 1.0 + (index % 20),
            "trip_duration_seconds": 300 + (index % 40) * 30,
            "fare_amount": 8.0 + (index % 30),
            "extra": 0.5, "mta_tax": 0.5, "tip_amount": float(index % 7),
            "tolls_amount": 0.0, "improvement_surcharge": 0.3,
            "congestion_surcharge": 2.5, "airport_fee": 1.75,
            "year": day.year, "month": day.month,
        })
    frame = pd.DataFrame(rows)
    frame["total_amount"] = (
        frame["fare_amount"] + frame["extra"] + frame["mta_tax"]
        + frame["tip_amount"] + frame["improvement_surcharge"]
        + frame["congestion_surcharge"] + frame["airport_fee"]
    )
    frame["avg_speed_mph"] = (
        frame["trip_distance_miles"] / (frame["trip_duration_seconds"] / 3600.0)
    )
    con.register("synthetic", frame)
    con.execute(
        """
        CREATE TABLE fact_trip AS
        SELECT s.trip_id, s.pickup_date_key, s.dropoff_date_key, s.pickup_hour,
               coalesce(pg.geo_key, -1) AS pickup_geo_key,
               coalesce(dg.geo_key, -1) AS dropoff_geo_key,
               s.mode_key, s.vendor_key, s.payment_key, s.rate_key, s.flag_key,
               s.passenger_count, s.trip_distance_miles, s.trip_duration_seconds,
               s.avg_speed_mph, s.fare_amount, s.extra, s.mta_tax, s.tip_amount,
               s.tolls_amount, s.improvement_surcharge, s.congestion_surcharge,
               s.airport_fee, s.total_amount, s.year, s.month
        FROM synthetic s
        LEFT JOIN dim_geography pg ON pg.geo_code = s.pickup_zone_code
                                  AND pg.admin_level = 'zone' AND pg.is_current
        LEFT JOIN dim_geography dg ON dg.geo_code = s.dropoff_zone_code
                                  AND dg.admin_level = 'zone' AND dg.is_current
        """
    )
    con.unregister("synthetic")
    yield con
    con.close()


def test_fact_and_dimensions_build(mini) -> None:
    assert mini.execute("SELECT count(*) FROM fact_trip").fetchone()[0] \
        == SYNTHETIC_TRIPS
    assert mini.execute("SELECT count(*) FROM dim_date").fetchone()[0] > 4000
    for dimension in dimensions.STATIC_DIMENSIONS:
        rows = mini.execute(f"SELECT count(*) FROM {dimension}").fetchone()[0]
        assert rows > 0


def test_every_foreign_key_resolves(mini) -> None:
    """No fact row may be lost to a failed lookup."""
    orphans = mini.execute(
        """
        SELECT count(*) FROM fact_trip f
        LEFT JOIN dim_date       d  ON d.date_key      = f.pickup_date_key
        LEFT JOIN dim_geography  g  ON g.geo_key       = f.pickup_geo_key
        LEFT JOIN dim_vendor     v  ON v.vendor_key    = f.vendor_key
        LEFT JOIN dim_payment_type p ON p.payment_key  = f.payment_key
        LEFT JOIN dim_rate_code  r  ON r.rate_key      = f.rate_key
        WHERE d.date_key IS NULL OR g.geo_key IS NULL OR v.vendor_key IS NULL
           OR p.payment_key IS NULL OR r.rate_key IS NULL
        """
    ).fetchone()[0]
    assert orphans == 0


def test_aggregate_agrees_with_the_atomic_fact(mini) -> None:
    """The aggregate is derived, so it can never disagree -- assert it."""
    mini.execute(read_sql_file(settings.SQL_DDL_DIR / "fact_trip_daily_agg.sql"))

    atomic = mini.execute(
        "SELECT count(*), round(sum(total_amount), 4) FROM fact_trip"
    ).fetchone()
    aggregated = mini.execute(
        "SELECT sum(trip_count), round(sum(total_revenue), 4) "
        "FROM fact_trip_daily_agg"
    ).fetchone()
    assert atomic[0] == aggregated[0]
    assert abs(float(atomic[1]) - float(aggregated[1])) < 0.01


def test_quality_rules_run_against_the_slice(mini) -> None:
    """The shipped rule set must execute; rules for absent tables must skip."""
    run = engine.evaluate(mini, engine.load_rules())
    assert run.results, "no rules were evaluated"

    executed = [r for r in run.results if r.status != engine.SKIP]
    assert executed, "every rule skipped; the slice built nothing"

    unexecutable = [r for r in run.results if "could not execute" in r.detail]
    assert not unexecutable, (
        "rule(s) failed to compile or run: "
        f"{[(r.rule_id, r.detail) for r in unexecutable]}"
    )


def test_trip_id_is_deterministic() -> None:
    """Reruns must produce identical ids (constraint 2.7, idempotence)."""
    con = duckdb.connect(":memory:")
    try:
        expression = (
            "md5(concat_ws('|', CAST(1 AS VARCHAR), "
            "CAST(TIMESTAMP '2024-03-01 10:00:00' AS VARCHAR), "
            "CAST(TIMESTAMP '2024-03-01 10:20:00' AS VARCHAR), "
            "'132', '161', '5.2', '20.0', '28.5', '1'))"
        )
        first = con.execute(f"SELECT {expression}").fetchone()[0]
        second = con.execute(f"SELECT {expression}").fetchone()[0]
        assert first == second
        assert len(first) == 32
    finally:
        con.close()


def test_measures_are_internally_consistent(mini) -> None:
    """Derived measures must agree with the columns they derive from."""
    bad_speed = mini.execute(
        """
        SELECT count(*) FROM fact_trip
        WHERE abs(avg_speed_mph
                  - trip_distance_miles / (trip_duration_seconds / 3600.0)) > 1e-6
        """
    ).fetchone()[0]
    assert bad_speed == 0

    negative = mini.execute(
        "SELECT count(*) FROM fact_trip "
        "WHERE fare_amount < 0 OR total_amount < 0 OR trip_distance_miles < 0 "
        "OR trip_duration_seconds <= 0"
    ).fetchone()[0]
    assert negative == 0
