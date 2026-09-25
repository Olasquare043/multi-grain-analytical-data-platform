"""An optional source must never break the core run.

Source E (weather) is non-blocking by specification. During a real cold run the
Open-Meteo API timed out, ``fact_weather_daily`` was correctly never built -- and
analysis A10 then crashed because its SQL referenced that table unconditionally.
These tests pin down the fix: optional SQL blocks are dropped, and reported,
when their table is absent, and kept verbatim when it is present.
"""
from __future__ import annotations

import duckdb

from src.analysis.run_analysis import strip_optional_blocks
from src.utils.db import read_sql_file
from config import settings

SQL = """SELECT 'core' AS source, count(*) AS n FROM core_table
-- optional:optional_table begin
UNION ALL
SELECT 'optional', count(*) FROM optional_table
-- optional:optional_table end
ORDER BY 1;
"""


def test_block_is_removed_and_reported_when_table_absent(memory_con) -> None:
    memory_con.execute("CREATE TABLE core_table AS SELECT 1 AS x")
    sql, omitted = strip_optional_blocks(SQL, memory_con)

    assert omitted == ["optional_table"]
    assert "optional_table" not in sql
    rows = memory_con.execute(sql).fetchall()
    assert rows == [("core", 1)]


def test_block_is_kept_when_table_present(memory_con) -> None:
    memory_con.execute("CREATE TABLE core_table AS SELECT 1 AS x")
    memory_con.execute("CREATE TABLE optional_table AS SELECT 1 AS y UNION ALL SELECT 2")
    sql, omitted = strip_optional_blocks(SQL, memory_con)

    assert omitted == []
    rows = memory_con.execute(sql).fetchall()
    assert rows == [("core", 1), ("optional", 2)]


def test_a10_runs_without_the_weather_fact() -> None:
    """The shipped A10 query must execute when Source E was skipped."""
    con = duckdb.connect(":memory:")
    try:
        for table, ddl in {
            "fact_trip": "pickup_date_key INT, pickup_geo_key INT, mode_key INT, flag_key INT",
            "fact_trip_daily_agg": "date_key INT, pickup_geo_key INT, mode_key INT",
            "fact_market_price_monthly": "month_key INT, geo_key INT, commodity_key INT",
            "fact_fuel_price_monthly": "month_key INT, geo_key INT",
        }.items():
            con.execute(f"CREATE TABLE {table} ({ddl})")

        raw = read_sql_file(settings.SQL_ANALYSIS_DIR / "A10_grain_comparison.sql")
        sql, omitted = strip_optional_blocks(raw, con)

        assert omitted == ["fact_weather_daily"]
        result = con.execute(sql).df()
        assert set(result["fact_table"]) == {
            "fact_trip", "fact_trip_daily_agg",
            "fact_market_price_monthly", "fact_fuel_price_monthly",
        }
    finally:
        con.close()
