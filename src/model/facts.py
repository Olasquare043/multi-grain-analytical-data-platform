"""Gold facts across three grains, plus one aggregate.

``fact_trip`` is written one month at a time. Two reasons, both engineering
rather than stylistic: peak memory stays bounded well inside the 8 GB budget,
and a single failed month can be rebuilt without touching the other eleven.
Idempotence (constraint 2.7) is enforced by clearing the partition directory
before the first month rather than relying on overwrite semantics.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import duckdb

from config import settings
from src.transform import silver
from src.utils.db import connect, read_sql_file, sql_df
from src.utils.io_utils import dir_size, human_bytes, reset_dir
from src.utils.logging_setup import get_logger, stage
from src.utils.manifest import get_manifest

LOG = get_logger(__name__)

FACT_TRIP_DIR = "fact_trip"
YELLOW_TAXI_MODE_KEY = 1


def _sql(path) -> str:
    return str(path).replace("\\", "/")


def _ddl(name: str) -> str:
    return read_sql_file(settings.SQL_DDL_DIR / f"{name}.sql")


def fact_trip_relation() -> str:
    """Hive-partitioned scan of the gold trip fact, for use in any query."""
    root = settings.GOLD_DIR / FACT_TRIP_DIR
    return (
        f"read_parquet('{_sql(root)}/**/*.parquet', hive_partitioning=true, "
        f"hive_types_autocast=1)"
    )


def build_fact_trip(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Write ``fact_trip`` as Hive-partitioned Parquet, one month per statement."""
    output_dir = settings.GOLD_DIR / FACT_TRIP_DIR
    reset_dir(output_dir)  # idempotence: no orphaned fragments from a prior run

    template = _ddl("fact_trip")
    per_month: list[dict[str, Any]] = []
    total_rows = 0

    for month in settings.NYC_TRIP_MONTHS:
        predicate = (
            f"year(t.pickup_ts) = {settings.NYC_TRIP_YEAR} "
            f"AND month(t.pickup_ts) = {month}"
        )
        statement = template.format(
            unknown_key=settings.UNKNOWN_KEY,
            mode_key=YELLOW_TAXI_MODE_KEY,
            month_predicate=predicate,
            output_dir=_sql(output_dir),
            filename_pattern=f"part-{settings.NYC_TRIP_YEAR}-{month:02d}-{{i}}",
        )
        started = time.perf_counter()
        con.execute(statement)
        elapsed = time.perf_counter() - started

        partition = output_dir / f"year={settings.NYC_TRIP_YEAR}" / f"month={month}"
        rows = int(
            con.execute(
                f"SELECT count(*) FROM read_parquet('{_sql(partition)}/*.parquet')"
            ).fetchone()[0]
        ) if partition.exists() else 0
        total_rows += rows
        per_month.append({
            "month": f"{settings.NYC_TRIP_YEAR}-{month:02d}",
            "rows": rows,
            "bytes": dir_size(partition),
            "seconds": round(elapsed, 2),
        })
        LOG.info("fact_trip %s-%02d: %s rows, %s, %.1fs",
                 settings.NYC_TRIP_YEAR, month, f"{rows:,}",
                 human_bytes(dir_size(partition)), elapsed)

    # Trips whose pickup falls outside 2024 land in no 2024 partition. They are
    # a genuine defect of the source and must not be lost, so they are written
    # to their own partitions by their true year.
    leftover_predicate = (
        f"NOT (year(t.pickup_ts) = {settings.NYC_TRIP_YEAR})"
    )
    statement = template.format(
        unknown_key=settings.UNKNOWN_KEY,
        mode_key=YELLOW_TAXI_MODE_KEY,
        month_predicate=leftover_predicate,
        output_dir=_sql(output_dir),
        filename_pattern="part-outofwindow-{i}",
    )
    con.execute(statement)

    grand_total = int(
        con.execute(f"SELECT count(*) FROM {fact_trip_relation()}").fetchone()[0]
    )
    out_of_window_rows = grand_total - total_rows
    partitions = sorted(
        p.relative_to(output_dir).as_posix()
        for p in output_dir.glob("year=*/month=*")
    )
    size = dir_size(output_dir)

    LOG.info(
        "fact_trip complete: %s rows in %d partition(s), %s on disk "
        "(%.1f bytes/row); %s row(s) sit outside the declared 2024 window and "
        "are retained in their true year's partition",
        f"{grand_total:,}", len(partitions), human_bytes(size),
        size / grand_total if grand_total else 0, f"{out_of_window_rows:,}",
    )
    return {
        "rows": grand_total,
        "rows_in_2024_partitions": total_rows,
        "rows_outside_declared_window": out_of_window_rows,
        "partition_count": len(partitions),
        "partitions": partitions,
        "bytes": size,
        "bytes_human": human_bytes(size),
        "bytes_per_row": round(size / grand_total, 2) if grand_total else 0.0,
        "per_month": per_month,
    }


def build_fact_market_price_monthly(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Build the market/commodity/month price fact."""
    con.execute(
        _ddl("fact_market_price_monthly").format(
            silver_wfp=_sql(settings.SILVER_DIR / silver.SILVER_WFP),
            unknown_key=settings.UNKNOWN_KEY,
        )
    )
    stats = sql_df(
        con,
        """
        SELECT count(*)                                         AS rows,
               count(DISTINCT month_key)                        AS months,
               count(DISTINCT geo_key)                          AS markets,
               count(DISTINCT commodity_key)                    AS commodities,
               count(*) FILTER (WHERE geo_key = -1)             AS unknown_geo,
               count(*) FILTER (WHERE commodity_key = -1)       AS unknown_commodity,
               sum(observation_count)                           AS source_rows,
               min(month_key)                                   AS month_min,
               max(month_key)                                   AS month_max
        FROM fact_market_price_monthly
        """,
    ).iloc[0].to_dict()
    LOG.info("fact_market_price_monthly: %s rows over %d months, %d markets, "
             "%d commodities (unknown geo %d, unknown commodity %d)",
             f"{int(stats['rows']):,}", int(stats["months"]), int(stats["markets"]),
             int(stats["commodities"]), int(stats["unknown_geo"]),
             int(stats["unknown_commodity"]))
    return {k: int(v) for k, v in stats.items()}


def build_fact_fuel_price_monthly(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Build the state/month petrol price fact."""
    con.execute(
        _ddl("fact_fuel_price_monthly").format(
            silver_nbs=_sql(settings.SILVER_DIR / silver.SILVER_NBS),
            unknown_key=settings.UNKNOWN_KEY,
            fuel_type=settings.NBS_FUEL_TYPE,
        )
    )
    stats = sql_df(
        con,
        """
        SELECT count(*)                                  AS rows,
               count(DISTINCT month_key)                 AS months,
               count(DISTINCT geo_key)                   AS states,
               count(*) FILTER (WHERE geo_key = -1)      AS unknown_geo,
               count(*) FILTER (WHERE mom_pct_change IS NULL) AS null_mom,
               count(*) FILTER (WHERE yoy_pct_change IS NULL) AS null_yoy,
               min(month_key)                            AS month_min,
               max(month_key)                            AS month_max
        FROM fact_fuel_price_monthly
        """,
    ).iloc[0].to_dict()
    LOG.info("fact_fuel_price_monthly: %d rows over %d months and %d states "
             "(unknown geo %d); %d null MoM and %d null YoY, which is the "
             "expected consequence of not interpolating absent prior periods",
             int(stats["rows"]), int(stats["months"]), int(stats["states"]),
             int(stats["unknown_geo"]), int(stats["null_mom"]),
             int(stats["null_yoy"]))
    return {k: int(v) for k, v in stats.items()}


def build_fact_trip_daily_agg(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Build the daily/zone/mode aggregate used by benchmark B3."""
    con.execute("CREATE OR REPLACE VIEW fact_trip AS SELECT * FROM "
                + fact_trip_relation())
    started = time.perf_counter()
    con.execute(_ddl("fact_trip_daily_agg"))
    elapsed = time.perf_counter() - started

    stats = sql_df(
        con,
        """
        SELECT count(*)                       AS rows,
               count(DISTINCT date_key)       AS dates,
               count(DISTINCT pickup_geo_key) AS zones,
               sum(trip_count)                AS trips_covered
        FROM fact_trip_daily_agg
        """,
    ).iloc[0].to_dict()
    LOG.info("fact_trip_daily_agg: %s rows covering %s trips across %d dates "
             "and %d zones, built in %.1fs",
             f"{int(stats['rows']):,}", f"{int(stats['trips_covered']):,}",
             int(stats["dates"]), int(stats["zones"]), elapsed)
    return {**{k: int(v) for k, v in stats.items()},
            "build_seconds": round(elapsed, 2)}


def build_fact_weather_daily(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Build the optional weather fact, or report it skipped."""
    source = settings.SILVER_DIR / silver.SILVER_WEATHER
    if not Path(source).exists():
        LOG.warning("fact_weather_daily skipped: %s absent (Source E is optional)",
                    source.name)
        return {"status": "skipped", "reason": "silver weather artefact absent"}
    con.execute(
        _ddl("fact_weather_daily").format(
            silver_weather=_sql(source),
            unknown_key=settings.UNKNOWN_KEY,
            weather_geo_code=settings.WEATHER_GEO_CODE,
        )
    )
    stats = sql_df(
        con,
        """
        SELECT count(*) AS rows,
               count(*) FILTER (WHERE geo_key = -1) AS unknown_geo,
               min(date_key) AS date_min, max(date_key) AS date_max
        FROM fact_weather_daily
        """,
    ).iloc[0].to_dict()
    LOG.info("fact_weather_daily: %d rows (%s to %s)", int(stats["rows"]),
             stats["date_min"], stats["date_max"])
    return {"status": "loaded", **{k: int(v) for k, v in stats.items()}}


def build(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, Any]:
    """Build every fact table in dependency order."""
    with stage("gold: facts", LOG):
        owned = con is None
        con = con or connect(settings.WAREHOUSE_DB)
        manifest = get_manifest()
        settings.GOLD_DIR.mkdir(parents=True, exist_ok=True)

        # silver_nyc_trip is a view; it must exist in this connection.
        silver.create_trip_view(con)

        diagnostics = {
            "fact_trip": build_fact_trip(con),
            "fact_market_price_monthly": build_fact_market_price_monthly(con),
            "fact_fuel_price_monthly": build_fact_fuel_price_monthly(con),
            "fact_trip_daily_agg": build_fact_trip_daily_agg(con),
            "fact_weather_daily": build_fact_weather_daily(con),
        }

        for table, detail in diagnostics.items():
            if "rows" in detail:
                manifest.record_rows(table, detail["rows"])
        manifest.record_artefact_size(
            "fact_trip", settings.GOLD_DIR / FACT_TRIP_DIR
        )
        manifest.data.setdefault("gold", {}).update(diagnostics)
        manifest.save()

        if owned:
            con.close()
        return diagnostics
