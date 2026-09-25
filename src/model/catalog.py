"""Warehouse catalogue: views over the gold layer, and a portable Parquet export.

``data/warehouse.duckdb`` is a *catalogue*, not the warehouse. It holds the small
dimension and monthly-fact tables plus a view over the partitioned trip Parquet.
Everything in it is also exported to ``data/gold/*.parquet``, so the gold layer
can be read by any engine -- including the BigQuery comparison in section 9 --
without DuckDB being installed.
"""
from __future__ import annotations

from typing import Any

import duckdb

from config import settings
from src.model import dimensions
from src.model.facts import fact_trip_relation
from src.utils.db import connect, sql_df, table_exists
from src.utils.io_utils import dir_size, human_bytes
from src.utils.logging_setup import get_logger, stage

LOG = get_logger(__name__)

#: Tables exported to Parquet. fact_trip is excluded because it is already
#: Parquet, and copying 41 M rows a second time would breach the disk budget.
EXPORTED_TABLES = (
    "dim_date", "dim_geography", "dim_transport_mode", "dim_commodity",
    "dim_payment_type", "dim_rate_code", "dim_vendor", "dim_trip_flags",
    "fact_market_price_monthly", "fact_fuel_price_monthly",
    "fact_trip_daily_agg", "fact_weather_daily",
)

#: Every gold relation, in the order the data dictionary presents them.
GOLD_RELATIONS = (*dimensions.ALL_DIMENSIONS, "fact_trip",
                  "fact_market_price_monthly", "fact_fuel_price_monthly",
                  "fact_trip_daily_agg", "fact_weather_daily")


def _sql(path) -> str:
    return str(path).replace("\\", "/")


#: Silver artefacts exposed as views so the silver-layer quality rules can be
#: expressed against table names rather than file paths.
SILVER_VIEWS = {
    "silver_nyc_zones": "silver_nyc_zones.parquet",
    "silver_wfp_prices": "silver_wfp_prices.parquet",
    "silver_wfp_markets": "silver_wfp_markets.parquet",
    "silver_nbs_pms": "silver_nbs_pms.parquet",
    "silver_weather_daily": "silver_weather_daily.parquet",
}


def register_views(con: duckdb.DuckDBPyConnection) -> None:
    """(Re)register the views that make the Parquet layers queryable by name."""
    con.execute(
        "CREATE OR REPLACE VIEW fact_trip AS SELECT * FROM " + fact_trip_relation()
    )
    LOG.info("registered view fact_trip over the partitioned Parquet dataset")

    for view, filename in SILVER_VIEWS.items():
        path = settings.SILVER_DIR / filename
        if not path.exists():
            LOG.warning("silver view %s not registered: %s absent", view, filename)
            continue
        con.execute(
            f"CREATE OR REPLACE VIEW {view} AS "
            f"SELECT * FROM read_parquet('{_sql(path)}')"
        )


def export_parquet(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Export catalogue tables to ``data/gold/*.parquet``."""
    exported: dict[str, Any] = {}
    for table in EXPORTED_TABLES:
        if not table_exists(con, table):
            LOG.warning("skipping export of %s: not present in the catalogue", table)
            continue
        destination = settings.GOLD_DIR / f"{table}.parquet"
        con.execute(
            f"COPY (SELECT * FROM {table}) TO '{_sql(destination)}' "
            f"(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        rows = int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        exported[table] = {
            "rows": rows,
            "bytes": destination.stat().st_size,
            "path": destination.name,
        }
    LOG.info("exported %d gold table(s) to Parquet", len(exported))
    return exported


def inventory(con: duckdb.DuckDBPyConnection) -> list[dict[str, Any]]:
    """Row count and on-disk footprint per gold relation.

    This is the raw material for analysis A10, and is computed here so that the
    analysis query and the run manifest cannot report different numbers.
    """
    records: list[dict[str, Any]] = []
    for relation in GOLD_RELATIONS:
        if not table_exists(con, relation):
            continue
        rows = int(con.execute(f"SELECT count(*) FROM {relation}").fetchone()[0])
        if relation == "fact_trip":
            path = settings.GOLD_DIR / "fact_trip"
        else:
            path = settings.GOLD_DIR / f"{relation}.parquet"
        size = dir_size(path)
        records.append({
            "relation": relation,
            "rows": rows,
            "bytes": size,
            "bytes_human": human_bytes(size),
            "bytes_per_row": round(size / rows, 3) if rows else None,
        })
    return records


def build(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, Any]:
    """Register views, export Parquet and report the gold inventory."""
    with stage("gold: catalogue and Parquet export", LOG):
        owned = con is None
        con = con or connect(settings.WAREHOUSE_DB)

        register_views(con)
        exported = export_parquet(con)
        records = inventory(con)

        for record in records:
            LOG.info("  %-28s %12s rows  %10s  %8s B/row",
                     record["relation"], f"{record['rows']:,}",
                     record["bytes_human"],
                     record["bytes_per_row"] if record["bytes_per_row"] else "-")

        if owned:
            con.close()
        return {"exported": exported, "inventory": records}
