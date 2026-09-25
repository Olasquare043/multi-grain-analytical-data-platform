"""Gold dimensions other than ``dim_geography``.

Each is created by executing its own file in ``sql/ddl/``, so the DDL an examiner
reads is the DDL that actually ran. Every one of these is Type 1; the reasoning
is in :mod:`src.model.dim_geography` and ``docs/scd_strategy.md``.
"""
from __future__ import annotations

from typing import Any

import duckdb

from config import settings
from src.transform import silver
from src.utils.db import connect, read_sql_file, sql_df
from src.utils.logging_setup import get_logger, stage

LOG = get_logger(__name__)

#: Dimensions built straight from a decoded dictionary, with no data dependency.
STATIC_DIMENSIONS = (
    "dim_payment_type", "dim_rate_code", "dim_vendor", "dim_transport_mode",
)
#: Dimensions derived from the silver layer.
DERIVED_DIMENSIONS = ("dim_commodity", "dim_trip_flags")

ALL_DIMENSIONS = ("dim_date", *STATIC_DIMENSIONS, *DERIVED_DIMENSIONS,
                  "dim_geography")

#: Surrogate key column per dimension, used by the uniqueness quality rule.
SURROGATE_KEYS = {
    "dim_date": "date_key",
    "dim_geography": "geo_key",
    "dim_transport_mode": "mode_key",
    "dim_commodity": "commodity_key",
    "dim_payment_type": "payment_key",
    "dim_rate_code": "rate_key",
    "dim_vendor": "vendor_key",
    "dim_trip_flags": "flag_key",
}


def _sql(path) -> str:
    return str(path).replace("\\", "/")


def _ddl(name: str) -> str:
    return read_sql_file(settings.SQL_DDL_DIR / f"{name}.sql")


def build_dim_date(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Generate the conformed calendar plus its Unknown member."""
    con.execute(
        _ddl("dim_date").format(
            start_date=settings.DATE_DIM_START,
            end_date=settings.DATE_DIM_END,
            unknown_key=settings.UNKNOWN_KEY,
        )
    )
    stats = sql_df(
        con,
        """
        SELECT count(*)                                        AS rows,
               count(*) FILTER (WHERE date_key > 0)            AS calendar_days,
               min(full_date)                                  AS first_date,
               max(full_date)                                  AS last_date,
               count(*) FILTER (WHERE is_weekend)              AS weekend_days,
               count(*) FILTER (WHERE is_month_end)            AS month_ends
        FROM dim_date
        """,
    ).iloc[0].to_dict()
    LOG.info("dim_date: %d rows (%s to %s), %d weekend days, %d month ends",
             int(stats["rows"]), stats["first_date"], stats["last_date"],
             int(stats["weekend_days"]), int(stats["month_ends"]))
    return {k: (int(v) if isinstance(v, (int, float)) else str(v))
            for k, v in stats.items()}


def build_static_dimensions(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Create the dictionary-decoded dimensions."""
    out: dict[str, Any] = {}
    for name in STATIC_DIMENSIONS:
        con.execute(_ddl(name).format(unknown_key=settings.UNKNOWN_KEY))
        rows = int(con.execute(f"SELECT count(*) FROM {name}").fetchone()[0])
        out[name] = {"rows": rows}
        LOG.info("%s: %d rows", name, rows)
    return out


def build_dim_commodity(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Conform priced goods at (commodity, unit) grain, flagging fuel."""
    con.execute(
        _ddl("dim_commodity").format(
            silver_wfp=_sql(settings.SILVER_DIR / silver.SILVER_WFP),
            unknown_key=settings.UNKNOWN_KEY,
            nonfood_category=settings.HDX_NONFOOD_CATEGORY,
            fuel_commodities=str(tuple(settings.HDX_FUEL_COMMODITIES)),
        )
    )
    stats = sql_df(
        con,
        """
        SELECT count(*)                              AS rows,
               count(DISTINCT commodity_name)        AS commodities,
               count(DISTINCT category)              AS categories,
               count(DISTINCT unit)                  AS units,
               count(*) FILTER (WHERE is_fuel)       AS fuel_members
        FROM dim_commodity
        """,
    ).iloc[0].to_dict()
    LOG.info("dim_commodity: %d members across %d commodities, %d units, "
             "%d categories (%d fuel)", int(stats["rows"]),
             int(stats["commodities"]), int(stats["units"]),
             int(stats["categories"]), int(stats["fuel_members"]))
    return {k: int(v) for k, v in stats.items()}


def build_dim_trip_flags(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Materialise the junk dimension over observed flag combinations only."""
    con.execute(_ddl("dim_trip_flags").format(unknown_key=settings.UNKNOWN_KEY))
    rows = int(con.execute("SELECT count(*) FROM dim_trip_flags").fetchone()[0])
    observed = rows - 1  # excluding the Unknown member
    theoretical = 3 * 2 * 2 * 2  # store_and_fwd in {Y, N, NULL} x three booleans
    LOG.info(
        "dim_trip_flags: %d observed combination(s) of a theoretical %d "
        "(%.0f%% of the Cartesian product materialised)",
        observed, theoretical, 100.0 * observed / theoretical,
    )
    return {
        "rows": rows,
        "observed_combinations": observed,
        "theoretical_combinations": theoretical,
        "materialised_fraction": round(observed / theoretical, 4),
    }


def build(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, Any]:
    """Build every dimension except ``dim_geography``.

    ``dim_geography`` is built separately because its Type 2 merge must run
    before the facts but has different rerun semantics from a full replace.
    """
    with stage("gold: dimensions", LOG):
        owned = con is None
        con = con or connect(settings.WAREHOUSE_DB)

        diagnostics: dict[str, Any] = {"dim_date": build_dim_date(con)}
        diagnostics.update(build_static_dimensions(con))
        diagnostics["dim_commodity"] = build_dim_commodity(con)
        diagnostics["dim_trip_flags"] = build_dim_trip_flags(con)

        if owned:
            con.close()
        return diagnostics
