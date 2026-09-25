"""Silver layer: cleansed, conformed, quality-gated.

Silver is where source vocabularies are reconciled onto the platform's own --
where ``FCT Abuja`` and ``Abuja`` become ``FCT``, where a price string becomes a
number, and where every rejected row acquires exactly one reason.

As in bronze, the layer is physical for the Nigerian sources and logical for the
41 M row NYC corpus. ``silver_nyc_trip`` is a view whose reject predicate is the
single authority consumed by both the gold fact build and the reconciliation
report, so the two cannot drift apart.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from config import settings
from config.nigeria_states import (
    CANONICAL_STATES, GEOPOLITICAL_ZONES, is_non_state_label, normalise_state,
)
from src.transform import bronze
from src.utils.db import connect, read_sql_file, sql_df
from src.utils.io_utils import write_text
from src.utils.logging_setup import get_logger, stage
from src.utils.manifest import get_manifest

LOG = get_logger(__name__)

SILVER_ZONES = "silver_nyc_zones.parquet"
SILVER_WFP = "silver_wfp_prices.parquet"
SILVER_WFP_REJECTS = "silver_wfp_prices_rejects.parquet"
SILVER_MARKETS = "silver_wfp_markets.parquet"
SILVER_NBS = "silver_nbs_pms.parquet"
SILVER_NBS_REJECTS = "silver_nbs_pms_rejects.parquet"
SILVER_WEATHER = "silver_weather_daily.parquet"
SILVER_TRIP_REJECTS = "silver_nyc_trip_reject_summary.parquet"
SILVER_TRIP_REJECT_SAMPLE = "silver_nyc_trip_reject_sample.parquet"

#: Rejected trips are counted exhaustively but only this many are materialised
#: for inspection, so a pathological month cannot blow the disk budget.
REJECT_SAMPLE_ROWS = 10_000


def _sql(path: Path) -> str:
    return str(path).replace("\\", "/")


def _trip_relation() -> str:
    pattern = _sql(settings.RAW_NYC / "yellow_tripdata_*.parquet")
    return f"read_parquet('{pattern}', union_by_name=true, filename=true)"


def _airport_fee_column(con: duckdb.DuckDBPyConnection) -> str:
    """Resolve the airport-fee column, whose case has drifted across vintages.

    2024 files ship ``Airport_fee``; other years ship ``airport_fee``. DuckDB
    quotes identifiers case-sensitively, so the name is resolved from the actual
    Parquet schema instead of being assumed.
    """
    sample = sorted(settings.RAW_NYC.glob("yellow_tripdata_*.parquet"))
    if not sample:
        return '"Airport_fee"'
    columns = con.execute(
        f"SELECT name FROM parquet_schema('{_sql(sample[0])}')"
    ).df()["name"].tolist()
    for candidate in ("Airport_fee", "airport_fee", "AIRPORT_FEE"):
        if candidate in columns:
            return f'"{candidate}"'
    LOG.warning("no airport fee column found in %s; substituting NULL", sample[0].name)
    return "NULL"


def create_trip_view(con: duckdb.DuckDBPyConnection) -> str:
    """Render and register ``silver_nyc_trip``. Returns the rendered SQL."""
    template = read_sql_file(settings.SQL_DDL_DIR / "silver_nyc_trip.sql")
    airport_ids = "(" + ", ".join(str(z) for z in settings.NYC_AIRPORT_ZONE_IDS) + ")"
    rendered = template.format(
        trip_relation=_trip_relation(),
        airport_fee_column=_airport_fee_column(con),
        window_start=settings.NYC_WINDOW_START,
        window_end=settings.NYC_WINDOW_END,
        max_duration_seconds=settings.TRIP_MAX_DURATION_SECONDS,
        max_distance_miles=settings.TRIP_MAX_DISTANCE_MILES,
        max_passengers=settings.TRIP_MAX_PASSENGERS,
        airport_zone_ids=airport_ids,
    )
    con.execute(rendered)
    return rendered


def _silver_zones(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Conform the taxi zone lookup; retain the two non-geographic ids."""
    source = settings.BRONZE_DIR / bronze.BRONZE_ZONES
    out = settings.SILVER_DIR / SILVER_ZONES
    non_geo = ", ".join(str(z) for z in settings.NYC_NONGEOGRAPHIC_ZONE_IDS)
    con.execute(
        f"""
        COPY (
            SELECT
                location_id,
                nullif(borough, '')                       AS borough,
                nullif(zone, '')                          AS zone,
                nullif(service_zone, '')                  AS service_zone,
                location_id IN ({non_geo})                AS is_non_geographic,
                source_file
            FROM read_parquet('{_sql(source)}')
            WHERE location_id IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY location_id
                                       ORDER BY zone) = 1
        ) TO '{_sql(out)}' (FORMAT PARQUET)
        """
    )
    stats = sql_df(
        con,
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT borough) AS boroughs,
               count(*) FILTER (WHERE is_non_geographic) AS non_geographic
        FROM read_parquet('{_sql(out)}')
        """,
    ).iloc[0].to_dict()
    LOG.info("silver zones: %d rows, %d boroughs, %d non-geographic",
             int(stats["rows"]), int(stats["boroughs"]), int(stats["non_geographic"]))
    return {k: int(v) for k, v in stats.items()}


def _state_lookup_relation(con: duckdb.DuckDBPyConnection) -> None:
    """Register the canonical state/zone lookup as a temp table for SQL joins."""
    rows = [
        {"state": state, "geopolitical_zone": zone}
        for zone, states in GEOPOLITICAL_ZONES.items()
        for state in states
    ]
    frame = pd.DataFrame(rows).sort_values("state").reset_index(drop=True)
    con.register("canonical_states_df", frame)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE canonical_states AS "
        "SELECT * FROM canonical_states_df"
    )
    con.unregister("canonical_states_df")


def _normalise_admin1_map(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Build a raw-``admin1`` to canonical-state map from observed values.

    Normalisation runs in Python (the authority is ``config.nigeria_states``) and
    the result is registered as a table, so the SQL stays declarative and the
    mapping is auditable as data rather than buried in a CASE expression.
    """
    source = settings.BRONZE_DIR / bronze.BRONZE_WFP_PRICES
    observed = sql_df(
        con,
        f"SELECT DISTINCT admin1 FROM read_parquet('{_sql(source)}') "
        f"WHERE admin1 IS NOT NULL",
    )
    observed["state"] = observed["admin1"].map(normalise_state)
    observed["is_aggregate_label"] = observed["admin1"].map(is_non_state_label)
    con.register("admin1_map_df", observed)
    con.execute(
        "CREATE OR REPLACE TEMP TABLE admin1_map AS SELECT * FROM admin1_map_df"
    )
    con.unregister("admin1_map_df")

    unresolved = observed[observed["state"].isna()]
    if len(unresolved):
        LOG.warning(
            "WFP admin1 values that did not resolve to a canonical state (%d): %s",
            len(unresolved), unresolved["admin1"].tolist(),
        )
    return observed


def _silver_wfp(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Cleanse the WFP price panel and quarantine unusable rows with reasons."""
    source = settings.BRONZE_DIR / bronze.BRONZE_WFP_PRICES
    markets_src = settings.BRONZE_DIR / bronze.BRONZE_WFP_MARKETS
    out = settings.SILVER_DIR / SILVER_WFP
    rejects_out = settings.SILVER_DIR / SILVER_WFP_REJECTS
    markets_out = settings.SILVER_DIR / SILVER_MARKETS

    admin1_map = _normalise_admin1_map(con)
    _state_lookup_relation(con)

    judged = f"""
        SELECT
            p.observation_date,
            p.year_month,
            CAST(strftime(date_trunc('month', p.observation_date), '%Y%m01')
                 AS INTEGER)                                   AS month_key,
            p.admin1                                           AS admin1_raw,
            m.state                                            AS state,
            cs.geopolitical_zone                               AS geopolitical_zone,
            p.admin2,
            p.market,
            p.market_id,
            p.latitude,
            p.longitude,
            p.category,
            p.commodity,
            p.commodity_id,
            p.unit,
            p.priceflag,
            p.pricetype                                        AS price_type,
            p.currency,
            p.price                                            AS price_ngn,
            p.usdprice                                         AS price_usd,
            (p.category = '{settings.HDX_NONFOOD_CATEGORY}')   AS is_non_food,
            (p.commodity IN {tuple(settings.HDX_FUEL_COMMODITIES)}) AS is_fuel,
            p.source_file,
            CASE
                WHEN p.observation_date IS NULL      THEN 'unparsable_date'
                WHEN p.market IS NULL OR p.market='' THEN 'missing_market'
                WHEN p.commodity IS NULL OR p.commodity='' THEN 'missing_commodity'
                WHEN p.price IS NULL                 THEN 'null_price'
                WHEN p.price <= 0                    THEN 'non_positive_price'
                WHEN m.state IS NULL                 THEN 'unresolvable_admin1'
                ELSE NULL
            END                                                AS reject_reason
        FROM read_parquet('{_sql(source)}') p
        LEFT JOIN admin1_map m ON m.admin1 = p.admin1
        LEFT JOIN canonical_states cs ON cs.state = m.state
    """

    con.execute(
        f"COPY (SELECT * EXCLUDE (reject_reason) FROM ({judged}) "
        f"WHERE reject_reason IS NULL) TO '{_sql(out)}' (FORMAT PARQUET)"
    )
    con.execute(
        f"COPY (SELECT * FROM ({judged}) WHERE reject_reason IS NOT NULL) "
        f"TO '{_sql(rejects_out)}' (FORMAT PARQUET)"
    )
    con.execute(
        f"""
        COPY (
            SELECT mk.market, mk.market_id, mk.admin1 AS admin1_raw,
                   am.state, cs.geopolitical_zone, mk.admin2,
                   mk.latitude, mk.longitude, mk.source_file
            FROM read_parquet('{_sql(markets_src)}') mk
            LEFT JOIN admin1_map am ON am.admin1 = mk.admin1
            LEFT JOIN canonical_states cs ON cs.state = am.state
            QUALIFY row_number() OVER (PARTITION BY mk.market, mk.admin1
                                       ORDER BY mk.market_id) = 1
        ) TO '{_sql(markets_out)}' (FORMAT PARQUET)
        """
    )

    loaded = sql_df(
        con,
        f"""
        SELECT count(*) AS rows,
               count(DISTINCT state)      AS states,
               count(DISTINCT market)     AS markets,
               count(DISTINCT commodity)  AS commodities,
               count(DISTINCT year_month) AS months,
               min(year_month)            AS month_min,
               max(year_month)            AS month_max,
               count(*) FILTER (WHERE is_fuel) AS fuel_rows
        FROM read_parquet('{_sql(out)}')
        """,
    ).iloc[0].to_dict()
    by_reason = sql_df(
        con,
        f"SELECT reject_reason, count(*) AS rows "
        f"FROM read_parquet('{_sql(rejects_out)}') GROUP BY 1 ORDER BY 2 DESC",
    )

    result: dict[str, Any] = {
        "rows_loaded": int(loaded["rows"]),
        "states": int(loaded["states"]),
        "markets": int(loaded["markets"]),
        "commodities": int(loaded["commodities"]),
        "months": int(loaded["months"]),
        "month_min": str(loaded["month_min"]),
        "month_max": str(loaded["month_max"]),
        "fuel_rows": int(loaded["fuel_rows"]),
        "rejects_by_reason": dict(zip(by_reason["reject_reason"],
                                      by_reason["rows"].astype(int))),
        "rows_rejected": int(by_reason["rows"].sum()) if len(by_reason) else 0,
        "unresolved_admin1_values": admin1_map[admin1_map["state"].isna()][
            "admin1"].tolist(),
    }
    LOG.info(
        "silver WFP: %s rows loaded, %s rejected %s; %d states, %d markets, "
        "%d commodities, %d months (%s to %s)",
        f"{result['rows_loaded']:,}", f"{result['rows_rejected']:,}",
        result["rejects_by_reason"], result["states"], result["markets"],
        result["commodities"], result["months"], result["month_min"],
        result["month_max"],
    )
    return result


def _silver_nbs(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Cleanse the NBS fuel panel and compute month-over-month / year-over-year.

    The change columns are computed here rather than read from the published
    sheets (section 4), so that every derived figure in the paper is traceable to
    the price observations this pipeline actually loaded.
    """
    source = settings.BRONZE_DIR / bronze.BRONZE_NBS_PMS
    out = settings.SILVER_DIR / SILVER_NBS
    rejects_out = settings.SILVER_DIR / SILVER_NBS_REJECTS
    _state_lookup_relation(con)

    judged = f"""
        SELECT
            n.year_month,
            CAST(replace(n.year_month, '-', '') || '01' AS INTEGER) AS month_key,
            n.state,
            n.state_raw,
            cs.geopolitical_zone,
            n.price_ngn,
            n.source_file,
            n.sheet_name,
            n.is_current_month,
            CASE
                WHEN n.state IS NULL OR n.state = ''      THEN 'unresolvable_state'
                WHEN cs.geopolitical_zone IS NULL         THEN 'state_not_in_zone_map'
                WHEN n.price_ngn IS NULL                  THEN 'null_price'
                WHEN n.price_ngn < {settings.NBS_PRICE_MIN_NGN}
                                                          THEN 'price_below_floor'
                WHEN n.price_ngn > {settings.NBS_PRICE_MAX_NGN}
                                                          THEN 'price_above_ceiling'
                ELSE NULL
            END                                            AS reject_reason
        FROM read_parquet('{_sql(source)}') n
        LEFT JOIN canonical_states cs ON cs.state = n.state
    """

    con.execute(
        f"""
        COPY (
            WITH clean AS (
                SELECT * EXCLUDE (reject_reason) FROM ({judged})
                WHERE reject_reason IS NULL
            )
            SELECT
                clean.*,
                lag(price_ngn) OVER w_prev                     AS price_prev_month,
                100.0 * (price_ngn - lag(price_ngn) OVER w_prev)
                      / nullif(lag(price_ngn) OVER w_prev, 0)  AS mom_pct_change,
                lag(price_ngn, 12) OVER w_prev                 AS price_prev_year,
                100.0 * (price_ngn - lag(price_ngn, 12) OVER w_prev)
                      / nullif(lag(price_ngn, 12) OVER w_prev, 0) AS yoy_pct_change
            FROM clean
            WINDOW w_prev AS (PARTITION BY state ORDER BY year_month)
        ) TO '{_sql(out)}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"COPY (SELECT * FROM ({judged}) WHERE reject_reason IS NOT NULL) "
        f"TO '{_sql(rejects_out)}' (FORMAT PARQUET)"
    )

    loaded = sql_df(
        con,
        f"""
        SELECT count(*) AS rows, count(DISTINCT year_month) AS months,
               count(DISTINCT state) AS states,
               min(year_month) AS month_min, max(year_month) AS month_max
        FROM read_parquet('{_sql(out)}')
        """,
    ).iloc[0].to_dict()
    by_reason = sql_df(
        con,
        f"SELECT reject_reason, count(*) AS rows "
        f"FROM read_parquet('{_sql(rejects_out)}') GROUP BY 1 ORDER BY 2 DESC",
    )
    national = sql_df(
        con,
        f"SELECT year_month, round(avg(price_ngn), 2) AS national_mean, "
        f"count(*) AS reporting_states "
        f"FROM read_parquet('{_sql(out)}') GROUP BY 1 ORDER BY 1",
    )

    result: dict[str, Any] = {
        "rows_loaded": int(loaded["rows"]),
        "months": int(loaded["months"]),
        "states": int(loaded["states"]),
        "month_min": str(loaded["month_min"]),
        "month_max": str(loaded["month_max"]),
        "rejects_by_reason": dict(zip(by_reason["reject_reason"],
                                      by_reason["rows"].astype(int))),
        "rows_rejected": int(by_reason["rows"].sum()) if len(by_reason) else 0,
        "national_mean_by_month": dict(zip(national["year_month"],
                                           national["national_mean"])),
        "reporting_states_by_month": dict(zip(national["year_month"],
                                              national["reporting_states"].astype(int))),
        "canonical_state_total": len(CANONICAL_STATES),
    }
    LOG.info("silver NBS: %d rows, %d months (%s to %s), %d states; rejected %d %s",
             result["rows_loaded"], result["months"], result["month_min"],
             result["month_max"], result["states"], result["rows_rejected"],
             result["rejects_by_reason"])
    return result


def _silver_weather(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Type the weather panel, if Source E landed."""
    source = settings.BRONZE_DIR / bronze.BRONZE_WEATHER
    if not source.exists():
        LOG.warning("silver weather: bronze artefact absent; skipping (optional)")
        return {"status": "skipped", "reason": "bronze artefact absent"}
    out = settings.SILVER_DIR / SILVER_WEATHER
    con.execute(
        f"""
        COPY (
            SELECT
                CAST(observation_date AS DATE)                      AS observation_date,
                CAST(strftime(CAST(observation_date AS DATE), '%Y%m%d')
                     AS INTEGER)                                    AS date_key,
                CAST(temperature_2m_max AS DOUBLE)                  AS temp_max_c,
                CAST(temperature_2m_min AS DOUBLE)                  AS temp_min_c,
                CAST(precipitation_sum  AS DOUBLE)                  AS precipitation_mm,
                CAST(snowfall_sum       AS DOUBLE)                  AS snowfall_mm,
                CAST(wind_speed_10m_max AS DOUBLE)                  AS wind_speed_max_kmh,
                source_file
            FROM read_parquet('{_sql(source)}')
            WHERE observation_date IS NOT NULL
        ) TO '{_sql(out)}' (FORMAT PARQUET)
        """
    )
    rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{_sql(out)}')"
    ).fetchone()[0]
    LOG.info("silver weather: %d rows", rows)
    return {"status": "loaded", "rows": int(rows)}


def _silver_trip_rejects(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Count every rejected trip by reason and materialise a bounded sample.

    Counting is exhaustive; materialisation is capped. Publishing millions of
    reject rows would breach the disk budget without telling an examiner
    anything the per-reason counts do not.
    """
    summary = sql_df(
        con,
        f"""
        SELECT
            coalesce(reject_reason, 'accepted')                AS reject_reason,
            count(*)                                           AS rows,
            count(*) FILTER (WHERE is_outside_declared_window) AS outside_window_rows,
            round(min(trip_distance_miles), 3)                 AS min_distance,
            round(max(trip_distance_miles), 3)                 AS max_distance,
            round(min(total_amount), 2)                        AS min_total,
            round(max(total_amount), 2)                        AS max_total
        FROM silver_nyc_trip
        GROUP BY 1
        ORDER BY rows DESC
        """,
    )
    out = settings.SILVER_DIR / SILVER_TRIP_REJECTS
    summary.to_parquet(out, index=False)

    con.execute(
        f"""
        COPY (
            SELECT * FROM silver_nyc_trip
            WHERE reject_reason IS NOT NULL
            LIMIT {REJECT_SAMPLE_ROWS}
        ) TO '{_sql(settings.SILVER_DIR / SILVER_TRIP_REJECT_SAMPLE)}'
          (FORMAT PARQUET)
        """
    )

    accepted = int(summary.loc[summary["reject_reason"] == "accepted", "rows"].sum())
    rejected = int(summary.loc[summary["reject_reason"] != "accepted", "rows"].sum())
    by_reason = dict(
        zip(summary.loc[summary["reject_reason"] != "accepted", "reject_reason"],
            summary.loc[summary["reject_reason"] != "accepted", "rows"].astype(int))
    )
    outside_window = int(summary["outside_window_rows"].sum())

    LOG.info("silver trips: %s accepted, %s rejected %s", f"{accepted:,}",
             f"{rejected:,}", by_reason)
    LOG.info("silver trips: %s row(s) carry a pickup outside the declared window "
             "and are RETAINED and flagged, not rejected", f"{outside_window:,}")
    return {
        "rows_accepted": accepted,
        "rows_rejected": rejected,
        "raw_rows": accepted + rejected,
        "rejects_by_reason": by_reason,
        "outside_declared_window_rows": outside_window,
        "reject_sample_rows": min(rejected, REJECT_SAMPLE_ROWS),
    }


def build(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, Any]:
    """Build the silver layer and return its diagnostics."""
    with stage("silver: cleanse and conform", LOG):
        owned = con is None
        con = con or connect(settings.WAREHOUSE_DB)
        settings.SILVER_DIR.mkdir(parents=True, exist_ok=True)
        manifest = get_manifest()

        rendered = create_trip_view(con)
        write_text(settings.SILVER_DIR / "silver_nyc_trip.rendered.sql", rendered)

        diagnostics: dict[str, Any] = {
            "nyc_zones": _silver_zones(con),
            "wfp": _silver_wfp(con),
            "nbs": _silver_nbs(con),
            "weather": _silver_weather(con),
            "nyc_trips": _silver_trip_rejects(con),
        }

        manifest.data.setdefault("silver", {}).update(diagnostics)
        manifest.save()
        if owned:
            con.close()
        return diagnostics
