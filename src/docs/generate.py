"""Generate ``docs/data_dictionary.md`` from the live warehouse.

The dictionary is generated rather than hand-maintained so it cannot drift from
the schema it documents: column names and types are read from the catalogue on
every run, and only the prose -- grain statements and column notes -- is
curated here. A column that exists but has no note is listed with an empty
description rather than omitted, so the gap is visible.

This file goes into the paper's appendix.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import duckdb
import pandas as pd

from config import settings
from src.model import catalog
from src.utils.db import connect, sql_df, table_exists
from src.utils.io_utils import human_bytes, write_text
from src.utils.logging_setup import get_logger, stage

LOG = get_logger(__name__)

DATA_DICTIONARY = "data_dictionary.md"

#: Grain statement and provenance per relation. Grain is the single most
#: important fact about a fact table and is stated first, everywhere.
TABLE_NOTES: dict[str, dict[str, str]] = {
    "dim_date": {
        "kind": "Dimension (conformed, Type 1)",
        "grain": "One row per calendar day, 2016-01-01 to 2026-12-31, plus one "
                 "Unknown member at date_key = -1.",
        "source": "Generated. No input data; cannot carry a fabricated value.",
    },
    "dim_geography": {
        "kind": "Dimension (conformed, ragged hierarchy, **Type 2 SCD**)",
        "grain": "One row per (geo_code, country, admin_level) VERSION. A "
                 "natural key may hold several versions, of which exactly one "
                 "is current.",
        "source": "NYC taxi zone lookup (Source B), WFP market register and "
                  "price panel (Source C), and config/nigeria_states.py for "
                  "the 36 states plus FCT.",
    },
    "dim_transport_mode": {
        "kind": "Dimension (Type 1)",
        "grain": "One row per transport mode. Currently one real member.",
        "source": "Defined by the platform, not sourced.",
    },
    "dim_commodity": {
        "kind": "Dimension (Type 1)",
        "grain": "One row per (commodity, unit). Unit is part of the identity: "
                 "1 KG of rice and 100 KG of rice are different products.",
        "source": "Derived from silver_wfp_prices (Source C).",
    },
    "dim_payment_type": {
        "kind": "Dimension (Type 1, decoded dictionary)",
        "grain": "One row per TLC payment_type code.",
        "source": "NYC TLC Yellow Trips data dictionary.",
    },
    "dim_rate_code": {
        "kind": "Dimension (Type 1, decoded dictionary)",
        "grain": "One row per TLC RatecodeID.",
        "source": "NYC TLC Yellow Trips data dictionary.",
    },
    "dim_vendor": {
        "kind": "Dimension (Type 1, decoded dictionary)",
        "grain": "One row per licensed technology provider code.",
        "source": "NYC TLC Yellow Trips data dictionary.",
    },
    "dim_trip_flags": {
        "kind": "Dimension (Type 1, **junk dimension**)",
        "grain": "One row per OBSERVED combination of four low-cardinality "
                 "trip attributes, plus an Unknown member.",
        "source": "Derived from silver_nyc_trip (Source A).",
    },
    "fact_trip": {
        "kind": "Fact (atomic, transactional)",
        "grain": "**ONE ROW PER COMPLETED TAXI TRIP.** Parquet, Hive-partitioned "
                 "year=YYYY/month=M.",
        "source": "NYC TLC Yellow Taxi trip records 2024 (Source A), after "
                  "structural rejection at the silver boundary.",
    },
    "fact_market_price_monthly": {
        "kind": "Fact (periodic snapshot)",
        "grain": "**ONE ROW PER (market, commodity, price type, month).**",
        "source": "WFP Nigeria market food prices via HDX (Source C).",
    },
    "fact_fuel_price_monthly": {
        "kind": "Fact (periodic snapshot)",
        "grain": "**ONE ROW PER (state, month)** for Premium Motor Spirit.",
        "source": "NBS PMS Price Watch (Source D).",
    },
    "fact_trip_daily_agg": {
        "kind": "Fact (aggregate, derived)",
        "grain": "**ONE ROW PER (date, pickup zone, mode).** Redundant by "
                 "design; exists so benchmark B3 can measure what aggregate "
                 "navigation buys.",
        "source": "Derived entirely from fact_trip.",
    },
    "fact_weather_daily": {
        "kind": "Fact (periodic snapshot, OPTIONAL)",
        "grain": "**ONE ROW PER (date, geography).** In practice one row per "
                 "date: Open-Meteo publishes a single point series.",
        "source": "Open-Meteo archive API (Source E). Non-blocking; the core "
                  "pipeline succeeds without it.",
    },
}

#: Curated column notes. Anything not listed renders with an empty description.
COLUMN_NOTES: dict[str, str] = {
    # keys
    "date_key": "Surrogate key, INTEGER YYYYMMDD. -1 = Unknown member.",
    "geo_key": "Surrogate key into dim_geography. -1 = Unknown member.",
    "pickup_geo_key": "dim_geography surrogate for the PICKUP zone.",
    "dropoff_geo_key": "dim_geography surrogate for the DROPOFF zone.",
    "pickup_date_key": "dim_date surrogate for the pickup date. -1 where the "
                       "pickup falls outside the generated calendar, which the "
                       "official 2024 files genuinely contain.",
    "dropoff_date_key": "dim_date surrogate for the dropoff date.",
    "commodity_key": "dim_commodity surrogate.",
    "mode_key": "dim_transport_mode surrogate.",
    "vendor_key": "dim_vendor surrogate.",
    "payment_key": "dim_payment_type surrogate.",
    "rate_key": "dim_rate_code surrogate.",
    "flag_key": "dim_trip_flags surrogate (junk dimension).",
    "month_key": "INTEGER YYYYMM01, the first day of the month.",
    "trip_id": "Degenerate dimension. Deterministic MD5 of the row's business "
               "content; stable across reruns, never random. TLC publishes no "
               "trip identifier.",
    # dim_date
    "full_date": "Calendar date. NULL on the Unknown member.",
    "day_of_week": "ISO numbering: 1 = Monday .. 7 = Sunday.",
    "year_month": "'YYYY-MM' string, for grouping and labelling.",
    "is_month_end": "True on the last calendar day of the month.",
    # dim_geography
    "geo_code": "Natural key within (country, admin_level): a TLC LocationID, a "
                "WFP market id, or a canonical Nigerian state name.",
    "admin_level": "zone | market | state | city. 'city' is a documented "
                   "extension carrying the single New York City member that "
                   "fact_weather_daily references.",
    "parent_geo_name": "Parent in the ragged hierarchy: borough for a zone, "
                       "state for a market, 'Nigeria' for a state. Type 2 tracked.",
    "region_group": "TLC service zone for a zone; geopolitical zone for a "
                    "Nigerian state or market. Type 2 tracked.",
    "valid_from": "Start of this version's validity (Type 2).",
    "valid_to": "End of validity; 9999-12-31 while current (Type 2).",
    "is_current": "True on exactly one version per natural key (Type 2).",
    # measures, trips
    "pickup_hour": "Hour of pickup, 0-23, local New York time.",
    "passenger_count": "As transmitted by the vendor. NULL for ~9.8% of 2024 "
                       "trips; never defaulted to zero. See analysis A11.",
    "trip_distance_miles": "Metered distance in miles.",
    "trip_duration_seconds": "Dropoff minus pickup, in seconds. Strictly "
                             "positive and at most 86,400 by construction.",
    "avg_speed_mph": "Derived: distance / duration. A congestion PROXY only.",
    "fare_amount": "Metered fare, USD. Non-negative by construction.",
    "extra": "Miscellaneous extras and surcharges, USD.",
    "mta_tax": "MTA tax, USD.",
    "tip_amount": "Tip, USD. CARD TIPS ONLY -- the meter does not record cash "
                  "tips. A zero on a cash trip is evidence about the recording "
                  "system, not about the passenger.",
    "tolls_amount": "Tolls, USD.",
    "improvement_surcharge": "Improvement surcharge, USD.",
    "congestion_surcharge": "Congestion surcharge, USD. NULL on the same rows "
                            "that lack passenger_count (see A11).",
    "airport_fee": "Airport pickup fee, USD.",
    "total_amount": "Total charged to the passenger, USD. Not operator net "
                    "revenue.",
    "year": "Hive partition column: year of pickup.",
    "month": "Hive partition column: month of pickup, 1-12.",
    # prices
    "price_ngn": "Price in Nigerian naira, as published.",
    "price_usd": "WFP's own USD conversion. Carried but never used to build an "
                 "index, since the exchange rate would then drive the series.",
    "unit": "Published unit of sale. Part of dim_commodity's grain.",
    "price_type": "Retail or Wholesale, as published by WFP.",
    "observation_count": "Source rows that collapsed into this cell. Normally "
                         "1; a rise would reveal a change in publication "
                         "frequency rather than being silently averaged.",
    "fuel_type": "'PMS' (Premium Motor Spirit, petrol).",
    "mom_pct_change": "Month-over-month % change, COMPUTED from the loaded "
                      "series, never read from the published sheet. NULL where "
                      "the prior month is absent; never interpolated.",
    "yoy_pct_change": "Year-over-year % change, computed on the same basis.",
    "source_file": "Originating file, for provenance.",
    # aggregate
    "trip_count": "Additive count of trips.",
    "total_distance_miles": "Additive.",
    "total_revenue": "Additive.",
    "avg_fare": "NOT re-aggregable: averaging across zones weights every zone "
                "equally. Roll up with total_revenue / trip_count instead.",
    "median_fare": "Computed exactly, not approximated. Not re-aggregable.",
    "avg_duration_seconds": "Not re-aggregable.",
    "is_fuel": "True for the two WFP fuel commodities; excluded from every food "
               "index.",
    "is_tip_observable": "True only for credit card, the one payment type whose "
                         "tips the meter records.",
    # weather
    "temp_max_c": "Daily maximum air temperature, degrees Celsius.",
    "temp_min_c": "Daily minimum air temperature, degrees Celsius.",
    "precipitation_mm": "Daily precipitation total, millimetres.",
    "snowfall_mm": "Daily snowfall total, millimetres.",
    "wind_speed_max_kmh": "Daily maximum 10 m wind speed, km/h.",
}


def _columns(con: duckdb.DuckDBPyConnection, relation: str) -> pd.DataFrame:
    frame = sql_df(con, f"DESCRIBE SELECT * FROM {relation}")
    frame = frame.rename(columns={"column_name": "column", "column_type": "type"})
    frame["description"] = frame["column"].map(
        lambda c: COLUMN_NOTES.get(c, "")
    )
    return frame[["column", "type", "description"]]


def build_data_dictionary(con: duckdb.DuckDBPyConnection) -> str:
    """Render the full dictionary for every gold relation."""
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    inventory = {row["relation"]: row for row in catalog.inventory(con)}

    lines = [
        "# Data Dictionary",
        "",
        "CSC 796 Advanced Data Engineering -- *Engineering a Multi-Grain "
        "Analytical Data Platform for the Economics of Movement*.",
        "",
        f"Generated from the live warehouse catalogue on {generated}. Column "
        "names and types are read from the schema on every run, so this "
        "document cannot drift from the tables it describes.",
        "",
        "## Conventions",
        "",
        "- Every dimension carries an **Unknown member** at surrogate key "
        "`-1`. Facts route unmatched codes there rather than losing rows to an "
        "inner join, which is what makes referential integrity total.",
        "- Surrogate keys are integers. Natural keys are retained as "
        "attributes.",
        "- `dim_geography` is **Type 2** slowly changing; every other dimension "
        "is Type 1. The reasoning is in `docs/scd_strategy.md`.",
        "- Money is USD for New York, NGN for Nigeria. No cross-currency "
        "measure is computed anywhere.",
        "",
        "## Contents",
        "",
    ]
    relations = [r for r in catalog.GOLD_RELATIONS if table_exists(con, r)]
    lines += [f"- [`{r}`](#{r.replace('_', '-')})" for r in relations]
    lines.append("")

    for relation in relations:
        notes = TABLE_NOTES.get(relation, {})
        stats = inventory.get(relation, {})
        lines += [
            f"## {relation}",
            "",
            f"**{notes.get('kind', 'Relation')}**",
            "",
            f"- **Grain:** {notes.get('grain', 'Not stated.')}",
            f"- **Source:** {notes.get('source', 'Not stated.')}",
            f"- **Rows:** {stats.get('rows', 0):,}",
            f"- **On disk:** {stats.get('bytes_human', 'n/a')}"
            + (f" ({stats['bytes_per_row']} bytes/row)"
               if stats.get("bytes_per_row") else ""),
            "",
            "| Column | Type | Description |",
            "| --- | --- | --- |",
        ]
        for row in _columns(con, relation).itertuples(index=False):
            lines.append(f"| `{row.column}` | {row.type} | {row.description} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def run(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, Any]:
    """Write the generated documentation."""
    with stage("docs: generated data dictionary", LOG):
        owned = con is None
        con = con or connect(settings.WAREHOUSE_DB)
        catalog.register_views(con)

        text = build_data_dictionary(con)
        path = write_text(settings.DOCS_DIR / DATA_DICTIONARY, text)

        documented = sum(1 for line in text.splitlines() if line.startswith("## "))
        LOG.info("wrote %s (%d relations, %s)", path.name, documented - 1,
                 human_bytes(path.stat().st_size))

        if owned:
            con.close()
        return {"data_dictionary": path.name, "relations_documented": documented - 1}
