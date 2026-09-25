"""``dim_geography``: one conformed, ragged, Type 2 slowly changing dimension.

Why Type 2 here and Type 1 everywhere else
------------------------------------------
Geography is the only dimension in this platform whose attributes can change
while its identity does not. Nigerian states are periodically reassigned in
statistical publications; WFP revises a market's ``admin1``; TLC re-labels a zone
or moves it between service zones. When that happens, trips already loaded were
genuinely made under the old attribution, and overwriting it would silently
rewrite history in every prior analysis.

Every other dimension here is either immutable by construction (``dim_date``),
decoded from a published dictionary that is versioned by the publisher
(``dim_payment_type``, ``dim_rate_code``, ``dim_vendor``), or a junk dimension of
observed flag combinations. Those are Type 1: a correction is a correction.

The merge below is the whole Type 2 mechanism, and
``tests/test_scd2_geography.py`` exercises it by reloading with a changed
``region_group`` and asserting one closed prior row plus one new current row.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import duckdb
import pandas as pd

from config import settings
from config.nigeria_states import GEOPOLITICAL_ZONES
from src.transform import silver
from src.utils.db import connect, read_sql_file, sql_df
from src.utils.logging_setup import get_logger, stage

LOG = get_logger(__name__)

#: Attributes whose change opens a new version. The natural key is excluded by
#: definition: a change to it is a different place, not a new version.
TRACKED_ATTRIBUTES = ("geo_name", "parent_geo_name", "region_group")
NATURAL_KEY = ("geo_code", "country", "admin_level")

US = "United States"
NG = "Nigeria"


def _sql(path) -> str:
    return str(path).replace("\\", "/")


def build_incoming(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Assemble the current-truth geography set from the silver layer.

    Returns one row per natural key with its tracked attributes, which the merge
    then compares against what is already stored.
    """
    zones_path = settings.SILVER_DIR / silver.SILVER_ZONES
    markets_path = settings.SILVER_DIR / silver.SILVER_MARKETS

    nyc_zones = sql_df(
        con,
        f"""
        SELECT
            CAST(location_id AS VARCHAR)                      AS geo_code,
            coalesce(zone, 'Unknown zone ' || location_id)    AS geo_name,
            '{US}'                                            AS country,
            'zone'                                            AS admin_level,
            borough                                           AS parent_geo_name,
            service_zone                                      AS region_group
        FROM read_parquet('{_sql(zones_path)}')
        ORDER BY location_id
        """,
    )

    markets = sql_df(
        con,
        f"""
        SELECT
            CAST(coalesce(CAST(market_id AS VARCHAR), market) AS VARCHAR) AS geo_code,
            market                                            AS geo_name,
            '{NG}'                                            AS country,
            'market'                                          AS admin_level,
            coalesce(state, admin1_raw)                       AS parent_geo_name,
            geopolitical_zone                                 AS region_group
        FROM read_parquet('{_sql(markets_path)}')
        WHERE market IS NOT NULL AND market <> ''
        ORDER BY market
        """,
    )

    # Markets that appear in the price panel but not in the market register are
    # added here rather than lost, so no price row can fail its geography join.
    prices_path = settings.SILVER_DIR / silver.SILVER_WFP
    extra_markets = sql_df(
        con,
        f"""
        SELECT DISTINCT
            CAST(coalesce(CAST(market_id AS VARCHAR), market) AS VARCHAR) AS geo_code,
            market                                            AS geo_name,
            '{NG}'                                            AS country,
            'market'                                          AS admin_level,
            state                                             AS parent_geo_name,
            geopolitical_zone                                 AS region_group
        FROM read_parquet('{_sql(prices_path)}')
        WHERE market IS NOT NULL AND market <> ''
        ORDER BY market
        """,
    )

    states = pd.DataFrame(
        [
            {
                "geo_code": state,
                "geo_name": state,
                "country": NG,
                "admin_level": "state",
                "parent_geo_name": NG,
                "region_group": zone,
            }
            for zone, members in GEOPOLITICAL_ZONES.items()
            for state in members
        ]
    ).sort_values("geo_code")

    city = pd.DataFrame(
        [{
            "geo_code": settings.WEATHER_GEO_CODE,
            "geo_name": "New York City",
            "country": US,
            "admin_level": "city",
            "parent_geo_name": US,
            "region_group": "NYC Metro",
        }]
    )

    incoming = pd.concat(
        [nyc_zones, markets, extra_markets, states, city], ignore_index=True
    )
    incoming = incoming.drop_duplicates(subset=list(NATURAL_KEY), keep="first")
    incoming = incoming.sort_values(
        ["country", "admin_level", "geo_code"], kind="mergesort"
    ).reset_index(drop=True)
    for column in TRACKED_ATTRIBUTES:
        incoming[column] = incoming[column].astype("string")
    return incoming


def _unknown_row() -> dict[str, Any]:
    return {
        "geo_key": settings.UNKNOWN_KEY,
        "geo_code": "UNKNOWN",
        "geo_name": "Unknown",
        "country": "Unknown",
        "admin_level": "unknown",
        "parent_geo_name": None,
        "region_group": None,
        "valid_from": settings.SCD_INITIAL_VALID_FROM,
        "valid_to": settings.SCD_HIGH_DATE,
        "is_current": True,
    }


def merge(
    con: duckdb.DuckDBPyConnection,
    incoming: pd.DataFrame,
    effective_date: dt.date | None = None,
) -> dict[str, Any]:
    """Apply a Type 2 merge of ``incoming`` into ``dim_geography``.

    Behaviour:

    * **Unchanged** natural key with identical tracked attributes -> untouched.
      This is what makes reruns idempotent (constraint 2.7).
    * **Changed** tracked attribute -> the current row is closed
      (``valid_to = effective_date - 1 day``, ``is_current = FALSE``) and a new
      current version is inserted.
    * **New** natural key -> inserted as a current version.
    * **Absent** natural key -> left current. Geography disappearing from one
      vintage of a source is not evidence the place ceased to exist, and closing
      it would orphan facts already loaded against it.

    Args:
        con: Warehouse connection with ``dim_geography`` created.
        incoming: Current-truth rows from :func:`build_incoming`.
        effective_date: Date stamped on changes. Defaults to the configured
            fixed epoch on first load, so a cold run is byte-reproducible, and
            to today's date once the dimension already holds rows.

    Returns:
        Counts of inserted, versioned and unchanged members.
    """
    existing = sql_df(con, "SELECT * FROM dim_geography")
    first_load = existing.empty

    effective = effective_date or (
        settings.SCD_INITIAL_VALID_FROM if first_load else dt.date.today()
    )

    if first_load:
        rows = incoming.copy()
        rows.insert(0, "geo_key", range(1, len(rows) + 1))
        rows["valid_from"] = settings.SCD_INITIAL_VALID_FROM
        rows["valid_to"] = settings.SCD_HIGH_DATE
        rows["is_current"] = True
        # The Unknown member is inserted on its own: concatenating its all-NULL
        # attributes with the real rows would let pandas infer column types from
        # an empty column, which is deprecated and fragile.
        unknown = _unknown_row()
        con.execute(
            "INSERT INTO dim_geography VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [unknown[c] for c in (
                "geo_key", "geo_code", "geo_name", "country", "admin_level",
                "parent_geo_name", "region_group", "valid_from", "valid_to",
                "is_current",
            )],
        )
        con.register("incoming_rows", rows)
        con.execute("INSERT INTO dim_geography SELECT * FROM incoming_rows")
        con.unregister("incoming_rows")
        total_inserted = len(rows) + 1  # + the Unknown member
        LOG.info("dim_geography initial load: %d member(s) incl. the Unknown member",
                 total_inserted)
        return {
            "first_load": True,
            "inserted": int(total_inserted),
            "versioned": 0,
            "unchanged": 0,
            "effective_date": str(settings.SCD_INITIAL_VALID_FROM),
        }

    current = existing[existing["is_current"]].copy()
    merged = incoming.merge(
        current, on=list(NATURAL_KEY), how="left", suffixes=("", "_existing")
    )

    def _differs(row: pd.Series) -> bool:
        for attribute in TRACKED_ATTRIBUTES:
            new = row[attribute]
            old = row.get(f"{attribute}_existing")
            new = None if pd.isna(new) else str(new)
            old = None if pd.isna(old) else str(old)
            if new != old:
                return True
        return False

    is_new = merged["geo_key"].isna()
    changed_mask = (~is_new) & merged.apply(_differs, axis=1)

    new_rows = merged[is_new]
    changed_rows = merged[changed_mask]
    unchanged = int(len(merged) - len(new_rows) - len(changed_rows))

    next_key = int(existing["geo_key"].max()) + 1
    to_insert: list[dict[str, Any]] = []

    for _, row in pd.concat([new_rows, changed_rows]).iterrows():
        to_insert.append({
            "geo_key": next_key,
            "geo_code": row["geo_code"],
            "geo_name": None if pd.isna(row["geo_name"]) else row["geo_name"],
            "country": row["country"],
            "admin_level": row["admin_level"],
            "parent_geo_name": None if pd.isna(row["parent_geo_name"])
            else row["parent_geo_name"],
            "region_group": None if pd.isna(row["region_group"])
            else row["region_group"],
            "valid_from": effective,
            "valid_to": settings.SCD_HIGH_DATE,
            "is_current": True,
        })
        next_key += 1

    if len(changed_rows):
        closing_keys = [int(k) for k in changed_rows["geo_key"].tolist()]
        placeholders = ", ".join("?" for _ in closing_keys)
        con.execute(
            f"""
            UPDATE dim_geography
               SET valid_to = ?, is_current = FALSE
             WHERE geo_key IN ({placeholders})
            """,
            [effective - dt.timedelta(days=1), *closing_keys],
        )

    if to_insert:
        frame = pd.DataFrame(to_insert)
        con.register("scd_inserts", frame)
        con.execute("INSERT INTO dim_geography SELECT * FROM scd_inserts")
        con.unregister("scd_inserts")

    LOG.info(
        "dim_geography merge on %s: %d new, %d versioned, %d unchanged",
        effective, len(new_rows), len(changed_rows), unchanged,
    )
    return {
        "first_load": False,
        "inserted": int(len(new_rows)),
        "versioned": int(len(changed_rows)),
        "unchanged": unchanged,
        "effective_date": str(effective),
    }


def build(con: duckdb.DuckDBPyConnection | None = None,
          effective_date: dt.date | None = None) -> dict[str, Any]:
    """Create the table if needed and merge the current geography truth into it."""
    with stage("gold: dim_geography (Type 2 SCD)", LOG):
        owned = con is None
        con = con or connect(settings.WAREHOUSE_DB)

        ddl = read_sql_file(settings.SQL_DDL_DIR / "dim_geography.sql")
        con.execute(ddl.format(unknown_key=settings.UNKNOWN_KEY))

        incoming = build_incoming(con)
        outcome = merge(con, incoming, effective_date=effective_date)

        profile = sql_df(
            con,
            """
            SELECT country, admin_level,
                   count(*)                                AS versions,
                   count(*) FILTER (WHERE is_current)      AS current_members
            FROM dim_geography
            GROUP BY ALL
            ORDER BY country, admin_level
            """,
        )
        outcome["profile"] = profile.to_dict("records")
        outcome["total_rows"] = int(
            con.execute("SELECT count(*) FROM dim_geography").fetchone()[0]
        )
        outcome["current_rows"] = int(
            con.execute(
                "SELECT count(*) FROM dim_geography WHERE is_current"
            ).fetchone()[0]
        )
        for record in profile.to_dict("records"):
            LOG.info("  %-14s %-7s %4d version(s), %4d current",
                     record["country"], record["admin_level"],
                     record["versions"], record["current_members"])

        if owned:
            con.close()
        return outcome
