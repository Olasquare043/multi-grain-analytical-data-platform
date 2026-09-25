"""Bronze layer: raw artefacts registered, typed and provenance-stamped.

Bronze keeps source shape. No renaming, no business logic, no filtering -- only
the minimum needed to make each artefact queryable and attributable to the file
it came from.

A deliberate asymmetry, and a finding in its own right
-----------------------------------------------------
The Nigerian sources are materialised into Parquet in full, because they are
small enough that a physical copy costs a few tens of megabytes and buys a
genuinely immutable landing record.

The 660 MB NYC Parquet corpus is **registered rather than copied**. Copying 41 M
rows into bronze, then again into silver, would consume roughly 2 GB of the 3 GB
disk budget (constraint 2.4) to produce two byte-identical restatements of an
already-columnar, already-immutable input. Bronze for Source A is therefore a
provenance table plus a view; the raw Parquet *is* the bronze artefact.

That the same architecture must be physical at one grain and logical at another
is precisely the engineering cost the paper sets out to measure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from config import settings
from src.utils.db import connect, sql_df
from src.utils.io_utils import sha256_file
from src.utils.logging_setup import get_logger, stage
from src.utils.manifest import get_manifest

LOG = get_logger(__name__)

NYC_TRIP_GLOB = "data/raw/nyc/yellow_tripdata_*.parquet"

# Bronze artefacts
BRONZE_TRIP_FILES = "bronze_nyc_trip_files.parquet"
BRONZE_ZONES = "bronze_nyc_zones.parquet"
BRONZE_WFP_PRICES = "bronze_wfp_prices.parquet"
BRONZE_WFP_MARKETS = "bronze_wfp_markets.parquet"
BRONZE_NBS_PMS = "bronze_nbs_pms.parquet"
BRONZE_WEATHER = "bronze_weather_daily.parquet"


def _require(path: Path, source: str, remedy: str) -> Path:
    """Fail with an actionable message rather than a bare stack trace."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} is missing. {source} has not been extracted. {remedy}"
        )
    return Path(path)


def _trip_glob(con: duckdb.DuckDBPyConnection) -> str:
    """Absolute glob for the raw trip Parquet, quoted for SQL."""
    pattern = str(settings.RAW_NYC / "yellow_tripdata_*.parquet").replace("\\", "/")
    files = sorted(settings.RAW_NYC.glob("yellow_tripdata_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"{settings.RAW_NYC} contains no yellow_tripdata_*.parquet. "
            "Run 'make extract-nyc' or check network access to "
            "d37ci6vzurychx.cloudfront.net."
        )
    return f"read_parquet('{pattern}', union_by_name=true, filename=true)"


def build_nyc_trip_registry(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Profile every raw trip file: rows, timestamp extent and out-of-window rows.

    This is where the confirmed January 2024 defect is quantified. The
    specification records a minimum ``tpep_pickup_datetime`` of
    ``2002-12-31 22:59:39``; the figure reported here is measured, not restated.
    """
    relation = _trip_glob(con)
    frame = sql_df(
        con,
        f"""
        SELECT
            regexp_extract(filename, '([0-9]{{4}}-[0-9]{{2}})\\.parquet$', 1)
                                                              AS file_month,
            regexp_replace(filename, '^.*/', '')               AS source_file,
            count(*)                                           AS raw_rows,
            min(tpep_pickup_datetime)                          AS pickup_min,
            max(tpep_pickup_datetime)                          AS pickup_max,
            min(tpep_dropoff_datetime)                         AS dropoff_min,
            max(tpep_dropoff_datetime)                         AS dropoff_max,
            count(*) FILTER (
                WHERE tpep_pickup_datetime <  TIMESTAMP '{settings.NYC_WINDOW_START}'
                   OR tpep_pickup_datetime >= TIMESTAMP '{settings.NYC_WINDOW_END}'
            )                                                  AS pickup_out_of_window,
            count(*) FILTER (
                WHERE regexp_extract(filename,
                        '([0-9]{{4}}-[0-9]{{2}})\\.parquet$', 1)
                      <> strftime(tpep_pickup_datetime, '%Y-%m')
            )                                                  AS pickup_month_mismatch,
            count(*) FILTER (WHERE tpep_pickup_datetime IS NULL)  AS null_pickup,
            count(*) FILTER (WHERE tpep_dropoff_datetime IS NULL) AS null_dropoff
        FROM {relation}
        GROUP BY 1, 2
        ORDER BY 1
        """,
    )
    return frame


def build(con: duckdb.DuckDBPyConnection | None = None) -> dict[str, Any]:
    """Materialise the bronze layer and return its diagnostics."""
    with stage("bronze: register and type raw artefacts", LOG):
        owned = con is None
        con = con or connect(settings.WAREHOUSE_DB)
        manifest = get_manifest()
        diagnostics: dict[str, Any] = {}

        settings.BRONZE_DIR.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------ Source A (registered)
        registry = build_nyc_trip_registry(con)
        registry_path = settings.BRONZE_DIR / BRONZE_TRIP_FILES
        registry.to_parquet(registry_path, index=False)

        total_raw = int(registry["raw_rows"].sum())
        out_of_window = int(registry["pickup_out_of_window"].sum())
        earliest = registry["pickup_min"].min()
        latest = registry["pickup_max"].max()

        diagnostics["nyc_trips"] = {
            "strategy": "registered in place (zero-copy bronze)",
            "rationale": (
                "copying 41 M rows into bronze and again into silver would spend "
                "~2 GB of the 3 GB disk budget restating an already-columnar, "
                "immutable input"
            ),
            "files": int(len(registry)),
            "raw_rows": total_raw,
            "expected_rows": settings.NYC_EXPECTED_TOTAL_ROWS,
            "rows_match_expected": total_raw == settings.NYC_EXPECTED_TOTAL_ROWS,
            "pickup_min_observed": str(earliest),
            "pickup_max_observed": str(latest),
            "pickup_out_of_window_rows": out_of_window,
            "pickup_out_of_window_pct": round(100.0 * out_of_window / total_raw, 6)
            if total_raw else 0.0,
            "pickup_month_mismatch_rows": int(registry["pickup_month_mismatch"].sum()),
            "null_pickup_rows": int(registry["null_pickup"].sum()),
            "null_dropoff_rows": int(registry["null_dropoff"].sum()),
            "window_start": str(settings.NYC_WINDOW_START),
            "window_end_exclusive": str(settings.NYC_WINDOW_END),
        }
        LOG.info(
            "bronze Source A: %s raw rows across %d files; pickup timestamps span "
            "%s to %s; %s row(s) (%.6f%%) fall outside the declared 2024 window",
            f"{total_raw:,}", len(registry), earliest, latest,
            f"{out_of_window:,}",
            diagnostics["nyc_trips"]["pickup_out_of_window_pct"],
        )
        manifest.note(
            f"NYC TLC 2024: {out_of_window:,} of {total_raw:,} trips carry a pickup "
            f"timestamp outside calendar 2024 (earliest observed {earliest}). This "
            f"is a defect of the official data, retained and flagged, not dropped."
        )

        # ------------------------------------------------------------ Source B
        zones_csv = _require(
            settings.RAW_NYC / "taxi_zone_lookup.csv", "Source B",
            "Run 'make extract-nyc'.",
        )
        con.execute(
            f"""
            COPY (
                SELECT
                    CAST(LocationID AS INTEGER)            AS location_id,
                    trim(CAST(Borough AS VARCHAR))         AS borough,
                    trim(CAST(Zone AS VARCHAR))            AS zone,
                    trim(CAST(service_zone AS VARCHAR))    AS service_zone,
                    '{zones_csv.name}'                     AS source_file
                FROM read_csv_auto('{str(zones_csv).replace(chr(92), "/")}',
                                   header=true)
            ) TO '{str(settings.BRONZE_DIR / BRONZE_ZONES).replace(chr(92), "/")}'
              (FORMAT PARQUET)
            """
        )
        zone_rows = con.execute(
            f"SELECT count(*) FROM read_parquet("
            f"'{str(settings.BRONZE_DIR / BRONZE_ZONES).replace(chr(92), '/')}')"
        ).fetchone()[0]
        diagnostics["nyc_zones"] = {"rows": int(zone_rows)}
        LOG.info("bronze Source B: %d zone rows", zone_rows)

        # ------------------------------------------------------------ Source C
        diagnostics["wfp"] = _bronze_wfp(con)

        # ------------------------------------------------------------ Source D
        diagnostics["nbs"] = _bronze_nbs(con)

        # ------------------------------------------------- Source E (optional)
        diagnostics["weather"] = _bronze_weather(con)

        manifest.data.setdefault("bronze", {}).update(diagnostics)
        manifest.save()

        if owned:
            con.close()
        return diagnostics


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _bronze_wfp(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Type the WFP price and market CSVs, dropping only the HXL tag row.

    The HXL row is removed by predicate (``date <> '#date'``) rather than by
    physical line offset, so the parser survives a vintage that omits it.
    """
    prices_csv = _require(
        settings.RAW_HDX / "wfp_food_prices_nga.csv", "Source C",
        "Run 'make extract-hdx'.",
    )
    markets_csv = _require(
        settings.RAW_HDX / "wfp_markets_nga.csv", "Source C (markets register)",
        "Run 'make extract-hdx'.",
    )
    prices_out = settings.BRONZE_DIR / BRONZE_WFP_PRICES
    markets_out = settings.BRONZE_DIR / BRONZE_WFP_MARKETS

    con.execute(
        f"""
        COPY (
            SELECT
                TRY_CAST(date AS DATE)                      AS observation_date,
                strftime(TRY_CAST(date AS DATE), '%Y-%m')   AS year_month,
                trim(admin1)                                AS admin1,
                trim(admin2)                                AS admin2,
                trim(market)                                AS market,
                TRY_CAST(market_id AS INTEGER)              AS market_id,
                TRY_CAST(latitude  AS DOUBLE)               AS latitude,
                TRY_CAST(longitude AS DOUBLE)               AS longitude,
                trim(category)                              AS category,
                trim(commodity)                             AS commodity,
                TRY_CAST(commodity_id AS INTEGER)           AS commodity_id,
                trim(unit)                                  AS unit,
                trim(priceflag)                             AS priceflag,
                trim(pricetype)                             AS pricetype,
                trim(currency)                              AS currency,
                TRY_CAST(price    AS DOUBLE)                AS price,
                TRY_CAST(usdprice AS DOUBLE)                AS usdprice,
                '{prices_csv.name}'                         AS source_file
            FROM read_csv_auto('{_sql_path(prices_csv)}', header=true,
                               all_varchar=true, sample_size=-1)
            WHERE trim(date) <> '{settings.HXL_SENTINEL}'
              AND trim(coalesce(date, '')) <> ''
        ) TO '{_sql_path(prices_out)}' (FORMAT PARQUET)
        """
    )
    con.execute(
        f"""
        COPY (
            SELECT
                trim(market)                       AS market,
                TRY_CAST(market_id AS INTEGER)     AS market_id,
                trim(admin1)                       AS admin1,
                trim(admin2)                       AS admin2,
                TRY_CAST(latitude  AS DOUBLE)      AS latitude,
                TRY_CAST(longitude AS DOUBLE)      AS longitude,
                '{markets_csv.name}'               AS source_file
            FROM read_csv_auto('{_sql_path(markets_csv)}', header=true,
                               all_varchar=true, sample_size=-1)
            WHERE trim(coalesce(market, '')) NOT IN ('', '#loc+market+name')
              AND trim(coalesce(market, '')) NOT LIKE '#%'
        ) TO '{_sql_path(markets_out)}' (FORMAT PARQUET)
        """
    )

    stats = sql_df(
        con,
        f"""
        SELECT count(*)                                    AS rows,
               count(*) FILTER (WHERE observation_date IS NULL) AS unparsable_dates,
               count(*) FILTER (WHERE price IS NULL)        AS null_prices,
               min(observation_date)                        AS date_min,
               max(observation_date)                        AS date_max,
               count(DISTINCT admin1)                       AS admin1_count,
               count(DISTINCT market)                       AS market_count,
               count(DISTINCT commodity)                    AS commodity_count
        FROM read_parquet('{_sql_path(prices_out)}')
        """,
    ).iloc[0].to_dict()
    market_rows = con.execute(
        f"SELECT count(*) FROM read_parquet('{_sql_path(markets_out)}')"
    ).fetchone()[0]

    result = {k: (int(v) if isinstance(v, (int, float)) and k.endswith(
        ("rows", "count", "dates", "prices")) else str(v))
        for k, v in stats.items()}
    result["market_register_rows"] = int(market_rows)
    LOG.info("bronze Source C: %s price rows (%s to %s), %s market register rows",
             f"{int(stats['rows']):,}", stats["date_min"], stats["date_max"],
             f"{int(market_rows):,}")
    return result


def _bronze_nbs(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Land the assembled NBS panel as Parquet."""
    assembled = _require(
        settings.RAW_NBS / "nbs_pms_assembled.csv", "Source D",
        "Run 'make extract-nbs'.",
    )
    out = settings.BRONZE_DIR / BRONZE_NBS_PMS
    con.execute(
        f"""
        COPY (
            SELECT
                CAST(month AS VARCHAR)              AS year_month,
                CAST(state AS VARCHAR)              AS state,
                CAST(state_raw AS VARCHAR)          AS state_raw,
                CAST(price_ngn AS DOUBLE)           AS price_ngn,
                CAST(source_file AS VARCHAR)        AS source_file,
                CAST(sheet_name AS VARCHAR)         AS sheet_name,
                CAST(is_current_month AS BOOLEAN)   AS is_current_month
            FROM read_csv_auto('{_sql_path(assembled)}', header=true)
        ) TO '{_sql_path(out)}' (FORMAT PARQUET)
        """
    )
    stats = sql_df(
        con,
        f"""
        SELECT count(*)                       AS rows,
               count(DISTINCT year_month)     AS months,
               count(DISTINCT state)          AS states,
               min(year_month)                AS month_min,
               max(year_month)                AS month_max
        FROM read_parquet('{_sql_path(out)}')
        """,
    ).iloc[0].to_dict()
    LOG.info("bronze Source D: %d observations, %d months (%s to %s), %d states",
             int(stats["rows"]), int(stats["months"]), stats["month_min"],
             stats["month_max"], int(stats["states"]))
    return {k: (int(v) if k in ("rows", "months", "states") else str(v))
            for k, v in stats.items()}


def _bronze_weather(con: duckdb.DuckDBPyConnection) -> dict[str, Any]:
    """Flatten the Open-Meteo parallel arrays into rows, if the source exists."""
    payload_path = settings.RAW_WEATHER / "open_meteo_nyc_2024.json"
    if not payload_path.exists():
        LOG.warning("bronze Source E: no payload at %s; weather is optional, "
                    "continuing without it", payload_path)
        return {"status": "skipped", "reason": "no extracted payload"}

    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        daily = payload["daily"]
        frame = pd.DataFrame(daily)
    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        LOG.warning("bronze Source E: unreadable payload (%s); continuing", exc)
        return {"status": "skipped", "reason": f"unreadable payload: {exc}"}

    frame = frame.rename(columns={"time": "observation_date"})
    frame["source_file"] = payload_path.name
    frame["latitude"] = payload.get("latitude")
    frame["longitude"] = payload.get("longitude")
    out = settings.BRONZE_DIR / BRONZE_WEATHER
    frame.to_parquet(out, index=False)
    LOG.info("bronze Source E: %d daily rows", len(frame))
    return {
        "status": "loaded",
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "units": payload.get("daily_units", {}),
    }
