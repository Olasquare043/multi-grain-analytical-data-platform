"""Execute analyses A1-A11 and write their result tables.

Each analysis is a single ``.sql`` file under ``sql/analysis/``. The runner
renders its placeholders from ``config/settings.py``, executes it, writes
``outputs/tables/<stem>.csv``, and records the shape in the run manifest.

A10 additionally receives measurements SQL cannot make -- on-disk bytes and the
wall-clock time to compute a comparable national aggregate -- which are attached
here and named with a ``_measured`` suffix so the provenance of every column is
unambiguous.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from config import settings
from src.model import catalog
from src.utils.db import connect, read_sql_file, table_exists
from src.utils.io_utils import dir_size, human_bytes
from src.utils.logging_setup import get_logger, stage
from src.utils.manifest import get_manifest

LOG = get_logger(__name__)

#: Analysis id -> SQL file stem. Order is the order the paper presents them.
ANALYSES: dict[str, str] = {
    "A1": "A1_demand_profile",
    "A2": "A2_od_corridors",
    "A3": "A3_congestion_proxy",
    "A4": "A4_tipping_behaviour",
    "A5": "A5_revenue_concentration",
    "A6": "A6_food_price_index",
    "A7": "A7_subsidy_structural_break",
    "A8": "A8_price_dispersion",
    "A9": "A9_fuel_food_passthrough",
    "A10": "A10_grain_comparison",
    "A11": "A11_missingness_structure",
    "A12": "A12_petrol_price_geography",
}

#: Analyses retained for transparency but NOT presented as findings. Their
#: outputs carry an analysis_status column saying so, and the manifest records
#: the status alongside the row counts.
EXPLORATORY_ANALYSES = {
    "A9": "exploratory -- null result; rests on 3 of 37 states and supports "
          "no inference (see docs/methodology_notes.md, Limitations)",
}

#: The "comparable national aggregate" of A10: one monthly national series per
#: fact table. Not the same question, but the same SHAPE of question, which is
#: what makes the timings comparable.
NATIONAL_AGGREGATE_QUERIES: dict[str, str] = {
    "fact_trip": """
        SELECT d.year_month, count(*) AS trips, sum(f.total_amount) AS revenue
        FROM fact_trip f JOIN dim_date d ON d.date_key = f.pickup_date_key
        GROUP BY 1 ORDER BY 1
    """,
    "fact_trip_daily_agg": """
        SELECT d.year_month, sum(a.trip_count) AS trips,
               sum(a.total_revenue) AS revenue
        FROM fact_trip_daily_agg a JOIN dim_date d ON d.date_key = a.date_key
        GROUP BY 1 ORDER BY 1
    """,
    "fact_market_price_monthly": """
        SELECT month_key, count(*) AS observations, avg(price_ngn) AS mean_price
        FROM fact_market_price_monthly GROUP BY 1 ORDER BY 1
    """,
    "fact_fuel_price_monthly": """
        SELECT month_key, count(*) AS observations, avg(price_ngn) AS mean_price
        FROM fact_fuel_price_monthly GROUP BY 1 ORDER BY 1
    """,
    "fact_weather_daily": """
        SELECT date_key / 100 AS year_month, count(*) AS observations,
               avg(temp_max_c) AS mean_value
        FROM fact_weather_daily GROUP BY 1 ORDER BY 1
    """,
}


def _month_index(year: int, month: int) -> int:
    """Dense month ordinal matching the expression used inside the SQL."""
    return year * 12 + month


def render_params() -> dict[str, Any]:
    """Every placeholder any analysis may reference, resolved from settings."""
    break_year, break_month = (
        int(settings.SUBSIDY_BREAK_YEAR_MONTH[:4]),
        int(settings.SUBSIDY_BREAK_YEAR_MONTH[5:7]),
    )
    return {
        "base_year_month": settings.PRICE_INDEX_BASE_YEAR_MONTH,
        "max_link_gap": settings.PRICE_INDEX_MAX_LINK_GAP_MONTHS,
        "persistence_share": settings.A12_PERSISTENCE_SHARE,
        "min_matched_cells": settings.PRICE_INDEX_MIN_MATCHED_CELLS,
        "staple_categories": str(tuple(settings.NIGERIA_STAPLE_CATEGORIES)),
        "price_type": settings.NIGERIA_PRICE_TYPE,
        "break_month_index": _month_index(break_year, break_month),
        "break_year_month": settings.SUBSIDY_BREAK_YEAR_MONTH,
        "break_date": settings.SUBSIDY_REMOVAL_DATE.isoformat(),
        "window": settings.SUBSIDY_WINDOW_MONTHS,
        "min_markets": settings.DISPERSION_MIN_MARKETS,
        "min_pairs": settings.PASSTHROUGH_MIN_PAIRS,
        "lag_values": ", ".join(f"({lag})" for lag in settings.PASSTHROUGH_LAGS),
    }


def _render(stem: str, params: dict[str, Any]) -> str:
    """Read an analysis file and substitute only the placeholders it uses."""
    sql = read_sql_file(settings.SQL_ANALYSIS_DIR / f"{stem}.sql")
    try:
        return sql.format(**params)
    except KeyError as exc:
        raise KeyError(
            f"{stem}.sql references placeholder {exc} which render_params() does "
            f"not provide. Add it to config/settings.py and render_params()."
        ) from exc


_OPTIONAL_BLOCK = re.compile(
    r"--\s*optional:(\w+)\s+begin\s*\n(.*?)--\s*optional:\1\s+end[^\n]*\n?",
    re.DOTALL,
)


def strip_optional_blocks(sql: str, con: duckdb.DuckDBPyConnection
                          ) -> tuple[str, list[str]]:
    """Remove ``-- optional:<table> begin/end`` blocks whose table is absent.

    Optional sources (Source E) must never break the core run. An analysis that
    can include an optional table marks that part of its SQL, and the block is
    dropped -- and the omission reported -- when the table was not built.

    Returns:
        ``(sql, omitted_tables)``.
    """
    omitted: list[str] = []

    def _resolve(match: re.Match[str]) -> str:
        table, body = match.group(1), match.group(2)
        if table_exists(con, table):
            return body
        omitted.append(table)
        return ""

    return _OPTIONAL_BLOCK.sub(_resolve, sql), omitted


_SECTION_MARKER = re.compile(r"^--\s*@(setup|output:\s*(\w+))\s*$", re.MULTILINE)


def split_sections(sql: str) -> list[tuple[str | None, str]]:
    """Split an analysis file into executable sections.

    An analysis with several result grains (A12) marks each statement:

    * ``-- @setup``          executed, produces no file (e.g. a TEMP VIEW)
    * ``-- @output: <name>`` executed, written to ``<stem>_<name>.csv``;
      the name ``main`` is written to ``<stem>.csv``

    A file with no markers is a single ``main`` output, which is every analysis
    other than A12. Returns ``[(output_name or None for setup, sql), ...]``.
    """
    markers = list(_SECTION_MARKER.finditer(sql))
    if not markers:
        return [("main", sql)]
    sections: list[tuple[str | None, str]] = []
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(sql)
        name = None if marker.group(1) == "setup" else marker.group(2)
        sections.append((name, sql[marker.end():end]))
    return sections


def execute_analysis(con: duckdb.DuckDBPyConnection,
                     sql: str) -> dict[str, pd.DataFrame]:
    """Run every section of a rendered analysis; return ``{output: frame}``."""
    outputs: dict[str, pd.DataFrame] = {}
    for name, body in split_sections(sql):
        if name is None:
            con.execute(body)
        else:
            outputs[name] = con.execute(body).df()
    return outputs


def output_path(stem: str, name: str) -> Path:
    """CSV path for one named output of an analysis."""
    suffix = "" if name == "main" else f"_{name}"
    return settings.TABLES_DIR / f"{stem}{suffix}.csv"


def _fact_artefact_path(fact_table: str) -> Path:
    if fact_table == "fact_trip":
        return settings.GOLD_DIR / "fact_trip"
    return settings.GOLD_DIR / f"{fact_table}.parquet"


def measure_grain_costs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Measure on-disk size and national-aggregate latency per fact table.

    The aggregate is timed three times and the median reported, matching the
    benchmark protocol in section 8 so the two sets of timings are comparable.
    """
    records = []
    for fact_table, query in NATIONAL_AGGREGATE_QUERIES.items():
        path = _fact_artefact_path(fact_table)
        size = dir_size(path)
        timings: list[float] = []
        for _ in range(settings.BENCHMARK_REPEATS):
            started = time.perf_counter()
            try:
                con.execute(query).fetchall()
            except duckdb.Error as exc:
                LOG.warning("national aggregate for %s failed: %s", fact_table, exc)
                timings = []
                break
            timings.append(time.perf_counter() - started)
        median = round(sorted(timings)[len(timings) // 2], 6) if timings else None
        records.append({
            "fact_table": fact_table,
            "bytes_measured": size,
            "bytes_human_measured": human_bytes(size),
            "national_aggregate_seconds_measured": median,
            "national_aggregate_repeats": len(timings),
        })
        LOG.info("A10 measurement: %-28s %10s, national aggregate %s",
                 fact_table, human_bytes(size),
                 f"{median:.4f}s" if median is not None else "n/a")
    return pd.DataFrame(records)


def _augment_a10(frame: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Attach filesystem and wall-clock measurements to the A10 result."""
    measured = measure_grain_costs(con)
    merged = frame.merge(measured, on="fact_table", how="left")
    merged["bytes_per_row_measured"] = [
        round(b / r, 4) if r and pd.notna(b) else None
        for b, r in zip(merged["bytes_measured"], merged["row_count"])
    ]
    # Relative cost against the coarsest fact table, which is the comparison the
    # paper's central argument actually turns on.
    baseline = merged.loc[
        merged["fact_table"] == "fact_fuel_price_monthly", "row_count"
    ]
    if len(baseline) and baseline.iloc[0]:
        merged["rows_vs_coarsest_fact"] = (
            merged["row_count"] / baseline.iloc[0]
        ).round(1)
    return merged


def run(con: duckdb.DuckDBPyConnection | None = None,
        only: str | None = None) -> dict[str, Any]:
    """Execute the analyses and write one CSV per analysis.

    Args:
        con: Warehouse connection; opened and closed here when omitted.
        only: Run a single analysis id (e.g. ``'A9'``) instead of all.
    """
    with stage("analysis: A1-A11", LOG):
        owned = con is None
        con = con or connect(settings.WAREHOUSE_DB)
        catalog.register_views(con)
        settings.TABLES_DIR.mkdir(parents=True, exist_ok=True)

        params = render_params()
        selected = {only: ANALYSES[only]} if only else ANALYSES
        if only and only not in ANALYSES:
            raise KeyError(f"unknown analysis {only!r}. Valid: {sorted(ANALYSES)}")

        results: dict[str, Any] = {}
        for analysis_id, stem in selected.items():
            sql, omitted = strip_optional_blocks(_render(stem, params), con)
            for table in omitted:
                LOG.warning("%s: optional relation %s is absent (its source was "
                            "skipped); that part of the analysis is omitted",
                            analysis_id, table)
                get_manifest().note(
                    f"{analysis_id} omitted optional relation {table} because "
                    f"its source was skipped in this run."
                )
            started = time.perf_counter()
            outputs = execute_analysis(con, sql)
            elapsed = time.perf_counter() - started

            if analysis_id == "A10":
                outputs["main"] = _augment_a10(outputs["main"], con)

            written: dict[str, int] = {}
            for name, frame in outputs.items():
                destination = output_path(stem, name)
                frame.to_csv(destination, index=False)
                written[destination.name] = int(len(frame))
                LOG.info("%-4s %-44s %7d row(s) x %2d column(s) -> %s",
                         analysis_id, f"{stem}[{name}]", len(frame),
                         frame.shape[1], destination.name)

            results[analysis_id] = {
                "sql_file": f"{stem}.sql",
                "outputs": written,
                "status": EXPLORATORY_ANALYSES.get(analysis_id, "result"),
                "seconds": round(elapsed, 3),
            }
            LOG.info("%-4s done in %.2fs%s", analysis_id, elapsed,
                     "  [EXPLORATORY -- not a finding]"
                     if analysis_id in EXPLORATORY_ANALYSES else "")

        manifest = get_manifest()
        manifest.data.setdefault("analyses", {}).update(results)
        manifest.save()

        if owned:
            con.close()
        return results
