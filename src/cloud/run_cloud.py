"""Optional cloud module (section 9): local DuckDB versus the BigQuery sandbox.

Run with ``make cloud`` (``python -m src.cloud.run_cloud``). The core pipeline
never imports this module and succeeds whether or not it has ever run.

Steps:

* **C0 verify** -- list the public NYC TLC tables and their row counts from the
  BigQuery metadata API (free), write ``outputs/cloud/table_inventory.csv``, and
  stop if the tables this module needs are absent.
* **C1 engine_comparison_cloud** -- the A1 demand-profile question on both
  engines: rows, bytes, latency (median of 3) and implied on-demand cost.
* **C2 scale_behaviour** -- one query over progressively larger cloud inputs,
  up to the whole yellow-trip archive, with a running byte budget.

**The comparison is not a controlled experiment**, and every output says so:
different data years, volumes, hardware, storage layouts and network paths.
"""
from __future__ import annotations

import datetime as dt
import re
import sys
import time
from pathlib import Path
from statistics import median
from typing import Any

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from config import settings
from src.cloud.bq_client import (
    BudgetExceeded, ByteBudget, CloudUnavailable, QueryFailed, Runner,
    make_client, monthly_bytes_used,
)
from src.utils.db import read_sql_file
from src.utils.io_utils import ensure_dirs, human_bytes
from src.utils.logging_setup import configure_logging, get_logger, stage
from src.utils.manifest import get_manifest

LOG = get_logger(__name__)

TIB = 1024 ** 4
REPEATS = settings.BENCHMARK_REPEATS
LOCAL_YEAR = settings.NYC_TRIP_YEAR
CLOUD_YEAR = 2019
PUBLIC = f"{settings.BQ_PUBLIC_PROJECT}.{settings.BQ_TLC_DATASET}"

CONFOUNDERS = [
    "Different data years: the public BigQuery TLC dataset ends in 2022 and "
    "holds no 2024 rows, so the cloud side reads 2019 and the local side 2024.",
    "Different volumes: the two years contain different numbers of trips.",
    "Different storage layouts: this pipeline's Hive-partitioned ZSTD Parquet "
    "versus Google-managed columnar storage.",
    "Different hardware: a 4-CPU / 8 GB Docker Desktop VM on one laptop versus "
    "a shared, elastically scheduled BigQuery slot pool of undisclosed size.",
    "Network: BigQuery latency includes job submission and completion polling "
    "over the internet; local latency has no network hop.",
    "Different byte definitions: BigQuery reports logical (uncompressed) bytes "
    "of the columns read; the local figures come from Parquet column-chunk "
    "metadata and are given both compressed (on disk) and uncompressed.",
    "Cache state: the BigQuery result cache is disabled; local runs read "
    "through a warm operating-system page cache.",
]


def _rel(path: Path) -> str:
    return str(path).replace("\\", "/")


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #
def resolve_pricing() -> dict[str, Any]:
    """Try to read the current on-demand rate; fall back to configuration.

    Section 9 forbids hard-coding a remembered price. The configured value is
    used only if the pricing page cannot be read, and the output records which
    source the rate came from and whether the page confirmed it.
    """
    configured = settings.BQ_ON_DEMAND_USD_PER_TIB
    result = {
        "usd_per_tib": configured,
        "source": "configuration value BQ_ON_DEMAND_USD_PER_TIB (not verified)",
        "reference_url": settings.BQ_PRICING_REFERENCE_URL,
        "page_rate_found": None,
        "checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        import requests

        html = requests.get(settings.BQ_PRICING_REFERENCE_URL, timeout=20,
                            headers={"User-Agent": settings.USER_AGENT}).text
        found = re.findall(r"\$\s?(\d+(?:\.\d+)?)\s*(?:USD\s*)?per\s*TiB", html)
        rates = sorted({float(value) for value in found})
        if rates:
            result["page_rate_found"] = rates
            if configured in rates:
                result["source"] = ("configuration value, CONFIRMED on the "
                                    "pricing page at runtime")
            else:
                result["source"] = ("configuration value; the pricing page listed "
                                    f"{rates} per TiB and did not confirm it -- "
                                    "check manually")
    except Exception as exc:  # noqa: BLE001 - pricing check is best effort
        result["source"] += f"; pricing page unreadable ({type(exc).__name__})"
    LOG.info("pricing: %.2f USD/TiB -- %s", result["usd_per_tib"], result["source"])
    return result


def implied_cost(billed_bytes: int, usd_per_tib: float) -> float:
    return round(billed_bytes / TIB * usd_per_tib, 6)


# --------------------------------------------------------------------------- #
# C0 -- verify before using
# --------------------------------------------------------------------------- #
def build_inventory(client: Any) -> pd.DataFrame:
    """Every table in the public dataset, from the free metadata API."""
    rows = []
    for item in client.list_tables(PUBLIC):
        table = client.get_table(item.reference)
        fields = {f.name: f.field_type for f in table.schema}
        partition = table.time_partitioning
        rows.append({
            "table_id": table.table_id,
            "num_rows": int(table.num_rows or 0),
            "logical_bytes": int(table.num_bytes or 0),
            "logical_bytes_human": human_bytes(int(table.num_bytes or 0)),
            "column_count": len(fields),
            "pickup_datetime_type": fields.get("pickup_datetime"),
            "pickup_location_id_type": fields.get("pickup_location_id"),
            "time_partitioning_field": (partition.field or "_PARTITIONTIME")
            if partition else None,
            "clustering_fields": ",".join(table.clustering_fields or []) or None,
            "last_modified": table.modified.isoformat() if table.modified else None,
        })
    frame = pd.DataFrame(rows).sort_values("table_id").reset_index(drop=True)
    frame.to_csv(settings.CLOUD_OUT_DIR / "table_inventory.csv", index=False)
    LOG.info("inventory: %d tables in %s", len(frame), PUBLIC)
    for row in frame.itertuples():
        LOG.info("  %-30s %14s rows  %10s", row.table_id, f"{row.num_rows:,}",
                 row.logical_bytes_human)
    return frame


def require_tables(inventory: pd.DataFrame) -> None:
    by_id = inventory.set_index("table_id")
    for table in (settings.BQ_COMPARISON_TABLE, settings.BQ_ZONE_GEOM_TABLE):
        if table not in by_id.index:
            raise CloudUnavailable(f"{PUBLIC}.{table} is not in the dataset; the "
                                   f"comparison cannot run. See table_inventory.csv.")
        if by_id.loc[table, "num_rows"] == 0:
            raise CloudUnavailable(f"{PUBLIC}.{table} exists but holds no rows.")


# --------------------------------------------------------------------------- #
# Local side
# --------------------------------------------------------------------------- #
def parquet_column_bytes(files: list[Path], columns: set[str]) -> tuple[int, int]:
    """Compressed and uncompressed bytes of the named columns, from metadata."""
    compressed = uncompressed = 0
    for path in files:
        meta = pq.ParquetFile(path).metadata
        for group_index in range(meta.num_row_groups):
            group = meta.row_group(group_index)
            for column_index in range(group.num_columns):
                column = group.column(column_index)
                if column.path_in_schema in columns:
                    compressed += column.total_compressed_size
                    uncompressed += column.total_uncompressed_size
    return compressed, uncompressed


def run_local(sql: str, columns: set[str], repeats: int) -> dict[str, Any]:
    """Time a query on the local warehouse and measure the bytes it reads."""
    files = sorted((settings.GOLD_DIR / "fact_trip" / f"year={LOCAL_YEAR}").rglob("*.parquet"))
    if not files:
        raise CloudUnavailable(f"no local fact_trip partitions for {LOCAL_YEAR}; "
                               f"run the core pipeline ('make run') first.")
    con = duckdb.connect(str(settings.WAREHOUSE_DB), read_only=True)
    try:
        timings, frame = [], None
        for _ in range(repeats):
            started = time.perf_counter()
            frame = con.execute(sql).df()
            timings.append(time.perf_counter() - started)
        rows_in_input = int(con.execute(
            f"SELECT count(*) FROM fact_trip WHERE year = {LOCAL_YEAR}").fetchone()[0])
    finally:
        con.close()
    compressed, uncompressed = parquet_column_bytes(files, columns)
    return {
        "frame": frame, "timings": timings, "rows_in_input": rows_in_input,
        "bytes_compressed": compressed, "bytes_uncompressed": uncompressed,
    }


# --------------------------------------------------------------------------- #
# C1
# --------------------------------------------------------------------------- #
def c1_bigquery_sql() -> str:
    return read_sql_file(settings.SQL_CLOUD_DIR / "C1_demand_profile_bigquery.sql").format(
        public_project=settings.BQ_PUBLIC_PROJECT, dataset=settings.BQ_TLC_DATASET,
        trip_table=settings.BQ_COMPARISON_TABLE, zone_table=settings.BQ_ZONE_GEOM_TABLE,
        start_date=f"{CLOUD_YEAR}-01-01", end_date=f"{CLOUD_YEAR + 1}-01-01",
        max_duration_seconds=settings.TRIP_MAX_DURATION_SECONDS,
        max_distance_miles=settings.TRIP_MAX_DISTANCE_MILES,
        max_passengers=settings.TRIP_MAX_PASSENGERS,
    )


def c1_local_sql() -> str:
    return read_sql_file(settings.SQL_CLOUD_DIR / "C1_demand_profile_duckdb.sql").format(
        local_year=LOCAL_YEAR)


def run_c1(runner: Runner, pricing: dict[str, Any]) -> pd.DataFrame:
    with stage("cloud: C1 engine_comparison_cloud", LOG):
        local = run_local(c1_local_sql(),
                          {"pickup_date_key", "pickup_geo_key", "pickup_hour"}, REPEATS)
        cloud_label = (f"{PUBLIC}.{settings.BQ_COMPARISON_TABLE}, pickups "
                       f"{CLOUD_YEAR}-01-01 to {CLOUD_YEAR}-12-31")
        results = [runner.run(c1_bigquery_sql(), f"C1 BigQuery run {i + 1}/{REPEATS}")
                   for i in range(REPEATS)]

        bq_times = [r.seconds_to_complete for r in results]
        bq = results[0]
        rows = [
            {
                "engine": "DuckDB (local)",
                "dataset": f"local gold fact_trip, NYC TLC yellow trips, pickups "
                           f"{LOCAL_YEAR}-01-01 to {LOCAL_YEAR}-12-31",
                "data_year": LOCAL_YEAR,
                "rows_in_input": local["rows_in_input"],
                "rows_counted": int(local["frame"]["trip_count"].sum()),
                "result_rows": int(len(local["frame"])),
                "bytes_processed_logical": local["bytes_uncompressed"],
                "bytes_read_compressed": local["bytes_compressed"],
                "bytes_billed": 0,
                "median_seconds": round(median(local["timings"]), 4),
                "min_seconds": round(min(local["timings"]), 4),
                "max_seconds": round(max(local["timings"]), 4),
                "repeats": REPEATS,
                "slot_ms_median": None,
                "implied_cost_usd": 0.0,
                "bytes_note": "Parquet column-chunk metadata for the columns read",
            },
            {
                "engine": "BigQuery (sandbox)",
                "dataset": cloud_label,
                "data_year": CLOUD_YEAR,
                "rows_in_input": None,
                "rows_counted": int(bq.frame["trip_count"].sum()),
                "result_rows": int(len(bq.frame)),
                "bytes_processed_logical": bq.bytes_processed,
                "bytes_read_compressed": None,
                "bytes_billed": bq.bytes_billed,
                "median_seconds": round(median(bq_times), 4),
                "min_seconds": round(min(bq_times), 4),
                "max_seconds": round(max(bq_times), 4),
                "repeats": REPEATS,
                "slot_ms_median": median([r.slot_ms or 0 for r in results]),
                "implied_cost_usd": implied_cost(bq.bytes_billed, pricing["usd_per_tib"]),
                "bytes_note": "BigQuery total_bytes_processed (logical); query cache disabled",
            },
        ]
        frame = pd.DataFrame(rows)
        frame["usd_per_tib_used"] = pricing["usd_per_tib"]
        frame["pricing_source"] = pricing["source"]
        frame["confounders"] = " | ".join(CONFOUNDERS)
        frame.to_csv(settings.CLOUD_OUT_DIR / "engine_comparison_cloud.csv", index=False)
        return frame


# --------------------------------------------------------------------------- #
# C2
# --------------------------------------------------------------------------- #
def _source_select(table: str, dtype: str, start: str | None, end: str | None) -> str:
    column = "pickup_datetime"
    where = ""
    if start and end:
        literal = "TIMESTAMP" if dtype == "TIMESTAMP" else "DATETIME"
        where = (f" WHERE {column} >= {literal} '{start}' "
                 f"AND {column} < {literal} '{end}'")
    return (f"    SELECT CAST({column} AS DATETIME) AS pickup_dt "
            f"FROM `{PUBLIC}.{table}`{where}")


def c2_plan(inventory: pd.DataFrame) -> list[dict[str, Any]]:
    yellow = inventory[
        inventory["table_id"].str.startswith("tlc_yellow_trips_")
        & (inventory["num_rows"] > 0)
        & inventory["pickup_datetime_type"].isin(["TIMESTAMP", "DATETIME"])
    ].sort_values("table_id")
    types = dict(zip(yellow["table_id"], yellow["pickup_datetime_type"]))
    comparison = settings.BQ_COMPARISON_TABLE
    largest = yellow.loc[yellow["num_rows"].idxmax(), "table_id"]
    years = [t.rsplit("_", 1)[1] for t in yellow["table_id"]]

    steps = [
        ("one month", [comparison], f"{CLOUD_YEAR}-01-01", f"{CLOUD_YEAR}-02-01",
         f"{comparison}, pickups {CLOUD_YEAR}-01-01 to {CLOUD_YEAR}-01-31"),
        ("one year", [comparison], None, None, f"{comparison}, all rows ({CLOUD_YEAR})"),
        ("largest single table", [largest], None, None, f"{largest}, all rows"),
        ("full yellow archive", list(yellow["table_id"]), None, None,
         f"{len(yellow)} tables tlc_yellow_trips_{years[0]} to _{years[-1]}, all rows"),
    ]
    template = read_sql_file(settings.SQL_CLOUD_DIR / "C2_scale_bigquery.sql")
    plan = []
    for label, tables, start, end, description in steps:
        sources = "\n    UNION ALL\n".join(
            _source_select(t, types[t], start, end) for t in tables)
        plan.append({
            "step": label, "tables": tables, "input": description,
            "table_rows": int(inventory.set_index("table_id").loc[tables, "num_rows"].sum()),
            # The template's only placeholders are {sources_description} (one
            # line, in the header) and {sources} (the FROM clause).
            "sql": template.replace("{sources_description}", description)
                           .replace("{sources}", sources),
        })
    return plan


def run_c2(runner: Runner, plan: list[dict[str, Any]],
           pricing: dict[str, Any]) -> pd.DataFrame:
    with stage("cloud: C2 scale_behaviour", LOG):
        rows = []
        for item in plan:
            try:
                result = runner.run(item["sql"], f"C2 {item['step']}")
            except BudgetExceeded as exc:
                LOG.error("C2 stopped before '%s': %s", item["step"], exc)
                rows.append({"step": item["step"], "input": item["input"],
                             "status": f"not run: {exc}"})
                break
            except QueryFailed as exc:
                LOG.error("C2 '%s' failed: %s", item["step"], exc)
                rows.append({"step": item["step"], "input": item["input"],
                             "status": f"failed: {exc}"})
                continue
            rows.append({
                "engine": "BigQuery (sandbox)",
                "step": item["step"],
                "input": item["input"],
                "tables": len(item["tables"]),
                "table_rows": item["table_rows"],
                "rows_counted": int(result.frame["trip_count"].sum()),
                "bytes_processed": result.bytes_processed,
                "bytes_processed_human": human_bytes(result.bytes_processed),
                "bytes_billed": result.bytes_billed,
                "seconds_to_complete": round(result.seconds_to_complete, 3),
                "server_seconds": result.server_seconds,
                "slot_ms": result.slot_ms,
                "implied_cost_usd": implied_cost(result.bytes_billed, pricing["usd_per_tib"]),
                "status": "ok (single run, cache disabled)",
            })

        # A local reference point: the same question over local 2024 data.
        local_sql = (
            "SELECT d.day_of_week, f.pickup_hour, count(*) AS trip_count "
            "FROM fact_trip f JOIN dim_date d ON d.date_key = f.pickup_date_key "
            f"WHERE f.year = {LOCAL_YEAR} GROUP BY 1, 2"
        )
        local = run_local(local_sql, {"pickup_date_key", "pickup_hour"}, REPEATS)
        rows.append({
            "engine": "DuckDB (local, reference)",
            "step": "local reference (one year)",
            "input": f"local gold fact_trip, pickups {LOCAL_YEAR}",
            "tables": 1,
            "table_rows": local["rows_in_input"],
            "rows_counted": int(local["frame"]["trip_count"].sum()),
            "bytes_processed": local["bytes_uncompressed"],
            "bytes_processed_human": human_bytes(local["bytes_uncompressed"]),
            "bytes_billed": 0,
            "seconds_to_complete": round(median(local["timings"]), 3),
            "implied_cost_usd": 0.0,
            "status": f"ok (median of {REPEATS})",
        })
        frame = pd.DataFrame(rows)
        frame["confounders"] = " | ".join(CONFOUNDERS)
        frame.to_csv(settings.CLOUD_OUT_DIR / "scale_behaviour.csv", index=False)
        return frame


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_engine(frame: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt
    from src.viz import style

    style.apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    labels = [f"{r.engine}\n{r.data_year} data" for r in frame.itertuples()]
    colours = [style.PALETTE[0], style.PALETTE[1]]
    errors = [frame["median_seconds"] - frame["min_seconds"],
              frame["max_seconds"] - frame["median_seconds"]]
    ax1.bar(labels, frame["median_seconds"], color=colours, yerr=errors,
            capsize=6, width=0.55)
    # Labels sit beside each bar, clear of the min-max whisker at its centre.
    for pos, row in enumerate(frame.itertuples()):
        ax1.text(pos + 0.31, row.median_seconds,
                 f"{row.median_seconds:.2f}s median\n"
                 f"range {row.min_seconds:.2f}-{row.max_seconds:.2f}s\n"
                 f"{row.rows_counted:,} trips",
                 ha="left", va="center", fontsize=8.3)
    ax1.set_xlim(-0.5, 1.95)
    ax1.set_ylabel(f"Latency, median of {REPEATS} runs (s)")
    ax1.set_title("Same question, two engines: latency", fontsize=11)

    local, cloud = frame.iloc[0], frame.iloc[1]
    bars = [
        ("local: on disk\n(compressed)", local["bytes_read_compressed"], style.PALETTE[0]),
        ("local: logical\n(uncompressed)", local["bytes_processed_logical"], style.PALETTE[5]),
        ("BigQuery:\nprocessed", cloud["bytes_processed_logical"], style.PALETTE[1]),
        ("BigQuery:\nbilled", cloud["bytes_billed"], style.PALETTE[4]),
    ]
    ax2.bar([b[0] for b in bars], [b[1] for b in bars],
            color=[b[2] for b in bars], width=0.6)
    for pos, (_, value, _) in enumerate(bars):
        ax2.text(pos, value, f"{human_bytes(value)}", ha="center", va="bottom",
                 fontsize=8.5)
    ax2.set_yscale("log")
    ax2.set_ylabel("Bytes (log scale)")
    ax2.set_title(f"Bytes read; BigQuery implied cost "
                  f"${cloud['implied_cost_usd']:.4f} per run", fontsize=11)
    ax2.tick_params(axis="x", labelsize=8.5)

    fig.suptitle("C1: A1 demand profile on local DuckDB versus BigQuery",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    path = settings.CLOUD_OUT_DIR / "engine_comparison_cloud.png"
    return style.finish(
        fig, path,
        f"Local: {local['dataset']}. Cloud: {cloud['dataset']}.",
        "ILLUSTRATIVE, NOT A CONTROLLED BENCHMARK: different data years (2024 "
        "local, 2019 cloud -- the public dataset ends in 2022), volumes, "
        "hardware, storage layouts and network paths. Pricing: "
        f"{cloud['usd_per_tib_used']:.2f} USD/TiB, {cloud['pricing_source']}.",
    )


def plot_scale(frame: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt
    from src.viz import style

    style.apply_style()
    ok = frame[frame["status"].astype(str).str.startswith("ok")]
    cloud = ok[ok["engine"] == "BigQuery (sandbox)"]
    local = ok[ok["engine"] != "BigQuery (sandbox)"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for ax, column, ylabel in (
        (ax1, "bytes_processed", "Bytes processed (log)"),
        (ax2, "seconds_to_complete", "Seconds to complete (log)"),
    ):
        ax.plot(cloud["rows_counted"], cloud[column], color=style.PALETTE[1],
                marker="o", markersize=7, linewidth=1.6, label="BigQuery sandbox")
        ax.plot(local["rows_counted"], local[column], color=style.PALETTE[0],
                marker="D", markersize=8, linestyle="none",
                label=f"local DuckDB, {LOCAL_YEAR} (reference)")
        for row in ok.itertuples():
            ax.annotate(row.step, (row.rows_counted, getattr(row, column)),
                        textcoords="offset points", xytext=(6, -12), fontsize=7.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Trips counted (log)")
        ax.set_ylabel(ylabel)
    ax1.set_title("Bytes processed against input size", fontsize=11)
    ax2.set_title("Latency against input size", fontsize=11)
    ax1.legend(fontsize=8.5, loc="upper left")
    fig.suptitle("C2: scale behaviour of one query, one month to the full archive",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    inputs = "; ".join(f"{r.step}: {r.input}" for r in ok.itertuples())
    return style.finish(
        fig, settings.CLOUD_OUT_DIR / "scale_behaviour.png",
        f"Inputs -- {inputs}.",
        "Cloud steps are single runs with the result cache disabled. The local "
        "point is a reference on different hardware and a different year, not "
        "a comparable measurement.",
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    configure_logging()
    ensure_dirs()
    args = sys.argv[1:] if argv is None else argv

    # Redraw the figures from the saved CSVs without running (or paying for)
    # a single query.
    if "--plots-only" in args:
        c1 = pd.read_csv(settings.CLOUD_OUT_DIR / "engine_comparison_cloud.csv")
        c2 = pd.read_csv(settings.CLOUD_OUT_DIR / "scale_behaviour.csv")
        written = [plot_engine(c1).name, plot_scale(c2).name]
        LOG.info("redrew %s from saved results; no queries run", written)
        return 0

    manifest = get_manifest()
    record: dict[str, Any] = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}

    try:
        client = make_client()
    except CloudUnavailable as exc:
        LOG.error("cloud module not run: %s", exc)
        return 1

    budget = ByteBudget(session_limit=settings.BQ_SESSION_BYTE_BUDGET)
    runner = Runner(client, budget)
    try:
        with stage("cloud: C0 verify public dataset inventory", LOG):
            inventory = build_inventory(client)
            require_tables(inventory)

        used, used_source = monthly_bytes_used(runner)
        budget.monthly_used_before = used
        LOG.info("free-tier usage before this run: %s (%s); session allowance %s",
                 human_bytes(used) if used is not None else "unknown", used_source,
                 human_bytes(budget.allowance()))
        record["monthly_usage_before"] = {"bytes": used, "source": used_source}

        pricing = resolve_pricing()
        record["pricing"] = pricing

        plan = c2_plan(inventory)
        projected = {"C1 x%d" % REPEATS: REPEATS * runner.dry_run(c1_bigquery_sql(), "C1")}
        for item in plan:
            projected[f"C2 {item['step']}"] = runner.dry_run(item["sql"], item["step"])
        total = sum(projected.values())
        LOG.info("projected bytes for the whole plan: %s (allowance %s)",
                 human_bytes(total), human_bytes(budget.headroom()))
        for label, value in projected.items():
            LOG.info("  %-32s %s", label, human_bytes(value))
        record["projected_bytes"] = projected
        budget.check(total, "whole cloud plan")

        c1 = run_c1(runner, pricing)
        c2 = run_c2(runner, plan, pricing)
        figures = [plot_engine(c1).name, plot_scale(c2).name]
    except (CloudUnavailable, BudgetExceeded, QueryFailed) as exc:
        LOG.error("cloud module stopped: %s", exc)
        record["stopped"] = str(exc)
        record["query_log"] = budget.log
        manifest.data["cloud"] = record
        manifest.save()
        return 1

    record.update({
        "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "session_bytes_spent": budget.spent,
        "session_headroom_bytes": budget.headroom(),
        "query_log": budget.log,
        "outputs": ["table_inventory.csv", "engine_comparison_cloud.csv",
                    "scale_behaviour.csv", *figures],
        "confounders": CONFOUNDERS,
    })
    manifest.data["cloud"] = record
    manifest.save()
    LOG.info("cloud module complete: %s spent this session; %s headroom left",
             human_bytes(budget.spent), human_bytes(budget.headroom()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
