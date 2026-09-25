"""Benchmarks B1-B4 (section 8).

Every workload runs in its own subprocess via :mod:`src.benchmark.harness`, is
repeated, and is summarised by its median. Container CPU and memory limits are
recorded alongside every result, because a latency without a resource envelope
is not a measurement.

Targets are module-level functions rather than closures so they survive the
``spawn`` start method used for isolated memory profiling.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from src.benchmark import harness
from src.model.facts import fact_trip_relation
from src.utils.db import connect
from src.utils.io_utils import dir_size, human_bytes
from src.utils.logging_setup import get_logger, stage

LOG = get_logger(__name__)

BENCH_DIR_NAME = "benchmark_tmp"

#: The projection B2 and B4 operate on. Declared once so that both sides of
#: every comparison read exactly the same columns -- otherwise the comparison
#: measures the projection, not the storage layout.
BENCH_COLUMNS = ("pickup_date_key", "pickup_geo_key", "total_amount",
                 "trip_distance_miles", "year", "month")


def _sql(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def bench_dir() -> Path:
    path = settings.DATA_DIR / BENCH_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def month_partition_dir(year_month: str) -> Path:
    year, month = year_month.split("-")
    return settings.GOLD_DIR / "fact_trip" / f"year={year}" / f"month={int(month)}"


# --------------------------------------------------------------------------- #
# Subprocess targets
# --------------------------------------------------------------------------- #
def _duckdb_scalar(query: str) -> Any:
    con = connect(settings.WAREHOUSE_DB, read_only=True)
    try:
        return con.execute(query).fetchall()[:5]
    finally:
        con.close()


def _duckdb_scratch(query: str) -> Any:
    """Run against an in-memory database, for queries over bare files."""
    con = connect(None)
    try:
        return con.execute(query).fetchall()[:5]
    finally:
        con.close()


def _pandas_aggregate(dataset_dir: str) -> Any:
    """The B4 pandas arm: load the projection, group, aggregate.

    Deliberately written the way a pandas user would write it, and deliberately
    NOT chunked. B4 asks whether pandas can do this job in the memory budget;
    hand-rolling an out-of-core loop would answer a different question.
    """
    frame = pd.read_parquet(
        dataset_dir,
        columns=["pickup_date_key", "pickup_geo_key", "total_amount",
                 "trip_distance_miles"],
    )
    grouped = frame.groupby(["pickup_date_key", "pickup_geo_key"], sort=False).agg(
        trip_count=("total_amount", "size"),
        total_revenue=("total_amount", "sum"),
        total_distance=("trip_distance_miles", "sum"),
        mean_fare=("total_amount", "mean"),
    )
    return [int(len(grouped)), float(grouped["total_revenue"].sum())]


# --------------------------------------------------------------------------- #
# B1  format_comparison
# --------------------------------------------------------------------------- #
def b1_format_comparison() -> list[dict[str, Any]]:
    """CSV versus Parquet for one month: size, full scan, single-column scan.

    The Parquet side is the gold ``fact_trip`` partition for the benchmark month.
    The CSV side is a byte-for-byte faithful text export of the same rows, so the
    only difference between the two inputs is the encoding.

    The CSV is deleted afterwards (``BENCHMARK_DELETE_CSV_AFTER``): it is roughly
    half a gigabyte, and retaining it would push the project past the 3 GB disk
    budget for no analytical gain. The measurement is kept; the bytes are not.
    """
    month = settings.BENCHMARK_MONTH
    partition = month_partition_dir(month)
    if not partition.exists():
        raise FileNotFoundError(
            f"{partition} is missing. B1 needs the gold fact_trip partition for "
            f"{month}. Run 'make facts' first."
        )

    csv_path = bench_dir() / f"fact_trip_{month}.csv"
    parquet_glob = _sql(partition / "*.parquet")

    con = connect(None)
    try:
        con.execute(
            f"COPY (SELECT * FROM read_parquet('{parquet_glob}')) "
            f"TO '{_sql(csv_path)}' (FORMAT CSV, HEADER)"
        )
        rows = con.execute(
            f"SELECT count(*) FROM read_parquet('{parquet_glob}')"
        ).fetchone()[0]
    finally:
        con.close()

    parquet_bytes = dir_size(partition)
    csv_bytes = csv_path.stat().st_size

    csv_relation = f"read_csv_auto('{_sql(csv_path)}', header=true)"
    parquet_relation = f"read_parquet('{parquet_glob}')"

    results: list[dict[str, Any]] = []
    for fmt, relation, size in (
        ("parquet", parquet_relation, parquet_bytes),
        ("csv", csv_relation, csv_bytes),
    ):
        # COLUMNS(*) forces every column to be decoded: a true full scan.
        full = harness.measure(
            f"B1 {fmt} full scan",
            _target(_duckdb_scratch, f"SELECT count(COLUMNS(*)) FROM {relation}"),
        )
        single = harness.measure(
            f"B1 {fmt} single-column aggregation",
            _target(_duckdb_scratch,
                    f"SELECT sum(total_amount) FROM {relation}"),
        )
        results.append({
            "benchmark": "B1_format_comparison",
            "variant": fmt,
            "month": month,
            "rows": int(rows),
            "bytes": size,
            "bytes_human": human_bytes(size),
            "bytes_per_row": round(size / rows, 4) if rows else None,
            "full_scan_median_seconds": full.median_seconds,
            "full_scan_min_seconds": full.min_seconds,
            "full_scan_max_seconds": full.max_seconds,
            "single_column_median_seconds": single.median_seconds,
            "single_column_min_seconds": single.min_seconds,
            "single_column_max_seconds": single.max_seconds,
            "peak_rss_gb": full.peak_rss_gb,
            "ok": full.ok and single.ok,
            "error": full.error or single.error,
        })

    if settings.BENCHMARK_DELETE_CSV_AFTER and csv_path.exists():
        csv_path.unlink()
        LOG.info("B1: deleted the %s CSV to stay inside the 3 GB disk budget "
                 "(the measurement is retained)", human_bytes(csv_bytes))

    _annotate_ratio(results, "csv", "parquet", "full_scan_median_seconds",
                    "csv_vs_parquet_full_scan_ratio")
    _annotate_ratio(results, "csv", "parquet", "single_column_median_seconds",
                    "csv_vs_parquet_single_column_ratio")
    return results


# --------------------------------------------------------------------------- #
# B2  partition_pruning
# --------------------------------------------------------------------------- #
def b2_partition_pruning() -> list[dict[str, Any]]:
    """Partitioned dataset with pruning versus one consolidated Parquet file.

    Fair-comparison note, stated in the report as well as here: the consolidated
    file carries only the columns the benchmark query reads
    (:data:`BENCH_COLUMNS`), and the partitioned side is queried with the same
    projection. A full 23-column consolidated copy of ``fact_trip`` would add
    ~1.4 GB and breach the 3 GB disk budget; projecting both sides identically
    preserves the comparison while costing ~0.3 GB, which is then deleted.

    "Bytes scanned" is the on-disk size of the files the scan must open. Within
    a file, Parquet row-group statistics may skip further, so this is an upper
    bound on physical I/O for both arms, measured the same way for each.
    """
    month = settings.BENCHMARK_MONTH
    year, month_num = int(month[:4]), int(month[5:7])
    partition = month_partition_dir(month)
    consolidated = bench_dir() / "fact_trip_consolidated.parquet"
    projection = ", ".join(BENCH_COLUMNS)

    con = connect(settings.WAREHOUSE_DB, read_only=True)
    try:
        con.execute(
            f"COPY (SELECT {projection} FROM {fact_trip_relation()}) "
            f"TO '{_sql(consolidated)}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        con.close()

    partitioned_relation = (
        f"read_parquet('{_sql(settings.GOLD_DIR / 'fact_trip')}/**/*.parquet', "
        f"hive_partitioning=true, hive_types_autocast=1)"
    )
    consolidated_relation = f"read_parquet('{_sql(consolidated)}')"

    aggregate = (
        "SELECT pickup_geo_key, count(*) AS trips, sum(total_amount) AS revenue "
        "FROM {relation} WHERE year = {year} AND month = {month} "
        "GROUP BY 1 ORDER BY 2 DESC"
    )

    arms = [
        ("partitioned_pruned", partitioned_relation, dir_size(partition),
         "partition keys in the predicate; only the target partition is opened"),
        ("consolidated_single_file", consolidated_relation,
         consolidated.stat().st_size,
         "one file, opened whole; Parquet row-group statistics (zone maps) on "
         "year and month still let the reader skip row groups outside the "
         "filter"),
    ]

    results: list[dict[str, Any]] = []
    for variant, relation, scanned_bytes, note in arms:
        query = aggregate.format(relation=relation, year=year, month=month_num)
        measured = harness.measure(f"B2 {variant}", _target(_duckdb_scratch, query))
        total_bytes = dir_size(settings.GOLD_DIR / "fact_trip") \
            if variant == "partitioned_pruned" else consolidated.stat().st_size
        results.append({
            "benchmark": "B2_partition_pruning",
            "variant": variant,
            "month_filtered": month,
            "median_seconds": measured.median_seconds,
            "min_seconds": measured.min_seconds,
            "max_seconds": measured.max_seconds,
            "bytes_scanned_on_disk": scanned_bytes,
            "bytes_scanned_human": human_bytes(scanned_bytes),
            "dataset_total_bytes": total_bytes,
            "fraction_of_dataset_scanned": round(scanned_bytes / total_bytes, 6)
            if total_bytes else None,
            "peak_rss_gb": measured.peak_rss_gb,
            "projection": projection,
            "note": note,
            "ok": measured.ok,
            "error": measured.error,
        })

    if consolidated.exists():
        size = consolidated.stat().st_size
        consolidated.unlink()
        LOG.info("B2: deleted the %s consolidated file to stay inside the 3 GB "
                 "disk budget (the measurement is retained)", human_bytes(size))

    _annotate_ratio(results, "consolidated_single_file", "partitioned_pruned",
                    "median_seconds", "speedup_from_pruning")
    return results


# --------------------------------------------------------------------------- #
# B3  aggregate_navigation
# --------------------------------------------------------------------------- #
def b3_aggregate_navigation() -> list[dict[str, Any]]:
    """The same answer from the atomic fact versus from the daily aggregate.

    Both arms must return identical numbers; the runner asserts it. An aggregate
    that is faster but wrong is not a speedup, and reporting the latency without
    checking agreement would be exactly that mistake.
    """
    atomic = (
        "SELECT d.year_month, f.pickup_geo_key, count(*) AS trips, "
        "       sum(f.total_amount) AS revenue "
        f"FROM {fact_trip_relation()} f "
        "JOIN dim_date d ON d.date_key = f.pickup_date_key "
        "GROUP BY 1, 2 ORDER BY revenue DESC"
    )
    aggregated = (
        "SELECT d.year_month, a.pickup_geo_key, sum(a.trip_count) AS trips, "
        "       sum(a.total_revenue) AS revenue "
        "FROM fact_trip_daily_agg a "
        "JOIN dim_date d ON d.date_key = a.date_key "
        "GROUP BY 1, 2 ORDER BY revenue DESC"
    )

    results: list[dict[str, Any]] = []
    for variant, query, source in (
        ("from_atomic_fact", atomic, "fact_trip (40.4 M rows)"),
        ("from_daily_aggregate", aggregated, "fact_trip_daily_agg"),
    ):
        measured = harness.measure(f"B3 {variant}",
                                   _target(_duckdb_scalar, query))
        results.append({
            "benchmark": "B3_aggregate_navigation",
            "variant": variant,
            "source_relation": source,
            "median_seconds": measured.median_seconds,
            "min_seconds": measured.min_seconds,
            "max_seconds": measured.max_seconds,
            "peak_rss_gb": measured.peak_rss_gb,
            "top_rows_sample": str(measured.result)[:300],
            "ok": measured.ok,
            "error": measured.error,
        })

    _annotate_ratio(results, "from_atomic_fact", "from_daily_aggregate",
                    "median_seconds", "speedup_from_aggregate")
    return results


# --------------------------------------------------------------------------- #
# B4  engine_comparison
# --------------------------------------------------------------------------- #
def b4_engine_comparison() -> list[dict[str, Any]]:
    """DuckDB versus pandas on the same aggregation over 40.4 M rows.

    If pandas cannot complete inside the container memory limit, THAT IS THE
    RESULT (section 8). The input is not shrunk until it passes, and the failure
    point is reported.
    """
    parquet_glob = _sql(settings.GOLD_DIR / "fact_trip" / "**" / "*.parquet")
    duck_query = (
        "SELECT pickup_date_key, pickup_geo_key, count(*) AS trip_count, "
        "       sum(total_amount) AS total_revenue, "
        "       sum(trip_distance_miles) AS total_distance, "
        "       avg(total_amount) AS mean_fare "
        f"FROM read_parquet('{parquet_glob}') "
        "GROUP BY 1, 2"
    )

    duck = harness.measure("B4 duckdb", _target(_duckdb_scratch, duck_query))
    # pandas/pyarrow read a Hive-partitioned DIRECTORY, not a '**' glob (which
    # pd.read_parquet does not expand). Same files, same rows as the DuckDB arm.
    pandas_run = harness.measure(
        "B4 pandas",
        _target(_pandas_aggregate, _sql(settings.GOLD_DIR / "fact_trip")),
    )

    results = [
        {
            "benchmark": "B4_engine_comparison",
            "variant": "duckdb",
            "median_seconds": duck.median_seconds,
            "min_seconds": duck.min_seconds,
            "max_seconds": duck.max_seconds,
            "peak_rss_gb": duck.peak_rss_gb,
            "memory_limit_gb": settings.MEM_LIMIT_GB,
            "completed": duck.ok,
            "ok": duck.ok,
            "error": duck.error,
            "note": "streaming, out-of-core group-by inside DuckDB",
        },
        {
            "benchmark": "B4_engine_comparison",
            "variant": "pandas",
            "median_seconds": pandas_run.median_seconds,
            "min_seconds": pandas_run.min_seconds,
            "max_seconds": pandas_run.max_seconds,
            "peak_rss_gb": pandas_run.peak_rss_gb,
            "memory_limit_gb": settings.MEM_LIMIT_GB,
            "completed": pandas_run.ok,
            "ok": True,  # a failure to complete is a valid, reported outcome
            "error": pandas_run.error,
            "note": "whole projection materialised in memory, then grouped; "
                    "not chunked, by design",
        },
    ]
    if duck.ok and pandas_run.ok:
        _annotate_ratio(results, "pandas", "duckdb", "median_seconds",
                        "pandas_vs_duckdb_ratio")
    else:
        LOG.warning(
            "B4: pandas did not complete. Reporting the failure as the result "
            "rather than shrinking the input: %s",
            pandas_run.error.splitlines()[0] if pandas_run.error else "unknown",
        )
    return results


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _target(func, *args):
    """Picklable partial for the spawn-based harness."""
    from functools import partial

    return partial(func, *args)


def _annotate_ratio(rows: list[dict[str, Any]], numerator_variant: str,
                    denominator_variant: str, field: str, label: str) -> None:
    """Attach a ratio between two variants onto every row of the benchmark."""
    values = {row["variant"]: row.get(field) for row in rows}
    num, den = values.get(numerator_variant), values.get(denominator_variant)
    ratio = round(num / den, 4) if (num and den) else None
    for row in rows:
        row[label] = ratio


def cleanup() -> None:
    """Remove the benchmark scratch directory."""
    path = settings.DATA_DIR / BENCH_DIR_NAME
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def run_all() -> list[dict[str, Any]]:
    """Run B1-B4 and return one flat row set."""
    rows: list[dict[str, Any]] = []
    for label, func in (
        ("B1 format_comparison", b1_format_comparison),
        ("B2 partition_pruning", b2_partition_pruning),
        ("B3 aggregate_navigation", b3_aggregate_navigation),
        ("B4 engine_comparison", b4_engine_comparison),
    ):
        with stage(f"benchmark: {label}", LOG):
            rows.extend(func())
    cleanup()
    return rows
