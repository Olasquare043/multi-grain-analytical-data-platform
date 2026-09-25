"""DuckDB connection factory and small query helpers.

The platform has no database server. DuckDB runs in-process against Parquet on
the medallion filesystem, and ``data/warehouse.duckdb`` holds nothing but views
over that Parquet plus the small dimension tables. That separation is what lets
the gold layer stay portable (any engine can read the Parquet) while analysis
SQL still reads like it is querying a warehouse.

Tuning applied here, and what it bought, is recorded in
``outputs/benchmarks/benchmark_results.md`` (section 12).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from config import settings
from src.utils.logging_setup import get_logger

LOG = get_logger(__name__)


def connect(
    database: Path | str | None = None,
    *,
    read_only: bool = False,
    threads: int | None = None,
    memory_limit: str | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open a tuned DuckDB connection.

    Args:
        database: Path to a persistent database, or ``None`` for in-memory.
        read_only: Open the file read-only (used by concurrent benchmarks).
        threads: Override the configured thread count; benchmarks vary this.
        memory_limit: Override the configured memory ceiling.
    """
    target = ":memory:" if database is None else str(database)
    if database is not None:
        Path(database).parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(target, read_only=read_only)
    Path(settings.DUCKDB_TEMP_DIR).mkdir(parents=True, exist_ok=True)

    con.execute(f"SET threads TO {threads or settings.DUCKDB_THREADS}")
    con.execute(f"SET memory_limit = '{memory_limit or settings.DUCKDB_MEMORY_LIMIT}'")
    con.execute(f"SET temp_directory = '{settings.DUCKDB_TEMP_DIR}'")
    # Insertion order is irrelevant to a dimensional model and preserving it
    # forces DuckDB to buffer whole result sets; disabling it is the single
    # largest memory saving on the 41 M row fact build.
    con.execute(
        "SET preserve_insertion_order = "
        f"{'true' if settings.DUCKDB_PRESERVE_INSERTION_ORDER else 'false'}"
    )
    # The interactive progress bar writes ANSI control characters into
    # logs/pipeline.log, which is meant to be a readable provenance record.
    con.execute("SET enable_progress_bar = false")
    return con


def sql_df(con: duckdb.DuckDBPyConnection, query: str,
           params: list[Any] | None = None) -> pd.DataFrame:
    """Execute a query and return a pandas DataFrame."""
    return con.execute(query, params or []).df()


def scalar(con: duckdb.DuckDBPyConnection, query: str,
           params: list[Any] | None = None) -> Any:
    """Execute a query expected to yield one row and one column."""
    row = con.execute(query, params or []).fetchone()
    return None if row is None else row[0]


def table_exists(con: duckdb.DuckDBPyConnection, name: str) -> bool:
    """True when a table or view of this name is visible in the catalogue."""
    found = con.execute(
        """
        SELECT count(*)
        FROM duckdb_tables()
        WHERE table_name = ?
        UNION ALL
        SELECT count(*)
        FROM duckdb_views()
        WHERE view_name = ?
        """,
        [name, name],
    ).fetchall()
    return sum(r[0] for r in found) > 0


def row_count(con: duckdb.DuckDBPyConnection, relation: str) -> int:
    """Row count of a table, view, or a parquet glob expressed as a relation."""
    return int(scalar(con, f"SELECT count(*) FROM {relation}") or 0)


def read_sql_file(path: Path) -> str:
    """Load a ``.sql`` file from disk with an actionable error when missing."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"SQL file not found: {p}. Expected it under {settings.SQL_DIR}; "
            "the repository ships every analysis query, so a missing file means "
            "an incomplete checkout."
        )
    return p.read_text(encoding="utf-8")


def explain_bytes_read(con: duckdb.DuckDBPyConnection, query: str) -> dict[str, Any]:
    """Collect a profiled plan for a query, used by the partition-pruning benchmark.

    DuckDB does not expose a single 'bytes read' counter, so the benchmark
    reports *files touched* and *rows scanned* from the profile instead, and says
    so explicitly in the benchmark report rather than inventing a byte figure.
    """
    con.execute("PRAGMA enable_profiling = 'json'")
    con.execute(f"PRAGMA profiling_output = '{settings.DUCKDB_TEMP_DIR}/profile.json'")
    try:
        con.execute(query).fetchall()
    finally:
        con.execute("PRAGMA disable_profiling")
    profile_path = Path(settings.DUCKDB_TEMP_DIR) / "profile.json"
    if not profile_path.exists():
        return {}
    import json

    try:
        return json.loads(profile_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
