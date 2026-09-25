"""Shared pytest fixtures.

Tests that need the warehouse open it **read-only**, so a test run cannot
corrupt a build and can run alongside one. Tests that need the warehouse but
find it absent skip with a message naming the stage that would produce it,
rather than failing for an environmental reason.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from config import settings


@pytest.fixture(scope="session")
def warehouse_path() -> Path:
    """Path to the built warehouse, skipping the test if it does not exist."""
    if not settings.WAREHOUSE_DB.exists():
        pytest.skip(
            f"{settings.WAREHOUSE_DB} not found. Run 'make facts' (or "
            f"'make run') before the tests that query the warehouse."
        )
    return settings.WAREHOUSE_DB


@pytest.fixture(scope="session")
def con(warehouse_path: Path) -> duckdb.DuckDBPyConnection:
    """Read-only connection to the built warehouse."""
    connection = duckdb.connect(str(warehouse_path), read_only=True)
    try:
        yield connection
    finally:
        connection.close()


@pytest.fixture()
def memory_con() -> duckdb.DuckDBPyConnection:
    """A scratch in-memory database for tests that build their own fixtures."""
    connection = duckdb.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def require_table(connection: duckdb.DuckDBPyConnection, name: str) -> None:
    """Skip a test when a relation it needs has not been built."""
    found = connection.execute(
        "SELECT count(*) FROM (SELECT table_name AS n FROM duckdb_tables() "
        "UNION ALL SELECT view_name FROM duckdb_views()) WHERE n = ?",
        [name],
    ).fetchone()[0]
    if not found:
        pytest.skip(f"relation {name} not present; build it before running "
                    f"this test")
