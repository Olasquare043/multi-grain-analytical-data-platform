"""The run manifest: ``docs/run_manifest.json``.

Section 10 requires a manifest written every run carrying the timestamp, input
files with row counts and checksums, rows loaded per table, quality outcomes,
stage timings and library versions. Because any stage can be rerun alone
(section 5, Makefile), the manifest is *merged* rather than overwritten: each
stage contributes its own facts and leaves the rest intact.

This file is the provenance record an examiner would use to decide whether a
number in the paper is traceable to a real input row.
"""
from __future__ import annotations

import datetime as dt
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

from config import settings
from src.utils.io_utils import describe_input, dir_size, human_bytes, write_json
from src.utils.logging_setup import get_logger, stage_timings

LOG = get_logger(__name__)

_TRACKED_LIBRARIES = (
    "duckdb", "pandas", "pyarrow", "requests", "matplotlib", "openpyxl",
    "beautifulsoup4", "lxml", "python-dateutil", "PyYAML", "numpy", "pytest",
    "google-cloud-bigquery", "db-dtypes",
)


def library_versions() -> dict[str, str]:
    """Installed version of every tracked dependency, or 'not installed'."""
    versions: dict[str, str] = {}
    for name in _TRACKED_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


class RunManifest:
    """Accumulating, mergeable provenance record for one pipeline run."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or settings.RUN_MANIFEST)
        self.data: dict[str, Any] = self._load()

    # ------------------------------------------------------------------ load
    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            try:
                existing = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(existing, dict):
                    return existing
            except json.JSONDecodeError:
                LOG.warning("existing manifest at %s was unreadable; starting fresh",
                            self.path)
        return self._skeleton()

    @staticmethod
    def _skeleton() -> dict[str, Any]:
        return {
            "project": "csc796-mobility-platform",
            "paper": (
                "Engineering a Multi-Grain Analytical Data Platform for the "
                "Economics of Movement: A Comparative Data Engineering Case Study "
                "of New York City Trip Records and Nigerian Price Statistics"
            ),
            "run_started_utc": None,
            "run_finished_utc": None,
            "environment": {},
            "inputs": {},
            "rows_loaded": {},
            "artefact_sizes": {},
            "quality": {},
            "reconciliation": {},
            "stage_timings_seconds": {},
            "skipped": {},
            "notes": [],
            "library_versions": {},
        }

    # ----------------------------------------------------------- contributors
    def start_run(self) -> None:
        """Stamp the start of a run and capture the execution environment."""
        self.data["run_started_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        # Run-scoped findings are reset at the start of a full run; stages
        # rerun individually still merge into the existing record.
        self.data["skipped"] = {}
        self.data["notes"] = []
        self.data["environment"] = {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_limit_declared": settings.CPU_LIMIT,
            "memory_limit_gb_declared": settings.MEM_LIMIT_GB,
            "duckdb_threads": settings.DUCKDB_THREADS,
            "duckdb_memory_limit": settings.DUCKDB_MEMORY_LIMIT,
            "disk_budget_gb": settings.DISK_BUDGET_GB,
            "runtime_budget_minutes": settings.RUNTIME_BUDGET_MINUTES,
        }
        self.data["library_versions"] = library_versions()
        self.save()

    def record_input(self, key: str, path: Path,
                     row_count: int | None = None, **extra: Any) -> None:
        """Record one input file with its size, checksum and row count."""
        record = describe_input(Path(path), row_count)
        record.update(extra)
        self.data.setdefault("inputs", {})[key] = record

    def record_inputs(self, key: str, records: list[dict[str, Any]]) -> None:
        """Record a collection of input files under one source key."""
        self.data.setdefault("inputs", {})[key] = records

    def record_rows(self, table: str, rows: int, **extra: Any) -> None:
        """Record rows loaded into a named table."""
        entry: dict[str, Any] = {"rows": int(rows)}
        entry.update(extra)
        self.data.setdefault("rows_loaded", {})[table] = entry

    def record_artefact_size(self, label: str, path: Path) -> None:
        """Record the on-disk footprint of a gold artefact."""
        size = dir_size(Path(path))
        self.data.setdefault("artefact_sizes", {})[label] = {
            "bytes": size,
            "bytes_human": human_bytes(size),
            "path": str(Path(path).name),
        }

    def record_quality(self, summary: dict[str, Any]) -> None:
        """Record the outcome of the quality framework."""
        self.data["quality"] = summary

    def record_reconciliation(self, summary: dict[str, Any]) -> None:
        """Record the NBS reconciliation outcome (the correctness proof)."""
        self.data["reconciliation"] = summary

    def record_skipped(self, key: str, reason: str) -> None:
        """Mark an optional component as skipped, with the reason."""
        self.data.setdefault("skipped", {})[key] = reason
        LOG.warning("marked skipped in manifest: %s -- %s", key, reason)

    def note(self, message: str) -> None:
        """Append a free-text finding worth carrying into the paper."""
        notes = self.data.setdefault("notes", [])
        if message not in notes:
            notes.append(message)

    # ------------------------------------------------------------------ save
    def finish_run(self) -> None:
        """Stamp completion, fold in stage timings and persist."""
        self.data["run_finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self.save()

    def save(self) -> Path:
        """Merge stage timings and write the manifest atomically."""
        timings = self.data.setdefault("stage_timings_seconds", {})
        timings.update(stage_timings())
        total = round(sum(timings.values()), 2)
        self.data["total_stage_seconds"] = total
        self.data["total_stage_minutes"] = round(total / 60.0, 2)
        if not self.data.get("library_versions"):
            self.data["library_versions"] = library_versions()
        return write_json(self.path, self.data)


_MANIFEST: RunManifest | None = None


def get_manifest() -> RunManifest:
    """Process-wide manifest singleton."""
    global _MANIFEST
    if _MANIFEST is None:
        _MANIFEST = RunManifest()
    return _MANIFEST
