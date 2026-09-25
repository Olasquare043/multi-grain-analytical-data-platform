"""Common shape for all four extraction patterns.

Section 3 closes by requiring the four structurally distinct extraction patterns
-- bulk binary download, open data API, REST/JSON, and HTML scrape with
heterogeneous Excel parsing -- to stay *structurally parallel*, so the ETL
chapter can compare them fairly rather than comparing four different coding
styles.

The contract every extractor honours:

* a module-level ``extract()`` returning an :class:`ExtractionResult`
* landing artefacts under ``data/raw/<source>/`` and nothing else
* caching by existence, so a warm rerun performs no network I/O
* required sources raise :class:`~src.utils.http.ExtractionError` on failure;
  optional sources return ``skipped=True`` with a reason
* every file that lands is described with size, checksum and row count
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.utils.io_utils import describe_input
from src.utils.logging_setup import get_logger

LOG = get_logger(__name__)


@dataclass
class ExtractionResult:
    """Outcome of one extraction, in the form the manifest and reports consume."""

    key: str
    name: str
    pattern: str
    files: list[Path] = field(default_factory=list)
    row_counts: dict[str, int] = field(default_factory=dict)
    ok: bool = True
    skipped: bool = False
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_rows(self) -> int:
        """Sum of rows across every landed artefact."""
        return sum(self.row_counts.values())

    def manifest_records(self) -> list[dict[str, Any]]:
        """Per-file provenance records for ``docs/run_manifest.json``."""
        records = []
        for path in self.files:
            rows = self.row_counts.get(Path(path).name)
            records.append(describe_input(Path(path), rows))
        return records

    def summary_line(self) -> str:
        """One-line human summary used in the console log and the run summary."""
        if self.skipped:
            return f"{self.key:<12} SKIPPED  ({self.reason})"
        return (
            f"{self.key:<12} OK       {len(self.files):>2} file(s), "
            f"{self.total_rows:,} row(s)  [{self.pattern}]"
        )


def compare_expected(
    label: str,
    observed: dict[str, int],
    expected: dict[str, int],
) -> dict[str, Any]:
    """Compare observed row counts against verified figures.

    A discrepancy is *reported*, never suppressed and never corrected by
    adjusting the expectation (constraint 2.5). The published counts in the
    specification were verified live; if the upstream file has since been
    revised, the pipeline must say so loudly and let a human decide.

    Returns:
        A diagnostics dict with per-key deltas and an overall match flag.
    """
    deltas: dict[str, dict[str, int]] = {}
    for key, expected_rows in expected.items():
        actual = observed.get(key)
        if actual is None:
            deltas[key] = {"expected": expected_rows, "observed": -1, "delta": 0,
                           "status": "missing"}
        elif actual != expected_rows:
            deltas[key] = {"expected": expected_rows, "observed": actual,
                           "delta": actual - expected_rows, "status": "mismatch"}

    matched = not deltas
    if matched:
        LOG.info("%s: all %d row-count assertions matched the verified figures",
                 label, len(expected))
    else:
        for key, detail in deltas.items():
            LOG.warning(
                "%s ROW COUNT DISCREPANCY for %s: expected %s, observed %s (delta %+d)",
                label, key, f"{detail['expected']:,}",
                f"{detail['observed']:,}" if detail["observed"] >= 0 else "absent",
                detail["delta"],
            )
    return {
        "assertions_checked": len(expected),
        "all_matched": matched,
        "discrepancies": deltas,
    }
