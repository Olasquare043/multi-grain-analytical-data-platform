"""Quality stage entrypoint: evaluate rules, report, then enforce the gate.

Order matters. The reports are written *before* the gate is enforced, so that a
blocked run still leaves behind the evidence explaining why it blocked.
"""
from __future__ import annotations

from typing import Any

import duckdb

from config import settings
from src.model import catalog
from src.quality import engine, report
from src.utils.db import connect
from src.utils.logging_setup import get_logger, stage
from src.utils.manifest import get_manifest

LOG = get_logger(__name__)


def run(con: duckdb.DuckDBPyConnection | None = None,
        enforce_gate: bool = True) -> dict[str, Any]:
    """Evaluate the rule set, write the reports and optionally enforce the gate.

    Args:
        con: Warehouse connection; opened and closed here when omitted.
        enforce_gate: When True, a ``fail``-severity breach raises
            :class:`~src.quality.engine.QualityGateFailure` after the reports
            have been written.
    """
    with stage("quality: rule evaluation and reconciliation", LOG):
        owned = con is None
        con = con or connect(settings.WAREHOUSE_DB)
        catalog.register_views(con)

        manifest = get_manifest()
        quality_run = engine.evaluate(con)
        summary = report.write_reports(con, quality_run, manifest.data)

        manifest.record_quality(summary)
        manifest.record_reconciliation({
            "nbs": summary.get("nbs_reconciliation", []),
            "all_within_tolerance":
                summary.get("nbs_reconciliation_all_within_tolerance", False),
            "per_source": summary.get("reconciliation", []),
        })
        manifest.save()

        if owned:
            con.close()

        if enforce_gate:
            engine.enforce(quality_run)
        return summary
