"""Quality and reconciliation reporting.

Emits four artefacts:

* ``outputs/quality/quality_report.csv`` -- one row per rule
* ``outputs/quality/quality_report.md``  -- the same, readable, with the
  reconciliation table and the NBS correctness proof
* ``outputs/quality/reconciliation.csv`` -- raw rows in, rejected by reason,
  rows loaded, per source
* ``outputs/quality/nbs_reconciliation.csv`` -- assembled vs published national
  means, the pipeline's proof of correctness against an external authority

Nothing here rounds a failure away or omits an inconvenient count.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import duckdb
import pandas as pd

from config import settings
from src.quality.engine import QualityRun
from src.utils.db import sql_df, table_exists
from src.utils.io_utils import write_text
from src.utils.logging_setup import get_logger

LOG = get_logger(__name__)

QUALITY_CSV = "quality_report.csv"
QUALITY_MD = "quality_report.md"
RECONCILIATION_CSV = "reconciliation.csv"
NBS_RECONCILIATION_CSV = "nbs_reconciliation.csv"


def build_reconciliation(manifest_data: dict[str, Any]) -> pd.DataFrame:
    """Raw rows in, rows rejected by reason, rows loaded -- per source.

    Reads the counts the bronze and silver stages already recorded, so the
    reconciliation cannot disagree with what the loaders actually did.
    """
    bronze = manifest_data.get("bronze", {})
    silver = manifest_data.get("silver", {})
    gold = manifest_data.get("gold", {})
    rows: list[dict[str, Any]] = []

    trips_silver = silver.get("nyc_trips", {})
    trip_reasons = trips_silver.get("rejects_by_reason", {})
    rows.append({
        "source": "A. NYC TLC Yellow Taxi 2024",
        "pattern": "bulk binary download",
        "raw_rows_in": trips_silver.get("raw_rows", 0),
        "rows_rejected": trips_silver.get("rows_rejected", 0),
        "rows_loaded": gold.get("fact_trip", {}).get("rows", 0),
        "target_table": "fact_trip",
        "reject_reasons": "; ".join(
            f"{reason}={count:,}" for reason, count in sorted(
                trip_reasons.items(), key=lambda kv: -kv[1]
            )
        ) or "none",
    })

    zones = silver.get("nyc_zones", {})
    rows.append({
        "source": "B. NYC Taxi Zone Lookup",
        "pattern": "bulk binary download",
        "raw_rows_in": bronze.get("nyc_zones", {}).get("rows", 0),
        "rows_rejected": max(
            bronze.get("nyc_zones", {}).get("rows", 0) - zones.get("rows", 0), 0
        ),
        "rows_loaded": zones.get("rows", 0),
        "target_table": "dim_geography (zone members)",
        "reject_reasons": "none",
    })

    wfp = silver.get("wfp", {})
    rows.append({
        "source": "C. WFP Nigeria Market Prices",
        "pattern": "open data API",
        "raw_rows_in": wfp.get("rows_loaded", 0) + wfp.get("rows_rejected", 0),
        "rows_rejected": wfp.get("rows_rejected", 0),
        "rows_loaded": gold.get("fact_market_price_monthly", {}).get("rows", 0),
        "target_table": "fact_market_price_monthly",
        "reject_reasons": "; ".join(
            f"{reason}={count:,}"
            for reason, count in sorted(wfp.get("rejects_by_reason", {}).items())
        ) or "none",
    })

    nbs = silver.get("nbs", {})
    rows.append({
        "source": "D. NBS PMS Price Watch",
        "pattern": "HTML scrape + Excel parse",
        "raw_rows_in": nbs.get("rows_loaded", 0) + nbs.get("rows_rejected", 0),
        "rows_rejected": nbs.get("rows_rejected", 0),
        "rows_loaded": gold.get("fact_fuel_price_monthly", {}).get("rows", 0),
        "target_table": "fact_fuel_price_monthly",
        "reject_reasons": "; ".join(
            f"{reason}={count:,}"
            for reason, count in sorted(nbs.get("rejects_by_reason", {}).items())
        ) or "none",
    })

    weather = silver.get("weather", {})
    weather_gold = gold.get("fact_weather_daily", {})
    rows.append({
        "source": "E. Open-Meteo NYC daily archive (optional)",
        "pattern": "REST/JSON",
        "raw_rows_in": weather.get("rows", 0),
        "rows_rejected": 0,
        "rows_loaded": weather_gold.get("rows", 0)
        if weather_gold.get("status") == "loaded" else 0,
        "target_table": "fact_weather_daily",
        "reject_reasons": weather.get("reason", "none"),
    })

    frame = pd.DataFrame(rows)
    frame["retention_pct"] = [
        round(100.0 * loaded / raw, 4) if raw else None
        for loaded, raw in zip(frame["rows_loaded"], frame["raw_rows_in"])
    ]
    return frame


def build_nbs_reconciliation(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Assembled national mean vs the NBS published figure, per target month.

    This is the pipeline's external correctness proof (section 3, Source D). The
    assembled figure is computed from ``fact_fuel_price_monthly``, i.e. from the
    fully modelled data rather than from an intermediate, so the check covers the
    whole chain from Excel cell to gold fact.
    """
    if not table_exists(con, "fact_fuel_price_monthly"):
        return pd.DataFrame()

    observed = sql_df(
        con,
        """
        SELECT CAST(month_key / 100 AS INTEGER)              AS ym,
               round(avg(price_ngn), 2)                      AS assembled_mean_ngn,
               count(*)                                      AS reporting_states
        FROM fact_fuel_price_monthly
        GROUP BY 1
        """,
    )
    observed["month"] = observed["ym"].apply(
        lambda v: f"{int(v) // 100:04d}-{int(v) % 100:02d}"
    )
    lookup = observed.set_index("month")

    records = []
    for month, published in sorted(settings.NBS_RECONCILIATION_TARGETS.items()):
        if month in lookup.index:
            assembled = float(lookup.loc[month, "assembled_mean_ngn"])
            states = int(lookup.loc[month, "reporting_states"])
            delta = round(assembled - published, 4)
            within = abs(delta) <= settings.NBS_RECONCILIATION_TOLERANCE
        else:
            assembled, states, delta, within = None, 0, None, False
        records.append({
            "month": month,
            "published_national_mean_ngn": published,
            "assembled_national_mean_ngn": assembled,
            "delta": delta,
            "tolerance": settings.NBS_RECONCILIATION_TOLERANCE,
            "within_tolerance": within,
            "reporting_states": states,
        })
    return pd.DataFrame(records)


def _md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table."""
    if frame.empty:
        return "_No rows._\n"
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = [
        "| " + " | ".join(
            "" if pd.isna(v) else (f"{v:,}" if isinstance(v, (int,)) else str(v))
            for v in row
        ) + " |"
        for row in frame.itertuples(index=False)
    ]
    return "\n".join([header, divider, *body]) + "\n"


def write_reports(
    con: duckdb.DuckDBPyConnection,
    run: QualityRun,
    manifest_data: dict[str, Any],
) -> dict[str, Any]:
    """Write every quality artefact and return a summary for the manifest."""
    settings.QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    rules_frame = pd.DataFrame([r.as_row() for r in run.results])
    rules_frame.to_csv(settings.QUALITY_DIR / QUALITY_CSV, index=False)

    reconciliation = build_reconciliation(manifest_data)
    reconciliation.to_csv(settings.QUALITY_DIR / RECONCILIATION_CSV, index=False)

    nbs = build_nbs_reconciliation(con)
    if not nbs.empty:
        nbs.to_csv(settings.QUALITY_DIR / NBS_RECONCILIATION_CSV, index=False)

    summary = run.summary()
    generated = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    by_status = rules_frame.groupby("status").size().to_dict() if len(rules_frame) else {}
    nbs_all_pass = bool(len(nbs)) and bool(nbs["within_tolerance"].all())

    lines = [
        "# Data Quality Report",
        "",
        "CSC 796 Advanced Data Engineering -- multi-grain mobility and price platform.",
        "",
        f"Generated: {generated}",
        "",
        "## Gate outcome",
        "",
        f"- **Gate: {summary['gate']}**",
        f"- Rules evaluated: {summary['rules_evaluated']}",
        f"- Passed: {summary['passed']} | Warned: {summary['warned']} | "
        f"Failed: {summary['failed']} | Skipped: {summary['skipped']}",
        "",
        "A `fail`-severity rule stops the pipeline before any figure is produced. "
        "A `warn` is a finding carried into the paper, not a defect hidden from it.",
        "",
        "## External correctness proof: NBS published national means",
        "",
        "The assembled state panel is aggregated to a national mean and compared "
        "against figures published independently by the National Bureau of "
        "Statistics. This is the only check in the platform that tests the whole "
        "chain -- HTML scrape, heterogeneous Excel parse, de-duplication, "
        "conforming and dimensional load -- against an outside authority.",
        "",
        _md_table(nbs),
        "",
        f"**All targets within tolerance: {nbs_all_pass}**",
        "",
        "## Reconciliation: rows in, rows rejected, rows loaded",
        "",
        "Every row entering the platform is accounted for. Rejections are "
        "attributed to exactly one reason, so the per-reason counts sum to the "
        "rejected total.",
        "",
        _md_table(reconciliation),
        "",
        "## Rule results",
        "",
        _md_table(
            rules_frame[[
                "rule_id", "rule_name", "layer", "table", "column", "severity",
                "rows_checked", "rows_failed", "failure_pct",
                "unknown_member_pct", "status",
            ]] if len(rules_frame) else rules_frame
        ),
    ]

    if run.failed:
        lines += ["", "## Failed rules (detail)", ""]
        for result in run.failed:
            lines.append(
                f"- **{result.rule_id} {result.name}** on `{result.table}`: "
                f"{result.rows_failed:,} of {result.rows_checked:,} rows "
                f"({result.failure_pct}%) violate `{result.threshold}`. "
                f"{result.detail}"
            )

    if run.warned:
        lines += ["", "## Warnings (findings, not defects)", ""]
        for result in run.warned:
            lines.append(
                f"- **{result.rule_id} {result.name}** on `{result.table}`: "
                f"{result.rows_failed:,} of {result.rows_checked:,} rows "
                f"({result.failure_pct}%). {result.description}"
            )

    if run.skipped:
        lines += ["", "## Skipped rules", ""]
        for result in run.skipped:
            lines.append(f"- {result.rule_id} {result.name}: {result.detail}")

    write_text(settings.QUALITY_DIR / QUALITY_MD, "\n".join(lines) + "\n")

    LOG.info("wrote %s, %s, %s%s", QUALITY_CSV, QUALITY_MD, RECONCILIATION_CSV,
             f", {NBS_RECONCILIATION_CSV}" if not nbs.empty else "")

    return {
        **summary,
        "by_status": by_status,
        "nbs_reconciliation_all_within_tolerance": nbs_all_pass,
        "nbs_reconciliation": nbs.to_dict("records") if not nbs.empty else [],
        "reconciliation": reconciliation.to_dict("records"),
    }
