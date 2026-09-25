"""Declarative data quality rule engine.

Rules live in ``config/quality_rules.yaml`` as data, not code, so that adding a
check is a configuration change and the full rule set can be printed into the
paper's appendix. Each rule is compiled to a single SQL statement returning
``(rows_checked, rows_failed)``; the engine does the arithmetic, the reporting
and the severity handling.

A ``fail`` rule that fails stops the pipeline. A ``warn`` rule that fails is
recorded and the run continues -- real anomaly volumes are findings, not
embarrassments (section 6).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb
import yaml

from config import settings
from config.nigeria_states import CANONICAL_STATES
from src.utils.db import table_exists
from src.utils.logging_setup import get_logger

LOG = get_logger(__name__)

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"

#: Named value sets a ``domain`` rule may reference via ``allowed_from``.
NAMED_DOMAINS: dict[str, tuple[str, ...]] = {
    "nigeria_states": CANONICAL_STATES,
}


class QualityGateFailure(RuntimeError):
    """Raised when one or more ``fail``-severity rules did not hold."""


@dataclass
class RuleResult:
    """Outcome of evaluating one rule."""

    rule_id: str
    name: str
    layer: str
    rule_type: str
    table: str
    column: str | None
    severity: str
    rows_checked: int
    rows_failed: int
    status: str
    threshold: str = ""
    detail: str = ""
    description: str = ""
    #: Rows routed to a dimension's Unknown member. Only meaningful for
    #: referential-integrity rules, where it is graded apart from orphans.
    unknown_rows: int = 0

    @property
    def failure_pct(self) -> float:
        """Share of checked rows that failed, as a percentage."""
        if not self.rows_checked:
            return 0.0
        return round(100.0 * self.rows_failed / self.rows_checked, 6)

    def as_row(self) -> dict[str, Any]:
        """Flat record for ``outputs/quality/quality_report.csv``."""
        return {
            "rule_id": self.rule_id,
            "rule_name": self.name,
            "layer": self.layer,
            "rule_type": self.rule_type,
            "table": self.table,
            "column": self.column or "",
            "severity": self.severity,
            "threshold": self.threshold,
            "rows_checked": self.rows_checked,
            "rows_failed": self.rows_failed,
            "failure_pct": self.failure_pct,
            "unknown_member_rows": self.unknown_rows,
            "unknown_member_pct": round(
                100.0 * self.unknown_rows / self.rows_checked, 6
            ) if self.rows_checked else 0.0,
            "status": self.status,
            "detail": self.detail,
            "description": " ".join(self.description.split()),
        }


@dataclass
class QualityRun:
    """Aggregate outcome of a full rule evaluation."""

    results: list[RuleResult] = field(default_factory=list)

    @property
    def failed(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == FAIL]

    @property
    def warned(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == WARN]

    @property
    def skipped(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == SKIP]

    @property
    def passed(self) -> list[RuleResult]:
        return [r for r in self.results if r.status == PASS]

    def summary(self) -> dict[str, Any]:
        return {
            "rules_evaluated": len(self.results),
            "passed": len(self.passed),
            "warned": len(self.warned),
            "failed": len(self.failed),
            "skipped": len(self.skipped),
            "gate": "BLOCKED" if self.failed else "CLEAR",
            "failed_rules": [r.rule_id for r in self.failed],
            "warned_rules": [r.rule_id for r in self.warned],
        }


def load_rules(path: Path | None = None) -> list[dict[str, Any]]:
    """Load and lightly validate the rule file."""
    rule_path = Path(path or settings.QUALITY_RULES_FILE)
    if not rule_path.exists():
        raise FileNotFoundError(
            f"{rule_path} is missing. The quality framework is declarative and "
            f"cannot run without it; restore the file from the repository."
        )
    document = yaml.safe_load(rule_path.read_text(encoding="utf-8")) or {}
    rules = document.get("rules") or []
    seen: set[str] = set()
    for rule in rules:
        for required in ("id", "name", "type", "table", "severity"):
            if required not in rule:
                raise ValueError(f"quality rule {rule!r} is missing '{required}'")
        if rule["severity"] not in ("fail", "warn"):
            raise ValueError(
                f"rule {rule['id']} has severity {rule['severity']!r}; "
                f"only 'fail' and 'warn' are defined (section 6)"
            )
        if rule["id"] in seen:
            raise ValueError(f"duplicate quality rule id {rule['id']}")
        seen.add(rule["id"])
    LOG.info("loaded %d quality rule(s) from %s", len(rules), rule_path.name)
    return rules


def _quote_list(values: tuple[str, ...] | list[str]) -> str:
    escaped = [str(v).replace("'", "''") for v in values]
    return "(" + ", ".join(f"'{v}'" for v in escaped) + ")"


def _compile(rule: dict[str, Any]) -> tuple[str, str]:
    """Compile a rule into ``(sql, threshold_description)``.

    The SQL must return exactly one row of ``(rows_checked, rows_failed)``.
    """
    table = rule["table"]
    column = rule.get("column")
    params = rule.get("params") or {}
    rule_type = rule["type"]
    where = f"WHERE {params['predicate']}" if params.get("predicate") else ""

    if rule_type == "row_count":
        low, high = params.get("min", 0), params.get("max")
        condition = f"cnt < {low}" + (f" OR cnt > {high}" if high is not None else "")
        return (
            f"WITH c AS (SELECT count(*) AS cnt FROM {table} {where}) "
            f"SELECT cnt AS rows_checked, "
            f"CASE WHEN {condition} THEN cnt ELSE 0 END AS rows_failed FROM c",
            f"rows between {low} and {high if high is not None else 'inf'}",
        )

    if rule_type == "null_rate":
        max_pct = float(params.get("max_pct", 0.0))
        return (
            f"SELECT count(*) AS rows_checked, "
            f"count(*) FILTER (WHERE {column} IS NULL) AS rows_failed "
            f"FROM {table} {where}",
            f"null rate <= {max_pct}%",
        )

    if rule_type == "referential_integrity":
        # Two distinct conditions, deliberately not conflated:
        #   ORPHAN  -- the key holds a value that exists in no dimension row.
        #              A genuine integrity breach; always a failure.
        #   UNKNOWN -- the key holds the Unknown surrogate because the source
        #              field was absent. The Unknown member exists precisely to
        #              absorb this; it is a rate to report, not a broken join.
        # Conflating them would either hide real breaks behind a tolerance or
        # fail the build over data the source never published.
        parent = params["parent"]
        parent_key = params["parent_key"]
        return (
            f"SELECT count(*) AS rows_checked, "
            f"count(*) FILTER (WHERE p.{parent_key} IS NULL) AS rows_failed, "
            f"count(*) FILTER (WHERE c.{column} = {settings.UNKNOWN_KEY}) "
            f"       AS extra_metric "
            f"FROM {table} c "
            f"LEFT JOIN {parent} p ON p.{parent_key} = c.{column}",
            f"zero orphans; unknown-member rate <= "
            f"{params.get('max_unknown_pct', 0.0)}%",
        )

    if rule_type == "range":
        low, high = params.get("min"), params.get("max")
        clauses = []
        if low is not None:
            clauses.append(f"{column} < {low}")
        if high is not None:
            clauses.append(f"{column} > {high}")
        violation = " OR ".join(clauses) or "FALSE"
        return (
            f"SELECT count(*) AS rows_checked, "
            f"count(*) FILTER (WHERE {column} IS NOT NULL AND ({violation})) "
            f"       AS rows_failed FROM {table} {where}",
            f"{low if low is not None else '-inf'} <= {column} <= "
            f"{high if high is not None else 'inf'}",
        )

    if rule_type == "domain":
        allowed = params.get("allowed")
        if allowed is None:
            allowed = NAMED_DOMAINS[params["allowed_from"]]
        return (
            f"SELECT count(*) AS rows_checked, "
            f"count(*) FILTER (WHERE {column} IS NOT NULL "
            f"                   AND {column} NOT IN {_quote_list(allowed)}) "
            f"       AS rows_failed FROM {table} {where}",
            f"{column} in a set of {len(allowed)} allowed value(s)",
        )

    if rule_type == "expression":
        predicate = params["predicate"]
        return (
            f"SELECT count(*) AS rows_checked, "
            f"count(*) FILTER (WHERE NOT ({predicate})) AS rows_failed "
            f"FROM {table}",
            predicate,
        )

    if rule_type in ("unique", "duplicates"):
        columns = ", ".join(params["columns"])
        return (
            f"WITH g AS (SELECT {columns}, count(*) AS n FROM {table} {where} "
            f"           GROUP BY ALL) "
            f"SELECT coalesce(sum(n), 0) AS rows_checked, "
            f"       coalesce(sum(CASE WHEN n > 1 THEN n - 1 ELSE 0 END), 0) "
            f"       AS rows_failed FROM g",
            f"unique on ({columns})",
        )

    raise ValueError(f"unknown quality rule type {rule_type!r} in rule {rule['id']}")


def _status(rule: dict[str, Any], checked: int, failed: int,
            extra: int = 0) -> str:
    """Decide PASS / WARN / FAIL given the rule's own tolerance.

    Args:
        checked: rows evaluated.
        failed: rows violating the rule's hard condition.
        extra: secondary metric. For referential integrity this is the count of
            rows routed to the Unknown member, which is graded separately from
            orphans.
    """
    params = rule.get("params") or {}
    pct = (100.0 * failed / checked) if checked else 0.0

    if rule["type"] == "null_rate":
        breached = pct > float(params.get("max_pct", 0.0))
    elif rule["type"] == "referential_integrity":
        if failed > 0:
            return FAIL if rule["severity"] == "fail" else WARN
        unknown_pct = (100.0 * extra / checked) if checked else 0.0
        # Unknown-member routing above tolerance is a finding, never a blocker:
        # no row was lost, and the source simply did not publish the field.
        return WARN if unknown_pct > float(
            params.get("max_unknown_pct", 0.0)) else PASS
    elif rule["type"] == "duplicates":
        # Reported, never enforced; see the rule's own description.
        return PASS if failed == 0 else WARN
    else:
        breached = failed > 0

    if not breached:
        return PASS
    return FAIL if rule["severity"] == "fail" else WARN


def evaluate(
    con: duckdb.DuckDBPyConnection,
    rules: list[dict[str, Any]] | None = None,
    layers: tuple[str, ...] = ("silver", "gold"),
) -> QualityRun:
    """Evaluate every rule for the requested layers against the warehouse."""
    rules = rules if rules is not None else load_rules()
    run = QualityRun()

    for rule in rules:
        layer = rule.get("layer", "gold")
        if layer not in layers:
            continue

        table = rule["table"]
        # A referential-integrity rule needs its parent dimension as well as the
        # fact; either being absent means the rule cannot be evaluated, which is
        # a SKIP (nothing was built), not an execution error.
        required = [table]
        if rule["type"] == "referential_integrity":
            required.append((rule.get("params") or {}).get("parent", ""))
        absent = [name for name in required if not table_exists(con, name)]
        if absent:
            run.results.append(RuleResult(
                rule_id=rule["id"], name=rule["name"], layer=layer,
                rule_type=rule["type"], table=table, column=rule.get("column"),
                severity=rule["severity"], rows_checked=0, rows_failed=0,
                status=SKIP,
                detail=f"relation(s) {', '.join(absent)} not present",
                description=rule.get("description", ""),
            ))
            LOG.warning("rule %s skipped: relation(s) %s not present",
                        rule["id"], ", ".join(absent))
            continue

        sql, threshold = _compile(rule)
        try:
            row = con.execute(sql).fetchone()
            checked, failed = row[0], row[1]
            extra = int(row[2] or 0) if len(row) > 2 else 0
        except duckdb.Error as exc:
            run.results.append(RuleResult(
                rule_id=rule["id"], name=rule["name"], layer=layer,
                rule_type=rule["type"], table=table, column=rule.get("column"),
                severity=rule["severity"], rows_checked=0, rows_failed=0,
                status=FAIL if rule["severity"] == "fail" else WARN,
                threshold=threshold, detail=f"rule could not execute: {exc}",
                description=rule.get("description", ""),
            ))
            LOG.error("rule %s failed to execute: %s", rule["id"], exc)
            continue

        checked, failed = int(checked or 0), int(failed or 0)
        detail = ""
        if rule["type"] == "referential_integrity":
            unknown_pct = round(100.0 * extra / checked, 6) if checked else 0.0
            detail = (
                f"{failed:,} orphan(s); {extra:,} row(s) ({unknown_pct}%) routed "
                f"to the Unknown member because the source field was absent"
            )
        result = RuleResult(
            rule_id=rule["id"], name=rule["name"], layer=layer,
            rule_type=rule["type"], table=table, column=rule.get("column"),
            severity=rule["severity"], rows_checked=checked, rows_failed=failed,
            status=_status(rule, checked, failed, extra), threshold=threshold,
            detail=detail, description=rule.get("description", ""),
        )
        result.unknown_rows = extra
        run.results.append(result)

        log = LOG.info if result.status == PASS else (
            LOG.warning if result.status == WARN else LOG.error
        )
        log("%-5s %-6s %-40s %12s checked, %10s failed (%.6f%%)",
            result.rule_id, result.status, result.name,
            f"{checked:,}", f"{failed:,}", result.failure_pct)

    summary = run.summary()
    LOG.info("quality gate %s: %d passed, %d warned, %d failed, %d skipped",
             summary["gate"], summary["passed"], summary["warned"],
             summary["failed"], summary["skipped"])
    return run


def enforce(run: QualityRun) -> None:
    """Stop the pipeline if any ``fail``-severity rule did not hold."""
    if not run.failed:
        return
    lines = [
        f"  {r.rule_id} {r.name}: {r.rows_failed:,} of {r.rows_checked:,} rows "
        f"({r.failure_pct}%) violate [{r.threshold}]{' -- ' + r.detail if r.detail else ''}"
        for r in run.failed
    ]
    raise QualityGateFailure(
        "Quality gate BLOCKED. The following fail-severity rule(s) did not hold, "
        "so no figure or table has been published from this run:\n"
        + "\n".join(lines)
        + "\n\nInspect outputs/quality/quality_report.md for the full evaluation."
    )
