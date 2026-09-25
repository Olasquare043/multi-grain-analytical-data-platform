"""The declarative quality rule engine.

The engine decides whether a run's figures may be published, so its own
behaviour needs pinning down on fixtures whose answers are known by
construction -- particularly the distinction between an **orphan** foreign key
(a real integrity breach) and **Unknown-member routing** (a null source field
absorbed by design). Conflating those two either hides real breakage behind a
tolerance or fails a build over data the source never published.
"""
from __future__ import annotations

import duckdb
import pytest

from config import settings
from src.quality import engine


@pytest.fixture()
def fixture_db(memory_con: duckdb.DuckDBPyConnection) -> duckdb.DuckDBPyConnection:
    """A tiny star: one dimension with an Unknown member, one fact."""
    memory_con.execute(
        """
        CREATE TABLE dim_thing AS
        SELECT * FROM (VALUES
            (-1, 'Unknown'),
            ( 1, 'Alpha'),
            ( 2, 'Beta')
        ) AS t(thing_key, thing_name)
        """
    )
    memory_con.execute(
        """
        CREATE TABLE fact_thing AS
        SELECT * FROM (VALUES
            (1,  1, 10.0, 'x'),
            (2,  1, 20.0, 'x'),
            (3,  2, 30.0, 'y'),
            (4, -1, 40.0, 'y'),   -- routed to the Unknown member, by design
            (5, 99, 50.0, 'z'),   -- ORPHAN: 99 exists in no dimension row
            (6,  2, NULL, 'y')
        ) AS t(id, thing_key, amount, label)
        """
    )
    return memory_con


def _rule(**overrides: object) -> dict:
    rule = {
        "id": "T001", "name": "test_rule", "layer": "gold",
        "type": "row_count", "table": "fact_thing", "severity": "fail",
        "params": {},
    }
    rule.update(overrides)
    return rule


def _run_one(con: duckdb.DuckDBPyConnection, rule: dict) -> engine.RuleResult:
    result = engine.evaluate(con, [rule])
    assert len(result.results) == 1
    return result.results[0]


# ----------------------------------------------------------------- rule types
def test_row_count_within_bounds_passes(fixture_db) -> None:
    outcome = _run_one(fixture_db, _rule(params={"min": 1, "max": 10}))
    assert outcome.status == engine.PASS
    assert outcome.rows_checked == 6


def test_row_count_out_of_bounds_fails(fixture_db) -> None:
    outcome = _run_one(fixture_db, _rule(params={"min": 100}))
    assert outcome.status == engine.FAIL


def test_null_rate_respects_its_tolerance(fixture_db) -> None:
    strict = _run_one(fixture_db, _rule(
        type="null_rate", column="amount", params={"max_pct": 0.0}))
    assert strict.status == engine.FAIL
    assert strict.rows_failed == 1

    tolerant = _run_one(fixture_db, _rule(
        type="null_rate", column="amount", params={"max_pct": 50.0}))
    assert tolerant.status == engine.PASS


def test_range_rule_counts_only_non_null_violations(fixture_db) -> None:
    outcome = _run_one(fixture_db, _rule(
        type="range", column="amount", params={"min": 0, "max": 35}))
    # the 40.0 and 50.0 rows violate; the NULL row is not a violation
    assert outcome.rows_failed == 2
    assert outcome.status == engine.FAIL


def test_domain_rule_detects_an_unlisted_value(fixture_db) -> None:
    outcome = _run_one(fixture_db, _rule(
        type="domain", column="label", params={"allowed": ["x", "y"]}))
    assert outcome.rows_failed == 1  # the 'z' row
    assert outcome.status == engine.FAIL


def test_domain_rule_can_use_a_named_domain(fixture_db) -> None:
    fixture_db.execute(
        "CREATE TABLE dim_state AS SELECT * FROM (VALUES "
        "('Lagos'), ('Kano'), ('Grand Total')) AS t(state)"
    )
    outcome = _run_one(fixture_db, _rule(
        table="dim_state", type="domain", column="state",
        params={"allowed_from": "nigeria_states"}))
    assert outcome.rows_failed == 1  # 'Grand Total' is not a federating unit


def test_expression_rule_evaluates_a_predicate_per_row(fixture_db) -> None:
    outcome = _run_one(fixture_db, _rule(
        type="expression", params={"predicate": "amount IS NULL OR amount < 45"}))
    assert outcome.rows_failed == 1
    assert outcome.status == engine.FAIL


def test_unique_rule_counts_surplus_rows(fixture_db) -> None:
    clean = _run_one(fixture_db, _rule(type="unique", params={"columns": ["id"]}))
    assert clean.status == engine.PASS

    duplicated = _run_one(fixture_db, _rule(
        type="unique", params={"columns": ["label"]}))
    assert duplicated.rows_failed == 3  # x:2->1 surplus, y:3->2 surplus
    assert duplicated.status == engine.FAIL


def test_duplicates_rule_warns_but_never_blocks(fixture_db) -> None:
    """Duplicate business keys are reported, not enforced (see Q083)."""
    outcome = _run_one(fixture_db, _rule(
        type="duplicates", severity="warn", params={"columns": ["label"]}))
    assert outcome.status == engine.WARN
    assert outcome.rows_failed > 0


# ------------------------------------------------- the orphan/unknown split
def test_orphan_foreign_key_fails_regardless_of_unknown_tolerance(
    fixture_db,
) -> None:
    """An orphan is a real integrity breach and must fail even at a high
    unknown-member tolerance."""
    outcome = _run_one(fixture_db, _rule(
        type="referential_integrity", column="thing_key", severity="fail",
        params={"parent": "dim_thing", "parent_key": "thing_key",
                "max_unknown_pct": 99.0}))
    assert outcome.rows_failed == 1, "the thing_key = 99 row is an orphan"
    assert outcome.status == engine.FAIL
    assert "orphan" in outcome.detail


def test_unknown_member_routing_warns_but_does_not_fail(fixture_db) -> None:
    """A null source field absorbed by the Unknown member is a rate to report."""
    fixture_db.execute("DELETE FROM fact_thing WHERE thing_key = 99")
    outcome = _run_one(fixture_db, _rule(
        type="referential_integrity", column="thing_key", severity="fail",
        params={"parent": "dim_thing", "parent_key": "thing_key",
                "max_unknown_pct": 0.0}))
    assert outcome.rows_failed == 0, "no orphans remain"
    assert outcome.unknown_rows == 1
    assert outcome.status == engine.WARN, (
        "unknown-member routing above tolerance is a finding, never a blocker: "
        "no row was lost and the source simply did not publish the field"
    )


def test_referential_integrity_passes_when_clean(fixture_db) -> None:
    fixture_db.execute("DELETE FROM fact_thing WHERE thing_key IN (99, -1)")
    outcome = _run_one(fixture_db, _rule(
        type="referential_integrity", column="thing_key", severity="fail",
        params={"parent": "dim_thing", "parent_key": "thing_key",
                "max_unknown_pct": 0.0}))
    assert outcome.status == engine.PASS


# ------------------------------------------------------------ engine behaviour
def test_missing_relation_is_skipped_not_failed(fixture_db) -> None:
    outcome = _run_one(fixture_db, _rule(table="fact_absent"))
    assert outcome.status == engine.SKIP
    assert "not present" in outcome.detail


def test_enforce_raises_only_on_fail_severity(fixture_db) -> None:
    run = engine.evaluate(fixture_db, [
        _rule(id="T010", severity="warn", params={"min": 100}),
    ])
    engine.enforce(run)  # a warn must not raise

    run = engine.evaluate(fixture_db, [
        _rule(id="T011", severity="fail", params={"min": 100}),
    ])
    with pytest.raises(engine.QualityGateFailure) as caught:
        engine.enforce(run)
    assert "T011" in str(caught.value)


def test_shipped_rule_file_is_valid() -> None:
    """The real rule set must load, and every rule must be well formed."""
    rules = engine.load_rules()
    assert len(rules) > 40
    identifiers = [rule["id"] for rule in rules]
    assert len(identifiers) == len(set(identifiers)), "rule ids must be unique"
    for rule in rules:
        assert rule["severity"] in ("fail", "warn")
        assert rule["type"] in {
            "row_count", "null_rate", "referential_integrity", "range",
            "domain", "expression", "unique", "duplicates",
        }
        # every rule must compile to SQL without raising
        sql, threshold = engine._compile(rule)
        assert "SELECT" in sql.upper()
        assert threshold


def test_failure_pct_is_zero_when_nothing_was_checked() -> None:
    result = engine.RuleResult(
        rule_id="T", name="n", layer="gold", rule_type="row_count",
        table="t", column=None, severity="warn", rows_checked=0,
        rows_failed=0, status=engine.PASS,
    )
    assert result.failure_pct == 0.0
