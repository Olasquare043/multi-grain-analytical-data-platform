"""BigQuery access for the OPTIONAL cloud module (section 9).

Nothing in the core pipeline imports this module, and ``google-cloud-bigquery``
is imported lazily inside the functions that need it, so the core run succeeds
whether or not the cloud dependencies are installed.

Three rules are enforced here rather than trusted to the caller:

* **Credentials are never printed.** Only the configured *path* is ever logged.
* **Every query is dry-run first.** The dry run costs nothing and returns the
  bytes the real query would process; the real query only runs if that fits the
  byte budget.
* **The query cache is disabled for measurements.** A cached result bills zero
  bytes and returns instantly, which would make every latency and cost figure
  meaningless.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from config import settings
from src.utils.io_utils import human_bytes
from src.utils.logging_setup import get_logger

LOG = get_logger(__name__)

SETUP_INSTRUCTIONS = """
The cloud module is optional and the core pipeline does not need it. To enable it:
  1. Create a BigQuery sandbox project (a Google account is enough; no billing).
  2. Create a service account with the roles BigQuery User and BigQuery Data
     Viewer, and download its JSON key into secrets/.
  3. In .env set:
       GOOGLE_APPLICATION_CREDENTIALS=/app/secrets/<your-key>.json
       GCP_PROJECT_ID=<your-project-id>
  4. Run: make cloud
Credentials are mounted read-only, excluded from git and from the image, and
never printed.
"""


class CloudUnavailable(RuntimeError):
    """The module cannot run in this environment; carries instructions."""


class BudgetExceeded(RuntimeError):
    """A query would take the session past its byte allowance."""


class QueryFailed(RuntimeError):
    """BigQuery rejected a query (for example, a schema mismatch)."""


def credentials_status() -> tuple[bool, str]:
    """Check the credential configuration without ever reading the key."""
    path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if not path:
        return False, "GOOGLE_APPLICATION_CREDENTIALS is not set."
    if not Path(path).exists():
        return False, (f"GOOGLE_APPLICATION_CREDENTIALS points to {path}, which "
                       f"does not exist inside the container.")
    if not os.getenv("GCP_PROJECT_ID", ""):
        return False, "GCP_PROJECT_ID is not set."
    return True, f"credential file present at {path} (contents not read)"


def make_client() -> Any:
    """Create a BigQuery client, or raise CloudUnavailable with instructions."""
    ok, message = credentials_status()
    if not ok:
        raise CloudUnavailable(message + "\n" + SETUP_INSTRUCTIONS)
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise CloudUnavailable(
            "google-cloud-bigquery is not installed. Install the optional "
            "dependencies with: pip install -r requirements-cloud.txt\n"
            + SETUP_INSTRUCTIONS
        ) from exc
    LOG.info("BigQuery client: %s; billing project set from GCP_PROJECT_ID",
             message)
    return bigquery.Client(project=os.environ["GCP_PROJECT_ID"],
                           location=settings.BQ_LOCATION)


@dataclass
class ByteBudget:
    """Session byte allowance, capped well below the free tier.

    The allowance is the smaller of the configured session budget and
    ``BQ_ABORT_THRESHOLD_FRACTION`` of whatever free-tier headroom remains this
    month. When this month's prior usage cannot be read, the whole free tier is
    assumed unused, and the manifest says that assumption was made.
    """

    session_limit: int
    monthly_used_before: int | None = None
    spent: int = 0
    log: list[dict[str, Any]] = field(default_factory=list)

    def allowance(self) -> int:
        free_left = settings.BQ_FREE_TIER_BYTES_PER_MONTH - (self.monthly_used_before or 0)
        cap = int(settings.BQ_ABORT_THRESHOLD_FRACTION * max(free_left, 0))
        return max(min(self.session_limit, cap), 0)

    def headroom(self) -> int:
        return self.allowance() - self.spent

    def check(self, projected: int, label: str) -> None:
        if self.spent + projected > self.allowance():
            raise BudgetExceeded(
                f"'{label}' would process {human_bytes(projected)}, taking the "
                f"session to {human_bytes(self.spent + projected)} against an "
                f"allowance of {human_bytes(self.allowance())}. Aborted before "
                f"running."
            )


@dataclass
class QueryResult:
    """One measured BigQuery job."""

    label: str
    bytes_processed: int
    bytes_billed: int
    slot_ms: int | None
    seconds_to_complete: float
    server_seconds: float | None
    cache_hit: bool
    job_id: str
    frame: pd.DataFrame | None


class Runner:
    """Dry-runs, budget-checks, executes and records BigQuery queries."""

    def __init__(self, client: Any, budget: ByteBudget) -> None:
        self.client = client
        self.budget = budget

    def dry_run(self, sql: str, label: str) -> int:
        """Bytes the query would process. Free; nothing is executed."""
        from google.cloud import bigquery

        config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            job = self.client.query(sql, job_config=config)
        except Exception as exc:  # noqa: BLE001 - surfaced as QueryFailed
            raise QueryFailed(f"dry run of '{label}' rejected: "
                              f"{str(exc).splitlines()[0]}") from exc
        return int(job.total_bytes_processed or 0)

    def run(self, sql: str, label: str, fetch: bool = True) -> QueryResult:
        """Execute with the cache disabled, after a budget-checked dry run."""
        from google.cloud import bigquery

        projected = self.dry_run(sql, label)
        self.budget.check(projected, label)

        config = bigquery.QueryJobConfig(use_query_cache=False)
        started = time.perf_counter()
        try:
            job = self.client.query(sql, job_config=config)
            rows = job.result()
        except Exception as exc:  # noqa: BLE001
            raise QueryFailed(f"'{label}' failed: {str(exc).splitlines()[0]}") from exc
        seconds = time.perf_counter() - started
        frame = rows.to_dataframe(create_bqstorage_client=False) if fetch else None

        processed = int(job.total_bytes_processed or 0)
        billed = int(job.total_bytes_billed or 0)
        # Track the larger of the two, so a sandbox that reports zero billed
        # bytes still counts against the allowance.
        self.budget.spent += max(processed, billed)
        server = None
        if job.started and job.ended:
            server = (job.ended - job.started).total_seconds()
        record = {
            "label": label, "projected_bytes": projected,
            "bytes_processed": processed, "bytes_billed": billed,
            "seconds_to_complete": round(seconds, 3),
            "session_spent_bytes": self.budget.spent,
            "headroom_bytes": self.budget.headroom(),
        }
        self.budget.log.append(record)
        LOG.info("BQ %-48s %10s processed, %10s billed, %6.2fs; headroom %s",
                 label, human_bytes(processed), human_bytes(billed), seconds,
                 human_bytes(self.budget.headroom()))
        return QueryResult(
            label=label, bytes_processed=processed, bytes_billed=billed,
            slot_ms=getattr(job, "slot_millis", None),
            seconds_to_complete=seconds, server_seconds=server,
            cache_hit=bool(getattr(job, "cache_hit", False)),
            job_id=job.job_id, frame=frame,
        )


def monthly_bytes_used(runner: Runner) -> tuple[int | None, str]:
    """This month's billed query bytes, from INFORMATION_SCHEMA, if readable.

    JOBS_BY_PROJECT needs a permission the sandbox service account may lack, so
    JOBS_BY_USER (this account's own jobs) is tried next. If neither is readable
    the caller must assume the whole free tier is unused, and says so.
    """
    for view in ("JOBS_BY_PROJECT", "JOBS_BY_USER"):
        sql = (
            f"SELECT COALESCE(SUM(total_bytes_billed), 0) AS billed "
            f"FROM `region-{settings.BQ_LOCATION.lower()}`.INFORMATION_SCHEMA.{view} "
            f"WHERE creation_time >= TIMESTAMP_TRUNC(CURRENT_TIMESTAMP(), MONTH) "
            f"AND job_type = 'QUERY'"
        )
        try:
            result = runner.run(sql, f"monthly usage via {view}")
        except (QueryFailed, BudgetExceeded) as exc:
            LOG.warning("monthly usage not readable via %s: %s", view, exc)
            continue
        billed = int(result.frame["billed"].iloc[0]) if result.frame is not None else 0
        scope = ("whole project" if view == "JOBS_BY_PROJECT"
                 else "this service account's own jobs only; usage by other "
                      "principals in the project is not visible")
        return billed, f"INFORMATION_SCHEMA.{view} ({scope})"
    return None, "unavailable -- assumed zero prior usage this month"
