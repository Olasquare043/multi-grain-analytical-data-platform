"""HTTP access shared by all four extraction patterns.

The four extractors differ in *what* they fetch (bulk Parquet, CSV over an open
data portal, JSON over REST, HTML to scrape) but not in how they should behave
when the network misbehaves. Keeping retry, timeout, caching and provenance
policy in one place is what makes the four patterns comparable in the ETL
chapter (section 3, closing note).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests

from config import settings
from src.utils.io_utils import file_size, human_bytes
from src.utils.logging_setup import get_logger

LOG = get_logger(__name__)


class ExtractionError(RuntimeError):
    """Raised when a required source cannot be obtained.

    Carries an actionable message naming the file and its expected location
    (section 11), because a bare stack trace is not an acceptable failure mode
    for a pipeline whose outputs are graded.
    """


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": settings.USER_AGENT})
    return session


def _retrying(description: str, func: Any, max_retries: int | None = None) -> Any:
    """Call ``func`` with bounded exponential backoff, logging each attempt.

    ``max_retries`` lets an optional source use a shorter leash than the
    required bulk downloads, so a failing optional API cannot stall the run.
    """
    attempts = max_retries or settings.HTTP_MAX_RETRIES
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except (requests.RequestException, OSError) as exc:
            last_error = exc
            # Exponential backoff: a transient network drop needs minutes to
            # clear, not the seconds a linear schedule would give it.
            wait = min(settings.HTTP_BACKOFF_SECONDS * 2 ** (attempt - 1),
                       settings.HTTP_BACKOFF_MAX_SECONDS)
            if attempt < attempts:
                LOG.warning(
                    "attempt %d/%d failed for %s: %s -- retrying in %.1fs",
                    attempt, attempts, description, exc, wait,
                )
                time.sleep(wait)
            else:
                LOG.error(
                    "attempt %d/%d failed for %s: %s -- no retries left",
                    attempt, attempts, description, exc,
                )
    raise ExtractionError(
        f"{description}: exhausted {attempts} attempts. "
        f"Last error: {last_error}. Check network access and retry."
    )


def download_file(
    url: str,
    destination: Path,
    *,
    force: bool = False,
    min_bytes: int = 1024,
    description: str | None = None,
) -> Path:
    """Stream a URL to disk, skipping the transfer when a good copy exists.

    Caching by existence is what keeps the 20-minute runtime budget honest: the
    budget excludes downloads (constraint 2.4), and a warm rerun must not refetch
    660 MB of Parquet.

    Args:
        url: Absolute URL to fetch.
        destination: Target path; parent directories are created.
        force: Refetch even when a plausible local copy exists.
        min_bytes: A local file smaller than this is treated as a failed
            partial download and refetched.
        description: Human label used in log lines and error messages.

    Returns:
        The destination path.

    Raises:
        ExtractionError: on non-2xx responses or exhausted retries.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    label = description or destination.name

    if destination.exists() and not force and file_size(destination) >= min_bytes:
        LOG.info("cached  : %s (%s)", label, human_bytes(file_size(destination)))
        return destination

    partial = destination.with_suffix(destination.suffix + ".part")

    def _fetch() -> Path:
        # Resume from whatever a previous attempt (or a previous run) already
        # wrote, so a dropped connection costs the missing bytes, not the file.
        resume_from = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
        with _session() as session:
            with session.get(
                url, stream=True, timeout=settings.HTTP_TIMEOUT_SECONDS,
                headers=headers,
            ) as response:
                if resume_from and response.status_code == 416:
                    # Range not satisfiable: the partial is unusable. Start over.
                    partial.unlink(missing_ok=True)
                    raise OSError(f"resume rejected for {label}; restarting")
                response.raise_for_status()
                resuming = bool(resume_from) and response.status_code == 206
                if resume_from and not resuming:
                    LOG.info("server ignored the Range request for %s; "
                             "restarting from byte 0", label)
                    resume_from = 0
                elif resuming:
                    LOG.info("resuming %s from byte %s", label, f"{resume_from:,}")
                length = int(response.headers.get("Content-Length") or 0)
                total = resume_from + length if length else 0
                written = resume_from
                with open(partial, "ab" if resuming else "wb") as handle:
                    for chunk in response.iter_content(settings.HTTP_CHUNK_BYTES):
                        if chunk:
                            handle.write(chunk)
                            written += len(chunk)
                if total and written != total:
                    raise OSError(
                        f"short read for {label}: have {written} of {total} bytes"
                    )
                if written < min_bytes:
                    raise OSError(f"implausibly small response for {label}: {written} B")
        partial.replace(destination)
        return destination

    started = time.perf_counter()
    result = _retrying(f"download {label} from {url}", _fetch)
    LOG.info(
        "fetched : %s (%s in %.1fs)",
        label, human_bytes(file_size(result)), time.perf_counter() - started,
    )
    return result


def get_json(url: str, params: dict[str, Any] | None = None,
             description: str | None = None,
             max_retries: int | None = None,
             timeout: float | None = None) -> dict[str, Any]:
    """GET a JSON document with retries. Used by the REST extraction pattern.

    ``max_retries`` and ``timeout`` override the defaults for optional sources.
    """
    label = description or url

    def _fetch() -> dict[str, Any]:
        with _session() as session:
            response = session.get(
                url, params=params,
                timeout=timeout or settings.HTTP_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return response.json()

    payload = _retrying(f"GET JSON {label}", _fetch, max_retries=max_retries)
    LOG.info("fetched : JSON from %s", label)
    return payload


def get_text(url: str, description: str | None = None) -> str:
    """GET an HTML/text document with retries. Used by the scrape pattern."""
    label = description or url

    def _fetch() -> str:
        with _session() as session:
            response = session.get(url, timeout=settings.HTTP_TIMEOUT_SECONDS)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text

    text = _retrying(f"GET {label}", _fetch)
    LOG.info("fetched : %d characters of markup from %s", len(text), label)
    return text
