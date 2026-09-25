"""Structured logging with explicit stage boundaries and timings.

Section 11 requires logging to both console and ``logs/pipeline.log`` with stage
boundaries and timings. Stage timings recorded here are also the source of the
``stage_timings`` block in ``docs/run_manifest.json``.
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator

from config import settings

_STAGE_TIMINGS: dict[str, float] = {}
_CONFIGURED = False

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO, quiet_libs: bool = True) -> None:
    """Attach console and file handlers exactly once per process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(settings.LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(console)
    root.addHandler(file_handler)

    if quiet_libs:
        for noisy in ("urllib3", "requests", "matplotlib", "matplotlib.font_manager",
                      "PIL", "google", "google.auth", "google.cloud"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger, configuring the root handlers on first use."""
    configure_logging()
    return logging.getLogger(name)


@contextmanager
def stage(name: str, logger: logging.Logger | None = None) -> Iterator[None]:
    """Bracket a pipeline stage with start/end log lines and record its duration.

    The duration is accumulated (not overwritten) so a stage invoked twice in one
    process reports total wall-clock time spent in it.
    """
    log = logger or get_logger("pipeline")
    log.info("=" * 78)
    log.info("STAGE START : %s", name)
    log.info("=" * 78)
    started = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        _STAGE_TIMINGS[name] = round(_STAGE_TIMINGS.get(name, 0.0) + elapsed, 3)
        log.error("STAGE FAILED: %s after %.2fs", name, elapsed)
        raise
    elapsed = time.perf_counter() - started
    _STAGE_TIMINGS[name] = round(_STAGE_TIMINGS.get(name, 0.0) + elapsed, 3)
    log.info("STAGE DONE  : %s in %.2fs", name, elapsed)


def stage_timings() -> dict[str, float]:
    """Snapshot of every stage duration recorded so far, in seconds."""
    return dict(_STAGE_TIMINGS)


def reset_stage_timings() -> None:
    """Clear recorded timings (used by tests)."""
    _STAGE_TIMINGS.clear()


class Timer:
    """Minimal wall-clock timer usable as a context manager.

    >>> with Timer() as t:
    ...     pass
    >>> t.elapsed >= 0
    True
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._started: float = 0.0

    def __enter__(self) -> "Timer":
        self._started = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._started
