"""Measurement harness shared by benchmarks B1-B4.

Two things this module exists to get right, because the benchmark chapter is
worthless if either is fudged:

**Repeats and medians.** Every measurement runs ``BENCHMARK_REPEATS`` times and
reports the median, plus the min and max so the spread is visible. A single
timing on a laptop under Docker Desktop is noise.

**Peak memory.** Peak resident set size is read from ``/proc/self/status``
(``VmHWM``) *inside a dedicated subprocess*, so each measurement reports its own
high-water mark rather than the parent's cumulative one. A measurement that dies
on memory reports the failure as its result -- it is never retried at a smaller
size to make it succeed (section 8, B4).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from config import settings
from src.utils.logging_setup import get_logger

LOG = get_logger(__name__)

#: Force a start method that works identically on Linux containers regardless of
#: how the parent was launched.
_CTX = mp.get_context("spawn")


def peak_rss_bytes() -> int | None:
    """Peak resident set size of the current process, from /proc/self/status.

    Returns ``None`` on platforms without procfs, in which case the benchmark
    reports memory as unavailable rather than guessing.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


@dataclass
class Measurement:
    """One timed, memory-profiled execution of a callable."""

    label: str
    ok: bool
    seconds: float | None
    peak_rss_bytes: int | None
    result: Any = None
    error: str = ""

    @property
    def peak_rss_gb(self) -> float | None:
        if self.peak_rss_bytes is None:
            return None
        return round(self.peak_rss_bytes / 1024 ** 3, 4)


@dataclass
class RepeatedMeasurement:
    """A set of repeats, summarised by median as section 8 requires."""

    label: str
    runs: list[Measurement] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.runs) and all(run.ok for run in self.runs)

    @property
    def timings(self) -> list[float]:
        return [r.seconds for r in self.runs if r.ok and r.seconds is not None]

    @property
    def median_seconds(self) -> float | None:
        values = sorted(self.timings)
        if not values:
            return None
        middle = len(values) // 2
        if len(values) % 2:
            return round(values[middle], 6)
        return round((values[middle - 1] + values[middle]) / 2.0, 6)

    @property
    def min_seconds(self) -> float | None:
        return round(min(self.timings), 6) if self.timings else None

    @property
    def max_seconds(self) -> float | None:
        return round(max(self.timings), 6) if self.timings else None

    @property
    def peak_rss_bytes(self) -> int | None:
        values = [r.peak_rss_bytes for r in self.runs if r.peak_rss_bytes is not None]
        return max(values) if values else None

    @property
    def peak_rss_gb(self) -> float | None:
        value = self.peak_rss_bytes
        return round(value / 1024 ** 3, 4) if value is not None else None

    @property
    def error(self) -> str:
        for run in self.runs:
            if not run.ok:
                return run.error
        return ""

    @property
    def result(self) -> Any:
        for run in self.runs:
            if run.ok:
                return run.result
        return None


def _child(target: Callable[[], Any], queue: "mp.Queue[Any]") -> None:
    """Subprocess body: run the callable, report time, peak RSS and outcome."""
    started = time.perf_counter()
    try:
        value = target()
        elapsed = time.perf_counter() - started
        queue.put({
            "ok": True, "seconds": elapsed, "peak": peak_rss_bytes(),
            "result": value, "error": "",
        })
    except MemoryError:
        queue.put({
            "ok": False, "seconds": time.perf_counter() - started,
            "peak": peak_rss_bytes(), "result": None,
            "error": "MemoryError: the engine could not complete this workload "
                     "inside the container memory limit",
        })
    except BaseException as exc:  # noqa: BLE001 - the failure IS the result
        queue.put({
            "ok": False, "seconds": time.perf_counter() - started,
            "peak": peak_rss_bytes(), "result": None,
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
        })


def measure_once(label: str, target: Callable[[], Any],
                 timeout_seconds: int = 1800) -> Measurement:
    """Run ``target`` in a subprocess, returning its timing and peak memory.

    An out-of-memory kill by the kernel (exit code -9) is reported as a failed
    measurement with an explanatory message, not as a crash of the pipeline.
    """
    queue: "mp.Queue[Any]" = _CTX.Queue()
    process = _CTX.Process(target=_child, args=(target, queue))
    started = time.perf_counter()
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join(10)
        return Measurement(label, False, time.perf_counter() - started, None,
                           error=f"timed out after {timeout_seconds}s")

    if queue.empty():
        code = process.exitcode
        oom = code == -9
        return Measurement(
            label, False, time.perf_counter() - started, None,
            error=(
                "process was killed by the kernel out-of-memory killer "
                f"(exit {code}); the workload does not fit in the "
                f"{settings.MEM_LIMIT_GB} GB container limit"
                if oom else f"process exited with code {code} and no result"
            ),
        )

    payload = queue.get()
    return Measurement(
        label=label, ok=payload["ok"], seconds=payload["seconds"],
        peak_rss_bytes=payload["peak"], result=payload["result"],
        error=payload["error"],
    )


def measure(label: str, target: Callable[[], Any],
            repeats: int | None = None,
            timeout_seconds: int = 1800) -> RepeatedMeasurement:
    """Run ``target`` ``repeats`` times and summarise.

    Stops early on the first failure: repeating an out-of-memory failure three
    times tells the reader nothing new and costs several minutes.
    """
    repeats = repeats or settings.BENCHMARK_REPEATS
    repeated = RepeatedMeasurement(label=label)
    for attempt in range(1, repeats + 1):
        run = measure_once(f"{label} (run {attempt})", target, timeout_seconds)
        repeated.runs.append(run)
        if not run.ok:
            LOG.warning("%s FAILED on run %d: %s", label, attempt,
                        run.error.splitlines()[0] if run.error else "unknown")
            break
        LOG.info("%s run %d/%d: %.4fs, peak RSS %s", label, attempt, repeats,
                 run.seconds or 0.0,
                 f"{run.peak_rss_gb} GB" if run.peak_rss_gb else "unavailable")
    if repeated.ok:
        LOG.info("%s median %.4fs (min %.4fs, max %.4fs), peak RSS %s",
                 label, repeated.median_seconds or 0.0, repeated.min_seconds or 0.0,
                 repeated.max_seconds or 0.0,
                 f"{repeated.peak_rss_gb} GB" if repeated.peak_rss_gb else "n/a")
    return repeated


def environment() -> dict[str, Any]:
    """Declared and observed resource limits, recorded with every benchmark."""
    return {
        "cpu_limit_declared": settings.CPU_LIMIT,
        "memory_limit_gb_declared": settings.MEM_LIMIT_GB,
        "duckdb_threads": settings.DUCKDB_THREADS,
        "duckdb_memory_limit": settings.DUCKDB_MEMORY_LIMIT,
        "os_cpu_count": os.cpu_count(),
        "repeats": settings.BENCHMARK_REPEATS,
        "summary_statistic": "median of repeats",
    }
