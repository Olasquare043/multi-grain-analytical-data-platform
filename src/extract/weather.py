"""Source E: Open-Meteo daily weather archive for New York City.

Extraction pattern: **REST/JSON**. Included to complete the comparison of
extraction patterns in the ETL chapter, not because the study depends on it: the
payload is a single JSON document of parallel arrays, keyless and uncapped.

This source is **optional and non-blocking** (section 3). Any failure warns,
marks the source skipped in the run manifest, and lets the pipeline continue.
Nothing downstream may assume ``fact_weather_daily`` exists.
"""
from __future__ import annotations

import json
from typing import Any

from config import settings
from src.extract.base import ExtractionResult
from src.utils.http import get_json
from src.utils.io_utils import write_json
from src.utils.logging_setup import get_logger, stage

LOG = get_logger(__name__)

RAW_FILE = "open_meteo_nyc_2024.json"


def _validate(payload: dict[str, Any]) -> dict[str, Any]:
    """Check the JSON document carries the daily block we asked for."""
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    expected_variables = settings.WEATHER_PARAMS["daily"].split(",")
    present = [v for v in expected_variables if v in daily]
    missing = [v for v in expected_variables if v not in daily]

    lengths = {v: len(daily.get(v) or []) for v in present}
    ragged = {v: n for v, n in lengths.items() if n != len(times)}

    return {
        "observed_days": len(times),
        "expected_days": settings.WEATHER_EXPECTED_DAYS,
        "days_match": len(times) == settings.WEATHER_EXPECTED_DAYS,
        "variables_present": present,
        "variables_missing": missing,
        "ragged_arrays": ragged,
        "date_min": times[0] if times else None,
        "date_max": times[-1] if times else None,
        "timezone": payload.get("timezone"),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "units": payload.get("daily_units", {}),
    }


def extract(force: bool = False) -> list[ExtractionResult]:
    """Fetch the daily archive, or return a skipped result on any failure."""
    with stage("extract: Open-Meteo NYC daily archive (REST/JSON, optional)", LOG):
        result = ExtractionResult(
            key="weather", name="Open-Meteo NYC daily archive", pattern="REST/JSON",
        )
        destination = settings.RAW_WEATHER / RAW_FILE

        if not settings.WEATHER_ENABLED:
            result.ok, result.skipped = True, True
            result.reason = "disabled by CSC796_WEATHER=0"
            LOG.warning("Source E skipped: %s", result.reason)
            return [result]

        if destination.exists() and not force:
            try:
                payload = json.loads(destination.read_text(encoding="utf-8"))
                LOG.info("cached  : %s", destination.name)
            except json.JSONDecodeError:
                LOG.warning("cached weather file was unreadable; refetching")
                payload = None
        else:
            payload = None

        if payload is None:
            try:
                payload = get_json(
                    settings.WEATHER_API_URL,
                    params=settings.WEATHER_PARAMS,
                    description="Open-Meteo NYC 2024 daily archive",
                    # Optional source: fail fast rather than stall the run.
                    max_retries=settings.WEATHER_MAX_RETRIES,
                    timeout=settings.WEATHER_TIMEOUT_SECONDS,
                )
                write_json(destination, payload)
            except Exception as exc:  # noqa: BLE001 - optional source, never fatal
                result.ok, result.skipped = False, True
                result.reason = f"fetch failed: {type(exc).__name__}: {exc}"
                LOG.warning(
                    "Source E is optional and failed; continuing without weather. %s",
                    result.reason,
                )
                return [result]

        diagnostics = _validate(payload)
        if diagnostics["observed_days"] == 0:
            result.ok, result.skipped = False, True
            result.reason = "response carried no daily observations"
            LOG.warning("Source E skipped: %s", result.reason)
            return [result]

        result.files.append(destination)
        result.row_counts[destination.name] = diagnostics["observed_days"]
        result.diagnostics = diagnostics

        LOG.info("Source E: %d daily record(s), %s to %s, variables=%s",
                 diagnostics["observed_days"], diagnostics["date_min"],
                 diagnostics["date_max"], diagnostics["variables_present"])
        if not diagnostics["days_match"]:
            LOG.warning("Source E day count %d differs from the verified %d",
                        diagnostics["observed_days"], diagnostics["expected_days"])
        return [result]
