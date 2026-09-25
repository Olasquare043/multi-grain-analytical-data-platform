"""Sources A and B: NYC TLC yellow taxi trips and the taxi zone lookup.

Extraction pattern: **bulk binary download** over a CDN. This is the highest
volume and lowest ceremony of the four patterns -- 12 immutable Parquet files
and one small CSV, no pagination, no authentication, no negotiation of schema.
The engineering cost sits downstream, in the 41 M row transform, not here.

Row counts are asserted against the figures verified on 13 September 2026. A
discrepancy is surfaced, never absorbed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from config import settings
from src.extract.base import ExtractionResult, compare_expected
from src.utils.http import ExtractionError, download_file
from src.utils.io_utils import human_bytes
from src.utils.logging_setup import get_logger, stage

LOG = get_logger(__name__)


def _month_key(month: int) -> str:
    return f"{settings.NYC_TRIP_YEAR}-{month:02d}"


def _trip_path(month: int) -> Path:
    return settings.RAW_NYC / f"yellow_tripdata_{_month_key(month)}.parquet"


def _parquet_row_count(path: Path) -> int:
    """Row count from Parquet footer metadata -- no data pages are read."""
    return pq.ParquetFile(path).metadata.num_rows


def _parquet_columns(path: Path) -> list[str]:
    return list(pq.ParquetFile(path).schema_arrow.names)


def extract_trips(force: bool = False) -> ExtractionResult:
    """Download the twelve 2024 monthly trip files and verify them.

    Verification performed here, before any transformation:

    * every file present and readable as Parquet
    * row count per month equal to the verified figure
    * the 19 expected columns present (case-insensitively, because the
      ``Airport_fee`` column has drifted in case across TLC vintages)
    """
    result = ExtractionResult(
        key="nyc_trips",
        name="NYC TLC Yellow Taxi 2024",
        pattern="bulk binary download",
    )
    schema_notes: dict[str, Any] = {}

    for month in settings.NYC_TRIP_MONTHS:
        key = _month_key(month)
        url = settings.NYC_TRIP_URL_TEMPLATE.format(
            year=settings.NYC_TRIP_YEAR, month=month
        )
        destination = _trip_path(month)
        try:
            path = download_file(
                url, destination, force=force,
                min_bytes=1_000_000, description=f"yellow_tripdata_{key}",
            )
        except ExtractionError as exc:
            raise ExtractionError(
                f"{exc}\nSource A is required. Expected file at {destination}. "
                f"Run 'make extract-nyc' with network access to "
                f"d37ci6vzurychx.cloudfront.net, or place the file manually."
            ) from exc

        rows = _parquet_row_count(path)
        result.files.append(path)
        result.row_counts[path.name] = rows

        columns = _parquet_columns(path)
        lowered = {c.lower() for c in columns}
        missing = [
            c for c in settings.NYC_EXPECTED_COLUMNS if c.lower() not in lowered
        ]
        extra = [
            c for c in columns
            if c.lower() not in {e.lower() for e in settings.NYC_EXPECTED_COLUMNS}
        ]
        if missing or extra:
            schema_notes[key] = {"missing": missing, "unexpected": extra,
                                 "column_count": len(columns)}
            LOG.warning(
                "schema drift in %s: missing=%s unexpected=%s", key, missing, extra
            )
        LOG.info("verified: %s -> %s rows, %d columns, %s",
                 path.name, f"{rows:,}", len(columns), human_bytes(path.stat().st_size))

    observed_by_month = {
        _month_key(m): result.row_counts[_trip_path(m).name]
        for m in settings.NYC_TRIP_MONTHS
    }
    diagnostics = compare_expected(
        "Source A", observed_by_month, settings.NYC_EXPECTED_ROW_COUNTS
    )
    diagnostics["observed_total_rows"] = sum(observed_by_month.values())
    diagnostics["expected_total_rows"] = settings.NYC_EXPECTED_TOTAL_ROWS
    diagnostics["schema_drift"] = schema_notes
    diagnostics["data_dictionary_url"] = settings.NYC_TLC_DATA_DICTIONARY
    result.diagnostics = diagnostics

    LOG.info(
        "Source A total: %s rows across %d files (expected %s)",
        f"{diagnostics['observed_total_rows']:,}", len(result.files),
        f"{settings.NYC_EXPECTED_TOTAL_ROWS:,}",
    )
    return result


def extract_zones(force: bool = False) -> ExtractionResult:
    """Download the taxi zone lookup and verify its shape.

    265 rows and 8 boroughs were verified live. The lookup carries two
    non-geographic ids (264 'Unknown', 265 'N/A') which are retained in
    ``dim_geography`` so that trips referencing them keep a valid foreign key.
    """
    import csv

    result = ExtractionResult(
        key="nyc_zones",
        name="NYC Taxi Zone Lookup",
        pattern="bulk binary download",
    )
    destination = settings.RAW_NYC / "taxi_zone_lookup.csv"
    try:
        path = download_file(
            settings.NYC_ZONE_LOOKUP_URL, destination, force=force,
            min_bytes=1024, description="taxi_zone_lookup.csv",
        )
    except ExtractionError as exc:
        raise ExtractionError(
            f"{exc}\nSource B is required. Expected file at {destination}."
        ) from exc

    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    boroughs = sorted({(r.get("Borough") or "").strip() for r in rows} - {""})

    result.files.append(path)
    result.row_counts[path.name] = len(rows)
    result.diagnostics = {
        "observed_rows": len(rows),
        "expected_rows": settings.NYC_ZONE_EXPECTED_ROWS,
        "rows_match": len(rows) == settings.NYC_ZONE_EXPECTED_ROWS,
        "boroughs": boroughs,
        "borough_count": len(boroughs),
        "boroughs_match": len(boroughs) == settings.NYC_ZONE_EXPECTED_BOROUGHS,
    }
    if len(rows) != settings.NYC_ZONE_EXPECTED_ROWS:
        LOG.warning("Source B row count %d differs from verified %d",
                    len(rows), settings.NYC_ZONE_EXPECTED_ROWS)
    LOG.info("Source B: %d zones across %d boroughs %s",
             len(rows), len(boroughs), boroughs)
    return result


def extract(force: bool = False) -> list[ExtractionResult]:
    """Run both NYC extractions (Sources A and B)."""
    with stage("extract: NYC TLC trips + zone lookup (bulk binary download)", LOG):
        return [extract_zones(force=force), extract_trips(force=force)]
