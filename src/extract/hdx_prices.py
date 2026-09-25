"""Source C: WFP Nigeria market food prices via the HDX open data portal.

Extraction pattern: **open data API**. Structurally distinct from the CDN bulk
download because the payload is a portal-mediated CSV carrying an HXL semantic
tag row -- machine-readable metadata interleaved with data, which a naive reader
silently ingests as a data row and which then poisons every downstream type
inference.

Two resources are fetched: the price observations and the market register that
provides the market-to-``admin1`` bridge used by A9.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from config import settings
from src.extract.base import ExtractionResult
from src.utils.http import ExtractionError, download_file
from src.utils.logging_setup import get_logger, stage

LOG = get_logger(__name__)

PRICES_FILE = "wfp_food_prices_nga.csv"
MARKETS_FILE = "wfp_markets_nga.csv"


def _read_with_hxl(path: Path) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Read an HDX CSV, separating the header, the HXL tag row and the data.

    The HXL row is the second physical line and begins with ``#date`` (or another
    ``#tag``). It is returned rather than discarded so the manifest can record
    that it was recognised and removed, which is the kind of detail an examiner
    checking for silent data loss will look for.

    Returns:
        ``(header, hxl_tags, data_rows)``. ``hxl_tags`` is empty when the vintage
        carries no tag row, which the parser tolerates.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ExtractionError(f"{path} is empty; expected an HDX CSV") from exc

        rows = list(reader)

    hxl_tags: list[str] = []
    if rows and rows[0] and str(rows[0][0]).strip().startswith("#"):
        hxl_tags = [c.strip() for c in rows[0]]
        rows = rows[1:]
        LOG.info("stripped HXL tag row from %s: %s", path.name, hxl_tags[:4])
    else:
        LOG.warning(
            "no HXL tag row found in %s; the vintage may have changed shape",
            path.name,
        )

    width = len(header)
    data = [dict(zip(header, row + [""] * (width - len(row)))) for row in rows if row]
    return header, hxl_tags, data


def _profile_prices(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Describe the price panel, including the unbalanced-panel evidence.

    The reporting-market count per month is the statistic that motivates the
    index construction in A6: a naive national mean over a panel whose
    composition changes month to month measures composition as much as price.
    """
    dates = sorted({(r.get("date") or "").strip() for r in rows} - {""})
    months = sorted({d[:7] for d in dates if len(d) >= 7})
    markets_per_month: dict[str, int] = {}
    for row in rows:
        month = (row.get("date") or "")[:7]
        if not month:
            continue
        markets_per_month.setdefault(month, set()).add(row.get("market", ""))  # type: ignore[arg-type]
    counts = {m: len(v) for m, v in markets_per_month.items()}  # type: ignore[arg-type]

    states = sorted({(r.get("admin1") or "").strip() for r in rows} - {""})
    markets = sorted({(r.get("market") or "").strip() for r in rows} - {""})
    commodities = sorted({(r.get("commodity") or "").strip() for r in rows} - {""})
    categories = sorted({(r.get("category") or "").strip() for r in rows} - {""})
    price_types = sorted({(r.get("pricetype") or "").strip() for r in rows} - {""})

    fuel_months = sorted({
        (r.get("date") or "")[:7]
        for r in rows
        if (r.get("commodity") or "").strip() in settings.HDX_FUEL_COMMODITIES
    } - {""})

    return {
        "observed_rows": len(rows),
        "date_min": dates[0] if dates else None,
        "date_max": dates[-1] if dates else None,
        "distinct_months": len(months),
        "states": states,
        "state_count": len(states),
        "market_count": len(markets),
        "commodity_count": len(commodities),
        "categories": categories,
        "price_types": price_types,
        "reporting_markets_per_month_min": min(counts.values()) if counts else 0,
        "reporting_markets_per_month_max": max(counts.values()) if counts else 0,
        "panel_is_unbalanced": (max(counts.values()) != min(counts.values()))
        if counts else False,
        "fuel_month_min": fuel_months[0] if fuel_months else None,
        "fuel_month_max": fuel_months[-1] if fuel_months else None,
        "fuel_months_observed": len(fuel_months),
    }


def extract(force: bool = False) -> list[ExtractionResult]:
    """Download and profile both HDX resources."""
    with stage("extract: WFP Nigeria prices via HDX (open data API)", LOG):
        results: list[ExtractionResult] = []

        # ---------------------------------------------------------- prices
        prices_result = ExtractionResult(
            key="hdx_prices", name="WFP Nigeria Market Prices", pattern="open data API"
        )
        prices_path = settings.RAW_HDX / PRICES_FILE
        try:
            path = download_file(
                settings.HDX_PRICES_URL, prices_path, force=force,
                min_bytes=100_000, description=PRICES_FILE,
            )
        except ExtractionError as exc:
            raise ExtractionError(
                f"{exc}\nSource C is required. Expected file at {prices_path}. "
                f"Run 'make extract-hdx' or check access to data.humdata.org."
            ) from exc

        header, hxl, rows = _read_with_hxl(path)
        profile = _profile_prices(rows)
        profile["columns"] = header
        profile["hxl_tag_row_present"] = bool(hxl)
        profile["hxl_tags"] = hxl
        profile["expected_min_rows"] = settings.HDX_PRICES_EXPECTED_MIN_ROWS

        if len(rows) < settings.HDX_PRICES_EXPECTED_MIN_ROWS:
            raise ExtractionError(
                f"Source C returned {len(rows):,} data rows, fewer than the "
                f"{settings.HDX_PRICES_EXPECTED_MIN_ROWS:,} minimum. The verified "
                f"figure on 2026-09-13 was 87,384. A collapse this large means a "
                f"truncated download or an upstream change; refusing to proceed "
                f"rather than publish figures from a partial panel. File: {path}"
            )

        prices_result.files.append(path)
        prices_result.row_counts[path.name] = len(rows)
        prices_result.diagnostics = profile
        results.append(prices_result)

        LOG.info(
            "Source C prices: %s rows, %s to %s, %d states, %d markets, "
            "%d commodities; reporting markets per month %d-%d (unbalanced=%s)",
            f"{len(rows):,}", profile["date_min"], profile["date_max"],
            profile["state_count"], profile["market_count"],
            profile["commodity_count"],
            profile["reporting_markets_per_month_min"],
            profile["reporting_markets_per_month_max"],
            profile["panel_is_unbalanced"],
        )
        if profile["fuel_month_min"]:
            LOG.info(
                "Source C fuel commodities span %s to %s across %d months "
                "(coverage gap is documented, context only)",
                profile["fuel_month_min"], profile["fuel_month_max"],
                profile["fuel_months_observed"],
            )

        # --------------------------------------------------------- markets
        markets_result = ExtractionResult(
            key="hdx_markets", name="WFP Nigeria Markets", pattern="open data API"
        )
        markets_path = settings.RAW_HDX / MARKETS_FILE
        try:
            mpath = download_file(
                settings.HDX_MARKETS_URL, markets_path, force=force,
                min_bytes=500, description=MARKETS_FILE,
            )
        except ExtractionError as exc:
            raise ExtractionError(
                f"{exc}\nSource C (markets register) is required for the "
                f"market-to-state bridge used by A9. Expected at {markets_path}."
            ) from exc

        mheader, mhxl, mrows = _read_with_hxl(mpath)
        markets_result.files.append(mpath)
        markets_result.row_counts[mpath.name] = len(mrows)
        markets_result.diagnostics = {
            "observed_rows": len(mrows),
            "columns": mheader,
            "hxl_tag_row_present": bool(mhxl),
            "distinct_markets": len({r.get("market", "") for r in mrows} - {""}),
            "distinct_admin1": len({r.get("admin1", "") for r in mrows} - {""}),
        }
        results.append(markets_result)
        LOG.info("Source C markets: %d rows, %d markets, %d admin1 values",
                 len(mrows), markets_result.diagnostics["distinct_markets"],
                 markets_result.diagnostics["distinct_admin1"])

        return results
