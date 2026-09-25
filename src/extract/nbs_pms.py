"""Source D: NBS Premium Motor Spirit (petrol) Price Watch.

Extraction pattern: **HTML scrape plus heterogeneous Excel parsing**. This is the
most expensive of the four patterns per row delivered, and that asymmetry is
itself a finding for the ETL chapter: roughly a thousand state-month
observations cost more parser code than 41 million taxi trips.

Why the parser is shaped the way it is
--------------------------------------
Releases are not a stable feed. Across the catalogue the following vary:

* sheet names (``Fuel January 2025``, ``PMS_OCT_2025``, ``Sheet1``)
* whether row 0 is the ``State`` header or a title banner
* column counts (4, 7 and 10 observed) and sheet shapes ((41,7) to (62,4))
* month header formatting (real Excel dates, ``Nov-23``, ``November 2023``)
* container format (bare ``.xlsx``, or a ``.zip`` holding one ``.xlsx`` + a PDF)

So the parser never addresses cells positionally. It locates the header row by
scanning for ``State``, then keeps only those columns whose header *parses as a
month*, and unpivots. Each release carries three month columns (same month last
year, previous month, current month), which is why ~18 files yield ~28 distinct
months.

The file list is discovered from the catalogue page on every run and is never
hard-coded, so new monthly releases are picked up automatically.
"""
from __future__ import annotations

import re
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urljoin

import pandas as pd
from bs4 import BeautifulSoup

from config import settings
from config.nigeria_states import is_non_state_label, normalise_state
from src.extract.base import ExtractionResult
from src.utils.http import ExtractionError, download_file, get_text
from src.utils.logging_setup import get_logger, stage

LOG = get_logger(__name__)

EXTRACT_DIR_NAME = "extracted"

_MONTH_NAMES = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}
_MONTH_ALTERNATION = "|".join(sorted(_MONTH_NAMES, key=len, reverse=True))
_RE_MONTH_YEAR = re.compile(
    rf"\b({_MONTH_ALTERNATION})\b[\s,._/-]*'?(\d{{2}}|\d{{4}})\b", re.IGNORECASE
)
_RE_YEAR_MONTH_ISO = re.compile(r"\b(\d{4})[-/](\d{1,2})\b")
_RE_MONTH_YEAR_NUM = re.compile(r"\b(\d{1,2})[-/](\d{4})\b")
_RE_NUMERIC = re.compile(r"-?\d+(?:\.\d+)?")

#: Excel's day-zero for the 1900 date system, accounting for its leap-year bug.
_EXCEL_EPOCH = datetime(1899, 12, 30)


# --------------------------------------------------------------------------- #
# Catalogue scraping
# --------------------------------------------------------------------------- #
def discover_catalogue_files(html: str) -> list[dict[str, str]]:
    """Parse the catalogue markup into a list of downloadable data files.

    Args:
        html: Raw markup of the NBS catalogue page.

    Returns:
        One dict per data file with ``url``, ``filename`` and ``resource_id``,
        de-duplicated by URL and stable in document order.
    """
    soup = BeautifulSoup(html, "lxml")
    pattern = re.compile(settings.NBS_DOWNLOAD_HREF_PATTERN)
    found: dict[str, dict[str, str]] = {}

    for anchor in soup.find_all("a", href=True):
        href = anchor["href"].strip()
        if not pattern.search(href):
            continue
        url = urljoin(settings.NBS_BASE_URL, href)
        filename = href.rstrip("/").split("/")[-1]
        if not filename.lower().endswith(settings.NBS_ACCEPTED_EXTENSIONS):
            continue
        resource_match = re.search(r"download/(\d+)", href)
        found.setdefault(url, {
            "url": url,
            "filename": filename,
            "resource_id": resource_match.group(1) if resource_match else "",
        })

    files = list(found.values())
    LOG.info("catalogue scrape found %d data file(s) matching the download pattern",
             len(files))
    return files


def _looks_like_zip_or_office(path: Path) -> bool:
    """True when the first bytes are the ZIP magic shared by .zip and .xlsx."""
    with open(path, "rb") as handle:
        return handle.read(2) == b"PK"


def _unpack(path: Path, extract_root: Path) -> list[Path]:
    """Return the spreadsheet(s) a downloaded artefact contributes.

    A ``.zip`` release contains one workbook plus an explanatory PDF; only the
    workbook is extracted. A bare workbook is returned as-is.
    """
    if path.suffix.lower() in (".xlsx", ".xls"):
        return [path]

    workbooks: list[Path] = []
    target = extract_root / path.stem
    target.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.endswith("/"):
                    continue
                if not member.lower().endswith((".xlsx", ".xls")):
                    continue
                safe_name = Path(member).name
                destination = target / safe_name
                if not destination.exists():
                    with archive.open(member) as src, open(destination, "wb") as dst:
                        dst.write(src.read())
                workbooks.append(destination)
    except zipfile.BadZipFile as exc:
        LOG.warning("skipping %s: not a readable ZIP (%s)", path.name, exc)
        return []

    if not workbooks:
        LOG.warning("archive %s contained no workbook", path.name)
    return workbooks


# --------------------------------------------------------------------------- #
# Heterogeneous workbook parsing
# --------------------------------------------------------------------------- #
def parse_month_header(value: Any) -> str | None:
    """Parse a column header into ``YYYY-MM``, or ``None`` if it is not a month.

    Deliberately conservative: a header only counts as a month when it carries an
    explicit month name or an unambiguous numeric month/year pair. A bare year,
    a rank, or a percentage-change column must not be mistaken for data.

    >>> parse_month_header("November 2023")
    '2023-11'
    >>> parse_month_header("Nov-23")
    '2023-11'
    >>> parse_month_header("% change")
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None

    if isinstance(value, (pd.Timestamp, datetime, date)):
        return f"{value.year:04d}-{value.month:02d}"

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        serial = float(value)
        # Excel serials for 2010-2035 fall in this band; anything else is a price,
        # a rank or a percentage and must not be read as a date.
        if 40_000 <= serial <= 60_000:
            stamp = _EXCEL_EPOCH + pd.Timedelta(days=serial)
            return f"{stamp.year:04d}-{stamp.month:02d}"
        return None

    text = str(value).strip()
    if not text or text.lower() in ("nan", "none"):
        return None

    match = _RE_MONTH_YEAR.search(text)
    if match:
        month = _MONTH_NAMES[match.group(1).lower()]
        year_token = match.group(2)
        year = int(year_token) if len(year_token) == 4 else 2000 + int(year_token)
        return f"{year:04d}-{month:02d}"

    match = _RE_YEAR_MONTH_ISO.search(text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12 and 2000 <= year <= 2100:
            return f"{year:04d}-{month:02d}"

    match = _RE_MONTH_YEAR_NUM.search(text)
    if match:
        month, year = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12 and 2000 <= year <= 2100:
            return f"{year:04d}-{month:02d}"

    return None


def _to_price(value: Any) -> float | None:
    """Coerce a cell into a price, tolerating thousands separators and dashes."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return None if pd.isna(value) else float(value)
    text = str(value).strip().replace(",", "").replace(" ", "")
    if not text or text in ("-", "--", "n/a", "N/A", "nan"):
        return None
    match = _RE_NUMERIC.search(text)
    return float(match.group(0)) if match else None


#: Labels that open a commentary block *below* the state table. Anything at or
#: after one of these is narrative, not data.
_BLOCK_TERMINATOR_TOKENS = (
    "highest", "lowest", "month on month", "year on year",
    "month-on-month", "year-on-year", "source:", "note:",
)


def _is_block_terminator(label: str) -> bool:
    """True when a label opens a commentary block beneath the state table."""
    folded = str(label).strip().lower()
    return any(token in folded for token in _BLOCK_TERMINATOR_TOKENS)


def _find_header_row(grid: pd.DataFrame, scan_rows: int) -> int | None:
    """Locate the row that carries the ``State`` header.

    Scans the first ``scan_rows`` rows and returns the index of the first row
    whose leading cells contain ``state``. Returns ``None`` when the sheet has no
    recognisable header, which is how metadata and notes sheets are skipped.
    """
    limit = min(scan_rows, len(grid))
    for row_index in range(limit):
        leading = [str(v).strip().lower() for v in grid.iloc[row_index, :4].tolist()]
        if any(cell == "state" or cell.startswith("state") for cell in leading):
            return row_index
    # Fallback: a sheet whose first column is already full of state names but
    # carries no literal 'State' header.
    for row_index in range(limit):
        cell = grid.iloc[row_index, 0] if grid.shape[1] else None
        if normalise_state(cell) is not None:
            return max(row_index - 1, 0)
    return None


def parse_workbook(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Unpivot one NBS workbook into ``(month, state, price)`` observations.

    Returns:
        ``(rows, diagnostics)``. ``rows`` carry ``month``, ``state_raw``,
        ``state``, ``price_ngn``, ``source_file``, ``sheet_name`` and
        ``is_current_month``; the last marks the release's own reference month,
        which wins during de-duplication because it is the authoritative
        publication of that month rather than a restated comparison column.
    """
    diagnostics: dict[str, Any] = {
        "file": path.name, "sheets": [], "rows_emitted": 0,
        "excluded_labels": {}, "months": [],
    }
    rows: list[dict[str, Any]] = []

    try:
        sheets = pd.read_excel(path, sheet_name=None, header=None, engine="openpyxl")
    except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
        LOG.warning("could not open workbook %s: %s", path.name, exc)
        diagnostics["error"] = str(exc)
        return rows, diagnostics

    for sheet_name, grid in sheets.items():
        sheet_note: dict[str, Any] = {
            "sheet_name": sheet_name,
            "shape": list(grid.shape),
            "status": "skipped",
        }
        if grid.empty:
            diagnostics["sheets"].append(sheet_note)
            continue

        header_row = _find_header_row(grid, settings.NBS_HEADER_SCAN_ROWS)
        if header_row is None:
            sheet_note["reason"] = "no 'State' header found in the first "
            sheet_note["reason"] += f"{settings.NBS_HEADER_SCAN_ROWS} rows"
            diagnostics["sheets"].append(sheet_note)
            continue

        header = grid.iloc[header_row].tolist()
        body = grid.iloc[header_row + 1:]

        # The state column is the one headed 'State', else the first column.
        state_col = 0
        for index, cell in enumerate(header):
            if str(cell).strip().lower().startswith("state"):
                state_col = index
                break

        month_columns: dict[int, str] = {}
        for index, cell in enumerate(header):
            if index == state_col:
                continue
            month = parse_month_header(cell)
            if month:
                month_columns[index] = month

        if not month_columns:
            sheet_note["reason"] = "no column header parsed as a month"
            sheet_note["headers"] = [str(h)[:40] for h in header]
            diagnostics["sheets"].append(sheet_note)
            continue

        current_month = max(month_columns.values())
        excluded: dict[str, int] = {}
        emitted = 0
        seen_states: set[str] = set()
        truncation: dict[str, Any] | None = None

        for row_offset, (_, record) in enumerate(body.iterrows()):
            raw_state = record.iloc[state_col] if state_col < len(record) else None
            if raw_state is None or (isinstance(raw_state, float) and pd.isna(raw_state)):
                continue
            raw_label = str(raw_state).strip()
            if not raw_label:
                continue

            # Several releases append commentary blocks below the state table --
            # "STATES WITH THE HIGHEST AVERAGE PRICES" and its counterpart --
            # which repeat a subset of states under the same column headers but
            # with a different column meaning. Reading them produces silently
            # wrong values (observed: the January 2026 release placed six states'
            # 2026-01 prices into the 2025-01 column). The state table carries
            # each state exactly once, so the first repeat is the block boundary.
            candidate = normalise_state(raw_label)
            if candidate is not None and candidate in seen_states:
                truncation = {
                    "row_offset": row_offset,
                    "trigger_label": raw_label,
                    "reason": "repeated state marks the end of the state table; "
                              "trailing commentary block not parsed",
                }
                break
            if _is_block_terminator(raw_label):
                truncation = {
                    "row_offset": row_offset,
                    "trigger_label": raw_label,
                    "reason": "recognised commentary banner below the state table",
                }
                break

            state = candidate
            if state is None:
                excluded[raw_label] = excluded.get(raw_label, 0) + 1
                continue
            seen_states.add(state)

            for col_index, month in month_columns.items():
                if col_index >= len(record):
                    continue
                price = _to_price(record.iloc[col_index])
                if price is None:
                    continue
                rows.append({
                    "month": month,
                    "state_raw": raw_label,
                    "state": state,
                    "price_ngn": price,
                    "source_file": path.name,
                    "sheet_name": str(sheet_name),
                    "is_current_month": month == current_month,
                })
                emitted += 1

        sheet_note.update({
            "status": "parsed",
            "header_row": header_row,
            "column_count": len(header),
            "month_columns": sorted(month_columns.values()),
            "current_month": current_month,
            "rows_emitted": emitted,
            "states_parsed": len(seen_states),
            "excluded_labels": excluded,
            "body_truncated_at": truncation,
        })
        diagnostics["sheets"].append(sheet_note)
        diagnostics["rows_emitted"] += emitted
        for label, count in excluded.items():
            diagnostics["excluded_labels"][label] = (
                diagnostics["excluded_labels"].get(label, 0) + count
            )

    diagnostics["months"] = sorted({r["month"] for r in rows})
    return rows, diagnostics


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def deduplicate(rows: Iterable[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse to one row per ``(month, state)``.

    Preference order, highest first:

    1. the observation from the release for which that month is the *current*
       month (the authoritative first publication);
    2. failing that, the observation from the most recent release, since NBS
       restates comparison columns from the latest available figures.

    De-duplication here is what makes the stage idempotent (constraint 2.7):
    re-running against the same catalogue cannot inflate the panel.
    """
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame, {"duplicates_collapsed": 0}

    before = len(frame)
    frame = frame.sort_values(
        ["month", "state", "is_current_month", "source_file"],
        ascending=[True, True, False, False],
        kind="mergesort",
    )
    deduped = frame.drop_duplicates(subset=["month", "state"], keep="first").copy()
    deduped = deduped.sort_values(["month", "state"], kind="mergesort").reset_index(
        drop=True
    )

    diagnostics = {
        "observations_before_dedup": before,
        "observations_after_dedup": len(deduped),
        "duplicates_collapsed": before - len(deduped),
        "from_current_month_column": int(deduped["is_current_month"].sum()),
    }
    return deduped, diagnostics


def _month_gaps(months: list[str]) -> list[str]:
    """Calendar months absent between the first and last observed month."""
    if not months:
        return []
    start, end = min(months), max(months)
    cursor = pd.Period(start, freq="M")
    last = pd.Period(end, freq="M")
    present = set(months)
    gaps = []
    while cursor <= last:
        label = str(cursor)
        if label not in present:
            gaps.append(label)
        cursor += 1
    return gaps


def extract(force: bool = False) -> list[ExtractionResult]:
    """Scrape the catalogue, download every release, and assemble the panel."""
    with stage("extract: NBS PMS price watch (HTML scrape + Excel parse)", LOG):
        result = ExtractionResult(
            key="nbs_pms", name="NBS PMS Price Watch",
            pattern="HTML scrape + heterogeneous Excel parse",
        )

        html = get_text(settings.NBS_CATALOGUE_URL, description="NBS catalogue 157")
        catalogue = discover_catalogue_files(html)
        if len(catalogue) < settings.NBS_EXPECTED_MIN_FILES:
            raise ExtractionError(
                f"NBS catalogue scrape found only {len(catalogue)} data file(s); "
                f"at least {settings.NBS_EXPECTED_MIN_FILES} were verified present "
                f"on 2026-09-13 at {settings.NBS_CATALOGUE_URL}. The page layout "
                f"may have changed. Inspect the markup before trusting any figure "
                f"derived from this source."
            )

        extract_root = settings.RAW_NBS / EXTRACT_DIR_NAME
        extract_root.mkdir(parents=True, exist_ok=True)

        downloaded: list[Path] = []
        workbooks: list[Path] = []
        for entry in catalogue:
            destination = settings.RAW_NBS / entry["filename"]
            try:
                path = download_file(
                    entry["url"], destination, force=force, min_bytes=2048,
                    description=entry["filename"],
                )
            except ExtractionError as exc:
                LOG.warning("could not download %s: %s", entry["filename"], exc)
                continue
            if not _looks_like_zip_or_office(path):
                LOG.warning(
                    "%s does not carry the ZIP/OOXML magic bytes; the catalogue "
                    "may have served an error page. Skipping.", path.name
                )
                continue
            downloaded.append(path)
            workbooks.extend(_unpack(path, extract_root))

        if not workbooks:
            raise ExtractionError(
                f"FileNotFoundError: {settings.RAW_NBS} contains no readable "
                f"workbook. Run 'make extract-nbs' or check network access to "
                f"microdata.nigerianstat.gov.ng."
            )

        all_rows: list[dict[str, Any]] = []
        per_file: list[dict[str, Any]] = []
        for workbook in sorted(set(workbooks)):
            rows, diagnostics = parse_workbook(workbook)
            all_rows.extend(rows)
            per_file.append(diagnostics)
            LOG.info("parsed %s -> %d observation(s) across months %s",
                     workbook.name, diagnostics["rows_emitted"],
                     diagnostics["months"])

        panel, dedup_diagnostics = deduplicate(all_rows)
        if panel.empty:
            raise ExtractionError(
                "NBS parsing produced zero observations. The workbook layouts "
                "have changed beyond what the header-scan parser tolerates; "
                f"inspect the sheets under {settings.RAW_NBS}."
            )

        months = sorted(panel["month"].unique().tolist())
        gaps = _month_gaps(months)
        excluded_labels: dict[str, int] = {}
        for diagnostics in per_file:
            for label, count in diagnostics["excluded_labels"].items():
                excluded_labels[label] = excluded_labels.get(label, 0) + count

        implausible = panel[
            (panel["price_ngn"] < settings.NBS_PRICE_MIN_NGN)
            | (panel["price_ngn"] > settings.NBS_PRICE_MAX_NGN)
        ]

        landing = settings.RAW_NBS / "nbs_pms_assembled.csv"
        panel.to_csv(landing, index=False)

        result.files = [landing] + downloaded
        result.row_counts[landing.name] = len(panel)

        national_by_month = (
            panel.groupby("month")["price_ngn"].mean().round(2).to_dict()
        )

        result.diagnostics = {
            "catalogue_url": settings.NBS_CATALOGUE_URL,
            "catalogue_files_discovered": len(catalogue),
            "files_downloaded": len(downloaded),
            "workbooks_parsed": len(set(workbooks)),
            "observations_assembled": len(panel),
            "distinct_months": len(months),
            "month_min": months[0],
            "month_max": months[-1],
            "distinct_states": int(panel["state"].nunique()),
            "month_gaps_observed": gaps,
            "month_gaps_claimed_by_spec": list(settings.NBS_SPEC_CLAIMED_MISSING_MONTHS),
            "gaps_claimed_but_not_observed": sorted(
                set(settings.NBS_SPEC_CLAIMED_MISSING_MONTHS) - set(gaps)
            ),
            "gaps_observed_but_not_claimed": sorted(
                set(gaps) - set(settings.NBS_SPEC_CLAIMED_MISSING_MONTHS)
            ),
            "excluded_non_state_labels": excluded_labels,
            "excluded_label_count": len(excluded_labels),
            "implausible_price_rows": len(implausible),
            "national_mean_by_month": national_by_month,
            "per_file": per_file,
            **dedup_diagnostics,
        }

        LOG.info(
            "Source D assembled: %d observations, %d months (%s to %s), %d states",
            len(panel), len(months), months[0], months[-1],
            panel["state"].nunique(),
        )
        LOG.info("Source D excluded %d non-state label(s) from the state column: %s",
                 len(excluded_labels), sorted(excluded_labels))
        if gaps:
            LOG.warning("Source D missing month(s), NOT interpolated: %s", gaps)
        else:
            LOG.info(
                "Source D has NO month gaps between %s and %s: the panel is "
                "complete at %d months x %d states = %d observations",
                months[0], months[-1], len(months), panel["state"].nunique(),
                len(panel),
            )
        phantom = sorted(set(settings.NBS_SPEC_CLAIMED_MISSING_MONTHS) - set(gaps))
        if phantom:
            LOG.warning(
                "month(s) %s were expected to be missing but ARE present. They "
                "were absent only under a shallow header scan; the gap was a "
                "parser artefact, not a gap in NBS publication.", phantom,
            )
        for month, target in settings.NBS_RECONCILIATION_TARGETS.items():
            observed = national_by_month.get(month)
            if observed is None:
                LOG.warning("reconciliation month %s absent from the panel", month)
            else:
                LOG.info("reconciliation %s: assembled %.2f vs published %.2f "
                         "(delta %+.2f)", month, observed, target, observed - target)

        return [result]
