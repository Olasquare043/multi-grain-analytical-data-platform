"""Central configuration for the CSC 796 mobility data platform.

Every tunable in the pipeline lives here. No module hard-codes a path, a URL,
a window boundary or a threshold (coding standard, section 11).

Paths resolve relative to the repository root, which is the parent of this
package, so the same settings work inside the container (/app) and on a host
checkout without modification.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------- #
# Filesystem layout (medallion architecture, section 5)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
RAW_NYC = RAW_DIR / "nyc"
RAW_HDX = RAW_DIR / "hdx"
RAW_NBS = RAW_DIR / "nbs"
RAW_WEATHER = RAW_DIR / "weather"

BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
BENCHMARKS_DIR = OUTPUTS_DIR / "benchmarks"
QUALITY_DIR = OUTPUTS_DIR / "quality"
CLOUD_OUT_DIR = OUTPUTS_DIR / "cloud"

SQL_DIR = ROOT / "sql"
SQL_DDL_DIR = SQL_DIR / "ddl"
SQL_ANALYSIS_DIR = SQL_DIR / "analysis"
SQL_CLOUD_DIR = SQL_DIR / "cloud"

DOCS_DIR = ROOT / "docs"
CONFIG_DIR = ROOT / "config"
LOGS_DIR = ROOT / "logs"

LOG_FILE = LOGS_DIR / "pipeline.log"
RUN_MANIFEST = DOCS_DIR / "run_manifest.json"
QUALITY_RULES_FILE = CONFIG_DIR / "quality_rules.yaml"

#: DuckDB catalogue holding views over the Parquet gold layer.
WAREHOUSE_DB = DATA_DIR / "warehouse.duckdb"

ALL_DIRS = [
    RAW_NYC, RAW_HDX, RAW_NBS, RAW_WEATHER,
    BRONZE_DIR, SILVER_DIR, GOLD_DIR,
    FIGURES_DIR, TABLES_DIR, BENCHMARKS_DIR, QUALITY_DIR, CLOUD_OUT_DIR,
    DOCS_DIR, LOGS_DIR,
]

# --------------------------------------------------------------------------- #
# Engine tuning (section 12: record what changed and what it bought)
# --------------------------------------------------------------------------- #
DUCKDB_THREADS = int(os.getenv("DUCKDB_THREADS", "4"))
DUCKDB_MEMORY_LIMIT = os.getenv("DUCKDB_MEMORY_LIMIT", "6GB")
DUCKDB_TEMP_DIR = str(DATA_DIR / "duckdb_tmp")
DUCKDB_PRESERVE_INSERTION_ORDER = False  # lowers peak memory on wide scans

CPU_LIMIT = os.getenv("CSC796_CPU_LIMIT", "4")
MEM_LIMIT_GB = os.getenv("CSC796_MEM_LIMIT_GB", "8")

#: Declared budgets from section 2, echoed into the run manifest.
DISK_BUDGET_GB = 3.0
RUNTIME_BUDGET_MINUTES = 20

# --------------------------------------------------------------------------- #
# Source A -- NYC TLC Yellow Taxi trip records (bulk binary download)
# --------------------------------------------------------------------------- #
NYC_TLC_LANDING_PAGE = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
NYC_TLC_DATA_DICTIONARY = (
    "https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf"
)
NYC_TRIP_URL_TEMPLATE = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_{year}-{month:02d}.parquet"
)
NYC_TRIP_YEAR = 2024
NYC_TRIP_MONTHS = list(range(1, 13))

#: Verified per-month row counts (section 3, Source A). Ingestion asserts against
#: these and reports any discrepancy rather than silently accepting it.
NYC_EXPECTED_ROW_COUNTS: dict[str, int] = {
    "2024-01": 2_964_624, "2024-02": 3_007_526, "2024-03": 3_582_628,
    "2024-04": 3_514_289, "2024-05": 3_723_833, "2024-06": 3_539_193,
    "2024-07": 3_076_903, "2024-08": 2_979_183, "2024-09": 3_633_030,
    "2024-10": 3_833_771, "2024-11": 3_646_369, "2024-12": 3_668_371,
}
NYC_EXPECTED_TOTAL_ROWS = sum(NYC_EXPECTED_ROW_COUNTS.values())  # 41,169,720

#: The 19 columns verified present in every 2024 monthly file.
NYC_EXPECTED_COLUMNS: tuple[str, ...] = (
    "VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime", "passenger_count",
    "trip_distance", "RatecodeID", "store_and_fwd_flag", "PULocationID",
    "DOLocationID", "payment_type", "fare_amount", "extra", "mta_tax",
    "tip_amount", "tolls_amount", "improvement_surcharge", "total_amount",
    "congestion_surcharge", "Airport_fee",
)

#: Declared analytical window. Trips outside it are a *confirmed* defect of the
#: official data (January 2024 carries a pickup stamped 2002-12-31 22:59:39).
#: They are quantified and reported, never silently dropped (section 3).
NYC_WINDOW_START = date(2024, 1, 1)
NYC_WINDOW_END = date(2025, 1, 1)  # exclusive

# Source B -- taxi zone lookup
NYC_ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
NYC_ZONE_EXPECTED_ROWS = 265
NYC_ZONE_EXPECTED_BOROUGHS = 8

#: Zones flagged by TLC as non-geographic ("Unknown", "N/A"). Retained in the
#: dimension but excluded from borough-level geography; see analysis headers.
NYC_NONGEOGRAPHIC_ZONE_IDS = (264, 265)
NYC_AIRPORT_ZONE_IDS = (1, 132, 138)  # Newark, JFK, LaGuardia

#: Structural validity thresholds for a trip record. A row failing any of these
#: is excluded from fact_trip and counted by reason in the reconciliation table;
#: none of them is a silent drop.
TRIP_MAX_DURATION_SECONDS = 24 * 60 * 60   # a metered yellow-cab trip over a day
TRIP_MAX_DISTANCE_MILES = 500.0            # NYC to Washington DC is ~230 miles
TRIP_MAX_PASSENGERS = 9                    # TLC vehicle licensing ceiling
#: Speeds above this are flagged by a *warn* rule rather than rejected, because
#: they usually indicate a bad odometer reading rather than an invalid trip.
TRIP_IMPLAUSIBLE_SPEED_MPH = 100.0

#: Distance bands used by A4 (tipping behaviour), in miles. Upper bound exclusive.
TRIP_DISTANCE_BANDS = (
    ("0-1 mi", 0.0, 1.0),
    ("1-3 mi", 1.0, 3.0),
    ("3-5 mi", 3.0, 5.0),
    ("5-10 mi", 5.0, 10.0),
    ("10+ mi", 10.0, float("inf")),
)

# --------------------------------------------------------------------------- #
# Source C -- WFP Nigeria market prices via HDX (open data API)
# --------------------------------------------------------------------------- #
HDX_DATASET_PAGE = "https://data.humdata.org/dataset/wfp-food-prices-for-nigeria"
HDX_PRICES_URL = (
    "https://data.humdata.org/dataset/42db041f-7aaf-4ab4-961f-2a12096861e7/resource/"
    "12b51155-0cd3-4806-9924-61ede4077591/download/wfp_food_prices_nga.csv"
)
HDX_MARKETS_URL = (
    "https://data.humdata.org/dataset/42db041f-7aaf-4ab4-961f-2a12096861e7/resource/"
    "5329e772-0b74-4f65-8cc0-37a0915cc7e4/download/wfp_markets_nga.csv"
)
HDX_PRICES_EXPECTED_MIN_ROWS = 80_000   # 87,384 verified 2026-09-13; grows over time
HDX_EXPECTED_MIN_STATES = 12
HDX_EXPECTED_MIN_MARKETS = 60

#: Row 1 of every HDX resource is an HXL tag row beginning with this token.
HXL_SENTINEL = "#date"

#: Fuel commodities live under the 'non-food' category and carry a verified
#: coverage gap (2023-02 through 2025). Context only; never used for A7/A9.
HDX_FUEL_COMMODITIES = ("Fuel (diesel)", "Fuel (petrol-gasoline)")
HDX_NONFOOD_CATEGORY = "non-food"

#: Basket used by A6/A7/A9: the commodities with the longest, densest panel
#: coverage that are unambiguously Nigerian dietary staples. Matching is by
#: case-insensitive containment because WFP commodity spellings drift.
NIGERIA_STAPLE_BASKET = (
    "Rice (imported)", "Rice (local)", "Maize (white)", "Maize (yellow)",
    "Millet", "Sorghum", "Beans (niebe)", "Beans (white)", "Gari (white)",
    "Gari (yellow)", "Yam", "Cassava meal (gari, white)",
)
#: Fallback token matching when exact basket names are absent from a vintage.
NIGERIA_STAPLE_TOKENS = (
    "rice", "maize", "millet", "sorghum", "beans", "gari", "yam", "cassava",
)

#: The staple basket used by A7 and A9 is defined by WFP's own published
#: CATEGORY rather than by a list of commodity names. Verified against the live
#: panel: 'cereals and tubers' (35 commodity-unit members) and 'pulses and nuts'
#: (11) together cover every commodity in NIGERIA_STAPLE_BASKET plus their
#: spelling variants (Cowpeas/Beans, Rice (milled, local)/Rice (local),
#: Cassava meal (gari, yellow)/Gari (yellow), Yam (Abuja)/Yam).
#: Selecting by category means a WFP rename cannot silently empty the basket,
#: which a hard-coded name list would.
NIGERIA_STAPLE_CATEGORIES = ("cereals and tubers", "pulses and nuts")

#: WFP publishes both Retail and Wholesale quotes. A7 and A9 use Retail only:
#: it is the price a household actually faces, it is far better covered
#: (66,232 vs 21,152 observations), and mixing the two would make a series that
#: moves when the retail/wholesale mix moves.
NIGERIA_PRICE_TYPE = "Retail"

#: A6 index base period. Jan 2016 = 100 (section 7).
PRICE_INDEX_BASE_YEAR_MONTH = "2016-01"

#: Chained-index linking rules (A6, A7). A cell's price relative is formed
#: against that cell's own most recent OBSERVED price, provided it is no more
#: than this many months old. The relative is therefore always a real observed
#: change -- nothing is interpolated -- but a cell that skips a month still
#: contributes instead of silently dropping out.
PRICE_INDEX_MAX_LINK_GAP_MONTHS = 3
#: A monthly link resting on fewer matched cells than this is treated as a
#: CHAIN BREAK rather than a price change. The chain is never carried across a
#: break (an earlier draft did so by assuming a relative of 1.0, which is
#: carry-forward interpolation and was removed); only the segment that links
#: unbroken to the base month is published as an index.
PRICE_INDEX_MIN_MATCHED_CELLS = 3

#: A7 structural break: petrol subsidy removal announced 29 May 2023.
SUBSIDY_REMOVAL_DATE = date(2023, 5, 29)
SUBSIDY_BREAK_YEAR_MONTH = "2023-05"
#: A7 is a WITHIN-MARKET analysis on a FIXED PANEL: only markets reporting
#: retail staple prices in EVERY month of a symmetric window around the break.
#: Pre-break = the W months ending with the break month (t = -(W-1) .. 0);
#: post-break = the W months after it (t = 1 .. W).
#:
#: Why W = 4, and why it is not larger: WFP's north-eastern retail reporting has
#: a panel-wide hole from June 2022 to January 2023 (only five markets report
#: through it, and four of those stop in January 2023). No market reports in
#: every month of any window reaching back before February 2023, so February to
#: September 2023 is the longest symmetric window in which a continuous panel
#: exists at all. A wider window would contain zero continuous markets.
SUBSIDY_WINDOW_MONTHS = 4

#: A12: a state holds a PERSISTENT premium (discount) when it sits above (below)
#: the national mean in at least this share of the months observed.
A12_PERSISTENCE_SHARE = 0.8

#: A8 requires at least this many reporting markets in a cell for the
#: coefficient of variation to be meaningful.
DISPERSION_MIN_MARKETS = 3

#: A9 lag structure.
PASSTHROUGH_LAGS = (0, 1, 2, 3)
PASSTHROUGH_MIN_PAIRS = 6  # minimum (state, month) pairs before reporting a correlation

# --------------------------------------------------------------------------- #
# Source D -- NBS Premium Motor Spirit Price Watch (HTML scrape + Excel)
# --------------------------------------------------------------------------- #
NBS_CATALOGUE_URL = "https://microdata.nigerianstat.gov.ng/index.php/catalog/157"
NBS_BASE_URL = "https://microdata.nigerianstat.gov.ng"
#: Anchors matching this pattern are data files: catalog/157/download/{id}/{name}
NBS_DOWNLOAD_HREF_PATTERN = r"catalog/157/download/\d+"
NBS_ACCEPTED_EXTENSIONS = (".xlsx", ".xls", ".zip")
#: Rows scanned for the 'State' header. The specification describes a six-row
#: banner, but the March 2026 release (PMS_March_ 2026.xlsx) places the header on
#: row 14 behind a block of empty rows, and a six-row scan silently yields zero
#: observations from it. Widened to 25 and reported as observed drift beyond the
#: documented layouts.
NBS_HEADER_SCAN_ROWS = 25
NBS_EXPECTED_MIN_MONTHS = 24
NBS_EXPECTED_MIN_FILES = 15
NBS_FUEL_TYPE = "PMS"

#: Month gaps the specification reported as verified. Retained as the stated
#: expectation so the pipeline can report the DIFFERENCE between what was
#: claimed and what it actually observes, rather than quietly adopting either.
#:
#: Observed result: with the header scan widened to 25 rows, none of these three
#: months is missing. All three were artefacts of a six-row header scan failing
#: on releases that place the 'State' header behind a deeper title banner. Gaps
#: are still computed and reported every run; they are never interpolated.
NBS_SPEC_CLAIMED_MISSING_MONTHS = ("2024-08", "2025-10", "2025-11")

#: Mandatory reconciliation against NBS published national means (NGN/litre).
NBS_RECONCILIATION_TARGETS: dict[str, float] = {
    "2023-11": 648.93,
    "2023-12": 671.86,
    "2024-05": 769.62,
}
NBS_RECONCILIATION_TOLERANCE = 0.05

#: Plausibility bounds for a Nigerian pump price in NGN/litre over the covered
#: window. Values outside are quarantined and reported, never silently dropped.
NBS_PRICE_MIN_NGN = 100.0
NBS_PRICE_MAX_NGN = 3000.0

# --------------------------------------------------------------------------- #
# Source E -- Open-Meteo archive (REST/JSON, optional, non-blocking)
# --------------------------------------------------------------------------- #
WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_PARAMS = {
    "latitude": 40.7128,
    "longitude": -74.0060,
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "daily": (
        "temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "snowfall_sum,wind_speed_10m_max"
    ),
    "timezone": "America/New_York",
}
WEATHER_EXPECTED_DAYS = 366
#: Source E is optional, so it gets a short leash: the patient retry schedule
#: that suits the required 60 MB TLC files once spent 15 minutes on this source
#: during a network outage before it was skipped.
WEATHER_MAX_RETRIES = 2
WEATHER_TIMEOUT_SECONDS = 30
WEATHER_ENABLED = os.getenv("CSC796_WEATHER", "1") != "0"
#: Weather attaches to a single synthetic geography row representing NYC.
WEATHER_GEO_CODE = "NYC"

# --------------------------------------------------------------------------- #
# dim_date
# --------------------------------------------------------------------------- #
DATE_DIM_START = date(2016, 1, 1)
DATE_DIM_END = date(2026, 12, 31)

#: Every dimension carries an Unknown member at this surrogate key so that facts
#: never lose rows to a failed lookup (section 6, referential integrity).
UNKNOWN_KEY = -1

#: Type 2 SCD open-ended high date.
SCD_HIGH_DATE = date(9999, 12, 31)
#: valid_from stamped on the initial load of every dimension member. A fixed
#: epoch rather than the run date, so that two runs on different days produce
#: byte-identical dimensions (constraint 2.6, determinism). Genuine subsequent
#: changes are stamped with the date the change was detected.
SCD_INITIAL_VALID_FROM = date(2016, 1, 1)

#: dim_geography admin levels. 'zone', 'market' and 'state' are the specified
#: domain; 'city' is a documented extension carrying the single New York City
#: member that the optional fact_weather_daily must reference, because
#: Open-Meteo publishes one point series rather than zone-level weather.
GEO_ADMIN_LEVELS = ("zone", "market", "state", "city")

# --------------------------------------------------------------------------- #
# HTTP behaviour
# --------------------------------------------------------------------------- #
HTTP_TIMEOUT_SECONDS = 120
#: Retries use exponential backoff (5 s, 10 s, 20 s ... capped at 2 min) and
#: downloads RESUME from the partial file via an HTTP Range request. Both were
#: added after a cold run on a flaky connection exhausted four linear 3/6/9 s
#: retries in half a minute, each of which had restarted a 60 MB file from zero.
HTTP_MAX_RETRIES = 8
HTTP_BACKOFF_SECONDS = 5.0
HTTP_BACKOFF_MAX_SECONDS = 120.0
HTTP_CHUNK_BYTES = 1 << 20
USER_AGENT = (
    "csc796-mobility-platform/1.0 (University of Ibadan, MSc AI; "
    "academic research)"
)

# --------------------------------------------------------------------------- #
# Benchmarks (section 8)
# --------------------------------------------------------------------------- #
BENCHMARK_REPEATS = 3
BENCHMARK_MONTH = "2024-01"          # month used for B1/B2
BENCHMARK_PANDAS_MEMORY_CEILING_GB = 8.0
#: B1 materialises a CSV copy of one month (~450 MB). Deleted afterwards to stay
#: inside the 3 GB disk budget; the measurement itself is retained.
BENCHMARK_DELETE_CSV_AFTER = True

# --------------------------------------------------------------------------- #
# Figures (section 12)
# --------------------------------------------------------------------------- #
FIGURE_DPI = 300
FIGURE_FORMAT = "png"
FIGURE_ATTRIBUTION_NYC = "Source: NYC TLC Yellow Taxi Trip Records, 2024"
FIGURE_ATTRIBUTION_WFP = "Source: WFP Food Prices for Nigeria (via HDX)"
FIGURE_ATTRIBUTION_NBS = "Source: NBS Premium Motor Spirit Price Watch"

# --------------------------------------------------------------------------- #
# Cloud module (section 9) -- never imported by the core pipeline
# --------------------------------------------------------------------------- #
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
BQ_PUBLIC_PROJECT = "bigquery-public-data"
BQ_TLC_DATASET = "new_york_taxi_trips"
BQ_LOCATION = "US"

#: The public TLC dataset was not maintained past 2022 and contains no 2024 data,
#: so the cloud comparison runs against 2019. This YEAR MISMATCH is a documented
#: confounder, stated in outputs and in docs/cloud_architecture.md (section 9).
BQ_COMPARISON_TABLE = "tlc_yellow_trips_2019"
BQ_SCALE_TABLES = ("tlc_yellow_trips_2022", "tlc_yellow_trips_2019", "tlc_yellow_trips_2014")
BQ_ZONE_GEOM_TABLE = "taxi_zone_geom"

#: On-demand analysis price, USD per TiB. This is a CONFIGURATION VALUE -- the
#: published list price as recorded when the module was written -- NOT a price
#: verified on any particular date. At runtime the cloud module tries to read the
#: current rate from the pricing page and records which of the two it used, and
#: whether the page confirmed the configured value. Override with BQ_USD_PER_TIB.
BQ_ON_DEMAND_USD_PER_TIB = float(os.getenv("BQ_USD_PER_TIB", "6.25"))
BQ_PRICING_REFERENCE_URL = "https://cloud.google.com/bigquery/pricing"
BQ_PRICING_AS_OF = "configuration value, not verified at build time"

#: Free-tier guard rails. The module aborts before approaching the allowance.
BQ_FREE_TIER_BYTES_PER_MONTH = 1024 ** 4          # 1 TiB
BQ_SESSION_BYTE_BUDGET = 200 * 1024 ** 3          # 200 GiB ceiling for one run
BQ_ABORT_THRESHOLD_FRACTION = 0.80


@dataclass(frozen=True)
class SourceSpec:
    """Declarative description of one extraction, used by the run manifest."""

    key: str
    name: str
    pattern: str
    url: str
    required: bool = True
    notes: str = ""
    landing_page: str = ""


SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("nyc_trips", "NYC TLC Yellow Taxi 2024", "bulk binary download",
               NYC_TRIP_URL_TEMPLATE, True,
               "12 monthly Parquet files, 41,169,720 rows verified",
               NYC_TLC_LANDING_PAGE),
    SourceSpec("nyc_zones", "NYC Taxi Zone Lookup", "bulk binary download",
               NYC_ZONE_LOOKUP_URL, True, "265 rows, 8 boroughs",
               NYC_TLC_LANDING_PAGE),
    SourceSpec("hdx_prices", "WFP Nigeria Market Prices", "open data API",
               HDX_PRICES_URL, True, "HXL tag row must be stripped",
               HDX_DATASET_PAGE),
    SourceSpec("hdx_markets", "WFP Nigeria Markets", "open data API",
               HDX_MARKETS_URL, True, "market to admin1 bridge",
               HDX_DATASET_PAGE),
    SourceSpec("nbs_pms", "NBS PMS Price Watch", "HTML scrape + Excel parse",
               NBS_CATALOGUE_URL, True,
               "heterogeneous layouts; file list is discovered, never hard-coded",
               NBS_CATALOGUE_URL),
    SourceSpec("weather", "Open-Meteo NYC daily archive", "REST/JSON",
               WEATHER_API_URL, False,
               "optional; failure warns and marks skipped in the manifest",
               "https://open-meteo.com/en/docs/historical-weather-api"),
)
