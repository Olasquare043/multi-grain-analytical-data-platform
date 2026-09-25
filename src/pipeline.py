"""Pipeline entrypoint.

``python -m src.pipeline --stage all`` runs the complete study end to end, which
is what ``docker compose up --build`` invokes. Every stage is also individually
addressable so any one can be rerun alone (section 5).

The cloud module (section 9) is **not** reachable from here. It lives behind
``make cloud`` and the core run must succeed whether or not it has ever been
executed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from config import settings
from src.utils.io_utils import dir_size, ensure_dirs, human_bytes, relative_to_root
from src.utils.logging_setup import (
    configure_logging, get_logger, stage, stage_timings,
)
from src.utils.manifest import get_manifest

LOG = get_logger(__name__)

#: Outputs the study promises. `verify` asserts each exists and reports any that
#: do not, with the reason (section 13, step 15).
PROMISED_OUTPUTS: dict[str, list[str]] = {
    "outputs/tables": [
        "A1_demand_profile.csv", "A2_od_corridors.csv", "A3_congestion_proxy.csv",
        "A4_tipping_behaviour.csv", "A5_revenue_concentration.csv",
        "A6_food_price_index.csv", "A7_subsidy_structural_break.csv",
        "A8_price_dispersion.csv", "A9_fuel_food_passthrough.csv",
        "A10_grain_comparison.csv", "A11_missingness_structure.csv",
        "A12_petrol_price_geography.csv",
        "A12_petrol_price_geography_national.csv",
        "A12_petrol_price_geography_states.csv",
        "A12_petrol_price_geography_zones.csv",
    ],
    "outputs/figures": [
        "A1_demand_profile_heatmap.png", "A1_demand_profile_lines.png",
        "A2_od_corridors.png", "A3_congestion_proxy.png",
        "A4_tipping_behaviour.png", "A5_revenue_concentration.png",
        "A6_food_price_index.png", "A7_subsidy_structural_break.png",
        "A8_price_dispersion.png", "A9_fuel_food_passthrough.png",
        "A10_grain_comparison.png", "A11_missingness_structure.png",
        "A12_petrol_price_national.png", "A12_petrol_price_state_premiums.png",
        "A12_petrol_price_rank_stability.png", "A12_petrol_price_zones.png",
    ],
    "outputs/quality": [
        "quality_report.csv", "quality_report.md", "reconciliation.csv",
        "nbs_reconciliation.csv",
    ],
    "outputs/benchmarks": ["benchmark_results.csv", "benchmark_results.md"],
    "docs": [
        "data_dictionary.md", "erd_conceptual.md", "erd_star_schema.md",
        "architecture.md", "scd_strategy.md", "methodology_notes.md",
        "run_manifest.json",
    ],
}

#: Optional outputs: absence is reported, never treated as failure. Everything
#: here belongs to the cloud module (section 9), which runs only via make cloud.
OPTIONAL_OUTPUTS: dict[str, list[str]] = {
    "docs": ["cloud_architecture.md"],
    "outputs/cloud": [
        "table_inventory.csv", "engine_comparison_cloud.csv",
        "scale_behaviour.csv",
    ],
}


# --------------------------------------------------------------------------- #
# Stage implementations
# --------------------------------------------------------------------------- #
def _extract(which: str | None = None, force: bool = False) -> Any:
    from src.extract import run_extract

    return run_extract.run(which=which, force=force)


def _bronze() -> Any:
    from src.transform import bronze
    from src.utils.db import connect

    con = connect(settings.WAREHOUSE_DB)
    try:
        return bronze.build(con)
    finally:
        con.close()


def _silver() -> Any:
    from src.transform import silver
    from src.utils.db import connect

    con = connect(settings.WAREHOUSE_DB)
    try:
        return silver.build(con)
    finally:
        con.close()


def _dims() -> Any:
    from src.model import dim_geography, dimensions
    from src.utils.db import connect

    con = connect(settings.WAREHOUSE_DB)
    try:
        result = dimensions.build(con)
        result["dim_geography"] = dim_geography.build(con)
        return result
    finally:
        con.close()


def _facts() -> Any:
    from src.model import catalog, facts
    from src.utils.db import connect

    con = connect(settings.WAREHOUSE_DB)
    try:
        result = facts.build(con)
        result["catalog"] = catalog.build(con)
        return result
    finally:
        con.close()


def _quality() -> Any:
    from src.quality import run_quality

    return run_quality.run()


def _analysis() -> Any:
    from src.analysis import run_analysis

    return run_analysis.run()


def _benchmarks() -> Any:
    from src.benchmark import run_benchmarks

    return run_benchmarks.run()


def _figures() -> Any:
    from src.viz import figures

    return figures.run()


def _docs() -> Any:
    from src.docs import generate

    return generate.run()


def verify(strict: bool = False) -> dict[str, Any]:
    """Assert every promised output exists and print the completion summary."""
    with stage("verify: completion summary", LOG):
        present: list[tuple[str, int]] = []
        missing: list[str] = []

        LOG.info("")
        LOG.info("%-58s %12s", "ARTEFACT", "SIZE")
        LOG.info("%s", "-" * 72)

        for directory, filenames in PROMISED_OUTPUTS.items():
            LOG.info("[%s]", directory)
            for filename in filenames:
                path = settings.ROOT / directory / filename
                if path.exists():
                    size = path.stat().st_size
                    present.append((f"{directory}/{filename}", size))
                    LOG.info("  %-56s %12s", filename, human_bytes(size))
                else:
                    missing.append(f"{directory}/{filename}")
                    LOG.error("  %-56s %12s", filename, "MISSING")

        skipped: list[str] = []
        for directory, filenames in OPTIONAL_OUTPUTS.items():
            LOG.info("[%s]  (optional)", directory)
            for filename in filenames:
                path = settings.ROOT / directory / filename
                if path.exists():
                    size = path.stat().st_size
                    present.append((f"{directory}/{filename}", size))
                    LOG.info("  %-56s %12s", filename, human_bytes(size))
                else:
                    skipped.append(f"{directory}/{filename}")
                    LOG.warning("  %-56s %12s", filename, "not produced")

        # Anything else that landed in outputs/ but was not promised.
        promised = {
            f"{d}/{f}" for d, files in
            {**PROMISED_OUTPUTS, **OPTIONAL_OUTPUTS}.items() for f in files
        }
        extras = [
            relative_to_root(p)
            for p in sorted(settings.OUTPUTS_DIR.rglob("*"))
            if p.is_file() and relative_to_root(p) not in promised
            and p.name != ".gitkeep"
        ]
        if extras:
            LOG.info("[additional artefacts produced]")
            for extra in extras:
                LOG.info("  %-56s %12s",
                         extra, human_bytes((settings.ROOT / extra).stat().st_size))

        manifest = get_manifest()
        data_bytes = dir_size(settings.DATA_DIR)
        LOG.info("%s", "-" * 72)
        LOG.info("promised artefacts present : %d", len(PROMISED_OUTPUTS) and
                 sum(len(v) for v in PROMISED_OUTPUTS.values()) - len(missing))
        LOG.info("promised artefacts MISSING : %d", len(missing))
        LOG.info("optional artefacts skipped : %d", len(skipped))
        LOG.info("data/ on disk              : %s of a %.1f GB budget (%.0f%%)",
                 human_bytes(data_bytes), settings.DISK_BUDGET_GB,
                 100.0 * data_bytes / (settings.DISK_BUDGET_GB * 1024 ** 3))

        quality = manifest.data.get("quality", {})
        if quality:
            LOG.info("quality gate               : %s (%s passed, %s warned, "
                     "%s failed)", quality.get("gate"), quality.get("passed"),
                     quality.get("warned"), quality.get("failed"))
        reconciliation = manifest.data.get("reconciliation", {})
        if reconciliation:
            LOG.info("NBS external reconciliation: %s",
                     "ALL WITHIN TOLERANCE"
                     if reconciliation.get("all_within_tolerance")
                     else "OUT OF TOLERANCE -- investigate")

        if missing:
            LOG.error("")
            LOG.error("The following promised artefacts were not produced:")
            for item in missing:
                LOG.error("  - %s", item)
        if skipped:
            LOG.warning("")
            LOG.warning("Optional artefacts not produced (cloud module is "
                        "opt-in via 'make cloud'):")
            for item in skipped:
                LOG.warning("  - %s", item)

        summary = {
            "present": len(present),
            "missing": missing,
            "optional_skipped": skipped,
            "extras": extras,
            "data_bytes": data_bytes,
            "within_disk_budget":
                data_bytes <= settings.DISK_BUDGET_GB * 1024 ** 3,
        }
        manifest.data["verification"] = summary
        manifest.save()

        if missing and strict:
            raise SystemExit(
                f"{len(missing)} promised artefact(s) missing; see the log above."
            )
        return summary


def clean(include_raw: bool = False) -> None:
    """Remove derived artefacts. Raw downloads are kept unless asked otherwise."""
    with stage("clean", LOG):
        targets = [
            settings.BRONZE_DIR, settings.SILVER_DIR, settings.GOLD_DIR,
            settings.TABLES_DIR, settings.FIGURES_DIR, settings.BENCHMARKS_DIR,
            settings.QUALITY_DIR, settings.CLOUD_OUT_DIR,
            Path(settings.DUCKDB_TEMP_DIR), settings.DATA_DIR / "benchmark_tmp",
        ]
        if include_raw:
            targets.append(settings.RAW_DIR)

        for target in targets:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
                LOG.info("removed %s", relative_to_root(target))
        for artefact in (settings.WAREHOUSE_DB, settings.RUN_MANIFEST):
            if artefact.exists():
                artefact.unlink()
                LOG.info("removed %s", relative_to_root(artefact))
        ensure_dirs()
        LOG.info("clean complete%s",
                 " (raw downloads removed: the next run is cold)"
                 if include_raw else " (raw downloads kept)")


# --------------------------------------------------------------------------- #
# Stage registry
# --------------------------------------------------------------------------- #
STAGES: dict[str, Callable[[], Any]] = {
    "extract": lambda: _extract(None),
    "extract-nyc": lambda: _extract("nyc"),
    "extract-hdx": lambda: _extract("hdx"),
    "extract-nbs": lambda: _extract("nbs"),
    "extract-weather": lambda: _extract("weather"),
    "bronze": _bronze,
    "silver": _silver,
    "dims": _dims,
    "facts": _facts,
    "quality": _quality,
    "analysis": _analysis,
    "benchmarks": _benchmarks,
    "figures": _figures,
    "docs": _docs,
    "verify": verify,
}

#: The full run, in dependency order.
ALL_STAGES = (
    "extract", "bronze", "silver", "dims", "facts", "quality",
    "analysis", "benchmarks", "figures", "docs", "verify",
)


def run_all() -> None:
    """Execute every stage of the core pipeline in order."""
    started = time.perf_counter()
    manifest = get_manifest()
    manifest.start_run()

    LOG.info("#" * 78)
    LOG.info("CSC 796 Advanced Data Engineering -- multi-grain mobility platform")
    LOG.info("Full pipeline run started %s",
             dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    LOG.info("Budgets: %.1f GB disk, %d minutes runtime (excluding downloads)",
             settings.DISK_BUDGET_GB, settings.RUNTIME_BUDGET_MINUTES)
    LOG.info("#" * 78)

    for name in ALL_STAGES:
        STAGES[name]()

    elapsed = time.perf_counter() - started
    # The declared budget EXCLUDES downloads (constraint 2.4), so extraction
    # time is measured and reported separately rather than counted against it.
    extract_seconds = sum(
        seconds for label, seconds in stage_timings().items()
        if label.startswith("extract:")
    )
    processing_seconds = elapsed - extract_seconds
    within_budget = processing_seconds / 60.0 <= settings.RUNTIME_BUDGET_MINUTES
    manifest.data["runtime"] = {
        "total_seconds": round(elapsed, 1),
        "extraction_seconds": round(extract_seconds, 1),
        "processing_seconds_excluding_downloads": round(processing_seconds, 1),
        "budget_minutes": settings.RUNTIME_BUDGET_MINUTES,
        "within_budget": within_budget,
    }
    manifest.finish_run()

    LOG.info("#" * 78)
    LOG.info("PIPELINE COMPLETE in %.1f minutes total", elapsed / 60.0)
    LOG.info("  extraction (downloads, excluded from budget): %.1f min",
             extract_seconds / 60.0)
    LOG.info("  processing (counted against the %d-minute budget): %.1f min",
             settings.RUNTIME_BUDGET_MINUTES, processing_seconds / 60.0)
    if not within_budget:
        LOG.warning(
            "Processing exceeded the declared %d-minute budget. Reported as "
            "measured; the budget is not met by shrinking the work.",
            settings.RUNTIME_BUDGET_MINUTES,
        )
    LOG.info("Manifest: %s", relative_to_root(settings.RUN_MANIFEST))
    LOG.info("#" * 78)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="CSC 796 multi-grain mobility and price platform.",
    )
    parser.add_argument(
        "--stage", default="all",
        choices=["all", "clean", *STAGES.keys()],
        help="Stage to run. 'all' runs the complete core pipeline.",
    )
    parser.add_argument("--force", action="store_true",
                        help="Refetch sources even when a cached copy exists.")
    parser.add_argument("--include-raw", action="store_true",
                        help="With --stage clean, also delete raw downloads.")
    parser.add_argument("--strict", action="store_true",
                        help="With --stage verify, exit non-zero if an artefact "
                             "is missing.")
    args = parser.parse_args(argv)

    configure_logging()
    ensure_dirs()

    try:
        if args.stage == "clean":
            clean(include_raw=args.include_raw)
        elif args.stage == "all":
            run_all()
        elif args.stage == "verify":
            verify(strict=args.strict)
        elif args.stage.startswith("extract"):
            which = None if args.stage == "extract" else args.stage.split("-", 1)[1]
            _extract(which, force=args.force)
        else:
            STAGES[args.stage]()
    except Exception as exc:  # noqa: BLE001 - top level: report, do not dump
        LOG.error("")
        LOG.error("PIPELINE FAILED in stage '%s'", args.stage)
        LOG.error("%s: %s", type(exc).__name__, exc)
        LOG.exception("Traceback follows for diagnosis:")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
