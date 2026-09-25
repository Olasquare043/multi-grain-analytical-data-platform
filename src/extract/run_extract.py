"""Orchestrates the four extraction patterns and records their provenance.

Keeping orchestration separate from the individual extractors is what lets any
one source be rerun alone (``make extract-nbs``) while still contributing to the
same run manifest.
"""
from __future__ import annotations

from typing import Callable

from src.extract import hdx_prices, nbs_pms, nyc_trips, weather
from src.extract.base import ExtractionResult
from src.utils.io_utils import ensure_dirs
from src.utils.logging_setup import get_logger
from src.utils.manifest import get_manifest

LOG = get_logger(__name__)

#: Ordered so that the cheapest, most likely to fail source runs last and the
#: expensive bulk download runs first while the operator is still watching.
EXTRACTORS: dict[str, Callable[..., list[ExtractionResult]]] = {
    "nyc": nyc_trips.extract,
    "hdx": hdx_prices.extract,
    "nbs": nbs_pms.extract,
    "weather": weather.extract,
}


def run(which: str | None = None, force: bool = False) -> list[ExtractionResult]:
    """Run one or all extractors and write their provenance to the manifest.

    Args:
        which: ``'nyc'``, ``'hdx'``, ``'nbs'``, ``'weather'`` or ``None`` for all.
        force: Refetch even when a cached copy exists.

    Returns:
        Every :class:`ExtractionResult` produced, in execution order.
    """
    ensure_dirs()
    manifest = get_manifest()

    selected = EXTRACTORS if which is None else {which: EXTRACTORS[which]}
    if which is not None and which not in EXTRACTORS:
        raise KeyError(
            f"unknown extractor '{which}'. Valid: {sorted(EXTRACTORS)}"
        )

    results: list[ExtractionResult] = []
    for name, extractor in selected.items():
        produced = extractor(force=force)
        results.extend(produced)
        for result in produced:
            if result.skipped:
                manifest.record_skipped(result.key, result.reason)
            else:
                # A source that succeeds now must not keep a 'skipped' entry
                # left over from an earlier run: the manifest is merged across
                # stage reruns, and a stale entry would contradict the
                # extraction record beside it.
                manifest.data.setdefault("skipped", {}).pop(result.key, None)
            manifest.record_inputs(result.key, result.manifest_records())
            manifest.data.setdefault("extraction", {})[result.key] = {
                "name": result.name,
                "pattern": result.pattern,
                "ok": result.ok,
                "skipped": result.skipped,
                "reason": result.reason,
                "file_count": len(result.files),
                "total_rows": result.total_rows,
                "diagnostics": result.diagnostics,
            }
        manifest.save()

    LOG.info("-" * 78)
    LOG.info("EXTRACTION SUMMARY")
    for result in results:
        LOG.info("  %s", result.summary_line())
    LOG.info("-" * 78)
    return results
