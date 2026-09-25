"""Canonical Nigerian sub-national geography.

Two independently collected sources (WFP market prices, NBS fuel price watch)
spell the same federating unit in different ways, and the NBS spreadsheets mix
geopolitical-zone subtotals and a ``Grand Total`` row into the same column as
real states. This module is the single authority that reconciles them, so the
conformed ``dim_geography`` has exactly one row per real place.

Nomenclature note for the write-up: Nigeria has **36 states plus the Federal
Capital Territory**, i.e. 37 federating units -- not 38. The build prompt's
phrase "37 states plus FCT" is a miscount; 37 is the correct total and is what
this module enforces.
"""
from __future__ import annotations

import re
import unicodedata

#: The six geopolitical zones and their constituent federating units.
#: Zones are an administrative convention rather than a tier of government,
#: which is precisely why they appear as subtotal rows in NBS releases.
GEOPOLITICAL_ZONES: dict[str, tuple[str, ...]] = {
    "North Central": ("Benue", "FCT", "Kogi", "Kwara", "Nasarawa", "Niger", "Plateau"),
    "North East": ("Adamawa", "Bauchi", "Borno", "Gombe", "Taraba", "Yobe"),
    "North West": ("Jigawa", "Kaduna", "Kano", "Katsina", "Kebbi", "Sokoto", "Zamfara"),
    "South East": ("Abia", "Anambra", "Ebonyi", "Enugu", "Imo"),
    "South South": ("Akwa Ibom", "Bayelsa", "Cross River", "Delta", "Edo", "Rivers"),
    "South West": ("Ekiti", "Lagos", "Ogun", "Ondo", "Osun", "Oyo"),
}

#: Canonical spellings, alphabetical. 36 states + FCT = 37.
CANONICAL_STATES: tuple[str, ...] = tuple(
    sorted({state for states in GEOPOLITICAL_ZONES.values() for state in states})
)

#: state -> zone, derived rather than restated so the two cannot drift apart.
STATE_TO_ZONE: dict[str, str] = {
    state: zone for zone, states in GEOPOLITICAL_ZONES.items() for state in states
}

#: Labels that occupy the state column in NBS workbooks but are not places.
#: Filtered out and *reported* in the quality output, never silently discarded.
NON_STATE_LABELS: frozenset[str] = frozenset(
    {
        "grand total", "total", "average", "national", "national average",
        "nigeria", "mean", "overall", "all states", "sum",
        # geopolitical zone subtotal rows
        "north central", "north east", "north-east", "north east zone",
        "north west", "north-west", "north central zone", "north west zone",
        "south east", "south-east", "south east zone",
        "south south", "south-south", "south south zone",
        "south west", "south-west", "south west zone",
        "northcentral", "northeast", "northwest",
        "southeast", "southsouth", "southwest",
    }
)

#: Known orthographic variants observed across the two sources, normalised form
#: on the left. Keys are already lower-cased and whitespace-collapsed.
STATE_VARIANTS: dict[str, str] = {
    # Federal Capital Territory -- the most variable label in both sources
    "fct": "FCT",
    "fct abuja": "FCT",
    "fct, abuja": "FCT",
    "abuja": "FCT",
    "abuja fct": "FCT",
    "federal capital territory": "FCT",
    "federal capital territory abuja": "FCT",
    "fct (abuja)": "FCT",
    # Nasarawa
    "nassarawa": "Nasarawa",
    "nassarawa state": "Nasarawa",
    "nasarawa": "Nasarawa",
    # Akwa Ibom
    "akwa ibom": "Akwa Ibom",
    "akwa-ibom": "Akwa Ibom",
    "akwaibom": "Akwa Ibom",
    # Cross River
    "cross river": "Cross River",
    "cross-river": "Cross River",
    "crossriver": "Cross River",
    # Frequently mis-keyed northern states
    "kastina": "Katsina",
    "katsina": "Katsina",
    "bornu": "Borno",
    "borno": "Borno",
    "zampara": "Zamfara",
    "niger state": "Niger",
    "benue state": "Benue",
    # Occasional WFP spellings
    "ebony": "Ebonyi",
    "ogun state": "Ogun",
}

_WHITESPACE = re.compile(r"\s+")
_STATE_SUFFIX = re.compile(r"\s+state$")


def _fold(value: str) -> str:
    """Lower-case, strip accents, collapse whitespace, drop a trailing 'State'."""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace(" ", " ").strip().lower()
    text = _WHITESPACE.sub(" ", text)
    text = _STATE_SUFFIX.sub("", text)
    return text.strip(" .,;:-")


#: Folded canonical name -> canonical name, so exact matches are free.
_CANONICAL_FOLDED: dict[str, str] = {_fold(s): s for s in CANONICAL_STATES}


def normalise_state(raw: object) -> str | None:
    """Map a raw state label onto its canonical spelling.

    Returns ``None`` when the label is blank, a subtotal/aggregate row, or a
    value that cannot be resolved to one of the 37 federating units. Callers are
    expected to count and report the ``None`` cases rather than drop them
    silently (section 6, data quality framework).

    >>> normalise_state("FCT Abuja")
    'FCT'
    >>> normalise_state("Nassarawa")
    'Nasarawa'
    >>> normalise_state("Grand Total") is None
    True
    """
    if raw is None:
        return None
    folded = _fold(raw)
    if not folded:
        return None
    if folded in NON_STATE_LABELS:
        return None
    if folded in _CANONICAL_FOLDED:
        return _CANONICAL_FOLDED[folded]
    if folded in STATE_VARIANTS:
        return STATE_VARIANTS[folded]
    # Last resort: a label that contains exactly one canonical name as a token
    # run (e.g. "Kano  " already handled, "Price - Lagos" is not).
    hits = [
        canonical
        for canonical_folded, canonical in _CANONICAL_FOLDED.items()
        if canonical_folded == folded.replace("-", " ")
    ]
    if len(hits) == 1:
        return hits[0]
    return None


def zone_for_state(state: str | None) -> str | None:
    """Return the geopolitical zone for a canonical state name."""
    if state is None:
        return None
    return STATE_TO_ZONE.get(state)


def is_non_state_label(raw: object) -> bool:
    """True when a label is a recognised aggregate row rather than a place."""
    return _fold(raw) in NON_STATE_LABELS


# Guard: the module's own arithmetic must hold, or the paper's counts are wrong.
assert len(CANONICAL_STATES) == 37, f"expected 37 units, got {len(CANONICAL_STATES)}"
assert len(STATE_TO_ZONE) == 37, "a unit is listed in more than one zone"
