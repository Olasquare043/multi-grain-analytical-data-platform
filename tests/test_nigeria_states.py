"""The Nigerian state normaliser, against the variants the sources actually use.

Two independently collected sources spell the same federating unit differently,
and the NBS workbooks mix geopolitical-zone subtotals and a Grand Total row into
the same column as real states. This module is the single authority that
reconciles them, so its edge cases are worth pinning down.
"""
from __future__ import annotations

import pytest

from config.nigeria_states import (
    CANONICAL_STATES, GEOPOLITICAL_ZONES, STATE_TO_ZONE, is_non_state_label,
    normalise_state, zone_for_state,
)


def test_nigeria_has_thirty_seven_federating_units() -> None:
    """36 states plus the FCT.

    The build specification says "37 states plus FCT", which implies 38. That is
    a miscount, and the paper's denominators depend on getting it right.
    """
    assert len(CANONICAL_STATES) == 37
    assert "FCT" in CANONICAL_STATES
    assert len(STATE_TO_ZONE) == 37
    assert sum(len(v) for v in GEOPOLITICAL_ZONES.values()) == 37


def test_every_state_belongs_to_exactly_one_zone() -> None:
    seen: dict[str, str] = {}
    for zone, states in GEOPOLITICAL_ZONES.items():
        for state in states:
            assert state not in seen, (
                f"{state} appears in both {seen[state]} and {zone}"
            )
            seen[state] = zone
    assert len(GEOPOLITICAL_ZONES) == 6


@pytest.mark.parametrize("raw,expected", [
    # Federal Capital Territory: the most variable label in both sources
    ("FCT", "FCT"),
    ("FCT Abuja", "FCT"),
    ("Abuja", "FCT"),
    ("Federal Capital Territory", "FCT"),
    ("fct, abuja", "FCT"),
    ("FCT (Abuja)", "FCT"),
    # Nasarawa, routinely doubled-s in NBS releases
    ("Nassarawa", "Nasarawa"),
    ("Nasarawa", "Nasarawa"),
    ("NASSARAWA", "Nasarawa"),
    # Hyphenated compounds
    ("Akwa-Ibom", "Akwa Ibom"),
    ("Akwa Ibom", "Akwa Ibom"),
    ("Cross-River", "Cross River"),
    ("Cross River", "Cross River"),
    # Whitespace, case and the trailing word 'State'
    ("  KANO  ", "Kano"),
    ("Lagos state", "Lagos"),
    ("Niger State", "Niger"),
    ("oyo", "Oyo"),
    # Common mis-keyings observed in the workbooks
    ("Kastina", "Katsina"),
    ("Bornu", "Borno"),
])
def test_known_variants_normalise(raw: str, expected: str) -> None:
    assert normalise_state(raw) == expected


@pytest.mark.parametrize("raw", [
    "Grand Total", "GRAND TOTAL", "Total", "Average", "AVERAGE",
    "North Central", "North East", "North West",
    "South East", "South South", "South West",
    "Zone", "Nigeria", "National", "All States",
    "", "   ", None,
])
def test_aggregate_and_blank_labels_are_rejected(raw: object) -> None:
    """Subtotal rows must never be mistaken for a place."""
    assert normalise_state(raw) is None


def test_every_canonical_name_is_a_fixed_point() -> None:
    """Normalising an already-canonical name must not change it."""
    for state in CANONICAL_STATES:
        assert normalise_state(state) == state


def test_zone_lookup_round_trips() -> None:
    assert zone_for_state("FCT") == "North Central"
    assert zone_for_state("Lagos") == "South West"
    assert zone_for_state(normalise_state("Nassarawa")) == "North Central"
    assert zone_for_state(None) is None
    assert zone_for_state("Atlantis") is None


def test_is_non_state_label_agrees_with_normalise() -> None:
    """The two helpers must not disagree about what counts as a subtotal."""
    for label in ("Grand Total", "South South", "Average"):
        assert is_non_state_label(label) is True
        assert normalise_state(label) is None
    for label in ("Lagos", "FCT Abuja", "Nassarawa"):
        assert is_non_state_label(label) is False
        assert normalise_state(label) is not None


def test_unresolvable_labels_return_none_rather_than_guessing() -> None:
    """A label that is not a state must not be forced onto the nearest match."""
    for label in ("Price - Lagos", "Lagos and Ogun", "Unknown", "N/A", "12345"):
        assert normalise_state(label) is None, (
            f"{label!r} should be reported as unresolvable, not guessed at"
        )
