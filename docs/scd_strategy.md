# Slowly Changing Dimension Strategy

One dimension in this platform is Type 2. Every other is Type 1. This document
states why, and works through the change the test suite actually exercises.

## The decision

| Dimension | SCD type | Why |
| --- | --- | --- |
| `dim_geography` | **Type 2** | Attributes change while identity does not, and facts already loaded were genuinely recorded under the old attribution. |
| `dim_date` | Type 1 (in practice immutable) | Generated from the calendar. A day's attributes cannot change. |
| `dim_payment_type` | Type 1 | Decoded from a publisher-versioned dictionary. A change here is a correction. |
| `dim_rate_code` | Type 1 | As above. |
| `dim_vendor` | Type 1 | As above. |
| `dim_transport_mode` | Type 1 | Defined by the platform itself. |
| `dim_commodity` | Type 1 | Derived from the published commodity and unit; a rename is a correction, and unit is part of the grain so a unit change creates a new member rather than altering one. |
| `dim_trip_flags` | Type 1 (append-only in practice) | A junk dimension of observed combinations. Combinations are never deleted, because facts reference them. |

## Why Type 2 for geography specifically

Geography is the only dimension here whose **attributes can change while its
identity stays the same**:

- The TLC re-labels a taxi zone, or moves it between service zones.
- WFP revises the `admin1` a market is assigned to.
- Nigerian statistical publications reassign a state's grouping.

When that happens, the trips and prices already loaded were genuinely recorded
under the *old* attribution. Overwriting it — Type 1 — would silently rewrite
history: an analysis of 2024 demand by borough, rerun next year, would return
different numbers with no record that anything changed. For a graded,
reproducible study that is unacceptable.

Type 2 keeps both truths and timestamps them. The cost is three extra columns
(`valid_from`, `valid_to`, `is_current`) on a dimension of a few hundred rows.
The asymmetry is the point: the cost lands on the tiny dimension, the benefit
lands on the 40-million-row fact.

The counter-argument, stated fairly: Type 2 on geography means every fact join
must filter `is_current` (or join on the surrogate captured at load time), and
every analyst must understand why a natural key can appear more than once.
That is a real usability cost, and it is why the other seven dimensions are
Type 1 rather than Type 2 "for consistency".

## Mechanics

Natural key: `(geo_code, country, admin_level)`. A change to any part of it is a
different place, not a new version.

Tracked attributes: `geo_name`, `parent_geo_name`, `region_group`. A change to
any one of them closes the current version and opens a new one.

| Incoming row | Action |
| --- | --- |
| Natural key present, tracked attributes identical | **Untouched.** This is what makes reruns idempotent. |
| Natural key present, a tracked attribute differs | Close the current row (`valid_to = effective_date - 1 day`, `is_current = FALSE`), insert a new current version with a fresh `geo_key`. |
| Natural key absent from the dimension | Insert as a current version. |
| Natural key in the dimension but absent from the source | **Left current.** A place vanishing from one vintage of a feed is not evidence that it ceased to exist, and closing it would orphan facts already loaded against it. |

`valid_from` on the initial load is a **fixed epoch** (`2016-01-01`), not the run
date, so two cold runs on different days produce byte-identical dimensions
(constraint 2.6, determinism). Genuine later changes are stamped with the date
the change was detected.

The invariant is enforced as a quality rule, not just asserted in prose:

> **Q078 `scd2_one_current_row_per_natural_key`** — `(geo_code, country,
> admin_level)` must be unique among rows where `is_current`. Severity: `fail`.

## Worked example, from the test suite

`tests/test_scd2_geography.py` loads the dimension, then reloads it with one
attribute changed, and asserts the versioning behaviour.

**Step 1 — initial load.** Lagos market `Mile 12` arrives with
`region_group = 'South West'`:

| geo_key | geo_code | geo_name | admin_level | region_group | valid_from | valid_to | is_current |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 312 | 42 | Mile 12 | market | South West | 2016-01-01 | 9999-12-31 | true |

**Step 2 — reload, unchanged.** The same row arrives again. Nothing happens: no
new key, no closed row. This is the idempotence case, and it is asserted
explicitly because a Type 2 implementation that versions on every run is the
most common way to get this wrong.

**Step 3 — reload with `region_group` changed** to `'South East'`, effective
`2026-06-01`:

| geo_key | geo_code | geo_name | admin_level | region_group | valid_from | valid_to | is_current |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 312 | 42 | Mile 12 | market | South West | 2016-01-01 | **2026-05-31** | **false** |
| 431 | 42 | Mile 12 | market | **South East** | **2026-06-01** | 9999-12-31 | **true** |

The test asserts all of it: exactly one current row for the natural key, the
prior row closed the day before the change, the new row carrying a *different*
surrogate key, and the total row count for that natural key rising from one to
two.

Facts loaded before the change keep pointing at `geo_key = 312` and therefore
still report under `South West`, which is what they were actually recorded
under. Facts loaded afterwards point at `431`. Neither analysis silently
changes.

## What this platform deliberately does *not* implement

- **Type 3** (a `previous_value` column). It records only one prior state, which
  is a poor fit for geography that may be reassigned repeatedly.
- **Type 6 / hybrid.** It would let an analyst query "as of today's hierarchy"
  without restating the load, which is genuinely useful, but it adds a
  current-value column to every row that must be updated across all versions on
  every change. For a dimension of a few hundred rows serving a study of fixed
  scope, the complexity is not earned.
- **Surrogate key reuse.** Keys are allocated monotonically and never recycled,
  so an old fact can never silently acquire a new meaning.
