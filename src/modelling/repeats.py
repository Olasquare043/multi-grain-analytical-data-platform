"""Aggregation of repeated ladder runs, and paired rung-to-rung comparison.

WHY THIS EXISTS. The ladder originally reported one number per rung. Task B's
largest improvement over its own baseline was about 3.4% of MAE, and Task B's
sampling was, at the time, not reproducible between runs -- so that 3.4% could
not be distinguished from the variation a rerun would have produced anyway.
An effect smaller than the noise it was never measured against is not a
finding. This module measures the noise.

Every rung of both ladders is now run five times (`splits.REPEAT_SEEDS`),
varying the model's `random_state` and -- for Task B -- the deterministic
sample bucket together, so a repeat reflects both the model's own randomness
and the luck of which rows were drawn.

TWO THINGS ARE REPORTED, AND THEY ANSWER DIFFERENT QUESTIONS.

`aggregate_repeats` gives each rung's mean, standard deviation, minimum and
maximum across repeats: how stable is this rung's error?

`paired_comparisons` answers the question the ladder actually asks, which is
comparative, and must be computed WITHIN a seed. Because every rung is run on
the same five seeds, rung B minus rung A can be taken seed by seed, which
cancels the shared run-to-run variation the two rungs experienced together.
Comparing two marginal distributions instead -- "B's mean looks lower than A's
mean, and the error bars overlap a bit" -- discards that pairing and is
strictly weaker evidence.

ON STATISTICAL TESTING. Five repeats is a small sample, and nothing here
reports a p-value. A t-test on five paired differences would invite exactly
the over-reading this module exists to prevent. What is reported instead is
descriptive and hard to misread:

  * `n_same_direction` -- how many of the five repeats moved the same way as
    the mean difference. Five of five means every repeat agreed on the sign.
    Three of five means the direction flipped in two repeats, and the effect
    should not be described as established, whatever its mean looks like.
  * `mean_over_sd` -- the mean difference in units of its own standard
    deviation. A mean difference far smaller than the spread of the
    differences it averages is not a dependable effect.

These are indications of stability, not inferential tests, and
docs/modelling_notes.md says so in the same words.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.modelling.ladder import RungResult

#: Standard deviations use ddof=1 (the sample standard deviation). With five
#: repeats the distinction from ddof=0 is not negligible, and ddof=1 is the
#: conservative choice: it reports the larger spread.
SD_DDOF = 1

#: Rungs excluded when searching for "the best rung": V3a is deliberately
#: leaky and its number is never to be trusted, and the REF_* rows are
#: reference predictors rather than ladder rungs.
NOT_COMPARABLE_RUNGS = ("V3a", "REF_mean", "REF_heuristic")


def results_to_long_frame(results: list[RungResult], mae_col: str = "mae") -> pd.DataFrame:
    """One row per (variant, rung, seed) -- the raw material for everything else."""
    return pd.DataFrame([
        {
            "variant": r.variant,
            "rung": r.rung,
            "seed": r.seed,
            "description": r.description,
            mae_col: r.mae,
            "mape_pct": r.mape,
            "notes": r.notes,
            "n_train": r.n_train,
            "n_test": r.n_test,
        }
        for r in results
    ])


def aggregate_repeats(results: list[RungResult], mae_col: str = "mae") -> pd.DataFrame:
    """Collapse repeated runs to one row per (variant, rung), with uncertainty.

    The single-run columns (`<mae_col>`, `mape_pct`) are kept and carry the
    FIRST repeat's values, so the CSV's original columns still mean something
    concrete and downstream readers of those columns do not break. Every
    claim in docs/modelling_notes.md is made against the `_mean` columns and
    qualified by the `_sd` columns, never against the single-run value.
    """
    long = results_to_long_frame(results, mae_col=mae_col)
    rows: list[dict] = []

    for (variant, rung), group in long.groupby(["variant", "rung"], sort=False):
        ordered = group.sort_values("seed")
        first = ordered.iloc[0]
        row = {
            "variant": variant,
            "rung": rung,
            "description": first["description"],
            # Single-run columns: the first repeat, kept for continuity.
            mae_col: round(float(first[mae_col]), 4),
            "mape_pct": round(float(first["mape_pct"]), 4),
            # Across-repeat summary: what every claim is actually made against.
            "mae_mean": round(float(ordered[mae_col].mean()), 4),
            "mae_sd": round(float(ordered[mae_col].std(ddof=SD_DDOF)), 4),
            "mae_min": round(float(ordered[mae_col].min()), 4),
            "mae_max": round(float(ordered[mae_col].max()), 4),
            "mape_mean": round(float(ordered["mape_pct"].mean()), 4),
            "mape_sd": round(float(ordered["mape_pct"].std(ddof=SD_DDOF)), 4),
            "mape_min": round(float(ordered["mape_pct"].min()), 4),
            "mape_max": round(float(ordered["mape_pct"].max()), 4),
            "n_repeats": int(len(ordered)),
            "seeds": ",".join(str(s) for s in ordered["seed"]),
            "notes": first["notes"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Pair:
    """One comparison to make: does moving from `rung_a` to `rung_b` help?"""

    rung_a: str
    rung_b: str
    label: str = ""


def adjacent_pairs(rungs: list[str]) -> list[Pair]:
    """Every step up the ladder, in order: V0->V1, V1->V2, and so on."""
    return [
        Pair(a, b, label="adjacent")
        for a, b in zip(rungs, rungs[1:])
    ]


def paired_comparisons(
    results: list[RungResult],
    pairs: list[Pair],
    task: str,
    mae_col: str = "mae",
) -> pd.DataFrame:
    """Compare rungs within seed, and report how consistently they differ.

    For each pair and each variant: take rung B's MAE minus rung A's MAE
    **for the same seed**, five times, then summarise those five differences.
    A negative mean difference means B has the lower error, i.e. the step
    from A to B helped.

    `n_same_direction` counts how many of the repeats agree with the mean's
    sign. When that is not all of them, the step's direction is not
    established and must not be reported as though it were.
    """
    long = results_to_long_frame(results, mae_col=mae_col)
    rows: list[dict] = []

    for variant in long["variant"].unique():
        scoped = long[long["variant"] == variant]
        for pair in pairs:
            a = scoped[scoped["rung"] == pair.rung_a].set_index("seed")[mae_col]
            b = scoped[scoped["rung"] == pair.rung_b].set_index("seed")[mae_col]
            shared_seeds = sorted(set(a.index) & set(b.index))
            if not shared_seeds:
                continue

            diffs = np.array([b[s] - a[s] for s in shared_seeds], dtype=float)
            mean_diff = float(diffs.mean())
            sd_diff = float(diffs.std(ddof=SD_DDOF)) if len(diffs) > 1 else 0.0
            # Sign agreement with the mean. A difference of exactly zero
            # agrees with nothing and is counted against consistency.
            if mean_diff == 0:
                n_same = 0
            else:
                n_same = int(np.sum(np.sign(diffs) == np.sign(mean_diff)))

            rows.append({
                "task": task,
                "variant": variant,
                "rung_a": pair.rung_a,
                "rung_b": pair.rung_b,
                "comparison": pair.label,
                "n_repeats": len(diffs),
                "mean_diff_mae": round(mean_diff, 4),
                "sd_diff_mae": round(sd_diff, 4),
                "min_diff_mae": round(float(diffs.min()), 4),
                "max_diff_mae": round(float(diffs.max()), 4),
                # Mean difference in units of its own spread. NaN when the
                # spread is zero (a deterministic comparison), which is not
                # the same as "infinitely reliable" and must not be read so.
                "mean_over_sd": (round(mean_diff / sd_diff, 3) if sd_diff > 0 else np.nan),
                "n_same_direction": n_same,
                "direction_consistent": bool(n_same == len(diffs)),
                "better_rung": (
                    pair.rung_b if mean_diff < 0 else pair.rung_a if mean_diff > 0 else "tie"
                ),
                "per_seed_diffs": ";".join(f"{d:.4f}" for d in diffs),
            })

    return pd.DataFrame(rows)


def best_rung_by_mean_mae(summary: pd.DataFrame, variant: str) -> str:
    """The lowest mean-MAE rung of one variant, ignoring leaky and reference rows.

    V3a is excluded because its number is knowingly inflated by leakage and
    was never a candidate; REF_* are excluded because they are reference
    points, not rungs.
    """
    scoped = summary[
        (summary["variant"] == variant) & (~summary["rung"].isin(NOT_COMPARABLE_RUNGS))
    ]
    return str(scoped.loc[scoped["mae_mean"].idxmin(), "rung"])
