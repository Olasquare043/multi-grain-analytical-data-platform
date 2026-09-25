"""Once-only, time-respecting hyperparameter selection for the ladder's model
capacity.

Task A's original ladder (`LGBM_PARAMS` in `ladder.py`: 300 trees,
`num_leaves=31`, no L1/L2) made every rung past V0 look *worse* than the
naive baseline. That is very likely a model-capacity problem, not a data
engineering one: a 300-tree, 31-leaf booster on a 925-row training panel
will overfit almost any feature set it is given, which means the original
ladder never actually isolated the effect it was built to isolate.

The fix is not to tune each rung -- that would reintroduce exactly the
confound fixing the model exists to remove (`ladder.py`'s own docstring).
Instead, hyperparameters suited to the training set's size are selected
**once**, before any rung is fit, using only **V0's own baseline
features and data**, via **expanding-window cross-validation confined to
the training period** (never a test-period row), optimising MAE. The
winning hyperparameters are then frozen and reused, unchanged, at every
rung of the ladder -- the "only the data changes" principle applied to a
model capacity that actually fits the sample size, rather than to a model
capacity picked once for both a 925-row and a 2,000,000-row task.

Both the original (untuned, 300-tree) and capacity-controlled ladders are
kept and reported side by side (`docs/modelling_notes.md`,
`outputs/tables/model_ladder_*.csv`): the original's overfitting is itself
a finding and a live example for the point-in-time-correctness discussion,
not an error to be erased once a better version exists.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error

#: A compact, curated search space spanning "very small capacity" (few
#: leaves, small min_child_samples floor loosened, heavier regularisation)
#: through "close to the original 300-tree/31-leaf default" -- deliberately
#: not an exhaustive grid, so the NYC search (run on a 2,000,000-row sample,
#: per the instruction to keep runtime reasonable) stays tractable.
CANDIDATE_PARAMS: list[dict[str, Any]] = [
    dict(num_leaves=7,  learning_rate=0.10, min_child_samples=5,  reg_alpha=0.0, reg_lambda=0.0),
    dict(num_leaves=7,  learning_rate=0.10, min_child_samples=20, reg_alpha=1.0, reg_lambda=1.0),
    dict(num_leaves=7,  learning_rate=0.03, min_child_samples=5,  reg_alpha=0.0, reg_lambda=0.0),
    dict(num_leaves=15, learning_rate=0.05, min_child_samples=10, reg_alpha=0.0, reg_lambda=0.0),
    dict(num_leaves=15, learning_rate=0.05, min_child_samples=20, reg_alpha=1.0, reg_lambda=1.0),
    dict(num_leaves=15, learning_rate=0.10, min_child_samples=30, reg_alpha=0.5, reg_lambda=0.5),
    dict(num_leaves=31, learning_rate=0.05, min_child_samples=20, reg_alpha=1.0, reg_lambda=1.0),
    dict(num_leaves=31, learning_rate=0.10, min_child_samples=50, reg_alpha=1.0, reg_lambda=1.0),
]

MAX_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 30


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold: train on everything up to a growing
    cutoff, validate on the next block of months strictly after it."""

    train_months: tuple
    val_months: tuple


def expanding_window_folds(months: Sequence, min_train_months: int,
                            val_block: int) -> list[Fold]:
    """Forward-chaining (expanding-window) folds over a sorted month list.

    `months` must already be restricted to the TRAINING period alone (a
    test-period month must never be passed in): fold k's validation months
    are always strictly after fold k's training months, and both are always
    a subset of `months`, so no fold can ever see a test-period row.
    """
    ordered = sorted(set(months))
    folds: list[Fold] = []
    cutoff = min_train_months
    while cutoff + val_block <= len(ordered):
        train_months = tuple(ordered[:cutoff])
        val_months = tuple(ordered[cutoff:cutoff + val_block])
        folds.append(Fold(train_months, val_months))
        cutoff += val_block
    return folds


@dataclass
class TuningResult:
    best_params: dict[str, Any]
    frozen_n_estimators: int
    cv_table: pd.DataFrame
    seconds: float
    n_folds: int


def select_hyperparameters_cv(
    frame: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    month_col: str,
    folds: list[Fold],
    candidates: list[dict[str, Any]] | None = None,
    base_params: dict[str, Any] | None = None,
) -> TuningResult:
    """Select LightGBM hyperparameters once, via expanding-window CV on a
    baseline feature set, optimising MAE, never touching a test-period row.

    Each candidate is fit per fold with `n_estimators=MAX_ESTIMATORS` and
    early stopping on that fold's validation MAE (`EARLY_STOPPING_ROUNDS`),
    so `n_estimators` is itself selected by the search rather than grid-swept
    explicitly. The winning candidate is the one with the lowest MEAN
    validation MAE across folds; its frozen `n_estimators` for the ladder
    proper is the MEAN of its best iterations on the two LARGEST-training-
    window folds (the most recent ones), not a median across all folds --
    the ladder always refits on the full training period, so the tree count
    should come from the fold(s) closest in training-set size to that full
    fit, not be pulled down by folds trained on far fewer months.
    """
    candidates = candidates or CANDIDATE_PARAMS
    base = dict(base_params or {})
    started = time.perf_counter()

    records: list[dict[str, Any]] = []
    candidate_refs: list[dict[str, Any]] = []
    candidate_fold_iters: list[list[int]] = []

    for candidate in candidates:
        fold_maes: list[float] = []
        fold_best_iters: list[int] = []
        for fold in folds:
            train = frame[frame[month_col].isin(fold.train_months)]
            val = frame[frame[month_col].isin(fold.val_months)]
            if train.empty or val.empty:
                continue
            params = {
                **base, **candidate,
                "n_estimators": MAX_ESTIMATORS,
                "random_state": 796, "n_jobs": 4, "verbosity": -1,
            }
            model = LGBMRegressor(**params)
            model.fit(
                train[feature_cols], train[target_col],
                eval_set=[(val[feature_cols], val[target_col])],
                eval_metric="mae",
                callbacks=[early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                           log_evaluation(period=0)],
            )
            preds = model.predict(val[feature_cols])
            fold_maes.append(mean_absolute_error(val[target_col], preds))
            fold_best_iters.append(model.best_iteration_ or MAX_ESTIMATORS)

        if not fold_maes:
            continue
        records.append({
            **candidate,
            "mean_cv_mae": float(np.mean(fold_maes)),
            "n_folds_used": len(fold_maes),
            "median_best_iteration": int(np.median(fold_best_iters)),
            "largest_fold_best_iteration": int(np.mean(fold_best_iters[-2:])),
        })
        candidate_refs.append(candidate)
        candidate_fold_iters.append(fold_best_iters)

    if not records:
        raise RuntimeError(
            "hyperparameter CV selection produced no usable fold -- check "
            "that `folds` was built from the training period alone and is "
            "non-empty"
        )

    cv_table = pd.DataFrame(records)
    order = np.argsort(cv_table["mean_cv_mae"].to_numpy())
    cv_table = cv_table.iloc[order].reset_index(drop=True)
    winner_pos = int(order[0])

    best_params = dict(candidate_refs[winner_pos])
    # NOT the median across all folds: the expanding-window folds have
    # training windows ranging from `min_train_months` up to nearly the
    # full training period, and the ladder's real fit always uses the FULL
    # training period. A median best-iteration is dominated by the smaller,
    # earlier folds (e.g. 15-19 months of a 25-month period), whose early
    # stopping picks a tree count suited to THAT smaller sample -- freezing
    # it systematically underfits once refit on the full, larger training
    # set. Averaging the two folds with the LARGEST training windows (the
    # most recent, and therefore closest in size to the full training
    # period) instead of the median avoids that size mismatch, while still
    # smoothing over one fold's single noisy early-stopping point.
    frozen_n_estimators = int(np.mean(candidate_fold_iters[winner_pos][-2:]))
    elapsed = time.perf_counter() - started

    return TuningResult(
        best_params=best_params,
        frozen_n_estimators=frozen_n_estimators,
        cv_table=cv_table,
        seconds=round(elapsed, 2),
        n_folds=len(folds),
    )
