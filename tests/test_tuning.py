"""Proves the expanding-window CV folds used for once-only hyperparameter
selection (src/modelling/tuning.py) never let a fold's validation months
precede or overlap its own training months, and never manufacture a month
outside the list they were built from -- the CV-split analogue of
tests/test_time_split.py's assertion for the outer train/test split.
"""
from __future__ import annotations

import pytest

from src.modelling.tuning import expanding_window_folds


def test_folds_are_strictly_forward_chaining() -> None:
    months = list(range(1, 11))  # a 10-month training period, e.g. NYC's
    folds = expanding_window_folds(months, min_train_months=6, val_block=1)

    assert len(folds) == 4  # validates months 7, 8, 9, 10
    for fold in folds:
        assert max(fold.train_months) < min(fold.val_months), (
            "a fold's validation months must be strictly after its training months"
        )
        assert set(fold.train_months) | set(fold.val_months) <= set(months)
        assert not (set(fold.train_months) & set(fold.val_months))


def test_folds_expand_and_never_shrink() -> None:
    months = list(range(1, 11))
    folds = expanding_window_folds(months, min_train_months=6, val_block=1)
    sizes = [len(f.train_months) for f in folds]
    assert sizes == sorted(sizes), "training windows must only ever grow, fold to fold"
    assert sizes[0] == 6


def test_folds_never_include_a_month_outside_the_supplied_training_period() -> None:
    # Nigeria-shaped month keys (YYYYMM01 integers), a 25-month training period.
    months = [20231101 + i for i in range(25)]  # not real calendar arithmetic;
    # only used here to check no fold ever invents a value outside `months`.
    folds = expanding_window_folds(months, min_train_months=15, val_block=3)
    all_seen = set()
    for fold in folds:
        all_seen |= set(fold.train_months) | set(fold.val_months)
    assert all_seen <= set(months)


def test_too_few_months_yields_no_folds_rather_than_a_bad_one() -> None:
    months = [1, 2, 3]
    folds = expanding_window_folds(months, min_train_months=6, val_block=1)
    assert folds == []


def test_duplicate_and_unsorted_month_values_are_handled() -> None:
    months = [3, 1, 2, 2, 5, 4, 1]
    folds = expanding_window_folds(months, min_train_months=3, val_block=1)
    assert folds[0].train_months == (1, 2, 3)
    assert folds[0].val_months == (4,)
