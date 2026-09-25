"""Runs one named rung of a feature-engineering ladder and returns its metrics.

Used identically by both ladder notebooks (Task A: Nigeria petrol price,
Task B: NYC trip duration), so a rung's number is never computed by
different code in the two notebooks. The model type and hyperparameters
(LGBM_PARAMS) are fixed across every rung of both ladders, so any change in
a rung's error is attributable to the data engineering step that produced
its features, not to a different model (section 3.3 / 4.4 of the v3
prompt). The one exception is the V4 encoding rung, which still fits an
LGBMRegressor with these same hyperparameters -- it just fits it twice, once
per encoding, since encoding is itself the thing being compared there.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

#: Fixed across every rung of both ladders. Chosen to be modest (this is a
#: 1,147-row panel for Task A) rather than tuned per rung, which would
#: reintroduce exactly the "different algorithm, not different data" ambiguity
#: fixing the model exists to remove.
LGBM_PARAMS: dict[str, Any] = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    random_state=796,
    # Explicit 4, not -1: n_jobs=-1 asks LightGBM to size its thread pool
    # from the HOST's core count, not the container's cgroup CPU limit
    # (config/settings.py CPU_LIMIT=4 / docker-compose.yml cpus: "4"), which
    # over-subscribes threads and their per-thread buffers on a host with
    # more cores than the container is actually allotted. This is an
    # infrastructure setting, identical across every rung of both ladders,
    # not a per-rung modelling choice.
    n_jobs=4,
    verbosity=-1,
)


#: The two ladder variants reported side by side (docs/modelling_notes.md):
#: the original, fixed-capacity ladder every rung was first run with, and
#: the capacity-controlled one built on hyperparameters selected once, via
#: expanding-window CV on V0's own baseline features (src/modelling/tuning.py),
#: then frozen and reused unchanged across every rung -- never re-tuned per
#: rung, for the same reason the model itself is never swapped per rung.
VARIANT_ORIGINAL = "original (untuned, 300 trees)"
VARIANT_CAPACITY_CONTROLLED = "capacity-controlled (CV-selected on V0)"

#: The reference predictors (src/modelling/baselines.py) fit no model, so
#: neither hyperparameter variant applies to them. They carry their own
#: variant label rather than being duplicated under both, which keeps them
#: out of every rung-to-rung comparison by construction: they are the
#: absolute anchor the ladder is measured against, not steps on it.
VARIANT_REFERENCE = "reference (no model fitted)"


@dataclass
class RungResult:
    """One rung's result. `model` and `predictions` are kept for diagnostic
    plots in the notebook but dropped before anything is written to CSV.

    `seed` records which repeat this result belongs to. Every rung of both
    ladders is run once per seed (`src/modelling/repeats.py`), so that a
    rung-to-rung difference can be compared *within* a seed rather than
    between two single runs whose difference might be nothing but the
    model's own bagging randomness.
    """

    rung: str
    description: str
    mae: float
    mape: float
    notes: str
    n_train: int
    n_test: int
    seconds: float
    variant: str = VARIANT_ORIGINAL
    seed: int = 796
    model: LGBMRegressor | None = field(default=None, repr=False)
    predictions: np.ndarray | None = field(default=None, repr=False)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error, in percent.

    Rows where y_true == 0 are excluded (percentage error is undefined
    there); the excluded count is returned to the caller's attention via a
    RuntimeWarning-free explicit check rather than silently propagating inf.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def score_predictions(
    rung: str,
    description: str,
    y_test: pd.Series,
    preds: np.ndarray,
    n_train: int,
    notes: str = "",
    variant: str = VARIANT_ORIGINAL,
    seed: int = 796,
    seconds: float = 0.0,
) -> RungResult:
    """Turn a vector of predictions into a scored `RungResult`.

    Exists so that the reference baselines (`src/modelling/baselines.py`),
    which fit no model at all, have their MAE and MAPE computed by exactly
    the same code as every fitted rung -- a baseline scored by a second,
    parallel implementation would not be a valid comparison point.
    """
    return RungResult(
        rung=rung,
        description=description,
        mae=float(mean_absolute_error(y_test, preds)),
        mape=mape(y_test.to_numpy(), preds),
        notes=notes,
        n_train=n_train,
        n_test=len(y_test),
        seconds=round(seconds, 3),
        variant=variant,
        seed=seed,
        model=None,
        predictions=preds,
    )


def run_rung(
    rung: str,
    description: str,
    X_train,
    y_train: pd.Series,
    X_test,
    y_test: pd.Series,
    notes: str = "",
    params: dict[str, Any] | None = None,
    variant: str = VARIANT_ORIGINAL,
    seed: int | None = None,
) -> RungResult:
    """Fit one LGBMRegressor on this rung's features and score it on the
    held-out test set. The single function both notebooks call for every
    rung of both ladders, and both ladder variants (original / capacity-
    controlled -- `params` is the only thing that differs between the two
    calls for the same rung; `variant` just labels the result for reporting).

    `X_train`/`X_test` are usually a `pandas.DataFrame`, but rung V4a (the
    one-hot encoding comparison, at NYC scale) passes a `scipy.sparse`
    matrix instead -- LightGBM accepts either natively, but a plain
    `len(X_train)` raises on a sparse matrix, so row counts are read via
    `.shape[0]`, which both support.

    `seed` overrides `random_state` for this fit. It is the *only* thing
    allowed to vary between repeats of the same rung: the hyperparameters
    stay exactly as `params` sets them, so a spread across seeds measures
    the model's own bagging/feature-sampling randomness and nothing else.
    """
    resolved = dict(params or LGBM_PARAMS)
    if seed is not None:
        resolved["random_state"] = seed

    started = time.perf_counter()
    model = LGBMRegressor(**resolved)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    elapsed = time.perf_counter() - started

    return RungResult(
        rung=rung,
        description=description,
        mae=float(mean_absolute_error(y_test, preds)),
        mape=mape(y_test.to_numpy(), preds),
        notes=notes,
        n_train=X_train.shape[0],
        n_test=X_test.shape[0],
        seconds=round(elapsed, 3),
        variant=variant,
        seed=int(resolved["random_state"]),
        model=model,
        predictions=preds,
    )


def results_to_frame(results: list[RungResult], mae_col: str = "mae") -> pd.DataFrame:
    """Assemble the ladder CSV: variant, rung, description, <mae_col>,
    mape_pct, notes.

    `mae_col` names the MAE column by unit (mae_ngn for Task A, mae_seconds
    for Task B) so a reader of the CSV never has to guess. `variant`
    distinguishes the original (untuned) ladder from the capacity-controlled
    one when both are written to the same CSV (docs/modelling_notes.md).
    """
    return pd.DataFrame([
        {
            "variant": r.variant,
            "rung": r.rung,
            "description": r.description,
            mae_col: round(r.mae, 4),
            "mape_pct": round(r.mape, 4),
            "notes": r.notes,
        }
        for r in results
    ])
