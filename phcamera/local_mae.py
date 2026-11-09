"""Local MAE: the model-selection metric introduced in the paper (Fig. 2).

The mean absolute error (MAE) is computed separately for every pH level in the
test data, and the *maximum* of these per-level values is taken as the
representative error of the model.  Unlike the overall MAE, this penalizes a
model that is accurate on average but poor in a particular pH range (for
example where the universal indicator changes color quickly).
"""

from __future__ import annotations

import numpy as np


def per_level_mae(y_true, y_pred, decimals: int = 6) -> dict[float, float]:
    """Return ``{pH level: MAE}``.

    Samples are grouped by identical ``y_true`` (rounded to ``decimals`` to
    absorb floating-point noise).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if y_true.size == 0:
        raise ValueError("empty input")
    keys = np.round(y_true, decimals)
    return {
        float(k): float(np.mean(np.abs(y_true[keys == k] - y_pred[keys == k]))) for k in np.unique(keys)
    }


def local_mae(y_true, y_pred, decimals: int = 6) -> float:
    """Return the maximum per-pH-level MAE (the paper's Local MAE)."""
    return max(per_level_mae(y_true, y_pred, decimals).values())
