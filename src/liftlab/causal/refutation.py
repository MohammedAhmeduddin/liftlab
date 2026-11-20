"""
Refutation tests: try to break an estimate.

Each test takes ``estimate_fn(df) -> float`` (e.g. a difference in means or
AIPW wrapped in a lambda), so any estimator can be checked.

- placebo_treatment:  shuffle the treatment labels. The "effect" of a
                      random label should be ~0; the real estimate should
                      sit far outside the placebo distribution.
- random_common_cause: add an independent random covariate. A sound
                      adjustment estimate should barely move.
- subset_stability:   re-estimate on random halves. Large swings mean the
                      estimate is fragile.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

EstimateFn = Callable[[pd.DataFrame], float]


def placebo_treatment(
    estimate_fn: EstimateFn, df: pd.DataFrame, treatment_col: str = "treatment",
    n_permutations: int = 50, seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    actual = float(estimate_fn(df))
    placebo = []
    for _ in range(n_permutations):
        d = df.copy()
        d[treatment_col] = rng.permutation(d[treatment_col].to_numpy())
        placebo.append(float(estimate_fn(d)))
    placebo = np.asarray(placebo)
    # permutation p-value with the +1 correction (never exactly 0)
    p = (np.sum(np.abs(placebo) >= abs(actual)) + 1) / (n_permutations + 1)
    return {"test": "placebo_treatment", "actual": actual,
            "placebo_mean": float(placebo.mean()), "placebo_std": float(placebo.std(ddof=1)),
            "p_value": float(p), "passed": bool(p < 0.05)}


def random_common_cause(
    estimate_fn: EstimateFn, df: pd.DataFrame, column: str = "_random_cause", seed: int = 0,
    tolerance: float = 0.1,
) -> dict:
    """``estimate_fn`` must use ``column`` as a covariate for this to be informative."""
    rng = np.random.default_rng(seed)
    actual = float(estimate_fn(df))
    d = df.copy()
    d[column] = rng.standard_normal(len(d))
    new = float(estimate_fn(d))
    rel = abs(new - actual) / abs(actual) if actual else float("inf")
    return {"test": "random_common_cause", "actual": actual, "with_random_cause": new,
            "relative_change": float(rel), "passed": bool(rel <= tolerance)}


def subset_stability(
    estimate_fn: EstimateFn, df: pd.DataFrame, frac: float = 0.5, n_subsets: int = 10,
    seed: int = 0,
) -> dict:
    rng = np.random.default_rng(seed)
    actual = float(estimate_fn(df))
    ests = np.asarray([
        float(estimate_fn(df.sample(frac=frac, random_state=int(rng.integers(1 << 31)))))
        for _ in range(n_subsets)
    ])
    same_sign = float(np.mean(np.sign(ests) == np.sign(actual)))
    return {"test": "subset_stability", "actual": actual, "subset_mean": float(ests.mean()),
            "subset_std": float(ests.std(ddof=1)), "share_same_sign": same_sign,
            "passed": bool(same_sign >= 0.9)}
