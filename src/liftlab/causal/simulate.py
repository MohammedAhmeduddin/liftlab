"""
Simulated data with KNOWN causal effects, used to verify every estimator.

Real data never tells you the true effect, so estimators are tested here
against a ground truth first, then run on Criteo.

- simulate_rct:        randomized trial shaped like Criteo (12 features,
                       ~85% treated, rare binary outcome, effect that
                       varies with f8).
- make_confounded:     turns an RCT sample into a biased "observational"
                       one by dropping control units depending on a
                       feature. Treated units are untouched, so the effect
                       on the treated equals the RCT effect: a known target.
- simulate_panel:      two-period panel for difference-in-differences,
                       with groups that differ in level (so a post-only
                       comparison is biased) but share a trend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [f"f{i}" for i in range(12)]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_rct(
    n: int = 20_000,
    treat_rate: float = 0.85,
    base_rate: float = 0.05,
    base_effect: float = 0.01,
    responder_multiplier: float = 4.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Randomized trial with a heterogeneous effect.

    Users in the lowest quartile of f8 respond ``responder_multiplier`` times
    more strongly. Returns the data plus the ground truth in ``_true_cate``
    (per-user effect on the probability of the outcome).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURES)))
    df = pd.DataFrame(X, columns=FEATURES)

    logit0 = np.log(base_rate / (1 - base_rate)) + 0.5 * df.f0 - 0.4 * df.f1 + 0.3 * df.f2
    p0 = _sigmoid(logit0.to_numpy())
    responder = (df.f8 < df.f8.quantile(0.25)).to_numpy()
    tau = np.where(responder, base_effect * responder_multiplier, base_effect)
    t = rng.binomial(1, treat_rate, n)  # randomized: independent of features
    p = np.clip(p0 + t * tau, 0, 1)

    df["treatment"] = t
    df["conversion"] = rng.binomial(1, p)
    df["visit"] = np.maximum(df["conversion"], rng.binomial(1, np.clip(p * 5, 0, 1)))
    df["exposure"] = t * rng.binomial(1, 0.9, n)
    df["_true_cate"] = tau
    return df


def make_confounded(
    df: pd.DataFrame, feature: str = "f0", strength: float = 1.5, seed: int = 0
) -> pd.DataFrame:
    """Biased subsample: control units are kept with probability that rises
    with ``feature``, so treated and control differ in that feature.

    ``feature`` should also drive the outcome (f0 does in simulate_rct and
    is a strong predictor in Criteo), otherwise the bias is invisible.
    """
    rng = np.random.default_rng(seed)
    is_control = df["treatment"].to_numpy() == 0
    z = (df[feature] - df[feature].mean()) / df[feature].std()
    keep_p = _sigmoid(strength * z.to_numpy())
    keep = ~is_control | (rng.random(len(df)) < keep_p)
    return df[keep].reset_index(drop=True)


def simulate_panel(
    n_units: int = 4_000,
    effect: float = 2.0,
    group_gap: float = 5.0,
    trend: float = 1.5,
    noise: float = 1.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Two-period panel (long format) with a known effect.

    Treated units start ``group_gap`` higher (selection on levels), both
    groups gain ``trend`` between periods, and treated units gain an extra
    ``effect`` in the post period. DiD should recover ``effect``; a
    post-period comparison of groups would report ``group_gap + effect``.
    """
    rng = np.random.default_rng(seed)
    unit = np.arange(n_units)
    treated = rng.binomial(1, 0.5, n_units)
    unit_level = rng.normal(0, 1, n_units) + group_gap * treated
    rows = []
    for post in (0, 1):
        y = unit_level + trend * post + effect * treated * post + rng.normal(0, noise, n_units)
        rows.append(pd.DataFrame({"unit": unit, "treated": treated, "post": post, "y": y}))
    return pd.concat(rows, ignore_index=True)
