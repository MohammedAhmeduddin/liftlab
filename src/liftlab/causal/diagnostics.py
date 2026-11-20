"""
Randomization diagnostics: is treatment independent of the features?

A randomized trial should pass both checks. When it does not, the raw
difference in means is biased by whatever features differ between arms,
and a covariate-adjusted estimator (AIPW) is the better reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from liftlab.causal.estimators import standardized_mean_differences


def classifier_two_sample_test(
    df: pd.DataFrame,
    feature_cols: list[str],
    treatment_col: str = "treatment",
    max_rows: int = 2_000_000,
    n_permutations: int = 20,
    seed: int = 42,
) -> dict:
    """Train a classifier to predict treatment on half the rows and score it
    on the other half. Under randomization the held-out AUC is ~0.5; the
    shuffled-label AUCs show how far from 0.5 noise alone gets."""
    s = df.sample(n=min(max_rows, len(df)), random_state=seed)
    X = s[feature_cols].to_numpy(dtype=float)
    t = s[treatment_col].to_numpy()
    half = len(s) // 2
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(X[:half], t[:half])
    score = clf.predict_proba(X[half:])[:, 1]
    auc = float(roc_auc_score(t[half:], score))
    rng = np.random.default_rng(seed)
    null = [float(roc_auc_score(rng.permutation(t[half:]), score)) for _ in range(n_permutations)]
    return {"rows": int(len(s)), "auc": auc, "null_auc_min": min(null),
            "null_auc_max": max(null), "detected": auc > max(null)}


def balance(df: pd.DataFrame, feature_cols: list[str], treatment_col: str = "treatment") -> dict:
    """|SMD| per feature between arms, plus its noise level under
    randomization (about sqrt(1/n1 + 1/n0)), so tiny but real differences
    in very large samples are visible."""
    t = df[treatment_col].to_numpy() == 1
    smd = standardized_mean_differences(df[t], df[~t], feature_cols)
    noise = float(np.sqrt(1 / t.sum() + 1 / (~t).sum()))
    return {"smd": smd, "max_smd": max(smd.values()), "smd_noise_sd": noise}
