# tests/test_refutation.py
"""Basic sanity tests for causal graph and uplift models."""
import warnings
import numpy as np
import pandas as pd
import pytest
from liftlab.causal.graph import build_causal_model, identify_estimand
from liftlab.causal.uplift import TLearnerUplift, CATEResult


def make_df(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = {f"f{i}": rng.standard_normal(n) for i in range(12)}
    df = pd.DataFrame(data)
    df["treatment"] = rng.binomial(1, 0.5, n)
    df["conversion"] = (
        0.3 * df["treatment"] + 0.2 * df["f0"] + rng.standard_normal(n) * 0.1 > 0
    ).astype(int)
    df["visit"] = rng.binomial(1, 0.3, n)
    df["exposure"] = rng.binomial(1, 0.8, n)
    return df


def test_causal_model_builds():
    df = make_df()
    model = build_causal_model(df)
    assert model is not None


def test_estimand_identified():
    df = make_df()
    model = build_causal_model(df)
    estimand = identify_estimand(model)
    assert estimand is not None
    assert "backdoor" in str(estimand).lower()


def test_tlearner_cate_shape():
    df = make_df()
    feature_cols = [f"f{i}" for i in range(12)]
    tl = TLearnerUplift()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = tl.fit_estimate(df, feature_cols)
    assert isinstance(result, CATEResult)
    assert result.cate.shape == (len(df),)
    assert not np.isnan(result.ate)
    assert result.ate_lower < result.ate_upper
