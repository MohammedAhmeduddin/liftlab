# tests/test_uplift_models.py
"""Additional uplift model tests."""
import warnings
import numpy as np
import pandas as pd
from liftlab.causal.uplift import XLearnerUplift, CausalForestUplift


def make_df(n=1000):
    rng = np.random.default_rng(42)
    data = {f"f{i}": rng.standard_normal(n) for i in range(4)}
    df = pd.DataFrame(data)
    df["treatment"] = rng.binomial(1, 0.5, n)
    df["conversion"] = (0.3 * df["treatment"] + rng.standard_normal(n) * 0.1 > 0).astype(int)
    return df


def test_xlearner_returns_cate():
    df = make_df()
    xl = XLearnerUplift()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = xl.fit_estimate(df, ["f0", "f1", "f2", "f3"])
    assert result.cate.shape == (len(df),)
    assert not np.isnan(result.ate)


def test_causalforest_returns_cate():
    df = make_df()
    cf = CausalForestUplift(n_estimators=48)  # ← changed from 50 to 48
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cf.fit_estimate(df, ["f0", "f1", "f2", "f3"])
    assert result.cate.shape == (len(df),)
    assert result.method == "CausalForest"


def test_causalforest_feature_importance():
    df = make_df()
    cf = CausalForestUplift(n_estimators=48)  # ← changed from 50 to 48
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cf.fit_estimate(df, ["f0", "f1", "f2", "f3"])
    imp = cf.feature_importance(["f0", "f1", "f2", "f3"])
    assert len(imp) == 4
    assert "shap_importance" in imp.columns


def test_cate_result_segment_effects():
    df = make_df()
    cf = CausalForestUplift(n_estimators=48)  # ← changed from 50 to 48
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cf.fit_estimate(df, ["f0", "f1", "f2", "f3"])
    segments = result.segment_effects(df, col="f0", n_bins=3)
    assert len(segments) <= 3
    assert "mean_cate" in segments.columns
    assert "lift_vs_average" in segments.columns
