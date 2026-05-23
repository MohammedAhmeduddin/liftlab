# tests/test_estimators.py
"""Unit tests for causal estimators."""
import warnings
import numpy as np
import pandas as pd
import pytest
from liftlab.causal.estimators import PropensityScoreMatching, DifferenceInDifferences, CausalEstimate


def make_synthetic_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 4))
    propensity = 1 / (1 + np.exp(-X[:, 0]))
    T = rng.binomial(1, propensity)
    Y = 0.5 * T + X[:, 0] * 0.3 + rng.standard_normal(n) * 0.1
    df = pd.DataFrame(X, columns=["f0", "f1", "f2", "f3"])
    df["treatment"] = T
    df["conversion"] = (Y > 0.5).astype(int)
    df["time_period"] = rng.binomial(1, 0.5, n)
    return df


@pytest.fixture
def synthetic_df():
    return make_synthetic_data()


def test_psm_returns_causal_estimate(synthetic_df):
    psm = PropensityScoreMatching(caliper=0.1, n_bootstrap=50)
    result = psm.estimate(synthetic_df, ["f0", "f1", "f2", "f3"])
    assert isinstance(result, CausalEstimate)
    assert result.method == "PSM"
    assert result.n_treated > 0
    assert result.n_control > 0
    assert result.ci_lower < result.ci_upper
    assert 0 <= result.p_value <= 1


def test_psm_ate_direction(synthetic_df):
    """True ATE is positive (~0.5) — PSM should recover positive direction."""
    psm = PropensityScoreMatching(caliper=0.1, n_bootstrap=50)
    result = psm.estimate(synthetic_df, ["f0", "f1", "f2", "f3"])
    assert result.ate > 0, f"Expected positive ATE, got {result.ate}"


def test_did_returns_causal_estimate(synthetic_df):
    did = DifferenceInDifferences()
    result = did.estimate(
        synthetic_df,
        outcome_col="conversion",
        treatment_col="treatment",
        time_col="time_period",
    )
    assert isinstance(result, CausalEstimate)
    assert result.method == "DiD"
    assert result.ci_lower < result.ci_upper
    assert "r_squared" in result.metadata


def test_did_with_covariates(synthetic_df):
    did = DifferenceInDifferences()
    result = did.estimate(
        synthetic_df,
        outcome_col="conversion",
        treatment_col="treatment",
        time_col="time_period",
        covariate_cols=["f0", "f1"],
    )
    assert result.ate is not None
    assert not np.isnan(result.ate)
