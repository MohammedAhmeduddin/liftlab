# tests/test_estimate_full_coverage.py
"""Full coverage tests for estimation endpoint."""
import warnings
import pandas as pd
import numpy as np
from liftlab.api.routes.estimate import _run_estimator
from liftlab.db.models import Experiment


def make_test_df(n=500):
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "f0": rng.standard_normal(n),
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
        "f3": rng.standard_normal(n),
        "treatment": rng.binomial(1, 0.5, n),
        "conversion": rng.binomial(1, 0.1, n),
    })


def test_run_estimator_tlearner():
    """Test T-Learner path."""
    df = make_test_df()
    exp = Experiment(
        name="test_tl",
        treatment_col="treatment",
        outcome_col="conversion",
        feature_cols=["f0", "f1", "f2", "f3"],
        estimator="tlearner",
        sample_frac=1.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _run_estimator(exp, df)
    
    assert "ate" in result
    assert "shap_importance" in result
    assert "segment_effects" in result


def test_run_estimator_xlearner():
    """Test X-Learner path."""
    df = make_test_df()
    exp = Experiment(
        name="test_xl",
        treatment_col="treatment",
        outcome_col="conversion",
        feature_cols=["f0", "f1", "f2", "f3"],
        estimator="xlearner",
        sample_frac=1.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _run_estimator(exp, df)
    
    assert result["ate"] is not None
    assert result["n_treated"] > 0
    assert result["n_control"] > 0


def test_run_estimator_causalforest():
    """Test CausalForest path."""
    df = make_test_df(n=300)
    exp = Experiment(
        name="test_cf",
        treatment_col="treatment",
        outcome_col="conversion",
        feature_cols=["f0", "f1", "f2", "f3"],
        estimator="causalforest",
        sample_frac=1.0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = _run_estimator(exp, df)
    
    assert "ate" in result
    assert "ci_lower" in result
    assert "ci_upper" in result
    assert result["segment_effects"] is not None or result["segment_effects"] == []


def test_run_estimator_unknown_raises():
    """Unknown estimator should raise ValueError."""
    df = make_test_df()
    exp = Experiment(
        name="test",
        treatment_col="treatment",
        outcome_col="conversion",
        feature_cols=["f0"],
        estimator="not_real",
        sample_frac=1.0,
    )
    
    try:
        _run_estimator(exp, df)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Unknown estimator" in str(e)
