# tests/test_uplift_error_paths.py
"""Error path coverage for uplift models."""
import warnings
import numpy as np
import pandas as pd
import pytest
from liftlab.causal.uplift import CausalForestUplift


def test_causalforest_feature_importance_before_fit_raises():
    """feature_importance should raise if called before fit."""
    cf = CausalForestUplift(n_estimators=48)
    with pytest.raises(RuntimeError, match="Call fit_estimate"):
        cf.feature_importance(["f0", "f1"])


def test_causalforest_handles_shap_failure_gracefully():
    """Should handle SHAP computation failure gracefully."""
    # SHAP might fail on very small datasets or edge cases
    # The code has try/except — verify it doesn't crash
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "f0": rng.standard_normal(100),
        "treatment": rng.binomial(1, 0.5, 100),
        "conversion": rng.binomial(1, 0.1, 100),
    })

    cf = CausalForestUplift(n_estimators=48)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cf.fit_estimate(df, ["f0"])

    # Should complete even if SHAP fails
    assert result.method == "CausalForest"
    # shap_values might be None if computation failed
    assert result.shap_values is None or isinstance(result.shap_values, np.ndarray)
