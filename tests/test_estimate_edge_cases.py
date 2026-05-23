# tests/test_estimate_edge_cases.py
"""Edge case tests for estimation logic."""
from liftlab.api.routes.estimate import _run_estimator
from liftlab.db.models import Experiment
import pandas as pd
import numpy as np


def test_run_estimator_psm():
    """_run_estimator should handle PSM."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "f0": rng.standard_normal(500),
        "f1": rng.standard_normal(500),
        "treatment": rng.binomial(1, 0.5, 500),
        "conversion": rng.binomial(1, 0.1, 500),
    })
    
    exp = Experiment(
        name="test",
        treatment_col="treatment",
        outcome_col="conversion",
        feature_cols=["f0", "f1"],
        estimator="psm",
        sample_frac=1.0,
    )
    
    result = _run_estimator(exp, df)
    assert "ate" in result
    assert "ci_lower" in result
    assert "n_treated" in result
