# tests/test_uplift_edge_cases.py
"""Edge case tests for uplift models."""
import warnings
import numpy as np
import pandas as pd
from liftlab.causal.uplift import CATEResult


def test_cate_result_segment_effects_handles_few_bins():
    """segment_effects should handle n_bins > unique values."""
    df = pd.DataFrame({
        "f0": [1, 1, 2, 2, 3, 3],
        "treatment": [0, 1, 0, 1, 0, 1],
        "conversion": [0, 1, 0, 1, 1, 1],
    })
    
    cate = np.array([0.1, 0.2, 0.15, 0.25, 0.3, 0.35])
    
    result = CATEResult(
        method="test",
        cate=cate,
        ate=cate.mean(),
        ate_lower=0.0,
        ate_upper=0.5,
        feature_cols=["f0"],
    )
    
    # Should not crash with n_bins=10 when only 3 unique values
    segments = result.segment_effects(df, col="f0", n_bins=10)
    assert len(segments) <= 3
    assert "lift_vs_average" in segments.columns
