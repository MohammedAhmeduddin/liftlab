# tests/test_drift_edge_cases.py
"""Edge case tests for drift detection."""
import numpy as np
import pandas as pd
from liftlab.monitoring.drift import compute_psi, DriftReport


def test_drift_report_summary():
    """DriftReport.summary() should return formatted string."""
    report = DriftReport(
        reference_size=1000,
        current_size=1000,
        psi_scores={"f0": 0.15},
        psi_flagged=[],
        mmd_score=0.03,
        mmd_flagged=False,
        drift_detected=False,
        severity="none",
        recommendation="No action",
    )
    summary = report.summary()
    assert "Drift Report" in summary
    assert "1,000" in summary
    assert "NONE" in summary


def test_psi_handles_constant_features():
    """PSI should handle features with zero variance."""
    ref = np.ones(1000)  # constant
    cur = np.ones(1000) + 0.01  # nearly constant
    psi = compute_psi(ref, cur)
    assert not np.isnan(psi)
    assert not np.isinf(psi)


def test_psi_with_small_samples():
    """PSI should work with small sample sizes."""
    ref = np.random.randn(50)
    cur = np.random.randn(50)
    psi = compute_psi(ref, cur, n_bins=5)
    assert psi >= 0
