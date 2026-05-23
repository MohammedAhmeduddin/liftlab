# tests/test_alerts_coverage.py
"""Coverage tests for alerts."""
from liftlab.monitoring.drift import DriftReport
from liftlab.monitoring.alerts import send_drift_alert


def test_send_drift_alert_with_flagged_features():
    """send_drift_alert should format PSI flagged features."""
    report = DriftReport(
        reference_size=5000,
        current_size=5000,
        psi_scores={"f0": 0.30, "f1": 0.25},
        psi_flagged=["f0", "f1"],
        mmd_score=0.12,
        mmd_flagged=True,
        drift_detected=True,
        severity="severe",
        recommendation="Retrain immediately",
    )
    # Should attempt to send but fail gracefully without webhook
    result = send_drift_alert(report, "test_exp")
    assert result is False
