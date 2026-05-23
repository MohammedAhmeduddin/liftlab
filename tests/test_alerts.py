# tests/test_alerts.py
"""Drift alert tests."""
from liftlab.monitoring.drift import DriftReport
from liftlab.monitoring.alerts import send_drift_alert


def test_send_drift_alert_skips_when_no_webhook():
    """Should skip gracefully when Slack webhook not configured."""
    report = DriftReport(
        reference_size=1000,
        current_size=1000,
        psi_scores={"f0": 0.25},
        psi_flagged=["f0"],
        mmd_score=0.08,
        mmd_flagged=True,
        drift_detected=True,
        severity="severe",
        recommendation="Retrain immediately",
    )
    result = send_drift_alert(report, "test_experiment")
    assert result is False


def test_send_drift_alert_skips_when_no_drift():
    """Should skip when drift not detected."""
    report = DriftReport(
        reference_size=1000,
        current_size=1000,
        drift_detected=False,
        severity="none",
        recommendation="No action",
    )
    result = send_drift_alert(report, "test_experiment")
    assert result is False
