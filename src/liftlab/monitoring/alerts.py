# src/liftlab/monitoring/alerts.py
"""
Slack alerting for drift events.
Gracefully skips if webhook URL is not configured.
"""
import httpx
from loguru import logger
from liftlab.config import get_settings
from liftlab.monitoring.drift import DriftReport

settings = get_settings()


def send_drift_alert(report: DriftReport, experiment_name: str) -> bool:
    """
    Send a Slack alert when drift is detected.
    Returns True if sent, False if skipped/failed.
    """
    if not settings.slack_webhook_url:
        logger.info("Slack webhook not configured — skipping alert")
        return False

    if not report.drift_detected:
        return False

    emoji = "🔴" if report.severity == "severe" else "🟡"
    message = {
        "text": (
            f"{emoji} *LiftLab Drift Alert* — `{experiment_name}`\n"
            f"*Severity*: {report.severity.upper()}\n"
            f"*MMD Score*: {report.mmd_score:.6f} "
            f"{'(flagged)' if report.mmd_flagged else ''}\n"
            f"*PSI Flagged Features*: "
            f"{', '.join(report.psi_flagged) if report.psi_flagged else 'none'}\n"
            f"*Action*: {report.recommendation}"
        )
    }

    try:
        r = httpx.post(settings.slack_webhook_url, json=message, timeout=5.0)
        if r.status_code == 200:
            logger.info("Drift alert sent to Slack ✅")
            return True
        else:
            logger.warning(f"Slack alert failed: {r.status_code}")
            return False
    except Exception as e:
        logger.warning(f"Slack alert error: {e}")
        return False
