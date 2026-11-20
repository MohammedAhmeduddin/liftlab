"""MLflow logging: fast failure when the server is down, real logging to a file store."""

import time

import mlflow

from liftlab import mlflow_logger


def test_unreachable_server_fails_fast(monkeypatch):
    monkeypatch.setattr(mlflow_logger.settings, "mlflow_tracking_uri", "http://127.0.0.1:9")
    start = time.perf_counter()
    assert mlflow_logger.setup_mlflow() is False
    assert mlflow_logger.log_causal_run("x", "aipw", 0.1, 0.0, 0.2, 10, 5) is None
    assert time.perf_counter() - start < 5


def test_logs_run_to_local_sqlite_store(monkeypatch, tmp_path):
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setattr(mlflow_logger.settings, "mlflow_tracking_uri", uri)
    run_id = mlflow_logger.log_causal_run(
        "exp", "causalforest", ate=0.001, ci_lower=None, ci_upper=None, n_treated=85,
        n_control=15, p_value=0.01, shap_importance={"f8": 0.4},
        extra_metrics={"qini_coefficient": 0.12})
    assert run_id
    run = mlflow.get_run(run_id)
    assert run.data.metrics["ate"] == 0.001
    assert "ci_width" not in run.data.metrics
    assert run.data.metrics["importance_f8"] == 0.4
