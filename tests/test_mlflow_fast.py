"""MLflow logging: fast failure when the server is down, real logging to a file store."""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

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


def _serve(status: int):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.end_headers()
            self.wfile.write(b"OK")

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_open_port_that_is_not_mlflow_is_unreachable(monkeypatch):
    """e.g. macOS AirPlay Receiver on :5000 answers 403; must not stall."""
    server = _serve(403)
    try:
        uri = f"http://127.0.0.1:{server.server_port}"
        monkeypatch.setattr(mlflow_logger.settings, "mlflow_tracking_uri", uri)
        start = time.perf_counter()
        assert mlflow_logger.setup_mlflow() is False
        assert mlflow_logger.log_causal_run("x", "aipw", 0.1, 0.0, 0.2, 10, 5) is None
        assert time.perf_counter() - start < 5
    finally:
        server.shutdown()


def test_healthy_server_is_reachable():
    server = _serve(200)
    try:
        assert mlflow_logger._server_reachable(f"http://127.0.0.1:{server.server_port}/")
    finally:
        server.shutdown()
