# tests/test_main_edge_cases.py
"""Edge case tests for API main."""
from fastapi.testclient import TestClient
from liftlab.api.main import app

client = TestClient(app)


def test_health_database_status():
    """Health endpoint should show database status."""
    response = client.get("/health")
    data = response.json()
    assert "database" in data
    assert data["database"] in ["healthy", "unhealthy"] or "unhealthy:" in data["database"]


def test_health_mlflow_status():
    """Health endpoint should show MLflow status."""
    response = client.get("/health")
    data = response.json()
    assert "mlflow" in data
    # Should be unavailable since MLflow not running in tests
    assert "unavailable" in data["mlflow"] or data["mlflow"] == "healthy"
