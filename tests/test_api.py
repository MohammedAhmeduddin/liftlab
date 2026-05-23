# tests/test_api.py
"""API endpoint tests."""
import pytest
from fastapi.testclient import TestClient
from liftlab.api.main import app
from liftlab.db.session import SessionLocal, Base, engine
from liftlab.db.models import Experiment

client = TestClient(app)


@pytest.fixture(scope="module")
def setup_db():
    """Create tables before tests, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


def test_health_endpoint(setup_db):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "database" in data
    assert "mlflow" in data


def test_create_experiment(setup_db):
    payload = {
        "name": "test_exp_001",
        "description": "Test experiment",
        "treatment_col": "treatment",
        "outcome_col": "conversion",
        "feature_cols": ["f0", "f1", "f2"],
        "estimator": "causalforest",
        "sample_frac": 0.01,
    }
    response = client.post("/experiments/", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "test_exp_001"
    assert data["status"] == "pending"


def test_list_experiments(setup_db):
    response = client.get("/experiments/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_create_duplicate_experiment_fails(setup_db):
    payload = {
        "name": "test_exp_duplicate",
        "treatment_col": "treatment",
        "outcome_col": "conversion",
        "feature_cols": ["f0"],
        "estimator": "tlearner",
        "sample_frac": 0.01,
    }
    # First creation succeeds
    response = client.post("/experiments/", json=payload)
    assert response.status_code == 201

    # Duplicate fails
    response = client.post("/experiments/", json=payload)
    assert response.status_code == 409


def test_get_experiment_by_id(setup_db):
    # Create an experiment
    payload = {
        "name": "test_exp_get_by_id",
        "treatment_col": "treatment",
        "outcome_col": "conversion",
        "feature_cols": ["f0"],
        "estimator": "xlearner",
        "sample_frac": 0.01,
    }
    create_response = client.post("/experiments/", json=payload)
    exp_id = create_response.json()["id"]

    # Get it
    response = client.get(f"/experiments/{exp_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "test_exp_get_by_id"


def test_get_nonexistent_experiment_404(setup_db):
    fake_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/experiments/{fake_id}")
    assert response.status_code == 404


def test_delete_experiment(setup_db):
    payload = {
        "name": "test_exp_to_delete",
        "treatment_col": "treatment",
        "outcome_col": "conversion",
        "feature_cols": ["f0"],
        "estimator": "psm",
        "sample_frac": 0.01,
    }
    create_response = client.post("/experiments/", json=payload)
    exp_id = create_response.json()["id"]

    # Delete
    response = client.delete(f"/experiments/{exp_id}")
    assert response.status_code == 204

    # Verify deleted
    get_response = client.get(f"/experiments/{exp_id}")
    assert get_response.status_code == 404
