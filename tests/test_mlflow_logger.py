# tests/test_mlflow_logger.py
"""MLflow logger tests - fast only."""
from liftlab.mlflow_logger import setup_mlflow


def test_setup_mlflow_returns_bool():
    """setup_mlflow should return bool indicating success/failure."""
    result = setup_mlflow()
    assert isinstance(result, bool)
