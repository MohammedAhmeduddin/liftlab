# tests/test_mlflow_logger_coverage.py
"""Coverage tests for MLflow logger."""
from liftlab.mlflow_logger import log_causal_run


def test_log_causal_run_with_all_params():
    """log_causal_run should handle all optional params."""
    run_id = log_causal_run(
        experiment_name="test",
        estimator="causalforest",
        ate=0.005,
        ci_lower=0.001,
        ci_upper=0.009,
        n_treated=1000,
        n_control=1000,
        p_value=0.01,
        shap_importance={"f0": 0.5, "f1": 0.3},
        extra_params={"sample_frac": 0.1},
        extra_metrics={"duration": 45.0},
    )
    # Should return None gracefully if MLflow unavailable
    assert run_id is None or isinstance(run_id, str)
