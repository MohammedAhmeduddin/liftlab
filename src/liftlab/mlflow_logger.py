# src/liftlab/mlflow_logger.py
"""
MLflow logging utilities for causal estimation runs.
MLflow is optional — if unavailable, estimation continues without logging.
"""
import mlflow
from loguru import logger
from liftlab.config import get_settings

settings = get_settings()


def setup_mlflow() -> bool:
    """
    Configure MLflow. Returns True if reachable, False otherwise.
    Never raises — MLflow is optional infrastructure.
    """
    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        return True
    except Exception as e:
        logger.warning(f"MLflow setup failed (skipping): {e}")
        return False


def log_causal_run(
    experiment_name: str,
    estimator: str,
    ate: float,
    ci_lower: float,
    ci_upper: float,
    n_treated: int,
    n_control: int,
    p_value: float | None = None,
    shap_importance: dict | None = None,
    extra_params: dict | None = None,
    extra_metrics: dict | None = None,
) -> str | None:
    """
    Log a causal estimation run to MLflow.
    Returns mlflow_run_id if successful, None if MLflow is unavailable.
    Never raises — estimation results are always returned regardless.
    """
    if not setup_mlflow():
        logger.warning("MLflow unavailable — skipping run logging")
        return None

    try:
        with mlflow.start_run(run_name=f"{experiment_name}__{estimator}") as run:
            # Parameters
            mlflow.log_params({
                "experiment_name": experiment_name,
                "estimator": estimator,
                "n_treated": n_treated,
                "n_control": n_control,
                **(extra_params or {}),
            })

            # Core causal metrics
            metrics = {
                "ate": ate,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "ci_width": ci_upper - ci_lower,
                **(extra_metrics or {}),
            }
            if p_value is not None:
                metrics["p_value"] = p_value
            mlflow.log_metrics(metrics)

            # SHAP importance as individual metrics
            if shap_importance:
                shap_metrics = {
                    f"shap_{feat}": float(imp)
                    for feat, imp in shap_importance.items()
                }
                mlflow.log_metrics(shap_metrics)

            run_id = run.info.run_id
            logger.info(f"MLflow run logged | run_id={run_id} | ATE={ate:.6f}")
            return run_id

    except Exception as e:
        logger.warning(f"MLflow logging failed (skipping): {e}")
        return None
