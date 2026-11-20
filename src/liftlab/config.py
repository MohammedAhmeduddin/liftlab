# src/liftlab/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # App
    app_name: str = "LiftLab"
    debug: bool = False
    log_level: str = "INFO"

    # Database
    database_url: str = "postgresql://liftlab:liftlab@localhost:5432/liftlab"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5001"
    mlflow_experiment_name: str = "liftlab-causal"


    # Drift thresholds
    psi_threshold: float = 0.2
    mmd_threshold: float = 0.05

    # Slack
    slack_webhook_url: str = ""

    # Data paths
    criteo_data_path: str = "data/criteo_uplift.csv"


@lru_cache
def get_settings() -> Settings:
    return Settings()
