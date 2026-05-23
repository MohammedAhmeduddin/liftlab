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
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "liftlab-causal"

    # AWS S3
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    s3_bucket: str = "liftlab-artifacts"
    s3_region: str = "us-east-1"

    # Drift thresholds
    psi_threshold: float = 0.2
    mmd_threshold: float = 0.05

    # Slack
    slack_webhook_url: str = ""

    # Data paths
    criteo_data_path: str = "data/criteo_uplift.csv"
    olist_data_path: str = "data/olist/"


@lru_cache
def get_settings() -> Settings:
    return Settings()
